"""Collect the paper's evaluation numbers from result files, and check the text against them."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import j0

from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
from neural_rx.benchmark.scoring import paired_summary

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep"
ORIGINAL = ROOT/"experiments/20261021_package_layout_validation/results/v2_confirmation_005000/results.json"
FRESH = ROOT/"experiments/20261021_current_code_reproduction/results/confirmation_005000_seed86000000/results.json"
FALLBACK = ROOT/"experiments/20261106_crc_fallback/results"
CONTROLLER = ROOT/"experiments/20261112_early_retransmission/results"
RETRAINED = ROOT/"experiments/20261113_retrained_k2_baseline/results/summary.json"
RETRAINED_TEST = ROOT/"experiments/20261113_retrained_k2_baseline/results/test"
NO_IDD = ROOT/"experiments/20261114_no_idd_ablation/results"
SCORING_START = 16
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
RECEIVERS = ("baseline_lslin_lmmse", "baseline_lmmse_kbest", "baseline_perf_csi_kbest", "frozen_cold_k2",
             "frozen_cold_k3", "frozen_cold_k8", "candidate_reset", "candidate_persistent")
OPERATING = (*CORE, "close_los", "near_far", "ray_traced_munich")
PAIRED = ("candidate_persistent", "candidate_reset", "frozen_cold_k8")


def channel_power_table():
    """TBLER per UE-episode, grouped by mean effective channel power, at each condition's highest SNR point."""
    rows = {}
    for condition in CORE:
        values = []
        for path in sorted((SWEEP/"results/sweep_high"/condition/"candidate_persistent").glob("episode_*.npz")):
            with np.load(path, allow_pickle=False) as data, \
                    np.load(str(path).replace("candidate_persistent", "baseline_perf_csi_kbest"),
                            allow_pickle=False) as perfect:
                active, ids = data["active"][SCORING_START:], data["ue_ids"][SCORING_START:]
                power = data["channel_power"][SCORING_START:]
                for ue in np.unique(ids):
                    mask = (ids == ue) & active
                    if mask.any():
                        values.append((10*np.log10(power[mask].mean()), data["errors"][-1, SCORING_START:][mask].mean(),
                                       perfect["errors"][-1, SCORING_START:][mask].mean()))
        rows[condition] = values
    return rows


def pooled_over_core():
    """Pooled TBLER, differences, and the paired criterion over the conditions reported in the paper."""
    pooled, differences, criteria = {}, {}, {}
    for name, directory in (("original", ORIGINAL.parent), ("fresh", FRESH.parent)):
        data = dict(np.load(directory/"errors.npz", allow_pickle=False))
        protocol = json.loads((directory/"protocol.json").read_text())
        order = protocol["stages"]["development"]["condition_names"]
        indices = [order.index(condition) for condition in CORE]
        rates = {}
        for method in PAIRED:
            mask = data["active"][indices][:, :, SCORING_START:]
            failures = data[f"errors_{method}"][indices, 0][:, :, SCORING_START:] & mask
            rates[method] = float(failures.sum()/mask.sum())
        pooled[name] = rates
        differences[name] = {right: 100*(rates["candidate_persistent"]-rates[right]) for right in PAIRED[1:]}
        protocol["core_conditions"] = list(CORE)
        rng = np.random.RandomState(protocol["statistics"]["bootstrap_seed"])
        summaries = {right: paired_summary([data], "candidate_persistent", right, protocol, "development",
                                           protocol["statistics"]["bootstrap_samples"], rng)
                     for right in PAIRED[1:]}
        criteria[name] = {right: bool(summary["ci95"][1] < 0) for right, summary in summaries.items()}
    return {"core": pooled, "differences_pp": differences, "criteria": criteria}


def memory_gain_by_speed():
    """Memory gain at TBLER=10% for UMi, per Doppler stratum, against effective SNR (SNR + channel power in dB)."""
    edges = np.arange(-6., 14.)
    centers = edges[:-1]+.5

    def required(method, stratum):
        failures, blocks = np.zeros(len(centers)), np.zeros(len(centers))
        for directory, first in (("sweep", 0), ("sweep_high", 1)):
            for path in sorted((SWEEP/"results"/directory/"natural_umi"/method).glob("episode_*.npz")):
                if int(path.stem.split("_")[1]) % 3 != stratum:
                    continue
                with np.load(path, allow_pickle=False) as data:
                    active = data["active"][SCORING_START:]
                    power = 10*np.log10(data["channel_power"][SCORING_START:])
                    for point in range(first, len(data["snr_db"])):
                        index = np.floor(data["snr_db"][point]+power-edges[0]).astype(int)
                        valid = (index >= 0) & (index < len(centers)) & active
                        np.add.at(failures, index[valid], data["errors"][point, SCORING_START:][valid].astype(float))
                        np.add.at(blocks, index[valid], 1)
        rates = failures/np.maximum(blocks, 1)
        return snr_at_target(centers, decreasing_isotonic(rates, np.maximum(blocks, 1)), .1)

    gains, limits = {}, {}
    for stratum, name in enumerate(("0-50", "50-175", "175-400")):
        none = required("candidate_reset", stratum)
        gains[name] = none-required("candidate_persistent", stratum)
        limits[name] = none-required("baseline_perf_csi_kbest", stratum)
    return {"memory_gain_db": gains, "gap_to_perfect_csi_db": limits}


def effective_snr(width=1.):
    """Required effective SNR (SNR + effective channel power of the UE in dB) at TBLER=10% per condition and receiver,
    pooled over all sequences of the sweep without power control, in bins of `width` dB as for the memory gain.
    Channel power comes from the stored files of the proposed NRX; every receiver decoded the same episodes."""
    edges = np.arange(-10., 30.+width/2, width)
    centers = edges[:-1]+width/2
    sources = {method: SWEEP/"results" for method in RECEIVERS}
    sources["retrained_k2"] = RETRAINED_TEST
    sources["candidate_persistent_no_idd"] = sources["candidate_reset_no_idd"] = NO_IDD/"test"
    result = {}
    for condition in CORE:
        result[condition] = {}
        for method, base in sources.items():
            failures, blocks = np.zeros(len(centers)), np.zeros(len(centers))
            complete = True
            for directory, first in (("sweep", 0), ("sweep_high", 1)):  # +3 dB appears in both sweeps
                for path in sorted((SWEEP/"results"/directory/condition/"candidate_persistent").glob("episode_*.npz")):
                    target = base/directory/condition/method/path.name
                    if not target.is_file():
                        complete = False
                        continue
                    with np.load(path, allow_pickle=False) as reference, np.load(target, allow_pickle=False) as data:
                        if not np.array_equal(reference["snr_db"], data["snr_db"]):
                            raise ValueError(f"{target}: SNR grid differs from the proposed NRX")
                        active = reference["active"][SCORING_START:]
                        power = 10*np.log10(reference["channel_power"][SCORING_START:])
                        for point in range(first, len(data["snr_db"])):
                            index = np.floor((data["snr_db"][point]+power-edges[0])/width).astype(int)
                            valid = (index >= 0) & (index < len(centers)) & active
                            np.add.at(failures, index[valid], data["errors"][point, SCORING_START:][valid].astype(float))
                            np.add.at(blocks, index[valid], 1)
            if not complete:
                continue
            rates = failures/np.maximum(blocks, 1)
            fitted = decreasing_isotonic(rates, np.maximum(blocks, 1))
            occupied = blocks >= 100
            result[condition][method] = {
                "required_db": float(snr_at_target(centers, fitted, .1)),
                # Width of the TBLER drop from 80% to 5% on the fitted curve, in dB.
                "drop_80_to_5_db": float(snr_at_target(centers, fitted, .05)-snr_at_target(centers, fitted, .8)),
                "centers": centers[occupied].tolist(), "tbler": rates[occupied].tolist()}
    return result


def no_idd():
    """Proposed NRX without IDD (with and without memory) from experiments/20261114_no_idd_ablation."""
    summary = NO_IDD/"summary.json"
    complexity = NO_IDD/"complexity.json"
    return {"summary": json.loads(summary.read_text()) if summary.is_file() else None,
            "complexity": json.loads(complexity.read_text()) if complexity.is_file() else None}


def memory_effect():
    """TBs decoded only with memory, only with memory emptied, and by neither, over all sweep SNR points."""
    counts = {"rescued": 0, "broke": 0, "both_failed": 0}
    for condition in CORE:
        for directory, first in (("sweep", 0), ("sweep_high", 1)):  # +3 dB appears in both sweeps
            for path in sorted((SWEEP/"results"/directory/condition/"candidate_persistent").glob("episode_*.npz")):
                with np.load(path, allow_pickle=False) as memory, \
                        np.load(str(path).replace("candidate_persistent", "candidate_reset"),
                                allow_pickle=False) as reset:
                    active = memory["active"][SCORING_START:]
                    with_memory = memory["errors"][first:, SCORING_START:] & active
                    without = reset["errors"][first:, SCORING_START:] & active
                    counts["rescued"] += int((~with_memory & without).sum())
                    counts["broke"] += int((with_memory & ~without).sum())
                    counts["both_failed"] += int((with_memory & without).sum())
    return counts


def validation():
    """Restored baselines against the curves stored in NVlabs/neural_rx (results/nrx_rt_results)."""
    runs = {name: json.loads((SWEEP/"validation"/f"{name}.json").read_text())
            for name in ("doubletdl", "doubletdl_perfect_recheck", "doubletdl_lmmse_final_covariance")}
    return {"ls_lmmse_3db": runs["doubletdl"]["baseline_lslin_lmmse"]["3.0"]["tbler"],
            "ls_lmmse_3db_upstream": .38864,
            "perfect_kbest_3db": runs["doubletdl_perfect_recheck"]["baseline_perf_csi_kbest"]["3.0"]["tbler"],
            "perfect_kbest_3db_upstream": .00723,
            "lmmse_kbest_3db": runs["doubletdl_lmmse_final_covariance"]["baseline_lmmse_kbest"]["3.0"]["tbler"],
            "lmmse_kbest_3db_upstream": .0402}


def network():
    """Results quoted in the Network Decisions section: CRC fallback, persistence, power control, controller."""
    fallback = json.loads((FALLBACK/"fallback.json").read_text())
    kbest = json.loads((FALLBACK/"latency.json").read_text())
    result = {"fallback": {c: {"gain_db": {k: v["gain_db_from_rounded"] for k, v in fallback[c].items()
                                           if isinstance(v, dict) and "gain_db_from_rounded" in v},
                               "unfixable_share": fallback[c]["unfixable_share_of_failures"],
                               "unfixable_also_perfect_csi": fallback[c]["unfixable_also_failed_by_perfect_csi"],
                               "unfixable_power_db": fallback[c]["median_power_unfixable_minus_all_db"]}
                           for c in CORE},
              "fallback_latency_median_ms": {m: v["median_ms"] for m, v in kbest.items()}}
    for name, path in (("controller", "controller.json"), ("controller_power_control", "controller_pc.json"),
                       ("receivers_power_control", "fallback_pc.json")):
        if (CONTROLLER/path).is_file():
            result[name] = json.loads((CONTROLLER/path).read_text())
    if (CONTROLLER/"evidence_pc").is_dir():
        result["power_changed_since_newest_entry"] = {c: power_changed(CONTROLLER/"evidence_pc"/c) for c in CORE}
    return result


def power_changed(directory, expiry=8, first=16):
    """Share of TBs 17-64 with a memory entry whose UE's transmit power changed since that newest entry."""
    changed = total = 0
    for path in sorted(directory.glob("episode_*.npz")):
        with np.load(path) as data:
            active, ids, gain = data["active"], data["ue_ids"], data["power_gain_db"]
        newest = {}
        for tb, pos in zip(*np.nonzero(active)):
            ue = int(ids[tb, pos])
            if tb >= first and ue in newest and tb-newest[ue][0] <= expiry:
                total += 1
                changed += abs(gain[tb, pos]-newest[ue][1]) > 1e-6
            newest[ue] = (tb, gain[tb, pos])
    return changed/total


def derived(numbers):
    """Differences quoted in the text, so that each one is checked rather than computed by hand."""
    snr = numbers["snr_at_10pct"]
    values = {}
    for condition in CORE:
        proposed = snr[condition]["candidate_persistent"]
        for method in RECEIVERS[:-1]:
            if snr[condition][method] is not None:
                values[f"{condition}/{method}"] = round(round(snr[condition][method], 2)-round(proposed, 2), 2)
    latency = numbers["latency_median_ms"]
    # Loss of cutting the Large NRX from N_it=8 to 2, quoted in the abstract, introduction, and related work.
    cut = {condition: round(round(snr[condition]["frozen_cold_k2"], 2)-round(snr[condition]["frozen_cold_k8"], 2), 2)
           for condition in CORE}
    # Same differences with power control (fallback_pc.json stores the required SNRs already rounded).
    controlled = numbers["network"].get("receivers_power_control")
    controlled_gap = {} if controlled is None else {
        f"{condition}/{method}": round(value-controlled[condition]["required_snr_db"]["candidate_persistent"], 2)
        for condition in CORE for method, value in controlled[condition]["required_snr_db"].items()
        if method != "candidate_persistent" and value is not None}
    # Retrained two-iteration baseline: its gain from retraining and the proposed NRX's remaining lead.
    retrained = numbers.get("retrained_k2")
    retrained_gap = {} if retrained is None else {
        f"{setting}/{condition}/{name}": round(round(row[a], 2)-round(row[b], 2), 2)
        for setting, rows in retrained.items() if setting != "checkpoint" for condition, row in rows.items()
        for name, a, b in (("narrow", "frozen_cold_k2", "retrained_k2"), ("lead", "retrained_k2", "candidate_persistent"),
                           ("no_memory_lead", "retrained_k2", "candidate_reset"))}
    # Jakes correlation at 400 Hz between DMRS symbols (symbols 2 and 11 of each 0.5 ms slot with 14 symbols):
    # previous slot's last to current slot's first (5 symbols), within a slot (9), and one slot apart (14).
    symbol_s = .5e-3/14
    correlation = {f"{gap}_symbols": float(j0(2*np.pi*400.*gap*symbol_s)) for gap in (5, 9, 14)}
    # Multiplications per slot (latency section) and effective-SNR differences (TBLER section).
    cost = (numbers.get("no_idd") or {}).get("complexity")
    multiplications = {} if cost is None else {
        "cgnn_iteration_in_1e9": cost["cgnn_iteration"]["flops"]/2/1e9,
        "filter_in_1e7": cost["channel_filter_counted"]["real_mult"]/1e7,
        "idd_in_1e8": (cost["idd_detector"]["real_mult"]+cost["idd_combining"]["real_mult"])/1e8,
        "share_of_iteration": (cost["channel_filter_counted"]["real_mult"]+cost["idd_detector"]["real_mult"]
                               +cost["idd_combining"]["real_mult"])/(cost["cgnn_iteration"]["flops"]/2)}
    effective = numbers.get("effective_snr", {})
    effective_gap = {f"{condition}/{method}": rows[method]["required_db"]-rows["candidate_persistent"]["required_db"]
                     for condition, rows in effective.items() for method in rows if method != "candidate_persistent"}
    if multiplications:
        multiplications["filter_share_of_iteration"] = (cost["channel_filter_counted"]["real_mult"]
                                                         / (cost["cgnn_iteration"]["flops"]/2))
    # Without IDD: memory gain (nominal and effective SNR) and the lead over the retrained baseline.
    plain = (numbers.get("no_idd") or {}).get("summary")
    without_idd = {} if plain is None else {
        condition: {
            "memory_gain_db": round(round(plain[condition]["candidate_reset_no_idd"]["snr_at_10pct_db"], 2)
                                    - round(plain[condition]["candidate_persistent_no_idd"]["snr_at_10pct_db"], 2), 2),
            "lead_over_retrained_db": round(round(plain[condition]["stored"]["retrained_k2"], 2)
                                            - round(plain[condition]["candidate_persistent_no_idd"]["snr_at_10pct_db"], 2), 2),
            "no_memory_lead_over_retrained_db": round(
                round(plain[condition]["stored"]["retrained_k2"], 2)
                - round(plain[condition]["candidate_reset_no_idd"]["snr_at_10pct_db"], 2), 2),
            "idd_gain_db": round(round(plain[condition]["candidate_persistent_no_idd"]["snr_at_10pct_db"], 2)
                                 - round(snr[condition]["candidate_persistent"], 2), 2),
            "effective_memory_gain_db": (effective[condition]["candidate_reset_no_idd"]["required_db"]
                                         - effective[condition]["candidate_persistent_no_idd"]["required_db"])
            if "candidate_reset_no_idd" in effective.get(condition, {}) else None}
        for condition in CORE}
    retrained_cut = {} if retrained is None else {
        condition: round(round(row["retrained_k2"], 2)-round(row["frozen_cold_k8"], 2), 2)
        for condition, row in retrained["no_power_control"].items()}
    return {"snr_gap_db": values,
            "multiplications": multiplications,
            "effective_snr_gap_db": effective_gap,
            "without_idd": without_idd,
            "retrained_iteration_cut_db": retrained_cut,
            "snr_gap_power_control_db": controlled_gap,
            "retrained_k2_gap_db": retrained_gap,
            "dmrs_correlation_400hz": correlation,
            "iteration_cut_db": cut,
            "latency_saving_vs_large_percent": 100*(1-latency["candidate_persistent"]/latency["frozen_cold_k8"]),
            "latency_added_vs_nit2_ms": latency["candidate_persistent"]-latency["frozen_cold_k2"],
            "latency_per_cgnn_iteration_ms": (latency["frozen_cold_k8"]-latency["frozen_cold_k2"])/6,
            "rescue_ratio": numbers["memory_effect"]["rescued"]/numbers["memory_effect"]["broke"]}


def collect():
    summary = json.loads((SWEEP/"results/summary.json").read_text())
    latency = json.loads((SWEEP/"results/latency.json").read_text())
    original, fresh = (json.loads(path.read_text()) for path in (ORIGINAL, FRESH))
    numbers = {"snr_at_10pct": {c: {m: summary[c][m]["snr_at_10pct_db"] for m in RECEIVERS} for c in CORE},
               "tbler": {c: {m: dict(zip(summary[c][m]["snr_db"], summary[c][m]["tbler"], strict=True))
                             for m in RECEIVERS} for c in CORE},
               "blocks": {c: summary[c]["candidate_persistent"]["blocks"] for c in CORE},
               "crc_disagreements": sum(summary[c][m]["crc_disagreements"] for c in CORE for m in RECEIVERS),
               "overlap_points": summary["overlap_points"],
               "latency_median_ms": {m: value["median_ms"] for m, value in latency.items()}}
    numbers["operating"] = {
        name: {method: {condition: result["summary"][method]["per_condition_tbler"].get(condition)
                        for condition in OPERATING}
               for method in PAIRED} for name, result in (("original", original), ("fresh", fresh))}
    numbers.update(pooled_over_core())
    power = channel_power_table()
    groups = {}
    for label, low, high in (("below_minus_10", -np.inf, -10), ("minus_10_to_0", -10, 0), ("above_0", 0, np.inf)):
        selected = [row for rows in power.values() for row in rows if low <= row[0] < high]
        groups[label] = {"count": len(selected), "proposed": float(np.mean([r[1] for r in selected])),
                         "perfect": float(np.mean([r[2] for r in selected]))}
    numbers["channel_power"] = groups
    numbers["channel_power_range_db"] = [float(min(r[0] for rows in power.values() for r in rows)),
                                         float(max(r[0] for rows in power.values() for r in rows))]
    numbers["memory_effect"] = memory_effect()
    numbers["doppler"] = memory_gain_by_speed()
    numbers["validation"] = validation()
    numbers["network"] = network()
    run = ROOT/"experiments/20261019_learned_blend_joint_training_6000"
    parent = json.loads((run/"inputs/training_recipe.json").read_text())["initializer_parent_updates"]
    selected = json.loads((ROOT/json.loads((run/"results/best.json").read_text())["checkpoint"]/"manifest.json")
                          .read_text())["update"]
    numbers["training_updates"] = {"parent": parent, "final_run": selected, "total": parent+selected,
                                   "total_in_1e4": (parent+selected)/1e4, "parent_in_1e4": parent/1e4}
    if RETRAINED.is_file():
        summary = json.loads(RETRAINED.read_text())
        numbers["retrained_k2"] = {"checkpoint": summary["checkpoint"], **{
            setting: {condition: {k: v for k, v in row.items() if k not in ("snr_db", "tbler")}
                      for condition, row in summary[setting].items()}
            for setting in ("no_power_control", "power_control")}}
    numbers["effective_snr"] = effective_snr()
    numbers["no_idd"] = no_idd()
    numbers["derived"] = derived(numbers)
    return numbers


def flatten(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def check(tex, numbers, allowed):
    """Every decimal number in the abstract, introduction, evaluation, and conclusion must round from a collected
    result or be a listed constant. Related work quotes other papers' numbers, so it is not checked here."""
    # main.tex pulls the sections in with \input{name}; expand them to check the full text.
    text = re.sub(r"\\input\{([^}]+)\}", lambda m: (Path(tex).parent/f"{m.group(1)}.tex").read_text(),
                  Path(tex).read_text())
    section = (text[text.index(r"\begin{abstract}"):text.index(r"\section{Related Work}")]
               + text[text.index(r"\section{Evaluation}"):text.index(r"\bibliography{references}")])
    section = re.sub(r"(?m)%.*$", "", section)
    values = list(flatten(numbers))
    values += [100*v for v in values]+[abs(v) for v in values]
    unknown = []
    for token in re.findall(r"(?<![\w.])[-−]?\d+\.\d+", section):
        target = abs(float(token.replace("−", "-")))
        digits = len(token.split(".")[1])
        if token in allowed or any(round(abs(v), digits) == target for v in values):
            continue
        unknown.append(token)
    return unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SWEEP/"results/paper_numbers.json")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--allowed", nargs="*", default=[])
    args = parser.parse_args()
    numbers = collect()
    args.output.write_text(json.dumps(numbers, indent=1))
    if args.check:
        unknown = check(args.check, numbers, set(args.allowed))
        print("unverified numbers:", unknown)
        raise SystemExit(1 if unknown else 0)


if __name__ == "__main__":
    main()
