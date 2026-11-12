"""Collect the paper's evaluation numbers from result files, and check the text against them."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
from neural_rx.benchmark.scoring import paired_summary

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep"
ORIGINAL = ROOT/"experiments/20261021_package_layout_validation/results/v2_confirmation_005000/results.json"
FRESH = ROOT/"experiments/20261021_current_code_reproduction/results/confirmation_005000_seed86000000/results.json"
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
    return {"snr_gap_db": values,
            "iteration_cut_db": cut,
            "latency_saving_vs_large_percent": 100*(1-latency["candidate_persistent"]/latency["frozen_cold_k8"]),
            "latency_added_vs_nit2_ms": latency["candidate_persistent"]-latency["frozen_cold_k2"],
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
