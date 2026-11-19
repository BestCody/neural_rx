"""Collect every number the paper states from the result files, and check the paper's text against them.

    python scripts/nrx_paper_numbers.py                          # writes paper/numbers.json
    python scripts/nrx_paper_numbers.py --check paper/main.tex   # also lists decimal numbers in the text that do not
                                                                 # round from a collected value

Sources: experiments/20261118_test_evaluation (proposed NRX with and without its memory, retrained Large NRX),
experiments/20261104_tbler_sweep and 20261112_early_retransmission (released Large NRX and classical receivers on
the same sequences, without and with power control), experiments/20261114_tensorrt_latency,
experiments/20261118_network_decisions, experiments/20261118_further_datasets, and the training runs
experiments/20261117_neural_rx_training and experiments/20261118_no_teacher_training.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT/"experiments/20261104_tbler_sweep/results"
TRAINED = ROOT/"experiments/20261118_test_evaluation/results"
TENSORRT = ROOT/"experiments/20261114_tensorrt_latency/results"
NETWORK = ROOT/"experiments/20261118_network_decisions/results"
FURTHER = ROOT/"experiments/20261118_further_datasets/results"
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
START = 16


def difference(a, b):
    """a-b of two values as the paper prints them (two decimals each)."""
    return round(round(a, 2)-round(b, 2), 2)


def required_snr():
    summary = json.loads((TRAINED/"summary.json").read_text())
    table = {setting: {c: summary[setting][c]["required_snr_db"] for c in CORE} for setting in summary}
    gains = {}
    for setting, rows in table.items():
        gains[setting] = {c: {
            "lead_over_retrained_db": difference(r["neural_rx_k2_trained"], r["pilot_memory"]),
            "lead_over_three_iterations_db": difference(r["frozen_cold_k3"], r["pilot_memory"]),
            "behind_eight_iterations_db": difference(r["pilot_memory"], r["frozen_cold_k8"]),
            "memory_gain_db": difference(r["pilot_memory_emptied"], r["pilot_memory"]),
            "no_memory_lead_over_retrained_db": difference(r["neural_rx_k2_trained"], r["pilot_memory_emptied"]),
            "retraining_gain_db": difference(r["frozen_cold_k2"], r["neural_rx_k2_trained"]),
            "iteration_cut_db": difference(r["frozen_cold_k2"], r["frozen_cold_k8"]),
            "retrained_behind_eight_iterations_db": difference(r["neural_rx_k2_trained"], r["frozen_cold_k8"]),
        } for c, r in rows.items()}
    blocks = {setting: {c: summary[setting][c]["blocks"] for c in CORE} for setting in summary}
    return table, gains, blocks


def effective_snr(width=1.):
    """Required effective SNR (SNR + effective channel power of the UE in dB) at TBLER=10% per condition and receiver
    without power control, pooled over all sequences in bins of `width` dB. Channel power comes from the stored sweep;
    every receiver decoded the same sequences."""
    from neural_rx.benchmark.plan import decreasing_isotonic, snr_at_target
    edges = np.arange(-10., 30.+width/2, width)
    centers = edges[:-1]+width/2
    sources = {m: SWEEP for m in ("frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8", "baseline_lslin_lmmse",
                                  "baseline_lmmse_kbest", "baseline_perf_csi_kbest")}
    sources.update({m: TRAINED for m in ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")})
    result = {}
    for condition in CORE:
        result[condition] = {}
        for method, base in sources.items():
            failures, blocks = np.zeros(len(centers)), np.zeros(len(centers))
            for directory, first in (("sweep", 0), ("sweep_high", 1)):       # +3 dB appears in both sweeps
                for path in sorted((SWEEP/directory/condition/"candidate_persistent").glob("episode_*.npz")):
                    with np.load(path) as reference, np.load(base/directory/condition/method/path.name) as data:
                        if not np.array_equal(reference["fingerprint"], data["fingerprint"]):
                            raise ValueError(f"{method}/{path.name}: decoded on a different sequence")
                        active = reference["active"][START:].astype(bool)
                        power = 10*np.log10(reference["channel_power"][START:])
                        for point in range(first, len(data["snr_db"])):
                            index = np.floor((data["snr_db"][point]+power-edges[0])/width).astype(int)
                            valid = (index >= 0) & (index < len(centers)) & active
                            np.add.at(failures, index[valid], data["errors"][point, START:][valid].astype(float))
                            np.add.at(blocks, index[valid], 1)
            rates = failures/np.maximum(blocks, 1)
            fitted = decreasing_isotonic(rates, np.maximum(blocks, 1))
            result[condition][method] = {
                "required_db": float(snr_at_target(centers, fitted, .1)),
                "drop_80_to_5_db": float(snr_at_target(centers, fitted, .05)-snr_at_target(centers, fitted, .8))}
    for condition, rows in result.items():
        value = lambda m: rows[m]["required_db"]
        rows["gains"] = {"lead_over_retrained_db": value("neural_rx_k2_trained")-value("pilot_memory"),
                         "memory_gain_db": value("pilot_memory_emptied")-value("pilot_memory"),
                         "behind_eight_iterations_db": value("pilot_memory")-value("frozen_cold_k8"),
                         "lead_over_three_iterations_db": value("frozen_cold_k3")-value("pilot_memory")}
    return result


def doppler():
    required = json.loads((TRAINED/"doppler.json").read_text())["required_effective_snr_db"]
    groups = ("0-50", "50-175", "175-400")
    return {"required_effective_snr_db": required,
            "memory_gain_db": {g: required["pilot_memory_emptied"][g]-required["pilot_memory"][g] for g in groups},
            "lead_over_retrained_db": {g: required["neural_rx_k2_trained"][g]-required["pilot_memory"][g] for g in groups},
            "behind_perfect_csi_db": {g: required["pilot_memory"][g]-required["baseline_perf_csi_kbest"][g] for g in groups},
            "no_memory_behind_retrained_db": {g: required["pilot_memory_emptied"][g]-required["neural_rx_k2_trained"][g]
                                              for g in groups}}


def latency():
    """Median TensorRT times on an A100 in the export of the code release of Wiesmayr et al. The proposed NRX is the
    kernels that apply its channel filter and the engine of its two iterations together; what only depends on the
    memory is computed before the slot and timed separately."""
    data = json.loads((TENSORRT/"fused_a100.json").read_text())
    if "A100" not in data["gpu"]:
        raise ValueError(f"the paper's latency is measured on an A100, not on {data['gpu']}")
    m = data["median_ms"]
    return {"gpu": data["gpu"], "tensorrt": data["tensorrt"],
            "large_k2_ms": m["large_k2"], "large_k3_ms": m["large_k3"], "large_k8_ms": m["large_k8"],
            "cgnn_iteration_ms": (m["large_k8"]-m["large_k2"])/6, "k8_over_k2": m["large_k8"]/m["large_k2"],
            "proposed_ms": m["proposed"], "proposed_minus_k2_ms": m["proposed"]-m["large_k2"],
            "before_slot_ms": m["fit_only"], "samples": data["samples_per_receiver"]}


def training():
    def best(experiment):
        return json.loads((ROOT/"experiments"/experiment/"results/best.json").read_text())
    one, two = best("20261117_neural_rx_training"), best("20261118_no_teacher_training")
    first = json.loads((ROOT/"experiments/20261117_neural_rx_training/inputs/training_recipe.json").read_text())
    recipe = json.loads((ROOT/"experiments/20261118_no_teacher_training/inputs/resolved_recipe.json").read_text())
    return {"retrained_updates": one["update"], "retrained_development_tbler_percent": 100*one["core_tbler"],
            "proposed_updates": two["update"], "proposed_development_tbler_percent": 100*two["core_tbler"],
            "learning_rate": recipe["learning_rate"], "channel_loss_weight": recipe["channel_loss_weight"],
            "teacher_weight_first_stage": first["distillation_weight"],
            "teacher_weight_second_stage": recipe["distillation_weight"],
            "sequences_per_update": recipe["effective_examples"]}


def network():
    """Fallback after a CRC failure and the early-retransmission controller (experiments/20261118_network_decisions)."""
    fallback = json.loads((NETWORK/"fallback.json").read_text())
    controller = json.loads((NETWORK/"controller.json").read_text())
    step = controller["step_by_step"]["CRC history and receiver evidence, budget of the rule"]
    single = controller["single_check"]
    return {"fallback": {setting: {c: {"kbest_gain_db": r["fallback_gain_db"]["baseline_lmmse_kbest"],
                                       "unfixable_share": r["unfixable_share_of_failures"],
                                       "after_failure_share": r["failures_after_failure_of_same_ue"]}
                                   for c, r in rows.items()} for setting, rows in fallback.items()},
            "controller": {"rule_caught": controller["rule"]["caught"], "rule_false_alarm": controller["rule"]["false_alarm"],
                           "caught": step["caught"], "false_alarm": step["false_alarm"],
                           "caught_first_failures": step["caught_first_failures"],
                           "caught_at_check_1_of_caught": step["caught_at"]["check 1"]/step["caught"],
                           "single_check_caught": {name: value["caught"] for name, value in single.items()},
                           "test_ue_tbs": controller["test"]["ue_tbs"], "test_failures": controller["test"]["failures"]}}


def further():
    """TBLER at a fixed SNR per condition on two further datasets (experiments/20261118_further_datasets): the proposed
    NRX, the same with the memory emptied, and the Large NRX with eight iterations."""
    methods = {"proposed": "candidate_persistent", "no_memory": "candidate_reset", "large_k8": "frozen_cold_k8"}
    report = {}
    for dataset in ("v2", "seed86"):
        result = json.loads((FURTHER/dataset/"results.json").read_text())
        data = np.load(FURTHER/dataset/"errors.npz")
        names = json.loads((FURTHER/dataset/"protocol.json").read_text())["condition_order"][:data["active"].shape[0]]
        rows = {}
        for c, name in enumerate(names):
            mask = data["active"][c, :, START:]
            rows[name] = {key: float((data[f"errors_{m}"][c, 0, :, START:] & mask).sum()/mask.sum())
                          for key, m in methods.items()}
            for key, m in methods.items():
                assert abs(rows[name][key]-result["rates"][m][c][0]) < 1e-12, (dataset, name, key)
        report[dataset] = rows
    cases = [row for rows in report.values() for row in rows.values()]
    report["cases"] = len(cases)
    report["memory_lowers_tbler"] = sum(row["proposed"] < row["no_memory"] for row in cases)
    report["memory_unchanged"] = sum(row["proposed"] == row["no_memory"] for row in cases)
    report["large_k8_ahead"] = sum(row["large_k8"] < row["proposed"] for row in cases)
    report["proposed_ahead"] = sum(row["proposed"] < row["large_k8"] for row in cases)
    return report


def collect():
    table, gains, blocks = required_snr()
    export = json.loads((TENSORRT/"export.json").read_text())["weights"]
    settings = json.loads((ROOT/"config/memory_receiver.json").read_text())["architecture"]
    cells, antennas, neighbors = 132*3, 4, 2*(2*settings["neighbor_radius"]+1)     # per DMRS symbol at 132 PRBs
    entries = settings["history_tbs"]-settings["fit_tbs"]
    return {"required_snr_db": table, "gains": gains, "blocks": blocks, "effective_snr": effective_snr(),
            "doppler": doppler(), "latency": latency(), "training": training(), "network": network(),
            "further_datasets": further(),
            "weights": {"large_nrx_two_iterations": export["large_k2"], "large_nrx_three_iterations": export["large_k3"],
                        "large_nrx_eight_iterations": export["large_k8"], "proposed": export["large_k2"]},
            # as the text states them, in units of 1e5
            "weights_1e5": {"two_iterations": export["large_k2"]/1e5, "eight_iterations": export["large_k8"]/1e5},
            "filter": {"entries_used": entries, "fitted_tbs": settings["fit_tbs"], "stored_tbs": settings["history_tbs"],
                       "weights_per_dmrs_symbol": (entries+1)*neighbors-1,
                       "estimates_per_fit": settings["fit_tbs"]*cells*antennas}}


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
    """Every decimal number in the abstract, introduction, receiver design, evaluation, and conclusion must round from
    a collected result or be a listed constant. Related work quotes other papers' numbers, so it is not checked."""
    # main.tex pulls the sections in with \\input{name}; expand them to check the full text.
    text = re.sub(r"\\input\{([^}]+)\}", lambda m: (Path(tex).parent/f"{m.group(1)}.tex").read_text(),
                  Path(tex).read_text())
    section = (text[text.index(r"\begin{abstract}"):text.index(r"\section{Related Work}")]
               + text[text.index(r"\section{Receiver Design}"):text.index(r"\bibliography{references}")])
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
    parser.add_argument("--output", type=Path, default=ROOT/"paper/numbers.json")
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
