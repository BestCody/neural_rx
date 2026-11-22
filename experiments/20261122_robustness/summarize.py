"""Required SNR at 10% TBLER per test, condition and receiver (neural_rx.benchmark.tbler_sweep.summarize), next to the
same receivers on the power-control sequences without these changes (experiments/20261118_test_evaluation, setting
power_control). Receivers with fewer than 32 sequences of a condition are listed as incomplete and get no value.
Writes results/summary.json.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT/"experiments/20261118_test_evaluation/results/power_control"
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
OURS = ("pilot_memory", "pilot_memory_uplink")


def required(rows, condition, receiver):
    value = rows.get(condition, {}).get(receiver)
    return None if value is None or value["episodes"] != 32 else value["snr_at_10pct_db"]


def gap(a, b):
    return None if a is None or b is None else round(a-b, 3)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_rx.benchmark.tbler_sweep import summarize
    reference = summarize(REFERENCE)
    report = {"reference": "power_control sequences of experiments/20261118_test_evaluation"}
    for test_dir in sorted(p for p in (HERE/"results").iterdir() if p.is_dir() and p.name != "harq"):
        rows = summarize(test_dir)
        report[test_dir.name] = {}
        for condition in CONDITIONS:
            receivers = sorted(rows.get(condition, {}))
            entry = {"required_snr_db": {r: required(rows, condition, r) for r in receivers},
                     "reference_snr_db": {r: required(reference, condition, r)
                                          for r in ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")},
                     "incomplete": [r for r in receivers if rows[condition][r]["episodes"] != 32]}
            snr, ref = entry["required_snr_db"], entry["reference_snr_db"]
            for ours in OURS:
                if ours in snr:
                    entry[f"{ours}_memory_gain_db"] = gap(snr.get("pilot_memory_emptied"), snr[ours])
                    entry[f"{ours}_lead_over_retrained_db"] = gap(snr.get("neural_rx_k2_trained"), snr[ours])
            entry["reference_memory_gain_db"] = gap(ref["pilot_memory_emptied"], ref["pilot_memory"])
            entry["reference_lead_over_retrained_db"] = gap(ref["neural_rx_k2_trained"], ref["pilot_memory"])
            report[test_dir.name][condition] = entry
            print(f"{test_dir.name:16s} {condition:27s} " + " | ".join(
                f"{key.removesuffix('_db')} {value:+.2f}" for key, value in entry.items()
                if key.endswith("_db") and isinstance(value, float)), flush=True)
    (HERE/"results/summary.json").write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
