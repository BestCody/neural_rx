"""Required SNR at 10% TBLER of every receiver at one radio configuration (evaluate.py), as in the 16-QAM summaries
(neural_rx.benchmark.tbler_sweep.summarize: monotone fit, log-TBLER interpolation, receivers paired by episode
fingerprint), and the differences the paper states for 16-QAM. Receivers with fewer than 32 sequences in a
condition are listed as incomplete there and get no value. Writes results/<label>/summary.json.

    summarize.py --config nrx_large_64qam.cfg
"""
import argparse
import json
import os
from pathlib import Path

from evaluate import RECEIVERS

HERE = Path(__file__).resolve().parent
SETTINGS = ("sweep", "power_control")
CONDITIONS = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")


def gap(a, b):
    """a-b, or None if a receiver's TBLER does not cross 10% on the grid."""
    return None if a is None or b is None else a-b


def show(value):
    return "  n/a" if value is None else f"{value:5.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from neural_rx.benchmark.tbler_sweep import summarize
    label = Path(args.config).stem
    report = {"config": args.config}
    for setting in SETTINGS:
        rows = summarize(HERE/"results"/label/setting)
        report[setting] = {}
        for condition in CONDITIONS:
            values = rows.get(condition, {})
            missing = [r for r in RECEIVERS if r not in values or values[r]["episodes"] != 32]
            if missing:
                print(f"{label} {setting} {condition}: not yet complete for {', '.join(missing)}", flush=True)
            required = {r: (None if r in missing else values[r]["snr_at_10pct_db"]) for r in RECEIVERS}
            ours = required["pilot_memory"]
            report[setting][condition] = {
                "required_snr_db": required, "incomplete": missing,
                "memory_gain_db": gap(required["pilot_memory_emptied"], ours),
                "lead_over_retrained_db": gap(required["neural_rx_k2_trained"], ours),
                "lead_over_three_iterations_db": gap(required["frozen_cold_k3"], ours),
                "behind_eight_iterations_db": gap(ours, required["frozen_cold_k8"]),
                "retraining_gain_db": gap(required["frozen_cold_k2"], required["neural_rx_k2_trained"])}
            r = report[setting][condition]
            print(f"{label} {setting:13s} {condition:27s} ours {show(ours)} dB | memory {show(r['memory_gain_db'])} | "
                  f"vs retrained {show(r['lead_over_retrained_db'])} | vs N=3 {show(r['lead_over_three_iterations_db'])} | "
                  f"behind N=8 {show(r['behind_eight_iterations_db'])}", flush=True)
    (HERE/"results"/label/"summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
