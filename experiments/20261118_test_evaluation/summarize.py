"""Required SNR at 10% TBLER of the receivers trained in November 2026, next to the stored receivers.

Uses the TBLER sweep's own summarize() (decreasing isotonic fit, log interpolation at 10% TBLER): without power
control on the merged "sweep" and "sweep_high" grids, as in experiments/20261104_tbler_sweep/results/summary.json,
and with power control on the power-controlled grid. Writes results/summary.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
NEW = ("pilot_memory", "pilot_memory_emptied", "neural_rx_k2_trained")
STORED = ("frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8", "baseline_lslin_lmmse", "baseline_lmmse_kbest",
          "baseline_perf_csi_kbest")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((ROOT/"experiments/20261104_tbler_sweep/results/summary.json").read_text())
    stored_pc = json.loads((ROOT/"experiments/20261112_early_retransmission/results/fallback_pc.json").read_text())
    plain = summarize([HERE/"results/sweep", HERE/"results/sweep_high"])
    controlled = summarize(HERE/"results/power_control")
    # "sweep_high" starts at the last point of "sweep" (as in the stored summary); the shared point must agree.
    if any(point["first"] != point["second"] for point in plain["overlap_points"]) or controlled["overlap_points"]:
        raise ValueError("Shared SNR points disagree")
    report = {}
    for name, summary in (("no_power_control", plain), ("power_control", controlled)):
        report[name] = {}
        for condition in CORE:
            new = {receiver: summary[condition][receiver] for receiver in NEW}
            if name == "no_power_control":
                others = {m: stored[condition][m]["snr_at_10pct_db"] for m in STORED}
                if any(v["blocks"] != stored[condition]["frozen_cold_k2"]["blocks"] for v in new.values()):
                    raise ValueError(f"{condition}: test blocks differ from the stored sweep")
            else:
                others = {m: stored_pc[condition]["required_snr_db"][m] for m in STORED
                          if m in stored_pc[condition]["required_snr_db"]}
            first = new[NEW[0]]
            report[name][condition] = {
                "required_snr_db": {**{r: v["snr_at_10pct_db"] for r, v in new.items()}, **others},
                "blocks": first["blocks"], "episodes": first["episodes"], "snr_db": first["snr_db"],
                "tbler": {r: v["tbler"] for r, v in new.items()},
                "crc_disagreements": {r: v["crc_disagreements"] for r, v in new.items()}}
    (HERE/"results/summary.json").write_text(json.dumps(report, indent=1))
    labels = {"pilot_memory": "ours", "pilot_memory_emptied": "ours, memory emptied", "neural_rx_k2_trained":
              "neural_rx 2 it, trained", "frozen_cold_k2": "neural_rx 2 it, released", "frozen_cold_k3":
              "neural_rx 3 it", "frozen_cold_k8": "neural_rx 8 it", "baseline_lslin_lmmse": "LS + LMMSE",
              "baseline_lmmse_kbest": "LMMSE + K-Best", "baseline_perf_csi_kbest": "perfect CSI + K-Best"}
    for name in report:
        print(name, "(required SNR at 10% TBLER, dB)")
        print(f"  {'':26s}" + "".join(f"{c[:22]:>24s}" for c in CORE))
        for receiver, label in labels.items():
            values = [report[name][c]["required_snr_db"].get(receiver) for c in CORE]
            print(f"  {label:26s}" + "".join(f"{'-' if v is None else format(v, '.2f'):>24s}" for v in values))


if __name__ == "__main__":
    main()
