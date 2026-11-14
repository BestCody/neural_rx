"""Required SNR of the retrained two-iteration baseline on the test sequences, next to the stored receivers.

Uses the TBLER sweep's own summarize() (decreasing isotonic fit, log interpolation at 10% TBLER): without power
control on the merged "sweep" and "sweep_high" grids, as in experiments/20261104_tbler_sweep/results/summary.json,
and with power control on the power-controlled grid. Writes results/summary.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((ROOT/"experiments/20261104_tbler_sweep/results/summary.json").read_text())
    stored_pc = json.loads((ROOT/"experiments/20261112_early_retransmission/results/fallback_pc.json").read_text())
    plain = summarize([HERE/"results/test/sweep", HERE/"results/test/sweep_high"])
    controlled = summarize(HERE/"results/test/power_control")
    # "sweep_high" starts at the last point of "sweep" (as in the stored summary); the shared point must agree.
    if any(point["first"] != point["second"] for point in plain["overlap_points"]) or controlled["overlap_points"]:
        raise ValueError("Shared SNR points disagree")
    report = {"checkpoint": json.loads((HERE/"results/best.json").read_text())["checkpoint"]}
    for name, summary, reference in (("no_power_control", plain, None), ("power_control", controlled, stored_pc)):
        report[name] = {}
        for condition in CORE:
            value = summary[condition]["retrained_k2"]
            if reference is None:
                others = {m: stored[condition][m]["snr_at_10pct_db"] for m in
                          ("candidate_persistent", "candidate_reset", "frozen_cold_k2", "frozen_cold_k8")}
                if value["blocks"] != stored[condition]["candidate_persistent"]["blocks"]:
                    raise ValueError(f"{condition}: test blocks differ from the stored sweep")
            else:
                others = {m: reference[condition]["required_snr_db"][m] for m in
                          ("candidate_persistent", "candidate_reset", "frozen_cold_k2", "frozen_cold_k8")}
            report[name][condition] = {"retrained_k2": value["snr_at_10pct_db"], **others,
                                       "blocks": value["blocks"], "episodes": value["episodes"],
                                       "snr_db": value["snr_db"], "tbler": value["tbler"]}
    (HERE/"results/summary.json").write_text(json.dumps(report, indent=1))
    for name in ("no_power_control", "power_control"):
        print(name)
        for condition in CORE:
            r = report[name][condition]
            print(f"  {condition:28s} retrained k2 {r['retrained_k2']:.2f}  original k2 {r['frozen_cold_k2']:.2f}  "
                  f"k8 {r['frozen_cold_k8']:.2f}  proposed {r['candidate_persistent']:.2f}  "
                  f"no memory {r['candidate_reset']:.2f}")


if __name__ == "__main__":
    main()
