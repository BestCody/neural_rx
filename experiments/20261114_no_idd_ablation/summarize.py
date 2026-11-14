"""Required SNR of the proposed NRX without IDD (with and without memory) on the test sequences, next to the stored
receivers. Uses the TBLER sweep's own summarize() (decreasing isotonic fit, log interpolation at 10% TBLER) on the
merged "sweep" and "sweep_high" grids, as in experiments/20261104_tbler_sweep/results/summary.json.
Writes results/summary.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
METHODS = ("candidate_persistent_no_idd", "candidate_reset_no_idd")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((ROOT/"experiments/20261104_tbler_sweep/results/summary.json").read_text())
    retrained = json.loads((ROOT/"experiments/20261113_retrained_k2_baseline/results/summary.json").read_text())
    plain = summarize([HERE/"results/test/sweep", HERE/"results/test/sweep_high"])
    # "sweep_high" starts at the last point of "sweep"; the shared point must agree.
    if any(point["first"] != point["second"] for point in plain["overlap_points"]):
        raise ValueError("Shared SNR points disagree")
    report = {}
    for condition in CORE:
        row = {}
        for method in METHODS:
            value = plain[condition][method]
            if value["blocks"] != stored[condition]["candidate_persistent"]["blocks"]:
                raise ValueError(f"{condition}/{method}: test blocks differ from the stored sweep")
            row[method] = {"snr_at_10pct_db": value["snr_at_10pct_db"], "snr_db": value["snr_db"],
                           "tbler": value["tbler"], "blocks": value["blocks"], "episodes": value["episodes"]}
        row["stored"] = {m: stored[condition][m]["snr_at_10pct_db"] for m in
                         ("candidate_persistent", "candidate_reset", "frozen_cold_k2", "frozen_cold_k8")}
        row["stored"]["retrained_k2"] = retrained["no_power_control"][condition]["retrained_k2"]
        report[condition] = row
    (HERE/"results/summary.json").write_text(json.dumps(report, indent=1))
    for condition in CORE:
        r = report[condition]
        s = r["stored"]
        print(f"{condition:28s} no IDD {r['candidate_persistent_no_idd']['snr_at_10pct_db']:.2f}  "
              f"no IDD, no memory {r['candidate_reset_no_idd']['snr_at_10pct_db']:.2f}  proposed {s['candidate_persistent']:.2f}  "
              f"no memory {s['candidate_reset']:.2f}  retrained k2 {s['retrained_k2']:.2f}  k2 {s['frozen_cold_k2']:.2f}  "
              f"k8 {s['frozen_cold_k8']:.2f}")


if __name__ == "__main__":
    main()
