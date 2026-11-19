"""What the pilot memory and channel filter give a classical receiver, and how the filter compares with classical
multi-slot estimation: required SNR at 10% TBLER of K-Best detection on the filter's refined estimate (filter_kbest)
and on a classical LMMSE estimate over the current and stored TBs from known statistics (wiener_kbest; both
filter_kbest.py), next to K-Best on the single-slot LMMSE estimate and on the true channel, and next to the NRXs.
Required SNR as in summarize.py, whose results/summary.json provides the other receivers. Writes
results/filter_kbest.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")
OTHERS = ("baseline_lslin_lmmse", "baseline_lmmse_kbest", "baseline_perf_csi_kbest", "pilot_memory",
          "frozen_cold_k8")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((HERE/"results/summary.json").read_text())
    summaries = {"no_power_control": summarize([HERE/"results/sweep", HERE/"results/sweep_high"]),
                 "power_control": summarize(HERE/"results/power_control")}
    report = {}
    for setting, summary in summaries.items():
        report[setting] = {}
        for condition in CORE:
            required, curves = {}, {}
            for name in ("filter_kbest", "wiener_kbest"):
                row = summary[condition][name]
                if row["blocks"] != stored[setting][condition]["blocks"] or row["episodes"] != 32:
                    raise ValueError(f"{setting}/{condition}: {name} decoded other TBs than the stored receivers")
                required[name] = row["snr_at_10pct_db"]
                curves[name] = {"snr_db": row["snr_db"], "tbler": row["tbler"]}
            required.update({name: stored[setting][condition]["required_snr_db"][name] for name in OTHERS})
            report[setting][condition] = {
                "required_snr_db": required, "curves": curves,
                "gain_over_lmmse_kbest_db": required["baseline_lmmse_kbest"]-required["filter_kbest"],
                "filter_ahead_of_wiener_db": required["wiener_kbest"]-required["filter_kbest"],
                "wiener_gain_over_lmmse_kbest_db": required["baseline_lmmse_kbest"]-required["wiener_kbest"],
                "behind_perfect_csi_db": required["filter_kbest"]-required["baseline_perf_csi_kbest"],
                "ahead_of_proposed_nrx_db": required["pilot_memory"]-required["filter_kbest"]}
            print(f"{setting:17s} {condition:27s} " + "  ".join(f"{name.replace('baseline_', '')} {value:5.2f}"
                                                               for name, value in required.items()))
    (HERE/"results/filter_kbest.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
