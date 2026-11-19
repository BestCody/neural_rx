"""Check of the classical multi-slot estimator (wiener.py): its single-slot version with K-Best (wiener_slot_kbest)
must match the sweep's LMMSE + K-Best, which uses the same covariance matrices and detector. Required SNR at 10% TBLER
without power control, as in summarize.py, and TBLER per SNR point. Writes results/wiener_check.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = ("natural_umi", "controlled_topology_change", "schedule_gap_reentry")


def main():
    from neural_rx.benchmark.tbler_sweep import summarize
    stored = json.loads((HERE.parent/"20261104_tbler_sweep/results/summary.json").read_text())
    summary = summarize([HERE/"results/sweep", HERE/"results/sweep_high"])
    report = {}
    for condition in CORE:
        new, old = summary[condition]["wiener_slot_kbest"], stored[condition]["baseline_lmmse_kbest"]
        if new["blocks"] != old["blocks"] or new["episodes"] != 32:
            raise ValueError(f"{condition}: different TBs")
        rates = dict(zip(old["snr_db"], old["tbler"]))
        shared = [(snr, rate, rates[snr]) for snr, rate in zip(new["snr_db"], new["tbler"]) if snr in rates]
        report[condition] = {"wiener_slot_kbest_db": new["snr_at_10pct_db"], "lmmse_kbest_db": old["snr_at_10pct_db"],
                             "difference_db": new["snr_at_10pct_db"]-old["snr_at_10pct_db"],
                             "largest_tbler_difference": max(abs(a-b) for _, a, b in shared), "snr_points": len(shared)}
        row = report[condition]
        print(f"{condition:27s} single-slot Wiener + K-Best {row['wiener_slot_kbest_db']:.2f} dB, LMMSE + K-Best "
              f"{row['lmmse_kbest_db']:.2f} dB, difference {row['difference_db']:+.2f} dB; largest TBLER difference "
              f"{100*row['largest_tbler_difference']:.2f} percentage points over {row['snr_points']} SNR points")
    (HERE/"results/wiener_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
