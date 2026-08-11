#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-dir", required=True)
p.add_argument("--baseline-json", required=True)
a = p.parse_args()
root = Path(a.results_dir)
with open(a.baseline_json) as f:
    base = json.load(f)
cold2 = base["cold_k2"]["snr_at_10pct_tbler_db"]
cold8 = base["cold_k8"]["snr_at_10pct_tbler_db"]
gap = cold2 - cold8
rows = []
raw = {}
for d in [1, 4, 8, 16]:
    path = root / f"eval_dedge_{d}.json"
    if not path.exists():
        continue
    with open(path) as f:
        x = json.load(f)
    raw[d] = x
    nc = x["normal"]["snr_at_10pct_tbler_db"]
    rc = x["reset_each_tb"]["snr_at_10pct_tbler_db"]
    rows.append({
        "d_edge": d,
        "snr10_normal_db": nc,
        "snr10_reset_db": rc,
        "temporal_gain_db": None if nc is None or rc is None else rc - nc,
        "fraction_k2_to_k8_gap_recovered": None if nc is None else (cold2 - nc) / gap,
        "tbler_normal_3db": x["normal"]["tbler_vs_ebno"].get("3.0", {}).get("tbler"),
        "tbler_reset_3db": x["reset_each_tb"]["tbler_vs_ebno"].get("3.0", {}).get("tbler"),
        "tbler_shuffle_3db": x["shuffle_previous_edge_at_3db"]["tbler"],
        "median_latency_ms": x["latency"]["median_ms"],
        "p90_latency_ms": x["latency"]["p90_ms"],
        "edge_parameters": x["edge_parameters"],
        "persistent_edge_bytes_per_sequence": x["persistent_edge_bytes_per_sequence"],
    })

valid = [r for r in rows if r["snr10_normal_db"] is not None]
best = min(valid, key=lambda r: r["snr10_normal_db"]) if valid else None
scalar = next((r for r in rows if r["d_edge"] == 1), None)
recommendation = "inconclusive"
reason = "No variant produced a valid 10% TBLER crossing."
if best is not None and scalar is not None and scalar["snr10_normal_db"] is not None:
    accuracy_delta = scalar["snr10_normal_db"] - best["snr10_normal_db"]
    if accuracy_delta <= 0.05 and scalar["median_latency_ms"] <= 1.05 * best["median_latency_ms"]:
        recommendation = "scalar_d1"
        reason = "dE=1 is within 0.05 dB of the best crossing and has no >5% latency penalty."
    else:
        recommendation = f"d{best['d_edge']}"
        reason = "A larger edge state gives a material matched-accuracy advantage over dE=1."

summary = {
    "baseline": {
        "cold_k2_snr10_db": cold2,
        "cold_k8_snr10_db": cold8,
        "iteration_gap_db": gap,
    },
    "rows": rows,
    "best_snr10_variant": best,
    "recommendation": recommendation,
    "recommendation_reason": reason,
    "decision_rule": "Prefer dE=1 unless a vector state improves SNR@10% TBLER materially after accounting for total latency; normal-vs-reset/shuffle must also show that prior edge memory is actually used.",
}
with open(root / "comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
if rows:
    with open(root / "comparison_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

try:
    import matplotlib.pyplot as plt
    if valid:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in valid:
            ax.scatter(r["median_latency_ms"], r["snr10_normal_db"])
            ax.annotate(f"dE={r['d_edge']}",
                        (r["median_latency_ms"], r["snr10_normal_db"]),
                        xytext=(5, 5), textcoords="offset points")
        ax.axhline(cold2, linestyle="--", linewidth=1, label="cold K=2")
        ax.axhline(cold8, linestyle=":", linewidth=1, label="cold K=8")
        ax.set_xlabel("Median neural-receiver latency per TB (ms)")
        ax.set_ylabel("SNR at 10% TBLER (dB, lower is better)")
        ax.set_title("Temporal edge memory: accuracy-latency tradeoff")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "accuracy_latency_tradeoff.png", dpi=180)
        plt.close(fig)
except Exception as e:
    summary["plot_error"] = repr(e)
    with open(root / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
