#!/usr/bin/env python3
"""Bar chart of memory_nrx and Cold K8 on paired evolving-channel episodes.

This renders existing measurements only; it does not run a receiver.
memory_nrx is the figure label for the historical optimized V18 checkpoint.
No latency is inferred from the neural round count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIRED_FILE = "cjlr_snr_confirmation_result.json"
V18 = "optimized_exact_cjlr_pd_l8_k2"
ROOT = Path(__file__).resolve().parents[1]


def crossing(curve: list[dict], target: float = 0.1) -> float:
    """Verify the saved half-count-corrected, log-linear threshold."""
    points = sorted(curve, key=lambda p: p["ebno_db"])
    for point in points:
        if not np.isclose(point["tbler"], point["errors"] / point["blocks"]):
            raise ValueError("Saved error counts and TBLER disagree")
    for left, right in zip(points, points[1:]):
        high = (left["errors"] + 0.5) / (left["blocks"] + 1.0)
        low = (right["errors"] + 0.5) / (right["blocks"] + 1.0)
        if high >= target > low > 0:
            fraction = np.log(target / high) / np.log(low / high)
            return float(left["ebno_db"] + fraction * (right["ebno_db"] - left["ebno_db"]))
    raise ValueError("Measured curve does not bracket 10% TBLER")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=ROOT / "docs/evidence/memory_nrx"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/v18_iteration_plot_20260923",
    )
    args = parser.parse_args()
    source = args.sources.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paired = json.loads((source / PAIRED_FILE).read_text())
    if not paired["pass"] or not paired["methodology"]["paired_across_methods_and_snr"]:
        raise ValueError("Require the completed, paired SNR confirmation")
    if paired["methodology"]["mature_tb_start"] != 9:
        raise ValueError("Figure caption assumes scoring starts at TB 9")

    rows = []
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.25, top=0.79)
    teal, blue = "#087f73", "#255e99"
    for position, (method, k, color) in enumerate((
        (V18, 2, teal),
        ("cold_k8", 8, blue),
    )):
        value = float(paired["thresholds_db"][method])
        if not np.isclose(crossing(paired["curves"][method]), value, rtol=0, atol=1e-12):
            raise ValueError("Saved threshold differs from the measured curve")
        ax.bar(position, value, width=0.48, color=color, zorder=3)
        ax.text(position, value + 0.065, f"{value:.3f} dB",
                ha="center", va="bottom", color=color,
                fontsize=15, fontweight="bold")
        rows.append({
            "experiment": "paired_evolving_channel_mature_tb9_64",
            "receiver": "memory_nrx" if k == 2 else "Cold K8",
            "neural_iterations": k, "snr_at_10pct_tbler_db": value,
            "source": PAIRED_FILE,
        })

    v18 = float(paired["thresholds_db"][V18])
    cold = float(paired["thresholds_db"]["cold_k8"])
    gain = cold - v18
    if not np.isclose(-gain, paired["gap_to_cold_k8_db"], atol=1e-12, rtol=0):
        raise ValueError("Paired improvement differs from the saved result")
    ax.set(xlim=(-0.65, 1.65), ylim=(0, 2.85), xticks=[0, 1],
           xticklabels=["memory_nrx\nK=2", "Cold K8\nK=8"],
           ylabel="SNR at 10% TBLER (dB) — lower is better")
    ax.tick_params(axis="x", length=0, pad=10, labelsize=12)
    ax.grid(True, axis="y", color="#e4e7eb", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.14, 0.935, "memory_nrx versus Cold K8", fontsize=19, fontweight="bold")
    fig.text(0.14, 0.888, "Matched evolving-channel episodes • two UEs • 16-QAM", color="#4b5563", fontsize=10.5)
    fig.text(0.14, 0.843, f"memory_nrx needs {gain:.3f} dB less SNR", color=teal, fontsize=12)
    fig.text(0.14, 0.135,
             "32 paired episodes; TBs 9–64 scored after memory buildup. MCS 14.\n"
             "memory_nrx: historical optimized V18. K counts neural rounds.\n"
             "SNR follows the saved Eb/N0 convention; accuracy comparison only.",
             fontsize=9.2, color="#4b5563", va="top", linespacing=1.6)
    stem = output / "v18_snr_vs_iterations"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=190, facecolor="white")
    plt.close(fig)
    with (output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = [source / PAIRED_FILE]
    manifest = {
        "new_experiment_run": False,
        "v18_result_is_historical_not_a_new_current_checkpoint_measurement": True,
        "only_paired_evolving_channel_measurements": True,
        "chart_type": "bar",
        "confidence_intervals_displayed": False,
        "displayed_statistic": "threshold_interpolated_from_pooled_tbler_curve",
        "display_names": {V18: "memory_nrx", "cold_k8": "Cold K8"},
        "paired_gain_db": gain,
        "interpolated_thresholds_verified": True,
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "paired_gain_db": gain}))


if __name__ == "__main__":
    main()
