#!/usr/bin/env python3
"""Merge corrected AE rows with the 24 valid writer/PCA factorial rows.

The historical exhaustive CSV contains 12 pre-fix autoencoder rows that are not
valid scientific evidence. This script drops those rows, requires the complete
24-row writer/PCA matrix, appends the 12 protocol-v2 corrected AE rows, and
regenerates professor-facing comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

POOLINGS = ("mean", "attention", "cnn")
COMPRESSIONS = ("writer", "pca", "autoencoder")
CAPACITIES = (8, 16, 32, 56)


def parse_args():
    p = argparse.ArgumentParser()
    default_base = Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"
    p.add_argument("--base-root", default=str(default_base))
    p.add_argument("--ae-root", default=str(default_base / "autoencoder_v2"))
    return p.parse_args()


def maybe_float(value):
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def normalize(row, compression=None):
    comp = compression or row["compression"]
    return {
        "group": "factorial",
        "pooling": row["pooling"],
        "compression": comp,
        "d_mem": int(float(row["d_mem"])),
        "scenario": row.get("scenario") or "fixed",
        "seed": int(float(row["seed"])),
        "temporal_snr10": maybe_float(row.get("temporal_snr10")),
        "cold_k2_snr10": maybe_float(row.get("cold_k2_snr10")),
        "cold_k8_snr10": maybe_float(row.get("cold_k8_snr10")),
        "gap_recovered_percent": maybe_float(row.get("gap_recovered_percent")),
        "memory_bits_per_ue": (
            int(float(row["memory_bits_per_ue"]))
            if row.get("memory_bits_per_ue") not in (None, "")
            else int(float(row["d_mem"])) * 32
        ),
    }


def require_matrix(rows, compressions, expected_count):
    keys = {
        (r["pooling"], r["compression"], int(r["d_mem"]))
        for r in rows
    }
    expected = {
        (pool, comp, d)
        for pool in POOLINGS
        for comp in compressions
        for d in CAPACITIES
    }
    if len(rows) != expected_count or keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RuntimeError(
            f"factorial matrix mismatch: rows={len(rows)} expected={expected_count}; "
            f"missing={missing}; extra={extra}"
        )


def main():
    a = parse_args()
    base_root = Path(a.base_root).expanduser().resolve()
    ae_root = Path(a.ae_root).expanduser().resolve()
    base_csv = base_root / "all_results.csv"
    ae_csv = ae_root / "corrected_autoencoder_results.csv"
    ae_summary = ae_root / "corrected_autoencoder_summary.json"

    if not base_csv.is_file():
        raise FileNotFoundError(base_csv)
    if not ae_csv.is_file() or not ae_summary.is_file():
        raise RuntimeError("corrected AE factorial has not completed yet")
    summary = json.loads(ae_summary.read_text())
    if summary.get("status") != "complete" or int(summary.get("cell_count", 0)) != 12:
        raise RuntimeError("corrected AE summary is not a complete 12-cell run")

    with base_csv.open(newline="") as f:
        historical = list(csv.DictReader(f))
    valid_24 = [
        normalize(r)
        for r in historical
        if r.get("group") == "factorial"
        and r.get("compression") in {"writer", "pca"}
    ]
    require_matrix(valid_24, ("writer", "pca"), 24)

    with ae_csv.open(newline="") as f:
        corrected = [normalize(r, "autoencoder") for r in csv.DictReader(f)]
    require_matrix(corrected, ("autoencoder",), 12)

    merged = valid_24 + corrected
    merged.sort(
        key=lambda r: (
            POOLINGS.index(r["pooling"]),
            COMPRESSIONS.index(r["compression"]),
            int(r["d_mem"]),
        )
    )

    out_csv = ae_root / "merged_valid_factorial_results.csv"
    fields = [
        "group",
        "pooling",
        "compression",
        "d_mem",
        "scenario",
        "seed",
        "temporal_snr10",
        "cold_k2_snr10",
        "cold_k8_snr10",
        "gap_recovered_percent",
        "memory_bits_per_ue",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    import matplotlib.pyplot as plt

    graphs = ae_root / "graphs"
    graphs.mkdir(exist_ok=True)

    def series(pool, comp):
        return sorted(
            [r for r in merged if r["pooling"] == pool and r["compression"] == comp],
            key=lambda r: r["d_mem"],
        )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for pool in POOLINGS:
        for comp in COMPRESSIONS:
            rows = series(pool, comp)
            ax.plot(
                [r["d_mem"] for r in rows],
                [float("nan") if r["temporal_snr10"] is None else r["temporal_snr10"] for r in rows],
                marker="o",
                label=f"{pool} + {comp}",
            )
    cold2 = [r["cold_k2_snr10"] for r in merged if r["cold_k2_snr10"] is not None]
    cold8 = [r["cold_k8_snr10"] for r in merged if r["cold_k8_snr10"] is not None]
    if cold2:
        ax.axhline(statistics.median(cold2), linestyle="--", label="median paired cold K=2")
    if cold8:
        ax.axhline(statistics.median(cold8), linestyle=":", label="median paired cold K=8")
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
    ax.set_title("Temporal UE memory: pooling × compression × capacity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graphs / "valid_factorial_snr10.png", dpi=180)
    fig.savefig(graphs / "valid_factorial_snr10.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for pool in POOLINGS:
        for comp in COMPRESSIONS:
            rows = series(pool, comp)
            ax.plot(
                [r["d_mem"] for r in rows],
                [float("nan") if r["gap_recovered_percent"] is None else r["gap_recovered_percent"] for r in rows],
                marker="o",
                label=f"{pool} + {comp}",
            )
    ax.axhline(0, linewidth=1)
    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("K=2 → K=8 gap recovered (%)")
    ax.set_title("Temporal UE memory gap recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graphs / "valid_factorial_gap_recovered.png", dpi=180)
    fig.savefig(graphs / "valid_factorial_gap_recovered.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for pool in POOLINGS:
        for comp in COMPRESSIONS:
            rows = series(pool, comp)
            ax.plot(
                [r["memory_bits_per_ue"] / 8.0 for r in rows],
                [float("nan") if r["temporal_snr10"] is None else r["temporal_snr10"] for r in rows],
                marker="o",
                label=f"{pool} + {comp}",
            )
    ax.set_xlabel("Persistent memory bytes / UE")
    ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
    ax.set_title("Temporal UE memory performance vs persistent memory")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graphs / "valid_factorial_performance_vs_bytes.png", dpi=180)
    fig.savefig(graphs / "valid_factorial_performance_vs_bytes.pdf")
    plt.close(fig)

    result = {
        "rows": len(merged),
        "historical_writer_pca_rows": len(valid_24),
        "corrected_autoencoder_rows": len(corrected),
        "null_corrected_ae_crossings": sum(r["temporal_snr10"] is None for r in corrected),
        "output_csv": str(out_csv),
        "graphs": [
            str(graphs / "valid_factorial_snr10.png"),
            str(graphs / "valid_factorial_snr10.pdf"),
            str(graphs / "valid_factorial_gap_recovered.png"),
            str(graphs / "valid_factorial_gap_recovered.pdf"),
            str(graphs / "valid_factorial_performance_vs_bytes.png"),
            str(graphs / "valid_factorial_performance_vs_bytes.pdf"),
        ],
    }
    (ae_root / "merged_valid_factorial_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("MERGED_VALID_FACTORIAL=" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
