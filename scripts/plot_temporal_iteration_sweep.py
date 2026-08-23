#!/usr/bin/env python3
"""Summarize and plot the temporal iteration sweep.

Outputs:
  summary/iteration_transfer.csv
  summary/diagonal_snr10_vs_k.png + .pdf
  summary/heatmap_<finalist>.png + .pdf

The diagonal plot is the fair train-K/eval-K comparison. Heatmaps show whether
a checkpoint trained at K continues to work when extra NRX iterations are added.
Existing cold NRX results are overlaid when their CSV can be parsed; cold NRX
is never executed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINALISTS = (
    ("mean_pca_d56", "Mean + PCA d56"),
    ("cnn_pca_d16", "CNN + PCA d16"),
    ("cnn_autoencoder_d56", "CNN + AE d56"),
    ("mean_writer_d32", "Best Writer (Mean d32)"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--cold-csv", default=None)
    p.add_argument(
        "--cold-model-contains",
        default="large",
        help="When cold CSV has multiple models, prefer rows containing this text.",
    )
    return p.parse_args()


A = parse_args()
ROOT = Path(A.root).expanduser().resolve()
SUMMARY = ROOT / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text())


def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def collect_rows():
    rows = []
    for slug, label in FINALISTS:
        base = ROOT / "evaluations" / slug
        for train_k in range(2, 9):
            for eval_k in range(train_k, 9):
                path = base / f"train_k{train_k}" / f"eval_k{eval_k}" / "evaluation.json"
                if not path.is_file():
                    continue
                try:
                    s = load_json(path)
                except Exception as exc:
                    print(f"WARNING: cannot parse {path}: {exc}")
                    continue
                crossing = s.get("snr_db_at_10pct_tbler")
                rows.append(
                    {
                        "finalist": slug,
                        "label": label,
                        "pooling": s.get("pooling"),
                        "compression": s.get("compression"),
                        "d_mem": s.get("d_mem"),
                        "train_k": int(s.get("train_num_it", train_k)),
                        "eval_k": int(s.get("eval_num_it", eval_k)),
                        "snr10_tb2plus_db": crossing,
                        "memory_bytes_per_ue": (
                            int(s.get("d_mem", 0)) * 4 if s.get("d_mem") is not None else None
                        ),
                        "checkpoint": s.get("checkpoint"),
                        "evaluation_json": str(path),
                        "imported_k2": bool(
                            (s.get("provenance") or {}).get("reused_existing_k2_evaluation")
                        ),
                    }
                )
    rows.sort(key=lambda r: (r["finalist"], r["train_k"], r["eval_k"]))
    return rows


def write_rows(rows):
    path = SUMMARY / "iteration_transfer.csv"
    fields = [
        "finalist",
        "label",
        "pooling",
        "compression",
        "d_mem",
        "train_k",
        "eval_k",
        "snr10_tb2plus_db",
        "memory_bytes_per_ue",
        "checkpoint",
        "evaluation_json",
        "imported_k2",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE={path}")


def norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_cold_csv(path_str):
    if not path_str:
        return {}
    path = Path(path_str).expanduser()
    if not path.is_file():
        print(f"WARNING: cold CSV not found; plotting temporal only: {path}")
        return {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        raw = list(reader)
        headers = reader.fieldnames or []
    if not raw or not headers:
        print(f"WARNING: cold CSV is empty: {path}")
        return {}

    normalized = {norm_key(h): h for h in headers}

    def choose(candidates):
        for c in candidates:
            if c in normalized:
                return normalized[c]
        # Fuzzy fallback for old study column naming.
        for nk, original in normalized.items():
            if any(c in nk for c in candidates):
                return original
        return None

    k_col = choose(["k", "num_it", "iterations", "num_iterations", "iteration"])
    snr_col = choose(
        [
            "snr10",
            "snr_10",
            "snr_db_10pct",
            "snr_at_10pct_tbler",
            "snr_db_at_10pct_tbler",
            "ebno_at_10pct_tbler",
            "ebno_db_at_10pct_tbler",
            "snr_10pct_tbler",
        ]
    )
    model_col = choose(["model", "variant", "receiver", "name", "architecture"])

    if k_col is None or snr_col is None:
        print(
            "WARNING: could not identify K/SNR@10% columns in cold CSV. "
            f"headers={headers}. Plotting temporal only."
        )
        return {}

    preferred = str(A.cold_model_contains).lower().strip()
    filtered = raw
    if model_col and preferred:
        matching = [r for r in raw if preferred in str(r.get(model_col, "")).lower()]
        if matching:
            filtered = matching

    out = {}
    for row in filtered:
        try:
            k = int(float(row[k_col]))
            snr = float(row[snr_col])
        except (TypeError, ValueError):
            continue
        if 1 <= k <= 8 and math.isfinite(snr):
            # If duplicates remain, preserve the first matching row rather than
            # silently average across potentially different NRX variants.
            out.setdefault(k, snr)

    if out:
        print(f"COLD_POINTS={json.dumps(out, sort_keys=True)}")
    else:
        print(f"WARNING: no usable cold points parsed from {path}")
    return out


def plot_heatmap(rows, slug, label):
    grid = np.full((7, 7), np.nan, dtype=float)
    for r in rows:
        if r["finalist"] != slug or not finite(r["snr10_tb2plus_db"]):
            continue
        i = r["train_k"] - 2
        j = r["eval_k"] - 2
        if 0 <= i < 7 and 0 <= j < 7:
            grid[i, j] = float(r["snr10_tb2plus_db"])

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, origin="upper", aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("SNR at 10% TB2+ TBLER (dB) — lower is better")

    ticks = list(range(7))
    labels = [str(k) for k in range(2, 9)]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Evaluation NRX iterations K")
    ax.set_ylabel("Training NRX iterations K")
    ax.set_title(f"Iteration transfer — {label}")

    for i in range(7):
        for j in range(7):
            value = grid[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    png = SUMMARY / f"heatmap_{slug}.png"
    pdf = SUMMARY / f"heatmap_{slug}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"WROTE={png}")


def plot_diagonal(rows, cold):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    if cold:
        xs = sorted(k for k in cold if 2 <= k <= 8)
        ax.plot(xs, [cold[k] for k in xs], marker="o", linewidth=2, label="Cold NRX")

    for slug, label in FINALISTS:
        points = [
            r
            for r in rows
            if r["finalist"] == slug
            and r["train_k"] == r["eval_k"]
            and finite(r["snr10_tb2plus_db"])
        ]
        points.sort(key=lambda r: r["eval_k"])
        if not points:
            continue
        ax.plot(
            [r["eval_k"] for r in points],
            [float(r["snr10_tb2plus_db"]) for r in points],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("NRX iterations K")
    ax.set_ylabel("SNR at 10% TB2+ TBLER (dB)")
    ax.set_title("Cold vs temporal NRX — models trained for their evaluation K")
    ax.set_xticks(range(2, 9))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    png = SUMMARY / "diagonal_snr10_vs_k.png"
    pdf = SUMMARY / "diagonal_snr10_vs_k.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"WROTE={png}")


def write_diagonal_table(rows, cold):
    path = SUMMARY / "diagonal_snr10_vs_k.csv"
    by = {(r["finalist"], r["train_k"]): r for r in rows if r["train_k"] == r["eval_k"]}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "k",
                "cold_snr10_db",
                "mean_pca_d56",
                "cnn_pca_d16",
                "cnn_autoencoder_d56",
                "mean_writer_d32",
            ]
        )
        for k in range(2, 9):
            row = [k, cold.get(k)]
            for slug, _ in FINALISTS:
                r = by.get((slug, k))
                row.append(None if r is None else r["snr10_tb2plus_db"])
            w.writerow(row)
    print(f"WROTE={path}")


def main():
    rows = collect_rows()
    if not rows:
        raise RuntimeError(f"No evaluation cells found under {ROOT / 'evaluations'}")
    write_rows(rows)
    cold = parse_cold_csv(A.cold_csv)
    for slug, label in FINALISTS:
        plot_heatmap(rows, slug, label)
    plot_diagonal(rows, cold)
    write_diagonal_table(rows, cold)

    complete = sum(1 for r in rows if finite(r["snr10_tb2plus_db"]))
    print(
        "ITERATION_SWEEP_PLOT_SUMMARY="
        + json.dumps(
            {
                "finite_cells": complete,
                "expected_cells": 112,
                "cold_points": cold,
                "summary_dir": str(SUMMARY),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
