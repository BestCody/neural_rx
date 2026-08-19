#!/usr/bin/env python3
"""Diagnose a completed temporal research suite without retraining models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"),
    )
    return p.parse_args()


def crossing_status(points, target=0.1):
    values = [
        float(p["bler_tb2plus"])
        for p in points
        if p.get("bler_tb2plus") is not None
    ]
    if not values:
        return "no_valid_points"
    has_above = any(v >= target for v in values)
    has_below = any(v <= target for v in values)
    if has_above and has_below:
        return "bracketed"
    if all(v > target for v in values):
        return "above_target_through_snr_max"
    if all(v < target for v in values):
        return "below_target_from_snr_min"
    return "unbracketed"


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary = json.loads((root / "suite_summary.json").read_text())
    rows = list(csv.DictReader((root / "all_results.csv").open()))

    factorial = []
    for row in rows:
        if row.get("group") != "factorial" or not row.get("temporal_snr10"):
            continue
        factorial.append(
            (
                float(row["temporal_snr10"]),
                row["pooling"],
                row["compression"],
                int(row["d_mem"]),
                float(row["gap_recovered_percent"]),
            )
        )
    factorial.sort()
    print("SUMMARY_WINNER=" + json.dumps(summary.get("winner"), sort_keys=True))
    print("CSV_BEST=" + json.dumps(factorial[0] if factorial else None))
    print("CSV_TOP5=" + json.dumps(factorial[:5]))

    eval_root = root / "evaluations" / "fixed" / "seed_20260816"
    for path in sorted(eval_root.glob("factorial_*_autoencoder_d*/evaluation.json")):
        data = json.loads(path.read_text())
        curve = data.get("curves", {}).get("temporal_k2", [])
        print(
            "AUTOENCODER="
            + json.dumps(
                {
                    "configuration": path.parent.name,
                    "crossing": data.get("snr_db_at_10pct_tbler", {}).get("temporal_k2"),
                    "crossing_status": crossing_status(curve),
                    "first_point": curve[0] if curve else None,
                    "last_point": curve[-1] if curve else None,
                    "min_tbler": min(
                        (p["bler_tb2plus"] for p in curve if p.get("bler_tb2plus") is not None),
                        default=None,
                    ),
                    "max_tbler": max(
                        (p["bler_tb2plus"] for p in curve if p.get("bler_tb2plus") is not None),
                        default=None,
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
