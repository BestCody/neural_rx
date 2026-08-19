#!/usr/bin/env python3
"""Repair and annotate a completed temporal research-suite summary in place.

This script never trains or evaluates a model. It only reads the already-written
suite_summary.json/all_results.csv/evaluation.json files, canonicalizes winner
metadata from the factorial rows, and records why any SNR@10% crossing is null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"),
    )
    p.add_argument("--diagnostics-output", default=None)
    return p.parse_args()


def crossing_status(points, target=0.1):
    values = [
        float(point["bler_tb2plus"])
        for point in points
        if point.get("bler_tb2plus") is not None
    ]
    if not values:
        return "no_valid_points"
    if any(v >= target for v in values) and any(v <= target for v in values):
        return "bracketed"
    if all(v > target for v in values):
        return "above_target_through_snr_max"
    if all(v < target for v in values):
        return "below_target_from_snr_min"
    return "unbracketed_nonmonotonic"


def canonical_winner(summary):
    candidates = []
    for row in summary.get("rows", []):
        if row.get("group") != "factorial":
            continue
        snr10 = row.get("temporal_snr10")
        if snr10 is None:
            continue
        candidates.append((float(snr10), row))
    if not candidates:
        raise RuntimeError("No finite factorial temporal_snr10 values in suite summary")
    _, best = min(
        candidates,
        key=lambda item: (
            item[0],
            str(item[1].get("pooling")),
            str(item[1].get("compression")),
            int(item[1].get("d_mem")),
        ),
    )
    return {
        "pooling": best["pooling"],
        "compression": best["compression"],
        "d_mem": int(best["d_mem"]),
        "fixed_seed": int(best["seed"]),
        "snr10_db": float(best["temporal_snr10"]),
        "gap_recovered_percent": best.get("gap_recovered_percent"),
    }


def collect_unbracketed(root, summary):
    diagnostics = []
    for row in summary.get("rows", []):
        if row.get("group") != "factorial" or row.get("temporal_snr10") is not None:
            continue
        pool = row["pooling"]
        comp = row["compression"]
        d_mem = int(row["d_mem"])
        seed = int(row["seed"])
        evaluation_path = (
            root
            / "evaluations"
            / "fixed"
            / f"seed_{seed}"
            / f"factorial_{pool}_{comp}_d{d_mem}"
            / "evaluation.json"
        )
        item = {
            "pooling": pool,
            "compression": comp,
            "d_mem": d_mem,
            "seed": seed,
            "evaluation_json": str(evaluation_path),
        }
        if not evaluation_path.exists():
            item["status"] = "missing_evaluation_json"
            diagnostics.append(item)
            continue

        evaluation = json.loads(evaluation_path.read_text())
        points = evaluation.get("curves", {}).get("temporal_k2", [])
        values = [
            (float(p["snr_db"]), float(p["bler_tb2plus"]))
            for p in points
            if p.get("bler_tb2plus") is not None
        ]
        item.update(
            {
                "status": crossing_status(points),
                "snr_grid_db": evaluation.get("snr_grid_db"),
                "first_temporal_tbler": values[0][1] if values else None,
                "last_temporal_tbler": values[-1][1] if values else None,
                "min_temporal_tbler": min((v for _, v in values), default=None),
                "max_temporal_tbler": max((v for _, v in values), default=None),
                "reported_crossing": evaluation.get("snr_db_at_10pct_tbler", {}).get(
                    "temporal_k2"
                ),
            }
        )
        diagnostics.append(item)
    return diagnostics


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summary_path = root / "suite_summary.json"
    summary = json.loads(summary_path.read_text())

    previous = summary.get("winner")
    winner = canonical_winner(summary)
    summary["winner"] = winner
    summary["winner_metadata_source"] = (
        "minimum finite temporal_snr10 among suite_summary.rows group=factorial"
    )
    summary["winner_metadata_corrected"] = previous != winner
    if previous != winner:
        summary["winner_metadata_previous"] = previous

    diagnostics = collect_unbracketed(root, summary)
    summary["unbracketed_factorial_diagnostics"] = diagnostics
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report = {
        "root": str(root),
        "winner_before": previous,
        "winner_after": winner,
        "winner_changed": previous != winner,
        "unbracketed_factorial_diagnostics": diagnostics,
    }
    out = (
        Path(args.diagnostics_output).expanduser().resolve()
        if args.diagnostics_output
        else root / "temporal_suite_diagnostics.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("TEMPORAL_SUITE_REPAIR=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
