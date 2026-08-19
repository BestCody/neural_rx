#!/usr/bin/env python3
"""Diagnose completed temporal autoencoder runs without retraining.

Reads the existing research-suite training/evaluation artifacts on Atlas and
summarizes whether null SNR@10% crossings are range misses, training failures,
or signs of unstable/ineffective autoencoder memory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

POOLINGS = ("mean", "attention", "cnn")
CAPS = (8, 16, 32, 56)
TARGET = 0.1


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"))
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--output", required=True)
    return p.parse_args()


def load(path: Path):
    return json.loads(path.read_text())


def finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def curve_status(points):
    vals = [float(p["bler_tb2plus"]) for p in points if p.get("bler_tb2plus") is not None]
    if not vals:
        return "no_valid_points"
    above = any(v >= TARGET for v in vals)
    below = any(v <= TARGET for v in vals)
    if above and below:
        return "bracketed"
    if all(v > TARGET for v in vals):
        return "above_10pct_at_snr_max"
    if all(v < TARGET for v in vals):
        return "below_10pct_at_snr_min"
    return "unbracketed_nonmonotonic"


def parse_json_markers(text: str):
    rows = []
    markers = (
        "TRAIN_STEP=",
        "TEMPORAL_COMPRESSION_GRADIENT_CHECK=",
        "TRAINING_SUMMARY=",
        "V3_TRAINING_SUMMARY=",
        "V4_POOLING_SUMMARY=",
    )
    for line in text.splitlines():
        for marker in markers:
            idx = line.find(marker)
            if idx < 0:
                continue
            raw = line[idx + len(marker):].strip()
            try:
                rows.append((marker[:-1], json.loads(raw)))
            except Exception:
                pass
            break
    return rows


def choose_training_rows(markers):
    step_rows = [obj for marker, obj in markers if marker == "TRAIN_STEP" and isinstance(obj, dict) and "step" in obj]
    if not step_rows:
        # Some trainer revisions print bare JSON rows with a different prefix.
        step_rows = [obj for _, obj in markers if isinstance(obj, dict) and "step" in obj and "loss" in obj]
    step_rows.sort(key=lambda x: int(x.get("step", -1)))
    if not step_rows:
        return {}
    first = step_rows[0]
    last = step_rows[-1]
    joint = next((r for r in step_rows if str(r.get("phase")) == "joint"), None)
    return {"first": first, "first_joint": joint, "last": last, "count": len(step_rows)}


def summarize_case(root: Path, seed: int, pooling: str, d_mem: int):
    name = f"{pooling}_autoencoder_d{d_mem}"
    train_dir = root / "trained" / "fixed" / f"seed_{seed}" / name
    eval_dir = root / "evaluations" / "fixed" / f"seed_{seed}" / f"factorial_{name}"
    summary_path = train_dir / "training_summary.json"
    log_path = train_dir / "train.log"
    eval_path = eval_dir / "evaluation.json"

    item = {
        "pooling": pooling,
        "d_mem": d_mem,
        "training_dir": str(train_dir),
        "evaluation_dir": str(eval_dir),
        "training_summary_exists": summary_path.exists(),
        "train_log_exists": log_path.exists(),
        "evaluation_exists": eval_path.exists(),
    }

    ckpts = sorted(train_dir.glob("*.weights.h5")) if train_dir.exists() else []
    item["checkpoint_count"] = len(ckpts)
    item["checkpoint_sizes_bytes"] = [p.stat().st_size for p in ckpts]

    if summary_path.exists():
        s = load(summary_path)
        item["training_metadata"] = {
            k: s.get(k)
            for k in (
                "architecture", "pooling", "compression", "d_mem", "num_it",
                "train_steps", "memory_only_steps", "batch_size", "seq_len",
                "seed", "dynamic_scheduling", "checkpoint",
                "ae_reconstruction_weight", "temporal_gradient_check",
            )
            if k in s
        }

    if log_path.exists():
        text = log_path.read_text(errors="replace")
        lower = text.lower()
        item["log_has_traceback"] = "traceback (most recent call last)" in lower
        item["log_has_nan_token"] = bool(re.search(r"(?<![a-z])nan(?![a-z])", lower))
        item["log_has_inf_token"] = bool(re.search(r"(?<![a-z])inf(?:inity)?(?![a-z])", lower))
        markers = parse_json_markers(text)
        item["gradient_checks"] = [obj for marker, obj in markers if marker == "TEMPORAL_COMPRESSION_GRADIENT_CHECK"]
        item["training_rows"] = choose_training_rows(markers)

    if eval_path.exists():
        e = load(eval_path)
        curves = e.get("curves", {})
        temporal = curves.get("temporal_k2", [])
        cold2 = curves.get("cold_k2", [])
        cold8 = curves.get("cold_k8", [])
        cross = e.get("snr_db_at_10pct_tbler", {})
        item["reported_crossings"] = cross
        item["curve_status"] = curve_status(temporal)
        item["snr_grid_db"] = e.get("snr_grid_db")
        item["temporal_curve"] = [
            {
                "snr_db": p.get("snr_db"),
                "bler_tb2plus": p.get("bler_tb2plus"),
                "errors_tb2plus": p.get("errors_tb2plus"),
                "blocks_tb2plus": p.get("blocks_tb2plus"),
            }
            for p in temporal
        ]
        if temporal:
            valid = [p for p in temporal if p.get("bler_tb2plus") is not None]
            if valid:
                item["temporal_bler_at_snr_min"] = valid[0]["bler_tb2plus"]
                item["temporal_bler_at_snr_max"] = valid[-1]["bler_tb2plus"]
                item["temporal_min_bler"] = min(float(p["bler_tb2plus"]) for p in valid)
        if temporal and cold2:
            tlast = temporal[-1].get("bler_tb2plus")
            c2last = cold2[-1].get("bler_tb2plus")
            if finite(tlast) and finite(c2last):
                item["temporal_minus_cold_k2_bler_at_snr_max"] = float(tlast) - float(c2last)
        if temporal and cold8:
            tlast = temporal[-1].get("bler_tb2plus")
            c8last = cold8[-1].get("bler_tb2plus")
            if finite(tlast) and finite(c8last):
                item["temporal_minus_cold_k8_bler_at_snr_max"] = float(tlast) - float(c8last)

    reasons = []
    if not item["training_summary_exists"] or item["checkpoint_count"] == 0:
        reasons.append("missing_training_artifact")
    if not item["evaluation_exists"]:
        reasons.append("missing_evaluation_artifact")
    if item.get("log_has_traceback"):
        reasons.append("training_traceback_present")
    if item.get("log_has_nan_token") or item.get("log_has_inf_token"):
        reasons.append("nonfinite_training_log_token")
    grad = item.get("gradient_checks") or []
    if grad and not all(bool(x.get("passed")) for x in grad if isinstance(x, dict)):
        reasons.append("temporal_gradient_check_failed")
    status = item.get("curve_status")
    if status == "above_10pct_at_snr_max":
        reasons.append("evaluation_range_too_low_for_crossing_or_model_is_poor")
    elif status == "below_10pct_at_snr_min":
        reasons.append("evaluation_range_too_high_for_crossing")
    item["flags"] = reasons
    return item


def main():
    a = args()
    root = Path(a.root).expanduser().resolve()
    cases = [summarize_case(root, a.seed, p, d) for p in POOLINGS for d in CAPS]

    null_cases = [c for c in cases if c.get("reported_crossings", {}).get("temporal_k2") is None]
    finite_cases = [c for c in cases if c.get("reported_crossings", {}).get("temporal_k2") is not None]
    status_counts = {}
    for c in cases:
        status_counts[c.get("curve_status", "missing")] = status_counts.get(c.get("curve_status", "missing"), 0) + 1

    report = {
        "root": str(root),
        "seed": a.seed,
        "target_bler": TARGET,
        "case_count": len(cases),
        "null_crossing_count": len(null_cases),
        "finite_crossing_count": len(finite_cases),
        "curve_status_counts": status_counts,
        "cases": cases,
        "code_risk_hypotheses": {
            "unbounded_bottleneck": "Autoencoder bottleneck uses activation=None; unlike writer candidate tanh, persistent memory has no explicit bound or normalization.",
            "reconstruction_objective_tradeoff": "Autoencoder adds reconstruction MSE with default weight 0.1 in addition to decoding/channel losses, which may compete with future-TB usefulness.",
            "reader_saturation_path": "Raw autoencoder memory is fed into a tanh memory reader and into the sigmoid gate input; large bottleneck magnitudes can saturate both paths.",
        },
    }
    out = Path(a.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("AUTOENCODER_DEBUG=" + json.dumps({
        "null_crossing_count": len(null_cases),
        "finite_crossing_count": len(finite_cases),
        "curve_status_counts": status_counts,
        "output": str(out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
