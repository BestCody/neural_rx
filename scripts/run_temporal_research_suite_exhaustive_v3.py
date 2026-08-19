#!/usr/bin/env python3
"""Exhaustive factorial with PCA poolers tuned separately per memory capacity.

The 36 base cells remain:
  3 poolings x 3 compressors x 4 d_mem values.

For Attention/CNN + PCA only, each d_mem gets its own learned-pooler calibration
through a temporary writer with that SAME d_mem. The pooler is then frozen, PCA
is fit once, and the normal 6000-step temporal run proceeds. No pooler is shared
across PCA capacities.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_temporal_research_suite_exhaustive as suite

base = suite.base
A = suite.A
ROOT = suite.ROOT
SCRIPT_DIR = suite.SCRIPT_DIR
PY = suite.PY


def capacity_pca_valid(out, pooling, d_mem, seed, dynamic):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_pca_idaware_d{d_mem}_k2.weights.h5"
    if not summary.exists() or not ckpt.exists():
        return False
    try:
        s = base.load_json(summary)
        protocol = s.get("pca_protocol", {})
        calibration = s.get("pooler_calibration", {})
        return all(
            [
                s.get("architecture")
                == "ue_identity_aware_temporal_memory_v7_pca_capacity_tuned_pooler",
                s.get("pooling") == pooling,
                s.get("compression") == "pca",
                int(s.get("d_mem", -1)) == int(d_mem),
                int(s.get("num_it", -1)) == 2,
                int(s.get("train_steps", -1)) == A.train_steps,
                int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
                int(s.get("seq_len", -1)) == A.seq_len,
                int(s.get("seed", -1)) == int(seed),
                bool(s.get("dynamic_scheduling")) == bool(dynamic),
                protocol.get("pooler_calibrated_before_fit") is True,
                protocol.get("pooler_frozen_before_fit") is True,
                protocol.get("pca_fitted_once") is True,
                protocol.get("pooler_frozen_during_temporal_training") is True,
                protocol.get("pca_basis_frozen_during_temporal_training") is True,
                protocol.get("pooler_tuned_to_target_d_mem") is True,
                protocol.get("shared_pooler_across_capacities") is False,
                int(protocol.get("target_d_mem", -1)) == int(d_mem),
                calibration.get("capacity_tuned") is True,
                int(calibration.get("proxy_memory_width", -1)) == int(d_mem),
            ]
        )
    except Exception:
        return False


def train_factorial(gpu, compression, pooling, d_mem, seed, dynamic=False):
    # Writer, autoencoder, and mean+PCA keep the established path.
    if compression != "pca" or pooling == "mean":
        return base.train_compressed(
            gpu, compression, pooling, d_mem, seed, dynamic=dynamic
        )

    if pooling not in {"attention", "cnn"}:
        raise ValueError(pooling)

    mode = "dynamic" if dynamic else "fixed"
    out = (
        ROOT
        / "trained"
        / mode
        / f"seed_{seed}"
        / f"{pooling}_{compression}_d{d_mem}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"ue_memory_{pooling}_pca_idaware_d{d_mem}_k2.weights.h5"

    if capacity_pca_valid(out, pooling, d_mem, seed, dynamic):
        base.safe_print("REUSE_CAPACITY_TUNED_PCA_TRAINING", out, flush=True)
        return ckpt

    # Never reuse v4/v5/v6 learned-pool PCA checkpoints here because they do not
    # prove that the pooler was calibrated under this exact target d_mem.
    label = f"train-{mode}-{pooling}-pca-capacity-tuned-d{d_mem}-seed{seed}"
    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py",
        "--pooling",
        pooling,
        "--compression",
        "pca",
        "--d-mem",
        d_mem,
        "--num-it",
        2,
        "--train-steps",
        A.train_steps,
        "--memory-only-steps",
        A.memory_only_steps,
        "--pooler-calibration-steps",
        A.memory_only_steps,
        "--batch-size",
        A.train_batch,
        "--seq-len",
        A.seq_len,
        "--min-ebno-db",
        1.0,
        "--max-ebno-db",
        5.0,
        "--memory-lr",
        1e-3,
        "--joint-lr",
        2e-5,
        "--ue-pool-size",
        4,
        "--schedule-switch-prob",
        0.65,
        "--schedule-reorder-prob",
        0.50,
        "--seed",
        seed,
        "--output-dir",
        out,
        "--log-every",
        25,
        "--gpu",
        gpu,
    ]
    if not dynamic:
        cmd.append("--fixed-scheduling")

    base.tee_run(cmd, out / "train.log", gpu, label)
    if not capacity_pca_valid(out, pooling, d_mem, seed, dynamic):
        raise RuntimeError(
            f"capacity-tuned learned-pool PCA output failed validation: {out}"
        )
    return ckpt


# The original exhaustive suite resolves this global at runtime for factorial,
# dynamic-winner, and extra-seed jobs, so replacing it updates every PCA path.
suite.learned_pca_valid = capacity_pca_valid
suite.train_factorial = train_factorial


def _canonicalize_winner(summary):
    """Derive winner labels and metrics from the same factorial row.

    This prevents stale/mislabelled winner metadata from disagreeing with the
    row that actually has the minimum finite temporal SNR@10% TBLER.
    """
    candidates = []
    for row in summary.get("rows", []):
        if row.get("group") != "factorial":
            continue
        snr10 = row.get("temporal_snr10")
        if snr10 is None:
            continue
        candidates.append((float(snr10), row))
    if not candidates:
        raise RuntimeError("suite summary contains no finite factorial winner")

    _, best = min(
        candidates,
        key=lambda item: (
            item[0],
            str(item[1].get("pooling")),
            str(item[1].get("compression")),
            int(item[1].get("d_mem")),
        ),
    )
    canonical = {
        "pooling": best["pooling"],
        "compression": best["compression"],
        "d_mem": int(best["d_mem"]),
        "fixed_seed": int(best["seed"]),
        "snr10_db": float(best["temporal_snr10"]),
        "gap_recovered_percent": best.get("gap_recovered_percent"),
    }
    previous = summary.get("winner")
    summary["winner"] = canonical
    summary["winner_metadata_source"] = (
        "minimum finite temporal_snr10 among suite_summary.rows group=factorial"
    )
    summary["winner_metadata_corrected"] = previous != canonical
    if previous != canonical:
        summary["winner_metadata_previous"] = previous


def _crossing_status(points, target=0.1):
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


def _annotate_unbracketed_factorial_rows(summary):
    """Explain null temporal crossings from the already-written eval curves."""
    diagnostics = []
    for row in summary.get("rows", []):
        if row.get("group") != "factorial" or row.get("temporal_snr10") is not None:
            continue

        pool = row["pooling"]
        comp = row["compression"]
        d_mem = int(row["d_mem"])
        seed = int(row["seed"])
        evaluation_path = (
            ROOT
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
        item["status"] = _crossing_status(points)
        item["snr_grid_db"] = evaluation.get("snr_grid_db")
        item["first_temporal_tbler"] = values[0][1] if values else None
        item["last_temporal_tbler"] = values[-1][1] if values else None
        item["min_temporal_tbler"] = min((v for _, v in values), default=None)
        item["max_temporal_tbler"] = max((v for _, v in values), default=None)
        diagnostics.append(item)

    summary["unbracketed_factorial_diagnostics"] = diagnostics


def main():
    suite.main()

    summary_path = ROOT / "suite_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["suite"] = "temporal_ue_memory_exhaustive_factorial_v3_capacity_tuned_pca"
    summary["pca_fairness_protocol"] = {
        "mean": "parameter-free mean pooled state; ordinary frozen PCA fit",
        "attention_cnn": (
            "for each d_mem separately: calibrate learned pooler through a "
            "same-d_mem temporal writer proxy, freeze pooler, fit PCA once, "
            "freeze PCA during temporal training"
        ),
    }
    summary["pca_pooler_shared_across_capacities"] = False
    summary["pca_pooler_tuned_per_capacity"] = True
    _canonicalize_winner(summary)
    _annotate_unbracketed_factorial_rows(summary)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("CAPACITY_TUNED_EXHAUSTIVE_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
