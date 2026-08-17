#!/usr/bin/env python3
"""Exhaustive temporal-memory suite with shared learned-pooler PCA calibration.

Before the 36-cell factorial starts, Attention and CNN are each calibrated once
for the base seed/fixed-scheduling regime. Their frozen weights are then reused
by every PCA capacity (8/16/32/56). If the eventual winner is learned-pool+PCA,
dynamic or extra-seed follow-up runs lazily create one matching calibration for
that new seed/scheduling regime before training the winner.
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
ORIGINAL_TRAIN_FACTORIAL = suite.train_factorial


def calibration_path(pooling, seed, dynamic):
    mode = "dynamic" if dynamic else "fixed"
    return ROOT / "pooler_calibration" / mode / f"seed_{seed}" / f"{pooling}.npz"


def calibration_valid(path, pooling, seed, dynamic):
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not meta_path.exists():
        return False
    try:
        m = json.loads(meta_path.read_text())
        return all(
            [
                m.get("protocol") == "shared_temporal_pooler_calibration_v1",
                m.get("pooling") == pooling,
                int(m.get("d_s", -1)) == 56,
                int(m.get("steps", -1)) == A.memory_only_steps,
                int(m.get("seed", -1)) == int(seed),
                int(m.get("seq_len", -1)) == A.seq_len,
                bool(m.get("dynamic_scheduling")) == bool(dynamic),
                m.get("proxy_compression") == "writer",
                int(m.get("proxy_memory_width", -1)) == 56,
                m.get("nrx_base_frozen") is True,
            ]
        )
    except Exception:
        return False


def ensure_calibration(gpu, pooling, seed, dynamic=False):
    if pooling not in {"attention", "cnn"}:
        raise ValueError(pooling)
    out = calibration_path(pooling, seed, dynamic)
    if calibration_valid(out, pooling, seed, dynamic):
        base.safe_print("REUSE_SHARED_POOLER_CALIBRATION", out, flush=True)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "dynamic" if dynamic else "fixed"
    cmd = [
        PY,
        SCRIPT_DIR / "calibrate_temporal_pooler.py",
        "--pooling",
        pooling,
        "--compression",
        "writer",
        "--d-mem",
        56,
        "--num-it",
        2,
        "--train-steps",
        A.train_steps,
        "--memory-only-steps",
        A.memory_only_steps,
        "--pooler-calibration-steps",
        A.memory_only_steps,
        "--pooler-calibration-output",
        out,
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
        "--log-every",
        25,
        "--gpu",
        gpu,
    ]
    if not dynamic:
        cmd.append("--fixed-scheduling")

    base.tee_run(
        cmd,
        out.parent / f"{pooling}_calibration.log",
        gpu,
        f"calibrate-{mode}-{pooling}-seed{seed}",
    )
    if not calibration_valid(out, pooling, seed, dynamic):
        raise RuntimeError(f"Shared pooler calibration failed validation: {out}")
    return out


def shared_pca_valid(out, pooling, d_mem, seed, dynamic, calibration_file):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_pca_idaware_d{d_mem}_k2.weights.h5"
    if not summary.exists() or not ckpt.exists():
        return False
    try:
        s = json.loads(summary.read_text())
        p = s.get("pca_protocol", {})
        cal = s.get("pooler_calibration", {})
        return all(
            [
                s.get("architecture")
                == "ue_identity_aware_temporal_memory_v6_pca_shared_pooler",
                s.get("pooling") == pooling,
                s.get("compression") == "pca",
                int(s.get("d_mem", -1)) == int(d_mem),
                int(s.get("train_steps", -1)) == A.train_steps,
                int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
                int(s.get("seed", -1)) == int(seed),
                bool(s.get("dynamic_scheduling")) == bool(dynamic),
                p.get("pooler_calibrated_before_fit") is True,
                p.get("pooler_frozen_before_fit") is True,
                p.get("pca_fitted_once") is True,
                p.get("shared_pooler_across_capacities") is True,
                Path(p.get("pooler_calibration_file", "")) == Path(calibration_file),
                cal.get("shared_across_pca_capacities") is True,
            ]
        )
    except Exception:
        return False


def train_factorial_shared(gpu, compression, pooling, d_mem, seed, dynamic=False):
    if compression != "pca" or pooling == "mean":
        return ORIGINAL_TRAIN_FACTORIAL(
            gpu, compression, pooling, d_mem, seed, dynamic=dynamic
        )

    calibration = ensure_calibration(gpu, pooling, seed, dynamic)
    mode = "dynamic" if dynamic else "fixed"
    out = (
        ROOT
        / "trained"
        / mode
        / f"seed_{seed}"
        / f"{pooling}_pca_d{d_mem}"
    )
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"ue_memory_{pooling}_pca_idaware_d{d_mem}_k2.weights.h5"

    if shared_pca_valid(out, pooling, d_mem, seed, dynamic, calibration):
        base.safe_print("REUSE_SHARED_POOLER_PCA_TRAINING", out, flush=True)
        return ckpt

    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_ue_memory_v6_pca_shared_pooler.py",
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
        "--pooler-calibration-file",
        calibration,
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

    base.tee_run(
        cmd,
        out / "train.log",
        gpu,
        f"train-{mode}-{pooling}-pca-shared-d{d_mem}-seed{seed}",
    )
    if not shared_pca_valid(out, pooling, d_mem, seed, dynamic, calibration):
        raise RuntimeError(f"Shared-pooler PCA training failed validation: {out}")
    return ckpt


def main():
    # Establish one fixed-scheduling learned representation per learned pooler
    # before any PCA capacity cell can begin.
    stage0 = [
        ("calibration:attention", lambda gpu: ensure_calibration(gpu, "attention", A.seed, False)),
        ("calibration:cnn", lambda gpu: ensure_calibration(gpu, "cnn", A.seed, False)),
    ]
    base.run_parallel("STAGE0_SHARED_POOLER_CALIBRATION", stage0)

    suite.train_factorial = train_factorial_shared
    suite.main()

    summary_path = ROOT / "suite_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["suite"] = (
            "temporal_ue_memory_exhaustive_factorial_v2_shared_calibrated_pca"
        )
        summary["pca_fairness_protocol"] = {
            "mean": "parameter-free mean pooling; fit PCA once",
            "attention_cnn": (
                "calibrate each learned pooler once per seed/scheduling regime, "
                "reuse identical frozen pooler for PCA d_mem 8/16/32/56, fit "
                "each PCA basis once, keep pooler and basis frozen"
            ),
        }
        summary["shared_pooler_calibrations"] = {
            pool: str(calibration_path(pool, A.seed, False))
            for pool in ["attention", "cnn"]
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        base.safe_print(
            "EXHAUSTIVE_V2_SUMMARY=" + json.dumps(summary, indent=2), flush=True
        )


if __name__ == "__main__":
    main()
