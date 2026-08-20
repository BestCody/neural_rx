#!/usr/bin/env python3
"""Strict, resumable runner for the 12 corrected protocol-v2 AE factorial cells.

This runner exists because the first corrected-AE launch exposed two orchestration
bugs that were unrelated to model training:
1) v3 used the requested seed but omitted it from training_summary.json, causing
   strict post-training validation to reject otherwise-complete protocol-v2
   checkpoints.
2) the queued "fail-fast" scheduler could start the next cell before the main
   thread observed a failure in a just-finished cell.

This version uses exact protocol manifests, checkpoint SHA256 binding, strict
full-run stability checks, and two-GPU waves. No next wave starts unless every
job in the current wave succeeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from temporal_eval_metrics import make_snr_grid
import run_temporal_research_suite_exhaustive_v4 as v4

base = v4.base
A = v4.A
ROOT = v4.ROOT
SCRIPT_DIR = base.SCRIPT_DIR
PY = base.PY
POOLINGS = ("mean", "attention", "cnn")
CAPACITIES = tuple(base.CAPS)

LEGACY_REPAIR_SOURCE_COMMIT = "928cab2ea1b1696d553040ad9b8fd0bf5ce80a8f"
TRAINING_MANIFEST_VERSION = 1
EVALUATION_STAMP_VERSION = 1

TRAIN = {
    "config": "nrx_large.cfg",
    "num_it": 2,
    "train_steps": 6000,
    "memory_only_steps": 1000,
    "batch_size": 8,
    "seq_len": 4,
    "min_ebno_db": 1.0,
    "max_ebno_db": 5.0,
    "memory_lr": 1e-3,
    "joint_lr": 2e-5,
    "chest_weight": 0.01,
    "ae_reconstruction_weight": 0.1,
    "ue_pool_size": 4,
    "memory_expiry_slots": 8,
    "schedule_switch_prob": 0.65,
    "schedule_reorder_prob": 0.50,
    "dynamic_scheduling": False,
    "seed": 20260816,
}
EVAL = {
    "config": "nrx_large.cfg",
    "num_it": 2,
    "seq_len": 4,
    "batch_size": 8,
    "snr_min": 1.5,
    "snr_max": 3.75,
    "snr_step": 0.25,
    "target_errors": 120,
    "max_batches": 32,
    "ue_pool_size": 4,
    "schedule_switch_prob": 0.65,
    "schedule_reorder_prob": 0.50,
    "dynamic_scheduling": False,
    "memory_expiry_slots": 8,
    "seed": 20260816,
}


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(SCRIPT_DIR.parent),
        text=True,
    ).strip()


def tracked_tree_clean() -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(SCRIPT_DIR.parent),
        text=True,
    )
    return not status.strip()


def close(a, b, tol=1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def training_dir(pooling: str, d_mem: int) -> Path:
    return (
        ROOT / "trained" / "fixed" / f"seed_{A.seed}"
        / f"{pooling}_autoencoder_d{d_mem}"
    )


def checkpoint_path(pooling: str, d_mem: int) -> Path:
    return training_dir(pooling, d_mem) / (
        f"ue_memory_{pooling}_autoencoder_idaware_d{d_mem}_k2.weights.h5"
    )


def manifest_path(pooling: str, d_mem: int) -> Path:
    return training_dir(pooling, d_mem) / "corrected_ae_training_manifest.json"


def evaluation_dir(pooling: str, d_mem: int) -> Path:
    return (
        ROOT / "evaluations" / "fixed" / f"seed_{A.seed}"
        / f"factorial_{pooling}_autoencoder_d{d_mem}"
    )


def stability(summary: dict, d_mem: int) -> dict:
    history = summary.get("history") or []
    if not history:
        return {"passed": False, "reason": "missing_training_history"}

    scalars = []
    memory_norms = []
    for row in history:
        for key in (
            "loss",
            "loss_data",
            "loss_chest",
            "compression_aux_loss",
            "gradient_norm",
        ):
            scalars.append(row.get(key))
        scalars.extend(row.get("reconstruction_mse_per_tb") or [])
        memory_norms.extend(row.get("memory_norm_per_tb") or [])

    all_finite = all(finite(x) for x in scalars + memory_norms)
    last = history[-1]
    last_data = float(last["loss_data"]) if finite(last.get("loss_data")) else 0.0
    last_aux = (
        float(last["compression_aux_loss"])
        if finite(last.get("compression_aux_loss"))
        else math.inf
    )
    weighted_ratio = (
        TRAIN["ae_reconstruction_weight"] * last_aux / last_data
        if last_data > 0 else math.inf
    )
    max_norm = max((float(x) for x in memory_norms), default=math.inf)
    norm_bound = math.sqrt(int(d_mem)) + 1e-3
    identity = summary.get("identity_routing_check") or {}
    grad = summary.get("temporal_compression_gradient_check") or {}

    passed = bool(
        all_finite
        and int(last.get("step", -1)) == TRAIN["train_steps"] - 1
        and last.get("phase") == "joint"
        and max_norm <= norm_bound
        and weighted_ratio < 1.0
        and identity.get("passed") is True
        and grad.get("passed") is True
        and finite(grad.get("compression_path_grad_norm"))
        and float(grad.get("compression_path_grad_norm", 0.0)) > 0.0
    )
    return {
        "passed": passed,
        "finite_logged_training": all_finite,
        "last_step": int(last.get("step", -1)),
        "last_phase": last.get("phase"),
        "max_memory_norm": max_norm,
        "memory_norm_bound": norm_bound,
        "last_weighted_reconstruction_over_data": weighted_ratio,
        "identity_routing_passed": identity.get("passed") is True,
        "temporal_gradient_passed": grad.get("passed") is True,
        "temporal_gradient_norm": grad.get("compression_path_grad_norm"),
    }


def summary_valid(
    summary: dict,
    pooling: str,
    d_mem: int,
    *,
    allow_missing_seed: bool,
) -> tuple[bool, list[str]]:
    failures = []

    def req(ok, label):
        if not ok:
            failures.append(label)

    req(
        summary.get("architecture")
        == "ue_identity_aware_temporal_memory_v4_pooling",
        "architecture",
    )
    req(summary.get("config") == TRAIN["config"], "config")
    req(summary.get("pooling") == pooling, "pooling")
    req(summary.get("compression") == "autoencoder", "compression")
    req(int(summary.get("d_mem", -1)) == int(d_mem), "d_mem")
    req(int(summary.get("num_it", -1)) == TRAIN["num_it"], "num_it")
    req(int(summary.get("train_steps", -1)) == TRAIN["train_steps"], "train_steps")
    req(
        int(summary.get("memory_only_steps", -1)) == TRAIN["memory_only_steps"],
        "memory_only_steps",
    )
    req(int(summary.get("batch_size", -1)) == TRAIN["batch_size"], "batch_size")
    req(int(summary.get("seq_len", -1)) == TRAIN["seq_len"], "seq_len")
    req(
        int(summary.get("ue_pool_size", -1)) == TRAIN["ue_pool_size"],
        "ue_pool_size",
    )
    req(
        int(summary.get("memory_expiry_slots", -1))
        == TRAIN["memory_expiry_slots"],
        "memory_expiry_slots",
    )
    req(
        bool(summary.get("dynamic_scheduling")) == TRAIN["dynamic_scheduling"],
        "dynamic_scheduling",
    )
    req(
        close(summary.get("schedule_switch_prob"), TRAIN["schedule_switch_prob"]),
        "schedule_switch_prob",
    )
    req(
        close(summary.get("schedule_reorder_prob"), TRAIN["schedule_reorder_prob"]),
        "schedule_reorder_prob",
    )
    req(
        close(
            summary.get("ae_reconstruction_weight"),
            TRAIN["ae_reconstruction_weight"],
        ),
        "ae_reconstruction_weight",
    )
    req(v4._autoencoder_protocol_valid(summary), "autoencoder_protocol_v2")

    if summary.get("seed") is None and allow_missing_seed:
        pass
    else:
        req(int(summary.get("seed", -1)) == A.seed, "seed")

    try:
        recorded_ckpt = Path(summary.get("checkpoint", "")).expanduser().resolve()
    except (TypeError, ValueError):
        recorded_ckpt = Path("/")
    req(recorded_ckpt == checkpoint_path(pooling, d_mem).resolve(), "checkpoint_path")
    req(stability(summary, d_mem).get("passed") is True, "full_training_stability")
    return not failures, failures


def write_manifest(
    pooling: str,
    d_mem: int,
    summary: dict,
    code_commit: str,
    provenance_repair: dict | None = None,
) -> dict:
    ckpt = checkpoint_path(pooling, d_mem)
    manifest = {
        "manifest_version": TRAINING_MANIFEST_VERSION,
        "purpose": "corrected_autoencoder_factorial_protocol_v2",
        "pooling": pooling,
        "compression": "autoencoder",
        "d_mem": int(d_mem),
        "training_semantics": dict(TRAIN),
        "autoencoder_protocol": summary.get("autoencoder_protocol"),
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_size_bytes": ckpt.stat().st_size,
        "checkpoint_sha256": sha256_file(ckpt),
        "training_code_commit": code_commit,
        "stability": stability(summary, d_mem),
        "provenance_repair": provenance_repair,
    }
    write_json(manifest_path(pooling, d_mem), manifest)
    return manifest


def manifest_valid(pooling: str, d_mem: int, manifest: dict) -> tuple[bool, list[str]]:
    failures = []

    def req(ok, label):
        if not ok:
            failures.append(label)

    req(
        int(manifest.get("manifest_version", -1)) == TRAINING_MANIFEST_VERSION,
        "manifest_version",
    )
    req(manifest.get("pooling") == pooling, "pooling")
    req(manifest.get("compression") == "autoencoder", "compression")
    req(int(manifest.get("d_mem", -1)) == int(d_mem), "d_mem")
    req(manifest.get("training_semantics") == TRAIN, "training_semantics")
    req(
        v4._autoencoder_protocol_valid(
            {"autoencoder_protocol": manifest.get("autoencoder_protocol")}
        ),
        "autoencoder_protocol_v2",
    )
    req((manifest.get("stability") or {}).get("passed") is True, "stability")

    ckpt = checkpoint_path(pooling, d_mem)
    req(ckpt.is_file(), "checkpoint_exists")
    if ckpt.is_file():
        req(
            Path(manifest.get("checkpoint", "")).expanduser().resolve()
            == ckpt.resolve(),
            "checkpoint_path",
        )
        req(
            int(manifest.get("checkpoint_size_bytes", -1)) == ckpt.stat().st_size,
            "checkpoint_size",
        )
        req(
            manifest.get("checkpoint_sha256") == sha256_file(ckpt),
            "checkpoint_sha256",
        )
    return not failures, failures


def repair_known_first_run() -> list[str]:
    """Repair only protocol-v2 artifacts from the known interrupted first run."""
    repaired = []
    for pooling in POOLINGS:
        for d_mem in CAPACITIES:
            summary_file = training_dir(pooling, d_mem) / "training_summary.json"
            ckpt = checkpoint_path(pooling, d_mem)
            if not summary_file.is_file() or not ckpt.is_file():
                continue
            if manifest_path(pooling, d_mem).is_file():
                continue

            summary = load_json(summary_file)
            if "seed" in summary:
                continue

            ok, failures = summary_valid(
                summary, pooling, d_mem, allow_missing_seed=True
            )
            if not ok:
                raise RuntimeError(
                    f"refusing provenance repair for {summary_file.parent}; "
                    f"failed checks={failures}"
                )

            repair = {
                "reason": (
                    "v4 protocol-v2 training used --seed 20260816 but v3 omitted "
                    "the seed field from training_summary.json"
                ),
                "source_runner_commit": LEGACY_REPAIR_SOURCE_COMMIT,
                "seed_scoped_output_directory": f"seed_{A.seed}",
            }
            summary["seed"] = A.seed
            summary["seed_provenance_repair"] = repair
            write_json(summary_file, summary)
            write_manifest(
                pooling,
                d_mem,
                summary,
                code_commit=LEGACY_REPAIR_SOURCE_COMMIT,
                provenance_repair=repair,
            )
            repaired.append(str(summary_file.parent))
            print(
                f"REPAIRED_LEGACY_PROTOCOL_V2_ARTIFACT={summary_file.parent}",
                flush=True,
            )
    return repaired


def training_valid(pooling: str, d_mem: int) -> tuple[bool, list[str]]:
    summary_file = training_dir(pooling, d_mem) / "training_summary.json"
    manifest_file = manifest_path(pooling, d_mem)
    if not summary_file.is_file() or not manifest_file.is_file():
        return False, ["missing_summary_or_manifest"]
    try:
        s_ok, s_fail = summary_valid(
            load_json(summary_file),
            pooling,
            d_mem,
            allow_missing_seed=False,
        )
        m_ok, m_fail = manifest_valid(
            pooling, d_mem, load_json(manifest_file)
        )
        return (
            s_ok and m_ok,
            [f"summary:{x}" for x in s_fail]
            + [f"manifest:{x}" for x in m_fail],
        )
    except Exception as exc:
        return False, [f"{type(exc).__name__}:{exc}"]


def train_cell(gpu: int, pooling: str, d_mem: int, current_commit: str) -> Path:
    out = training_dir(pooling, d_mem)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(pooling, d_mem)

    valid, failures = training_valid(pooling, d_mem)
    if valid:
        print(f"REUSE_CORRECTED_AE_TRAINING={out}", flush=True)
        return ckpt

    if (out / "training_summary.json").exists() or ckpt.exists():
        print(
            "RETRAIN_INVALID_CORRECTED_AE_ARTIFACT="
            + json.dumps({"path": str(out), "failures": failures}),
            flush=True,
        )

    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_ue_memory_v4.py",
        "--gpu", gpu,
        "--pooling", pooling,
        "--compression", "autoencoder",
        "--d-mem", d_mem,
        "--num-it", TRAIN["num_it"],
        "--train-steps", TRAIN["train_steps"],
        "--memory-only-steps", TRAIN["memory_only_steps"],
        "--batch-size", TRAIN["batch_size"],
        "--seq-len", TRAIN["seq_len"],
        "--min-ebno-db", TRAIN["min_ebno_db"],
        "--max-ebno-db", TRAIN["max_ebno_db"],
        "--memory-lr", TRAIN["memory_lr"],
        "--joint-lr", TRAIN["joint_lr"],
        "--chest-weight", TRAIN["chest_weight"],
        "--ae-reconstruction-weight", TRAIN["ae_reconstruction_weight"],
        "--ue-pool-size", TRAIN["ue_pool_size"],
        "--memory-expiry-slots", TRAIN["memory_expiry_slots"],
        "--schedule-switch-prob", TRAIN["schedule_switch_prob"],
        "--schedule-reorder-prob", TRAIN["schedule_reorder_prob"],
        "--fixed-scheduling",
        "--seed", A.seed,
        "--output-dir", out,
        "--log-every", 25,
    ]
    base.tee_run(
        cmd,
        out / "train.log",
        gpu,
        f"train-fixed-{pooling}-autoencoder-d{d_mem}-seed{A.seed}",
    )

    summary_file = out / "training_summary.json"
    if not summary_file.is_file() or not ckpt.is_file():
        raise RuntimeError(f"incomplete training artifacts: {out}")

    summary = load_json(summary_file)
    ok, failures = summary_valid(
        summary, pooling, d_mem, allow_missing_seed=False
    )
    if not ok:
        raise RuntimeError(
            f"training failed corrected-AE validation: {out}; {failures}"
        )
    write_manifest(pooling, d_mem, summary, current_commit)

    valid, failures = training_valid(pooling, d_mem)
    if not valid:
        raise RuntimeError(
            f"training manifest failed self-check: {out}; {failures}"
        )
    return ckpt


def evaluation_valid(
    evaluation: dict,
    pooling: str,
    d_mem: int,
    checkpoint_sha: str,
) -> tuple[bool, list[str]]:
    failures = []

    def req(ok, label):
        if not ok:
            failures.append(label)

    req(
        evaluation.get("experiment")
        == "temporal_ue_memory_132prb_evaluation_v2",
        "experiment",
    )
    req(evaluation.get("config") == EVAL["config"], "config")
    req(evaluation.get("parameter_mode") == "training=False", "parameter_mode")
    req(int(evaluation.get("n_size_bwp", -1)) == 132, "n_size_bwp")
    req(evaluation.get("compression") == "autoencoder", "compression")
    req(evaluation.get("pooling") == pooling, "pooling")
    req(int(evaluation.get("d_mem", -1)) == int(d_mem), "d_mem")
    req(int(evaluation.get("seq_len", -1)) == EVAL["seq_len"], "seq_len")
    req(int(evaluation.get("batch_size", -1)) == EVAL["batch_size"], "batch_size")
    req(
        int(evaluation.get("target_errors", -1)) == EVAL["target_errors"],
        "target_errors",
    )
    req(
        int(evaluation.get("max_batches", -1)) == EVAL["max_batches"],
        "max_batches",
    )
    req(int(evaluation.get("seed", -1)) == A.seed, "seed")
    req(
        bool(evaluation.get("dynamic_scheduling")) == EVAL["dynamic_scheduling"],
        "dynamic_scheduling",
    )
    req(
        int(evaluation.get("ue_pool_size", -1)) == EVAL["ue_pool_size"],
        "ue_pool_size",
    )
    req(
        close(evaluation.get("schedule_switch_prob"), EVAL["schedule_switch_prob"]),
        "schedule_switch_prob",
    )
    req(
        close(evaluation.get("schedule_reorder_prob"), EVAL["schedule_reorder_prob"]),
        "schedule_reorder_prob",
    )

    expected_grid = make_snr_grid(
        EVAL["snr_min"], EVAL["snr_max"], EVAL["snr_step"]
    )
    req(evaluation.get("snr_grid_db") == expected_grid, "snr_grid_db")
    req(
        str(evaluation.get("crossing_method", "")).startswith(
            "log-BLER interpolation"
        ),
        "crossing_method",
    )

    try:
        recorded_ckpt = Path(
            evaluation.get("checkpoint", "")
        ).expanduser().resolve()
    except (TypeError, ValueError):
        recorded_ckpt = Path("/")
    req(recorded_ckpt == checkpoint_path(pooling, d_mem).resolve(), "checkpoint_path")

    stamp = evaluation.get("corrected_ae_evaluation_stamp") or {}
    req(
        int(stamp.get("version", -1)) == EVALUATION_STAMP_VERSION,
        "evaluation_stamp_version",
    )
    req(stamp.get("checkpoint_sha256") == checkpoint_sha, "checkpoint_sha256")
    req(int(stamp.get("num_it", -1)) == EVAL["num_it"], "stamp_num_it")
    req(
        int(stamp.get("memory_expiry_slots", -1))
        == EVAL["memory_expiry_slots"],
        "stamp_memory_expiry_slots",
    )

    curves = evaluation.get("curves") or {}
    req(set(curves) == {"cold_k2", "cold_k8", "temporal_k2"}, "curve_methods")
    for method in ("cold_k2", "cold_k8", "temporal_k2"):
        points = curves.get(method) or []
        req(len(points) == len(expected_grid), f"{method}:point_count")
        if len(points) != len(expected_grid):
            continue
        for idx, (point, snr) in enumerate(zip(points, expected_grid)):
            prefix = f"{method}:{idx}"
            req(close(point.get("snr_db"), snr, 1e-9), f"{prefix}:snr")
            bler = point.get("bler_tb2plus")
            req(
                bler is not None
                and finite(bler)
                and 0.0 <= float(bler) <= 1.0,
                f"{prefix}:bler",
            )
            req(int(point.get("blocks_tb2plus", 0)) > 0, f"{prefix}:blocks")
            req(int(point.get("errors_tb2plus", -1)) >= 0, f"{prefix}:errors")
            req(
                1 <= int(point.get("batches", 0)) <= EVAL["max_batches"],
                f"{prefix}:batches",
            )

    crossing = evaluation.get("snr_db_at_10pct_tbler") or {}
    c2, c8, ct = (
        crossing.get("cold_k2"),
        crossing.get("cold_k8"),
        crossing.get("temporal_k2"),
    )
    if c2 is not None and c8 is not None:
        req(finite(c2) and finite(c8), "cold_crossings_finite")
        if finite(c2) and finite(c8):
            req(float(c2) > float(c8), "positive_k2_to_k8_gap")
            req(
                close(
                    evaluation.get("cold_iteration_gap_db"),
                    float(c2) - float(c8),
                    1e-9,
                ),
                "cold_gap_consistency",
            )
    if ct is not None:
        req(finite(ct), "temporal_crossing_finite")

    return not failures, failures


def evaluate_cell(
    gpu: int,
    ckpt: Path,
    pooling: str,
    d_mem: int,
    current_commit: str,
) -> dict:
    manifest = load_json(manifest_path(pooling, d_mem))
    checkpoint_sha = manifest["checkpoint_sha256"]
    out = evaluation_dir(pooling, d_mem)
    out.mkdir(parents=True, exist_ok=True)
    evaluation_file = out / "evaluation.json"

    if evaluation_file.is_file():
        try:
            existing = load_json(evaluation_file)
            valid, failures = evaluation_valid(
                existing, pooling, d_mem, checkpoint_sha
            )
            if valid:
                print(f"REUSE_CORRECTED_AE_EVALUATION={out}", flush=True)
                return existing
            print(
                "RERUN_INVALID_CORRECTED_AE_EVALUATION="
                + json.dumps({"path": str(out), "failures": failures}),
                flush=True,
            )
        except Exception as exc:
            print(
                "RERUN_UNREADABLE_CORRECTED_AE_EVALUATION="
                + json.dumps({"path": str(out), "error": repr(exc)}),
                flush=True,
            )

    cmd = [
        PY,
        SCRIPT_DIR / "evaluate_temporal_ue_memory_v2.py",
        "--checkpoint", ckpt,
        "--config", EVAL["config"],
        "--gpu", gpu,
        "--compression", "autoencoder",
        "--pooling", pooling,
        "--d-mem", d_mem,
        "--num-it", EVAL["num_it"],
        "--seq-len", EVAL["seq_len"],
        "--batch-size", EVAL["batch_size"],
        "--snr-min", EVAL["snr_min"],
        "--snr-max", EVAL["snr_max"],
        "--snr-step", EVAL["snr_step"],
        "--target-errors", EVAL["target_errors"],
        "--max-batches", EVAL["max_batches"],
        "--seed", A.seed,
        "--ue-pool-size", EVAL["ue_pool_size"],
        "--schedule-switch-prob", EVAL["schedule_switch_prob"],
        "--schedule-reorder-prob", EVAL["schedule_reorder_prob"],
        "--output-dir", out,
    ]
    base.tee_run(
        cmd,
        out / "eval.log",
        gpu,
        f"eval-fixed-{pooling}-autoencoder-d{d_mem}-seed{A.seed}",
    )

    if not evaluation_file.is_file():
        raise RuntimeError(f"evaluation.json missing after evaluation: {out}")
    evaluation = load_json(evaluation_file)
    evaluation["corrected_ae_evaluation_stamp"] = {
        "version": EVALUATION_STAMP_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "runner_commit": current_commit,
        "num_it": EVAL["num_it"],
        "memory_expiry_slots": EVAL["memory_expiry_slots"],
    }
    write_json(evaluation_file, evaluation)

    valid, failures = evaluation_valid(
        evaluation, pooling, d_mem, checkpoint_sha
    )
    if not valid:
        raise RuntimeError(
            f"evaluation failed corrected-AE validation: {out}; {failures}"
        )
    return evaluation


def result_row(pooling: str, d_mem: int, evaluation: dict) -> dict:
    crossing = evaluation.get("snr_db_at_10pct_tbler") or {}
    return {
        "pooling": pooling,
        "compression": "autoencoder",
        "d_mem": int(d_mem),
        "seed": A.seed,
        "temporal_snr10": crossing.get("temporal_k2"),
        "cold_k2_snr10": crossing.get("cold_k2"),
        "cold_k8_snr10": crossing.get("cold_k8"),
        "gap_recovered_percent": evaluation.get("gap_recovered_percent"),
        "memory_bits_per_ue": evaluation.get("memory_bits_per_ue"),
    }


def run_cell(gpu: int, pooling: str, d_mem: int, current_commit: str) -> dict:
    ckpt = train_cell(gpu, pooling, d_mem, current_commit)
    evaluation = evaluate_cell(
        gpu, ckpt, pooling, d_mem, current_commit
    )
    return {
        "pooling": pooling,
        "d_mem": int(d_mem),
        "checkpoint": str(ckpt),
        "training_manifest": load_json(manifest_path(pooling, d_mem)),
        "evaluation": evaluation,
    }


def write_progress(results: dict, current_commit: str) -> None:
    rows = [
        result_row(v["pooling"], v["d_mem"], v["evaluation"])
        for v in results.values()
    ]
    rows.sort(key=lambda r: (POOLINGS.index(r["pooling"]), r["d_mem"]))
    write_json(
        ROOT / "corrected_autoencoder_progress.json",
        {
            "suite": "corrected_autoencoder_factorial_protocol_v2",
            "status": "complete" if len(results) == 12 else "running",
            "runner_commit": current_commit,
            "completed_cells": len(results),
            "total_cells": 12,
            "completed": [
                f'{r["pooling"]}_autoencoder_d{r["d_mem"]}' for r in rows
            ],
            "rows": rows,
        },
    )


def run_waves(jobs, current_commit: str) -> dict:
    """Run at most one cell per GPU and never queue the next wave early."""
    results = {}
    gpus = list(base.GPUS)
    wave_size = len(gpus)

    for start in range(0, len(jobs), wave_size):
        wave = jobs[start : start + wave_size]
        print(
            "CORRECTED_AE_WAVE_START="
            + json.dumps([name for name, _, _ in wave]),
            flush=True,
        )
        wave_results = {}
        failures = []

        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = []
            for gpu, (name, pooling, d_mem) in zip(gpus, wave):
                print(
                    f"corrected-ae-factorial: START {name} on GPU {gpu}",
                    flush=True,
                )
                futures.append(
                    (
                        name,
                        gpu,
                        executor.submit(
                            run_cell,
                            gpu,
                            pooling,
                            d_mem,
                            current_commit,
                        ),
                    )
                )

            for name, gpu, future in futures:
                try:
                    wave_results[name] = future.result()
                    print(
                        f"corrected-ae-factorial: DONE  {name} on GPU {gpu}",
                        flush=True,
                    )
                except BaseException as exc:
                    failures.append((name, gpu, exc))
                    print(
                        "corrected-ae-factorial: FAIL "
                        + json.dumps(
                            {
                                "name": name,
                                "gpu": gpu,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )

        if failures:
            write_progress(results, current_commit)
            name, gpu, exc = failures[0]
            raise RuntimeError(
                f"corrected AE wave failed at {name} on GPU {gpu}"
            ) from exc

        results.update(wave_results)
        write_progress(results, current_commit)
        print(
            "CORRECTED_AE_WAVE_DONE="
            + json.dumps(
                {
                    "completed_cells": len(results),
                    "total_cells": len(jobs),
                    "wave": list(wave_results),
                }
            ),
            flush=True,
        )

    return results


def preflight(current_commit: str) -> dict:
    main_root = (
        Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"
    ).resolve()
    if ROOT == main_root:
        raise SystemExit(
            "Refusing to write corrected AE outputs into the completed research_suite root"
        )
    if not tracked_tree_clean():
        raise SystemExit(
            "Refusing to run from a worktree with tracked modifications"
        )

    checks = {
        "config": A.config == TRAIN["config"],
        "train_steps": A.train_steps == TRAIN["train_steps"],
        "memory_only_steps": A.memory_only_steps == TRAIN["memory_only_steps"],
        "train_batch": A.train_batch == TRAIN["batch_size"],
        "seq_len": A.seq_len == TRAIN["seq_len"],
        "eval_batch": A.eval_batch == EVAL["batch_size"],
        "target_errors": A.target_errors == EVAL["target_errors"],
        "max_batches": A.max_batches == EVAL["max_batches"],
        "snr_min": close(A.snr_min, EVAL["snr_min"]),
        "snr_max": close(A.snr_max, EVAL["snr_max"]),
        "snr_step": close(A.snr_step, EVAL["snr_step"]),
        "seed": A.seed == TRAIN["seed"],
        "capacities": len(CAPACITIES) == 4
        and set(CAPACITIES) == {8, 16, 32, 56},
        "two_distinct_gpus": len(base.GPUS) == 2
        and len(set(base.GPUS)) == 2,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(
            "Corrected-AE factorial protocol mismatch: " + ", ".join(bad)
        )

    compile_targets = [
        SCRIPT_DIR / "temporal_compression.py",
        SCRIPT_DIR / "temporal_pooling.py",
        SCRIPT_DIR / "train_temporal_ue_memory_v3.py",
        SCRIPT_DIR / "train_temporal_ue_memory_v4.py",
        SCRIPT_DIR / "evaluate_temporal_ue_memory_v2.py",
        Path(__file__).resolve(),
    ]
    subprocess.run(
        [PY, "-m", "py_compile", *map(str, compile_targets)],
        cwd=str(SCRIPT_DIR),
        check=True,
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    regression = {}
    for name in (
        "test_temporal_compression.py",
        "test_temporal_pooling.py",
        "test_ue_memory_manager.py",
    ):
        proc = subprocess.run(
            [PY, str(SCRIPT_DIR / name)],
            cwd=str(SCRIPT_DIR),
            env=env,
            text=True,
            capture_output=True,
        )
        regression[name] = proc.returncode
        if proc.returncode:
            write_json(
                ROOT / "corrected_autoencoder_preflight_failure.json",
                {
                    "runner_commit": current_commit,
                    "failed_test": name,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                },
            )
            raise RuntimeError(f"preflight regression test failed: {name}")

    smoke_path = Path("/tmp/temporal_autoencoder_stability/summary.json")
    smoke = load_json(smoke_path) if smoke_path.is_file() else None
    if smoke is not None and smoke.get("passed") is not True:
        raise RuntimeError(f"existing stability smoke failed: {smoke_path}")

    report = {
        "runner_commit": current_commit,
        "root": str(ROOT),
        "protocol_checks": checks,
        "regression_tests": regression,
        "stability_smoke_present": smoke is not None,
        "stability_smoke_passed": None if smoke is None else smoke.get("passed"),
        "passed": True,
    }
    write_json(ROOT / "corrected_autoencoder_preflight.json", report)
    print("CORRECTED_AE_PREFLIGHT=" + json.dumps(report, indent=2), flush=True)
    return report


def final_summary(results: dict, current_commit: str, repaired: list[str]) -> dict:
    rows = [
        result_row(v["pooling"], v["d_mem"], v["evaluation"])
        for v in results.values()
    ]
    rows.sort(key=lambda r: (POOLINGS.index(r["pooling"]), r["d_mem"]))
    finite_rows = [r for r in rows if r["temporal_snr10"] is not None]
    best = (
        min(
            finite_rows,
            key=lambda r: (
                float(r["temporal_snr10"]),
                r["pooling"],
                r["d_mem"],
            ),
        )
        if finite_rows
        else None
    )

    csv_path = ROOT / "corrected_autoencoder_results.csv"
    fields = [
        "pooling",
        "compression",
        "d_mem",
        "seed",
        "temporal_snr10",
        "cold_k2_snr10",
        "cold_k8_snr10",
        "gap_recovered_percent",
        "memory_bits_per_ue",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "suite": "corrected_autoencoder_factorial_protocol_v2",
        "status": "complete",
        "root": str(ROOT),
        "runner_commit": current_commit,
        "gpus": list(base.GPUS),
        "training_semantics": dict(TRAIN),
        "evaluation_semantics": dict(EVAL),
        "repaired_legacy_protocol_v2_artifacts": repaired,
        "cell_count": len(rows),
        "finite_crossing_count": len(finite_rows),
        "null_crossing_count": len(rows) - len(finite_rows),
        "best_corrected_autoencoder": best,
        "rows": rows,
    }
    write_json(ROOT / "corrected_autoencoder_summary.json", summary)
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    current_commit = git_commit()
    preflight(current_commit)
    repaired = repair_known_first_run()

    jobs = [
        (f"{pool}_autoencoder_d{d_mem}", pool, d_mem)
        for pool in POOLINGS
        for d_mem in CAPACITIES
    ]
    results = run_waves(jobs, current_commit)
    summary = final_summary(results, current_commit, repaired)
    write_progress(results, current_commit)

    print(
        "CORRECTED_AUTOENCODER_FACTORIAL_SUMMARY="
        + json.dumps(summary, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
