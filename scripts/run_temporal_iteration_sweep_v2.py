#!/usr/bin/env python3
"""Corrected launcher for the temporal NRX K=2..8 finalist sweep.

This wrapper keeps the v1 orchestration/evaluator but fixes three provenance
issues before a multi-day run:

1. K=2 finalist checkpoints are bound to the exact audited artifacts instead of
   recursively searching one root. In particular, mean+writer+d32 is the
   historical legacy checkpoint outside research_suite, and CNN+AE+d56 is the
   corrected protocol-v2 AE checkpoint under autoencoder_v2.
2. CNN+PCA+d16 is retrained with the same capacity-tuned learned-pooler PCA
   protocol that produced the finalist (v7), rather than the generic v4 PCA
   path. Mean+PCA remains on v4 because mean pooling has no learned pooler.
3. K2/K2 temporal accuracy is re-evaluated with the new fixed-batch CRN
   evaluator instead of importing the older early-stopping evaluation. K=2
   TRAINING is still reused; only the four K2/K2 evaluation points are rerun.

Everything else (two-GPU fail-fast waves, K3..8 training, triangular transfer
matrix, plotting, and cold-baseline reuse) comes from run_temporal_iteration_sweep.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_temporal_iteration_sweep as base


A = base.A
SCRIPT_DIR = base.SCRIPT_DIR
FINALISTS = base.FINALISTS
TEMPORAL_ROOT = Path.home() / "sionna-srsran" / "temporal_reuse"
RESEARCH_ROOT = TEMPORAL_ROOT / "research_suite"
CORRECTED_AE_ROOT = RESEARCH_ROOT / "autoencoder_v2"


def exact_k2_checkpoint(f) -> Path:
    slug = f["slug"]
    if slug == "mean_pca_d56":
        path = (
            RESEARCH_ROOT / "trained" / "fixed" / f"seed_{A.seed}"
            / "mean_pca_d56" / base.checkpoint_name(f, 2)
        )
    elif slug == "cnn_pca_d16":
        path = (
            RESEARCH_ROOT / "trained" / "fixed" / f"seed_{A.seed}"
            / "cnn_pca_d16" / base.checkpoint_name(f, 2)
        )
    elif slug == "cnn_autoencoder_d56":
        path = (
            CORRECTED_AE_ROOT / "trained" / "fixed" / f"seed_{A.seed}"
            / "cnn_autoencoder_d56" / base.checkpoint_name(f, 2)
        )
    elif slug == "mean_writer_d32":
        # Audited legacy artifact intentionally reused by the original suite.
        path = (
            TEMPORAL_ROOT / "ue_memory" / "mean" / "writer"
            / "ue_memory_mean_writer_idaware_d32_k2.weights.h5"
        )
    else:
        raise KeyError(f"Unknown finalist: {slug}")

    if not path.is_file():
        raise FileNotFoundError(
            f"Exact audited K=2 checkpoint missing for {f['label']}: {path}"
        )
    return path.resolve()


def checkpoint_for(f, k: int) -> Path:
    if k == 2:
        return exact_k2_checkpoint(f)
    return base.training_dir(f, k) / base.checkpoint_name(f, k)


def _common_summary_valid(s: dict, f, k: int, ckpt: Path) -> bool:
    return bool(
        s.get("pooling") == f["pooling"]
        and s.get("compression") == f["compression"]
        and int(s.get("d_mem", -1)) == f["d_mem"]
        and int(s.get("num_it", -1)) == k
        and int(s.get("train_steps", -1)) == A.train_steps
        and int(s.get("memory_only_steps", -1)) == A.memory_only_steps
        and int(s.get("batch_size", -1)) == A.batch_size
        and int(s.get("seq_len", -1)) == A.seq_len
        and int(s.get("seed", -1)) == A.seed
        and bool(s.get("dynamic_scheduling")) is False
        and Path(s.get("checkpoint", "")).expanduser().resolve() == ckpt.resolve()
    )


def training_valid(f, k: int) -> bool:
    if k == 2:
        try:
            return exact_k2_checkpoint(f).is_file()
        except FileNotFoundError:
            return False

    out = base.training_dir(f, k)
    summary_file = out / "training_summary.json"
    ckpt = checkpoint_for(f, k)
    if not summary_file.is_file() or not ckpt.is_file():
        return False

    try:
        s = base.load_json(summary_file)
        if not _common_summary_valid(s, f, k, ckpt):
            return False

        if f["slug"] == "cnn_pca_d16":
            if s.get("architecture") != (
                "ue_identity_aware_temporal_memory_v7_pca_capacity_tuned_pooler"
            ):
                return False
            protocol = s.get("pca_protocol") or {}
            calibration = s.get("pooler_calibration") or {}
            return bool(
                protocol.get("pooler_calibrated_before_fit") is True
                and protocol.get("pooler_frozen_before_fit") is True
                and protocol.get("pca_fitted_once") is True
                and protocol.get("pooler_frozen_during_temporal_training") is True
                and protocol.get("pca_basis_frozen_during_temporal_training") is True
                and protocol.get("pooler_tuned_to_target_d_mem") is True
                and protocol.get("shared_pooler_across_capacities") is False
                and int(protocol.get("target_d_mem", -1)) == f["d_mem"]
                and calibration.get("capacity_tuned") is True
                and int(calibration.get("proxy_memory_width", -1)) == f["d_mem"]
            )

        if s.get("architecture") != "ue_identity_aware_temporal_memory_v4_pooling":
            return False

        if f["compression"] == "autoencoder":
            p = s.get("autoencoder_protocol") or {}
            return bool(
                int(p.get("version", -1)) == 2
                and p.get("bounded_tanh_bottleneck") is True
                and p.get("reconstruction_upstream_state_detached") is True
                and p.get("scale_normalized_reconstruction_aux_loss") is True
                and p.get("raw_reconstruction_mse_retained_for_diagnostics") is True
            )
        return True
    except Exception:
        return False


def train_one(gpu: int, f, k: int):
    if k == 2:
        ckpt = exact_k2_checkpoint(f)
        print(f"REUSE_EXACT_AUDITED_K2={f['label']}::{ckpt}", flush=True)
        return str(ckpt)

    if training_valid(f, k):
        ckpt = checkpoint_for(f, k)
        print(f"REUSE_TRAINING={f['label']} K={k}::{ckpt}", flush=True)
        return str(ckpt)

    out = base.training_dir(f, k)
    out.mkdir(parents=True, exist_ok=True)

    trainer = "train_temporal_ue_memory_v4.py"
    extra = []
    if f["slug"] == "cnn_pca_d16":
        trainer = "train_temporal_ue_memory_v7_pca_capacity_tuned_pooler.py"
        extra = ["--pooler-calibration-steps", A.memory_only_steps]

    cmd = [
        A.python,
        SCRIPT_DIR / trainer,
        "--config", A.config,
        "--gpu", gpu,
        "--pooling", f["pooling"],
        "--compression", f["compression"],
        "--d-mem", f["d_mem"],
        "--num-it", k,
        "--train-steps", A.train_steps,
        "--memory-only-steps", A.memory_only_steps,
        "--batch-size", A.batch_size,
        "--seq-len", A.seq_len,
        "--min-ebno-db", A.min_ebno_db,
        "--max-ebno-db", A.max_ebno_db,
        "--memory-lr", A.memory_lr,
        "--joint-lr", A.joint_lr,
        "--chest-weight", A.chest_weight,
        "--ae-reconstruction-weight", A.ae_reconstruction_weight,
        "--pca-fit-batches", A.pca_fit_batches,
        "--pca-fit-batch-size", A.pca_fit_batch_size,
        "--ue-pool-size", A.ue_pool_size,
        "--memory-expiry-slots", A.memory_expiry_slots,
        "--schedule-switch-prob", A.schedule_switch_prob,
        "--schedule-reorder-prob", A.schedule_reorder_prob,
        "--fixed-scheduling",
        "--seed", A.seed,
        "--output-dir", out,
        "--log-every", 25,
        *extra,
    ]
    base.tee_run(cmd, out / "train.log", gpu, f"train-{f['slug']}-k{k}")
    if A.dry_run:
        return str(checkpoint_for(f, k))
    if not training_valid(f, k):
        raise RuntimeError(f"Training artifact failed v2 validation: {f['label']} K={k}")
    return str(checkpoint_for(f, k))


def no_old_k2_import(_f) -> bool:
    # Fixed-batch CRN is a different evaluation protocol from the old
    # target-error early stop. Rerun only the four K2/K2 temporal evaluations.
    return False


def main():
    # Patch the v1 module globals used by its orchestration functions.
    base.find_k2_checkpoint = exact_k2_checkpoint
    base.checkpoint_for = checkpoint_for
    base.training_valid = training_valid
    base.train_one = train_one
    base.import_existing_k2_evaluation = no_old_k2_import

    print("ITERATION_SWEEP_V2_PROVENANCE=" + json.dumps({
        "k2": {f["slug"]: str(exact_k2_checkpoint(f)) for f in FINALISTS},
        "cnn_pca_training": "v7_capacity_tuned_pooler",
        "k2_k2_evaluation": "rerun_fixed_batch_crn",
        "cold_policy": "reuse_existing_only",
    }, indent=2), flush=True)

    base.main()


if __name__ == "__main__":
    main()
