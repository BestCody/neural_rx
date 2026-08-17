#!/usr/bin/env python3
"""Train PCA temporal UE memory with a calibrated learned pooler.

This fixes the unfair PCA + learned-pooling protocol from v4. Attention/CNN
poolers must not be random when PCA is fitted, and the pooler must not drift
after PCA is fitted.

Protocol
--------
1. Build the requested Attention/CNN pooler.
2. Calibrate that pooler for temporal utility with a temporary uncompressed
   d_s-wide learned-writer memory path while the shipped NRX base stays frozen.
3. Freeze the calibrated pooler.
4. Fit PCA once on cold final NRX states after that frozen pooler.
5. Train the requested d_mem PCA temporal-memory reader for the normal 6000
   steps. The frozen pooler and frozen PCA basis remain stationary throughout.

Only PCA + attention/cnn should use this script. Mean + PCA has no learned
pooler and should continue to use train_temporal_ue_memory_v4.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _extract_calibration_args(argv):
    steps = 1000
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--pooler-calibration-steps":
            if i + 1 >= len(argv):
                raise SystemExit("--pooler-calibration-steps requires an integer")
            steps = int(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--pooler-calibration-steps="):
            steps = int(arg.split("=", 1)[1])
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    if steps <= 0:
        raise SystemExit("--pooler-calibration-steps must be positive")
    return steps, cleaned


CALIBRATION_STEPS, _CLEAN_ARGV = _extract_calibration_args(sys.argv)
sys.argv[:] = _CLEAN_ARGV

# v4 extracts --pooling, then v3 parses the established training CLI.
import train_temporal_ue_memory_v4 as v4

v3 = v4.v3
tf = v4.tf
np = v3.np
sn = v3.sn


def _build_actual_model():
    """Build the requested pooling-aware PCA model and training generator."""
    v3.TemporalUEMemoryCGNN = v4.PooledTemporalUEMemoryCGNN
    return v3.build()


def _calibrate_pooler(p, e2e, actual_model, generator):
    """Learn the pooler through temporal loss before PCA is ever fitted.

    A temporary d_s-wide writer is used only as a differentiable transport path
    so the pooler receives a future-TB learning signal without imposing the PCA
    bottleneck. Only the calibrated pooler is transferred to the PCA model.
    """
    receiver = e2e._receiver
    base = e2e._receiver._neural_rx._cgnn

    calibration_model = v4.PooledTemporalUEMemoryCGNN(
        base,
        d_mem=int(p.d_s),
        d_s=int(p.d_s),
        compression="writer",
        name=f"pooler_calibration_{v4.POOLING}",
    )
    # Share the exact pooler object with the eventual PCA model. This avoids
    # any ambiguity from copying weights by name/order.
    calibration_model.pooler = actual_model.pooler

    calibration_memory = v3.DifferentiableUEMemoryManager(
        capacity=v3.ARGS.ue_pool_size,
        d_mem=int(p.d_s),
        expiry_slots=v3.ARGS.memory_expiry_slots,
    )

    # Build all memory-side variables before optimizer variable selection.
    warmup = generator.sample_batch(1, v3.ARGS.seq_len, 3.0)
    _ = v3.make_losses(receiver, calibration_model, calibration_memory, warmup)

    optimizer = tf.keras.optimizers.Adam(v3.ARGS.memory_lr)
    history = []
    start = time.time()
    v3.set_seed(v3.ARGS.seed + 7919)

    for step in range(CALIBRATION_STEPS):
        ebno = float(np.random.uniform(v3.ARGS.min_ebno_db, v3.ARGS.max_ebno_db))
        batch = generator.sample_batch(v3.ARGS.batch_size, v3.ARGS.seq_len, ebno)
        with tf.GradientTape() as tape:
            total, loss_data, loss_chest, per_tb, _ = v3.make_losses(
                receiver, calibration_model, calibration_memory, batch
            )

        # Keep the shipped NRX frozen. The temporary reader/writer and the
        # shared pooler learn solely to expose temporally useful information.
        variables = calibration_model.memory_variables
        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients reached learned-pooler calibration")
        optimizer.apply_gradients(pairs)

        if step % v3.ARGS.log_every == 0 or step == CALIBRATION_STEPS - 1:
            row = {
                "step": step,
                "pooling": v4.POOLING,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
                "gradient_norm": float(
                    tf.linalg.global_norm([g for g, _ in pairs]).numpy()
                ),
                "seconds": time.time() - start,
            }
            history.append(row)
            print("POOLER_CALIBRATION=" + json.dumps(row), flush=True)

    # Critical invariant: PCA will be fitted only after this line and this
    # representation is never allowed to drift afterward.
    actual_model.pooler.trainable = False
    return {
        "method": "temporal_writer_d_s_proxy_then_freeze",
        "pooling": v4.POOLING,
        "steps": int(CALIBRATION_STEPS),
        "proxy_memory_width": int(p.d_s),
        "nrx_base_frozen": True,
        "pooler_frozen_before_pca_fit": True,
        "history": history,
    }


def main():
    if v3.ARGS.compression != "pca":
        raise ValueError(
            "train_temporal_ue_memory_v5_pca_pooler_calibrated.py is only for --compression pca"
        )
    if v4.POOLING not in {"attention", "cnn"}:
        raise ValueError(
            "This calibrated-PCA trainer is only needed for --pooling attention or cnn; mean has no learned pooler"
        )
    if v3.ARGS.d_mem <= 0:
        raise ValueError("--d-mem must be positive")
    if v3.ARGS.pca_fit_batches <= 0:
        raise ValueError("--pca-fit-batches must be positive for PCA")

    v3.set_seed(v3.ARGS.seed)
    p, e2e, model, generator, memory_manager = _build_actual_model()
    receiver = e2e._receiver

    calibration = _calibrate_pooler(p, e2e, model, generator)
    print("POOLER_CALIBRATION_SUMMARY=" + json.dumps(calibration), flush=True)

    # Fit exactly once, after the learned representation is established and
    # frozen. No periodic PCA re-fit is used because basis rotations would
    # change memory coordinates while the reader is learning.
    v3.set_seed(v3.ARGS.seed + 104729)
    pca_stats = v3.fit_pca_from_generator(receiver, model, generator)
    if pca_stats is None:
        raise RuntimeError("PCA fitting unexpectedly returned None")
    pca_stats["fit_after_pooler_calibration"] = True
    pca_stats["pooler_frozen_during_fit"] = True
    pca_stats["pooler_frozen_after_fit"] = True
    print("PCA_FIT=" + json.dumps(pca_stats), flush=True)

    # Build the actual PCA memory reader/compressor before selecting variables.
    warmup = generator.sample_batch(1, v3.ARGS.seq_len, 3.0)
    _ = v3.make_losses(receiver, model, memory_manager, warmup)

    if model.pooler.trainable_variables:
        raise RuntimeError("Calibrated pooler must remain frozen before PCA training")

    identity_check = v3.identity_routing_check(
        d_mem=v3.ARGS.d_mem, capacity=v3.ARGS.ue_pool_size
    )
    print("IDENTITY_ROUTING_CHECK=" + json.dumps(identity_check), flush=True)
    if not identity_check["passed"]:
        raise RuntimeError("UE identity/memory routing correctness check failed")

    gradient_check = v3.temporal_gradient_check(
        receiver, model, generator, memory_manager
    )
    print(
        "TEMPORAL_COMPRESSION_GRADIENT_CHECK=" + json.dumps(gradient_check),
        flush=True,
    )
    if not gradient_check["passed"]:
        raise RuntimeError("TB2 loss did not cross the selected TB1 PCA memory path")

    memory_opt = tf.keras.optimizers.Adam(v3.ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(v3.ARGS.joint_lr)

    out = Path(
        v3.ARGS.output_dir
        or (
            Path.home()
            / "sionna-srsran"
            / "temporal_reuse"
            / "ue_memory"
            / v4.POOLING
            / "pca"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    history = []
    start = time.time()
    v3.set_seed(v3.ARGS.seed)

    for step in range(v3.ARGS.train_steps):
        ebno = float(np.random.uniform(v3.ARGS.min_ebno_db, v3.ARGS.max_ebno_db))
        batch = generator.sample_batch(v3.ARGS.batch_size, v3.ARGS.seq_len, ebno)

        with tf.GradientTape() as tape:
            total, loss_data, loss_chest, per_tb, diagnostics = v3.make_losses(
                receiver, model, memory_manager, batch
            )

        if step < v3.ARGS.memory_only_steps:
            variables = model.memory_variables
            optimizer = memory_opt
            phase = "memory_only"
        else:
            variables = model.trainable_variables
            optimizer = joint_opt
            phase = "joint"

        # The frozen pooler must never re-enter either optimization phase.
        pooler_ids = {id(x) for x in model.pooler.weights}
        if any(id(x) in pooler_ids for x in variables):
            raise RuntimeError("Frozen PCA pooler leaked into optimizer variables")

        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients reached the selected PCA variables")
        optimizer.apply_gradients(pairs)
        grad_norm = float(tf.linalg.global_norm([g for g, _ in pairs]).numpy())

        if step % v3.ARGS.log_every == 0 or step == v3.ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "pooling": v4.POOLING,
                "compression": "pca",
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "reconstruction_mse_per_tb": [
                    float(x.numpy()) for x in diagnostics["reconstruction_mses"]
                ],
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
                "memory_norm_per_tb": [
                    float(x.numpy()) for x in diagnostics["memory_norms"]
                ],
                "memory_valid_fraction_per_tb": [
                    float(x.numpy())
                    for x in diagnostics["memory_valid_fractions"]
                ],
                "memory_gap_mean_per_tb": [
                    float(x.numpy()) for x in diagnostics["memory_gap_means"]
                ],
                "schedule_change_fraction": v3.schedule_change_fraction(
                    batch["ue_ids"]
                ),
                "schedule_example": batch["ue_ids"][0].numpy().tolist(),
                "gradient_norm": grad_norm,
                "seconds": time.time() - start,
            }
            history.append(row)
            print("TRAIN=" + json.dumps(row), flush=True)

    checkpoint = out / (
        f"ue_memory_{v4.POOLING}_pca_idaware_d{v3.ARGS.d_mem}_k{v3.ARGS.num_it}.weights.h5"
    )
    model.save_weights(str(checkpoint))

    summary = {
        "architecture": "ue_identity_aware_temporal_memory_v5_pca_pooler_calibrated",
        "pooling": v4.POOLING,
        "pooling_semantics": {
            "attention": "learned softmax weighting over final NRX time/frequency locations",
            "cnn": "learned local 3x3 time/frequency features followed by global mean",
        }[v4.POOLING],
        "compression": "pca",
        "compression_semantics": "frozen PCA of a temporally calibrated, frozen learned-pooling representation",
        "config": v3.ARGS.config,
        "d_mem": v3.ARGS.d_mem,
        "memory_dtype": "float32",
        "memory_cap_bits_per_ue": int(v3.ARGS.d_mem * 32),
        "memory_cap_bytes_per_ue": int(v3.ARGS.d_mem * 4),
        "num_it": v3.ARGS.num_it,
        "train_steps": v3.ARGS.train_steps,
        "memory_only_steps": v3.ARGS.memory_only_steps,
        "batch_size": v3.ARGS.batch_size,
        "seq_len": v3.ARGS.seq_len,
        "seed": v3.ARGS.seed,
        "ue_pool_size": v3.ARGS.ue_pool_size,
        "dynamic_scheduling": not v3.ARGS.fixed_scheduling,
        "memory_expiry_slots": v3.ARGS.memory_expiry_slots,
        "schedule_switch_prob": v3.ARGS.schedule_switch_prob,
        "schedule_reorder_prob": v3.ARGS.schedule_reorder_prob,
        "pooler_calibration": calibration,
        "pca_fit": pca_stats,
        "pca_protocol": {
            "pooler_calibrated_before_fit": True,
            "pooler_frozen_before_fit": True,
            "pca_fitted_once": True,
            "pooler_frozen_during_temporal_training": True,
            "pca_basis_frozen_during_temporal_training": True,
        },
        "identity_routing_check": identity_check,
        "temporal_compression_gradient_check": gradient_check,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAINING_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
