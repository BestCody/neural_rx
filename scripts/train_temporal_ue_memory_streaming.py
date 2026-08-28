#!/usr/bin/env python3
"""Train temporal Neural RX as a persistent K=1/K=2 streaming receiver.

This method samples one genuinely continuous channel episode (64 TBs by
default) and keeps each UE's numerical memory for that entire episode.
Backpropagation is truncated to short windows:
the state value crosses a window boundary, but ``stop_gradient`` prevents the
training graph and GPU memory from growing with the episode length.

At window boundaries, valid UE memories are independently cold-reset with a
small probability. This teaches recovery from a discarded/stale state without
giving the receiver an artificial transport-block-position feature.

``--train-steps`` counts optimizer updates (TBPTT windows), not episodes.
Only one or two receiver/GNN iterations are accepted because those are the
deployment-latency operating points this experiment is intended to test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a persistent K=1/K=2 temporal Neural RX"
    )
    parser.add_argument("--config", default="nrx_large.cfg")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--pooling", choices=["mean", "attention", "cnn"], default="mean"
    )
    parser.add_argument(
        "--compression",
        choices=["writer", "pca", "autoencoder"],
        default="writer",
    )
    parser.add_argument("--d-mem", type=int, default=32)
    parser.add_argument("--num-it", type=int, choices=[1, 2], default=2)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--memory-only-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stream-len", type=int, default=64)
    parser.add_argument("--tbptt-window", type=int, default=4)
    parser.add_argument("--memory-reset-prob", type=float, default=0.05)
    parser.add_argument("--min-ebno-db", type=float, default=1.0)
    parser.add_argument("--max-ebno-db", type=float, default=5.0)
    parser.add_argument("--memory-lr", type=float, default=1e-3)
    parser.add_argument("--joint-lr", type=float, default=2e-5)
    parser.add_argument("--chest-weight", type=float, default=0.01)
    parser.add_argument("--ae-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-dir")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--pca-fit-batches", type=int, default=16)
    parser.add_argument("--pca-fit-batch-size", type=int, default=8)
    parser.add_argument("--pooler-calibration-steps", type=int, default=1000)
    parser.add_argument("--ue-pool-size", type=int, default=4)
    parser.add_argument("--memory-expiry-slots", type=int, default=8)
    parser.add_argument(
        "--dynamic-scheduling",
        action="store_true",
        help="opt into UE switching/reordering; fixed UEs are the default",
    )
    parser.add_argument("--schedule-switch-prob", type=float, default=0.65)
    parser.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from temporal_compression import fit_pca_numpy
from temporal_training_data import TemporalTrainingDataGenerator
from temporal_ue_memory_model import (
    TemporalUEMemoryCGNN,
    build_backbone,
    identity_routing_check,
    prepare_cgnn_inputs,
    schedule_change_fraction,
    set_seed,
    temporal_forward,
    temporal_gradient_check,
)
from ue_memory_manager import DifferentiableUEMemoryManager

from temporal_streaming_training import (
    detach_and_randomly_reset,
    iter_tbptt_windows,
    validate_streaming_config,
)


def make_streaming_chunk_losses(
    receiver,
    model,
    memory_manager,
    episode,
    state,
    start_tb: int,
    end_tb: int,
):
    """Train on one window and return its final differentiable memory state."""
    if end_tb <= start_tb:
        raise ValueError("A TBPTT chunk must contain at least one TB")

    data_losses = []
    chest_losses = []
    aux_losses = []
    reconstruction_mses = []
    memory_norms = []
    memory_valid_fractions = []
    memory_gap_means = []

    for t in range(int(start_tb), int(end_tb)):
        bits_t = episode["bits"][:, t]
        y_t = episode["y"][:, t]
        ls_t = episode["ls"][:, t]
        h_t = episode["h"][:, t]
        active_t = episode["active"][:, t]
        ue_ids_t = episode["ue_ids"][:, t]

        # The absolute episode index is important: gaps and expiry must not
        # restart at every TBPTT window.
        state, prev_memory, memory_gap, memory_valid = memory_manager.gather(
            state, ue_ids_t, t
        )

        coded_t = receiver._tb_encoders[0](bits_t)
        h_true_t = receiver.preprocess_channel_ground_truth(h_t)
        (
            llr_t,
            h_ref_t,
            updated_memory,
            aux_loss_t,
            reconstruction_mse_t,
        ) = temporal_forward(
            receiver,
            model,
            y_t,
            ls_t,
            active_t,
            prev_memory,
            memory_gap,
            memory_valid,
            training=True,
        )

        state = memory_manager.scatter(
            state, ue_ids_t, updated_memory, active_t, t
        )
        data_losses.append(
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(coded_t, tf.float32), logits=llr_t
                )
            )
        )
        chest_losses.append(tf.reduce_mean(tf.square(h_ref_t - h_true_t)))
        aux_losses.append(aux_loss_t)
        reconstruction_mses.append(reconstruction_mse_t)
        memory_norms.append(tf.reduce_mean(tf.norm(updated_memory, axis=-1)))
        memory_valid_fractions.append(
            tf.reduce_mean(tf.cast(memory_valid, tf.float32))
        )

        valid_gap = tf.where(
            memory_valid,
            tf.cast(memory_gap, tf.float32),
            tf.zeros_like(tf.cast(memory_gap, tf.float32)),
        )
        denom = tf.maximum(
            tf.reduce_sum(tf.cast(memory_valid, tf.float32)), 1.0
        )
        memory_gap_means.append(tf.reduce_sum(valid_gap) / denom)

    chunk_len = int(end_tb) - int(start_tb)
    loss_data = tf.add_n(data_losses) / float(chunk_len)
    loss_chest = tf.add_n(chest_losses) / float(chunk_len)
    loss_aux = tf.add_n(aux_losses) / float(chunk_len)
    aux_weight = (
        ARGS.ae_reconstruction_weight
        if ARGS.compression == "autoencoder"
        else 0.0
    )
    total = loss_data + ARGS.chest_weight * loss_chest + aux_weight * loss_aux
    diagnostics = {
        "memory_norms": memory_norms,
        "memory_valid_fractions": memory_valid_fractions,
        "memory_gap_means": memory_gap_means,
        "reconstruction_mses": reconstruction_mses,
        "compression_aux_loss": loss_aux,
    }
    return total, loss_data, loss_chest, data_losses, diagnostics, state


def _calibrate_learned_pca_pooler(p, e2e, actual_model, generator):
    """Capacity-tune a CNN/attention PCA pooler with streaming supervision."""
    if ARGS.pooler_calibration_steps <= 0:
        raise ValueError(
            "Learned pooling + PCA requires --pooler-calibration-steps > 0"
        )

    receiver = e2e._receiver
    base = receiver._neural_rx._cgnn
    calibration_model = TemporalUEMemoryCGNN(
        base,
        d_mem=ARGS.d_mem,
        d_s=int(p.d_s),
        compression="writer",
        pooling=ARGS.pooling,
        name=f"streaming_pooler_calibration_{ARGS.pooling}_d{ARGS.d_mem}",
    )
    # The proxy and final PCA receiver use the exact same pooler object.
    calibration_model.pooler = actual_model.pooler
    calibration_memory = DifferentiableUEMemoryManager(
        capacity=ARGS.ue_pool_size,
        d_mem=ARGS.d_mem,
        expiry_slots=ARGS.memory_expiry_slots,
    )

    warmup = generator.sample_batch(1, ARGS.tbptt_window, 3.0)
    warm_state = calibration_memory.zero_state(1, tf.float32)
    _ = make_streaming_chunk_losses(
        receiver,
        calibration_model,
        calibration_memory,
        warmup,
        warm_state,
        0,
        ARGS.tbptt_window,
    )

    optimizer = tf.keras.optimizers.Adam(ARGS.memory_lr)
    history = []
    started = time.time()
    set_seed(ARGS.seed + 7919)
    step = 0
    episode_index = 0
    while step < ARGS.pooler_calibration_steps:
        ebno = float(
            np.random.uniform(ARGS.min_ebno_db, ARGS.max_ebno_db)
        )
        episode = generator.sample_batch(
            ARGS.batch_size, ARGS.stream_len, ebno
        )
        state = calibration_memory.zero_state(ARGS.batch_size, tf.float32)

        for start_tb, end_tb in iter_tbptt_windows(
            ARGS.stream_len, ARGS.tbptt_window
        ):
            if step >= ARGS.pooler_calibration_steps:
                break
            reset_fraction = tf.constant(0.0, tf.float32)
            if start_tb > 0:
                state, reset_fraction = detach_and_randomly_reset(
                    state, ARGS.memory_reset_prob
                )

            with tf.GradientTape() as tape:
                (
                    total,
                    loss_data,
                    loss_chest,
                    per_tb,
                    _,
                    state,
                ) = make_streaming_chunk_losses(
                    receiver,
                    calibration_model,
                    calibration_memory,
                    episode,
                    state,
                    start_tb,
                    end_tb,
                )

            variables = calibration_model.memory_variables
            grads = tape.gradient(total, variables)
            pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if not pairs:
                raise RuntimeError("No gradients reached streaming pooler calibration")
            optimizer.apply_gradients(pairs)

            if (
                step % ARGS.log_every == 0
                or step == ARGS.pooler_calibration_steps - 1
            ):
                row = {
                    "step": step,
                    "episode": episode_index,
                    "tb_range": [start_tb, end_tb],
                    "pooling": ARGS.pooling,
                    "target_d_mem": ARGS.d_mem,
                    "ebno_db": ebno,
                    "loss": float(total.numpy()),
                    "loss_data": float(loss_data.numpy()),
                    "loss_chest": float(loss_chest.numpy()),
                    "loss_per_tb": [float(x.numpy()) for x in per_tb],
                    "reset_fraction": float(reset_fraction.numpy()),
                    "gradient_norm": float(
                        tf.linalg.global_norm([g for g, _ in pairs]).numpy()
                    ),
                    "seconds": time.time() - started,
                }
                history.append(row)
                print("STREAMING_POOLER_CALIBRATION=" + json.dumps(row), flush=True)
            step += 1
        episode_index += 1

    actual_model.pooler.trainable = False
    return {
        "method": "same_capacity_streaming_writer_proxy_then_freeze",
        "pooling": ARGS.pooling,
        "steps": ARGS.pooler_calibration_steps,
        "stream_len": ARGS.stream_len,
        "tbptt_window": ARGS.tbptt_window,
        "proxy_memory_width": ARGS.d_mem,
        "target_pca_d_mem": ARGS.d_mem,
        "capacity_tuned": True,
        "pooler_frozen_before_pca_fit": True,
        "history": history,
    }


def build_training_components():
    """Build the pretrained backbone, temporal model, data, and UE state table."""
    parameters, e2e = build_backbone(
        ARGS.config, num_it=ARGS.num_it, training=True
    )
    model = TemporalUEMemoryCGNN(
        e2e._receiver._neural_rx._cgnn,
        d_mem=ARGS.d_mem,
        d_s=parameters.d_s,
        compression=ARGS.compression,
        pooling=ARGS.pooling,
        name=(
            f"temporal_streaming_{ARGS.pooling}_{ARGS.compression}_"
            f"d{ARGS.d_mem}_k{ARGS.num_it}"
        ),
    )
    generator = TemporalTrainingDataGenerator(
        parameters,
        e2e,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=ARGS.dynamic_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
    )
    memory_manager = DifferentiableUEMemoryManager(
        capacity=ARGS.ue_pool_size,
        d_mem=ARGS.d_mem,
        expiry_slots=ARGS.memory_expiry_slots,
    )
    return parameters, e2e, model, generator, memory_manager


def fit_pca(receiver, model, generator):
    """Fit the frozen PCA compressor on cold K-step receiver states."""
    if ARGS.compression != "pca":
        return None

    samples = []
    for _ in range(ARGS.pca_fit_batches):
        ebno = float(np.random.uniform(ARGS.min_ebno_db, ARGS.max_ebno_db))
        batch = generator.sample_batch(
            ARGS.pca_fit_batch_size, ARGS.tbptt_window, ebno
        )
        for t in range(ARGS.tbptt_window):
            inputs = prepare_cgnn_inputs(
                receiver,
                batch["y"][:, t],
                batch["ls"][:, t],
                batch["active"][:, t],
            )
            pooled = model.cold_pooled_final(inputs)
            samples.append(tf.reshape(pooled, [-1, model.d_s]).numpy())

    values = np.concatenate(samples, axis=0)
    mean, components, eigenvalues, stats = fit_pca_numpy(values, model.d_mem)
    model.compressor.set_basis(mean, components, eigenvalues)
    stats.update(
        {
            "fit_batches": ARGS.pca_fit_batches,
            "fit_batch_size": ARGS.pca_fit_batch_size,
            "frozen_after_fit": True,
        }
    )
    return stats


def _validate_args():
    validate_streaming_config(
        ARGS.stream_len, ARGS.tbptt_window, ARGS.memory_reset_prob
    )
    if ARGS.d_mem <= 0:
        raise ValueError("--d-mem must be positive")
    if ARGS.train_steps <= 0:
        raise ValueError("--train-steps must be positive")
    if not 0 <= ARGS.memory_only_steps <= ARGS.train_steps:
        raise ValueError("--memory-only-steps must be in [0, train-steps]")
    if ARGS.compression == "pca" and ARGS.pca_fit_batches <= 0:
        raise ValueError("--pca-fit-batches must be positive for PCA")


def main():
    _validate_args()
    set_seed(ARGS.seed)
    p, e2e, model, generator, memory_manager = build_training_components()
    receiver = e2e._receiver

    pooler_calibration = None
    if ARGS.compression == "pca" and ARGS.pooling in {"attention", "cnn"}:
        pooler_calibration = _calibrate_learned_pca_pooler(
            p, e2e, model, generator
        )
        print(
            "STREAMING_POOLER_CALIBRATION_SUMMARY="
            + json.dumps(pooler_calibration),
            flush=True,
        )

    set_seed(ARGS.seed + 104729)
    pca_stats = fit_pca(receiver, model, generator)
    if pca_stats is not None:
        pca_stats.update(
            {
                "fit_after_pooler_calibration": pooler_calibration is not None,
                "pooler_has_no_trainable_variables_after_fit": (
                    len(model.pooler.trainable_variables) == 0
                ),
            }
        )
        print("PCA_FIT=" + json.dumps(pca_stats), flush=True)

    # Build reader/compressor variables using the same loss path as training.
    warmup = generator.sample_batch(1, ARGS.tbptt_window, 3.0)
    warm_state = memory_manager.zero_state(1, tf.float32)
    _ = make_streaming_chunk_losses(
        receiver,
        model,
        memory_manager,
        warmup,
        warm_state,
        0,
        ARGS.tbptt_window,
    )

    identity_check = identity_routing_check(
        d_mem=ARGS.d_mem, capacity=ARGS.ue_pool_size
    )
    print("IDENTITY_ROUTING_CHECK=" + json.dumps(identity_check), flush=True)
    if not identity_check["passed"]:
        raise RuntimeError("UE identity/memory routing correctness check failed")

    gradient_check = temporal_gradient_check(
        receiver, model, generator, memory_manager
    )
    print(
        "TEMPORAL_COMPRESSION_GRADIENT_CHECK=" + json.dumps(gradient_check),
        flush=True,
    )
    if not gradient_check["passed"]:
        raise RuntimeError("TB2 loss did not cross the TB1 memory path")

    memory_opt = tf.keras.optimizers.Adam(ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(ARGS.joint_lr)
    out = Path(
        ARGS.output_dir
        or (
            HERE.parent
            / "outputs"
            / "temporal_ue_memory_streaming"
            / ARGS.pooling
            / ARGS.compression
            / f"k{ARGS.num_it}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    history = []
    started = time.time()
    set_seed(ARGS.seed)
    step = 0
    episode_index = 0
    while step < ARGS.train_steps:
        ebno = float(
            np.random.uniform(ARGS.min_ebno_db, ARGS.max_ebno_db)
        )
        # One generator call produces one continuous TDL trajectory. Never
        # carry memory into a separately sampled episode.
        episode = generator.sample_batch(
            ARGS.batch_size, ARGS.stream_len, ebno
        )
        state = memory_manager.zero_state(ARGS.batch_size, tf.float32)

        for start_tb, end_tb in iter_tbptt_windows(
            ARGS.stream_len, ARGS.tbptt_window
        ):
            if step >= ARGS.train_steps:
                break

            reset_fraction = tf.constant(0.0, tf.float32)
            if start_tb > 0:
                state, reset_fraction = detach_and_randomly_reset(
                    state, ARGS.memory_reset_prob
                )

            with tf.GradientTape() as tape:
                (
                    total,
                    loss_data,
                    loss_chest,
                    per_tb,
                    diagnostics,
                    state,
                ) = make_streaming_chunk_losses(
                    receiver,
                    model,
                    memory_manager,
                    episode,
                    state,
                    start_tb,
                    end_tb,
                )

            if step < ARGS.memory_only_steps:
                variables = model.memory_variables
                optimizer = memory_opt
                phase = "memory_only"
            else:
                variables = model.trainable_variables
                optimizer = joint_opt
                phase = "joint"

            grads = tape.gradient(total, variables)
            pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if not pairs:
                raise RuntimeError("No gradients reached the selected variables")
            optimizer.apply_gradients(pairs)
            grad_norm = float(
                tf.linalg.global_norm([g for g, _ in pairs]).numpy()
            )

            if step % ARGS.log_every == 0 or step == ARGS.train_steps - 1:
                row = {
                    "step": step,
                    "phase": phase,
                    "episode": episode_index,
                    "tb_range": [start_tb, end_tb],
                    "pooling": ARGS.pooling,
                    "compression": ARGS.compression,
                    "num_it": ARGS.num_it,
                    "ebno_db": ebno,
                    "loss": float(total.numpy()),
                    "loss_data": float(loss_data.numpy()),
                    "loss_chest": float(loss_chest.numpy()),
                    "compression_aux_loss": float(
                        diagnostics["compression_aux_loss"].numpy()
                    ),
                    "loss_per_tb": [float(x.numpy()) for x in per_tb],
                    "memory_norm_per_tb": [
                        float(x.numpy()) for x in diagnostics["memory_norms"]
                    ],
                    "memory_valid_fraction_per_tb": [
                        float(x.numpy())
                        for x in diagnostics["memory_valid_fractions"]
                    ],
                    "memory_gap_mean_per_tb": [
                        float(x.numpy())
                        for x in diagnostics["memory_gap_means"]
                    ],
                    "reconstruction_mse_per_tb": [
                        float(x.numpy())
                        for x in diagnostics["reconstruction_mses"]
                    ],
                    "reset_fraction": float(reset_fraction.numpy()),
                    "schedule_change_fraction_episode": schedule_change_fraction(
                        episode["ue_ids"]
                    ),
                    "gradient_norm": grad_norm,
                    "seconds": time.time() - started,
                }
                history.append(row)
                print("STREAMING_TRAIN=" + json.dumps(row), flush=True)
            step += 1
        episode_index += 1

    checkpoint = out / (
        f"ue_memory_streaming_{ARGS.pooling}_{ARGS.compression}_"
        f"idaware_d{ARGS.d_mem}_k{ARGS.num_it}.weights.h5"
    )
    model.save_weights(str(checkpoint))
    summary = {
        "architecture": "ue_identity_aware_temporal_memory_streaming_tbptt_v1",
        "training_method": {
            "continuous_channel_per_episode": True,
            "state_carried_across_tbptt_windows": True,
            "gradient_detached_between_tbptt_windows": True,
            "state_reset_between_independent_episodes": True,
            "transport_block_position_input": False,
            "random_valid_ue_reset_at_window_boundaries": True,
        },
        "config": ARGS.config,
        "pooling": ARGS.pooling,
        "compression": ARGS.compression,
        "d_mem": ARGS.d_mem,
        "memory_dtype": "float32",
        "memory_cap_bits_per_ue": int(ARGS.d_mem * 32),
        "memory_cap_bytes_per_ue": int(ARGS.d_mem * 4),
        "num_it": ARGS.num_it,
        "train_steps": ARGS.train_steps,
        "train_steps_semantics": "optimizer_updates_or_tbptt_windows",
        "memory_only_steps": ARGS.memory_only_steps,
        "batch_size": ARGS.batch_size,
        "stream_len": ARGS.stream_len,
        "tbptt_window": ARGS.tbptt_window,
        "memory_reset_prob": ARGS.memory_reset_prob,
        "episodes_sampled": episode_index,
        "seed": ARGS.seed,
        "ue_pool_size": ARGS.ue_pool_size,
        "dynamic_scheduling": ARGS.dynamic_scheduling,
        "memory_expiry_slots": ARGS.memory_expiry_slots,
        "schedule_switch_prob": ARGS.schedule_switch_prob,
        "schedule_reorder_prob": ARGS.schedule_reorder_prob,
        "pooler_calibration": pooler_calibration,
        "pca_fit": pca_stats,
        "identity_routing_check": identity_check,
        "temporal_compression_gradient_check": gradient_check,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("STREAMING_TRAINING_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
