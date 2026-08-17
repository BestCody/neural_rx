#!/usr/bin/env python3
"""Train the raw, uncompressed full-state temporal-memory upper bound."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-it", type=int, default=2)
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--memory-only-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--min-ebno-db", type=float, default=1.0)
    p.add_argument("--max-ebno-db", type=float, default=5.0)
    p.add_argument("--memory-lr", type=float, default=1e-3)
    p.add_argument("--joint-lr", type=float, default=2e-5)
    p.add_argument("--chest-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--memory-expiry-slots", type=int, default=8)
    p.add_argument("--fixed-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    return p.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import sionna as sn

from temporal_full_state import build_system, make_manager, temporal_forward

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def make_losses(receiver, model, manager, batch):
    batch_size = tf.shape(batch["bits"])[0]
    seq_len = int(batch["bits"].shape[1])
    state = manager.zero_state(batch_size, tf.float32)
    data_losses = []
    chest_losses = []
    valid_fracs = []
    memory_norms = []

    for t in range(seq_len):
        state, prev, gap, valid = manager.gather(
            state, batch["ue_ids"][:, t], t
        )
        coded = receiver._tb_encoders[0](batch["bits"][:, t])
        h_true = receiver.preprocess_channel_ground_truth(batch["h"][:, t])
        llr, h_ref, updated = temporal_forward(
            receiver,
            model,
            batch["y"][:, t],
            batch["ls"][:, t],
            batch["active"][:, t],
            prev,
            gap,
            valid,
            training=True,
        )
        state = manager.scatter(
            state,
            batch["ue_ids"][:, t],
            updated,
            batch["active"][:, t],
            t,
        )
        data_losses.append(
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(coded, tf.float32), logits=llr
                )
            )
        )
        chest_losses.append(tf.reduce_mean(tf.square(h_ref - h_true)))
        valid_fracs.append(tf.reduce_mean(tf.cast(valid, tf.float32)))
        memory_norms.append(tf.reduce_mean(tf.norm(updated, axis=-1)))

    loss_data = tf.add_n(data_losses) / float(seq_len)
    loss_chest = tf.add_n(chest_losses) / float(seq_len)
    total = loss_data + ARGS.chest_weight * loss_chest
    return total, loss_data, loss_chest, data_losses, valid_fracs, memory_norms


def temporal_gradient_check(receiver, model, manager, generator):
    batch = generator.sample_batch(1, 2, 3.0)
    state = manager.zero_state(1, tf.float32)
    with tf.GradientTape() as tape:
        state, m0, g0, v0 = manager.gather(state, batch["ue_ids"][:, 0], 0)
        _, _, updated0 = temporal_forward(
            receiver,
            model,
            batch["y"][:, 0],
            batch["ls"][:, 0],
            batch["active"][:, 0],
            m0,
            g0,
            v0,
            training=True,
        )
        state = manager.scatter(
            state, batch["ue_ids"][:, 0], updated0, batch["active"][:, 0], 0
        )
        state, m1, g1, v1 = manager.gather(state, batch["ue_ids"][:, 1], 1)
        coded1 = receiver._tb_encoders[0](batch["bits"][:, 1])
        llr1, _, _ = temporal_forward(
            receiver,
            model,
            batch["y"][:, 1],
            batch["ls"][:, 1],
            batch["active"][:, 1],
            m1,
            g1,
            v1,
            training=True,
        )
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded1, tf.float32), logits=llr1
            )
        )
    grads = [g for g in tape.gradient(loss, model.memory_variables) if g is not None]
    norm = float(tf.linalg.global_norm(grads).numpy()) if grads else 0.0
    return {
        "tb2_only_loss": float(loss.numpy()),
        "read_gate_grad_norm": norm,
        "tb2_memory_valid": v1.numpy().tolist(),
        "passed": bool(norm > 0.0 and np.any(v1.numpy())),
    }


def validate_args():
    if ARGS.num_it != 2:
        raise ValueError("Full-state upper bound is defined for temporal K=2")
    if ARGS.train_steps <= 0 or ARGS.batch_size <= 0 or ARGS.seq_len < 2:
        raise ValueError("train-steps/batch-size must be positive and seq-len >= 2")
    if not 0 <= ARGS.memory_only_steps <= ARGS.train_steps:
        raise ValueError("memory-only-steps must be in [0, train-steps]")
    if ARGS.min_ebno_db > ARGS.max_ebno_db:
        raise ValueError("min-ebno-db must be <= max-ebno-db")
    if ARGS.memory_lr <= 0 or ARGS.joint_lr <= 0:
        raise ValueError("learning rates must be positive")
    if ARGS.ue_pool_size < 2:
        raise ValueError("ue-pool-size must be >= 2")
    if ARGS.memory_expiry_slots < 1:
        raise ValueError("memory-expiry-slots must be >= 1")
    for name in ("schedule_switch_prob", "schedule_reorder_prob"):
        value = float(getattr(ARGS, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def main():
    validate_args()
    set_seed(ARGS.seed)
    p, e2e, model, generator = build_system(
        config=ARGS.config,
        num_it=ARGS.num_it,
        training=True,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=not ARGS.fixed_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
    )
    receiver = e2e._receiver
    warm = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, 3.0)
    manager, d_mem_train, state_shape = make_manager(
        receiver,
        model,
        warm,
        capacity=ARGS.ue_pool_size,
        expiry_slots=ARGS.memory_expiry_slots,
    )
    _ = make_losses(receiver, model, manager, warm)
    check = temporal_gradient_check(receiver, model, manager, generator)
    print("FULL_STATE_TEMPORAL_GRADIENT_CHECK=" + json.dumps(check), flush=True)
    if not check["passed"]:
        raise RuntimeError("TB2 loss does not reach the full-state memory read path")

    memory_opt = tf.keras.optimizers.Adam(ARGS.memory_lr)
    joint_opt = tf.keras.optimizers.Adam(ARGS.joint_lr)
    out = Path(ARGS.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.time()
    set_seed(ARGS.seed)

    for step in range(ARGS.train_steps):
        ebno = float(np.random.uniform(ARGS.min_ebno_db, ARGS.max_ebno_db))
        batch = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, ebno)
        with tf.GradientTape() as tape:
            total, data, chest, per_tb, valid_fracs, memory_norms = make_losses(
                receiver, model, manager, batch
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
            raise RuntimeError("No gradients reached selected variables")
        optimizer.apply_gradients(pairs)

        if step % ARGS.log_every == 0 or step == ARGS.train_steps - 1:
            row = {
                "step": step,
                "phase": phase,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(data.numpy()),
                "loss_chest": float(chest.numpy()),
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
                "memory_valid_fraction_per_tb": [float(x.numpy()) for x in valid_fracs],
                "memory_norm_per_tb": [float(x.numpy()) for x in memory_norms],
                "seconds": time.time() - start,
            }
            history.append(row)
            print("TRAIN_FULL_STATE=" + json.dumps(row), flush=True)

    checkpoint = out / "ue_memory_full_state_raw_k2.weights.h5"
    model.save_weights(str(checkpoint))
    summary = {
        "architecture": "ue_identity_aware_temporal_full_state_v1",
        "representation": "raw_final_cgnn_state_no_pooling_no_compression",
        "config": ARGS.config,
        "d_s": int(p.d_s),
        "training_state_shape_per_ue": list(state_shape),
        "training_memory_floats_per_ue": int(d_mem_train),
        "training_memory_bytes_per_ue": int(d_mem_train * 4),
        "num_it": ARGS.num_it,
        "train_steps": ARGS.train_steps,
        "memory_only_steps": ARGS.memory_only_steps,
        "batch_size": ARGS.batch_size,
        "seq_len": ARGS.seq_len,
        "seed": ARGS.seed,
        "ue_pool_size": ARGS.ue_pool_size,
        "memory_expiry_slots": ARGS.memory_expiry_slots,
        "dynamic_scheduling": not ARGS.fixed_scheduling,
        "schedule_switch_prob": ARGS.schedule_switch_prob,
        "schedule_reorder_prob": ARGS.schedule_reorder_prob,
        "min_ebno_db": ARGS.min_ebno_db,
        "max_ebno_db": ARGS.max_ebno_db,
        "memory_lr": ARGS.memory_lr,
        "joint_lr": ARGS.joint_lr,
        "chest_weight": ARGS.chest_weight,
        "temporal_gradient_check": check,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FULL_STATE_TRAINING_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
