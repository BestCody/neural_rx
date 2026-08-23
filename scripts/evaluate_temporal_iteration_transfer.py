#!/usr/bin/env python3
"""Evaluate one temporal UE-memory checkpoint at a chosen NRX iteration count.

This evaluator is intentionally temporal-only. It does not recompute cold NRX
baselines. Every cell uses the same fixed number of batches per SNR and resets
its RNG at each SNR point, so separately launched cells see common random
numbers as long as they use the same seed and evaluation settings.

The intended sweep is triangular:

    train K=2 -> evaluate K=2..8
    train K=3 -> evaluate K=3..8
    ...
    train K=8 -> evaluate K=8

The checkpoint may contain a backbone fine-tuned at train K. Evaluation K may
be larger; the additional pretrained NRX iteration blocks are then executed
without retraining, which measures forward transfer to extra inference rounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from temporal_eval_metrics import log_bler_crossing, make_snr_grid, wilson_interval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-num-it", type=int, required=True)
    p.add_argument("--eval-num-it", type=int, required=True)
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--compression",
        choices=["writer", "pca", "autoencoder"],
        required=True,
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "attention", "cnn"],
        default="mean",
    )
    p.add_argument("--d-mem", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--snr-min", type=float, default=1.5)
    p.add_argument("--snr-max", type=float, default=3.75)
    p.add_argument("--snr-step", type=float, default=0.25)
    p.add_argument(
        "--batches-per-snr",
        type=int,
        default=32,
        help="Fixed batch count. Do not early-stop; fixed counts preserve CRN.",
    )
    p.add_argument(
        "--min-errors-warning",
        type=int,
        default=120,
        help="Warn when TB2+ errors are below this count at an SNR point.",
    )
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--memory-expiry-slots", type=int, default=8)
    p.add_argument("--dynamic-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


ARGS = parse_args()
if not 1 <= ARGS.train_num_it <= 8:
    raise ValueError("train-num-it must be in [1, 8]")
if not ARGS.train_num_it <= ARGS.eval_num_it <= 8:
    raise ValueError("eval-num-it must be >= train-num-it and <= 8")
if ARGS.seq_len < 2:
    raise ValueError("seq-len must be >= 2")
if ARGS.batch_size <= 0 or ARGS.batches_per_snr <= 0:
    raise ValueError("batch-size and batches-per-snr must be positive")
if ARGS.d_mem <= 0:
    raise ValueError("d-mem must be positive")
for name in ("schedule_switch_prob", "schedule_reorder_prob"):
    value = float(getattr(ARGS, name))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")

SNR_GRID = make_snr_grid(ARGS.snr_min, ARGS.snr_max, ARGS.snr_step)

os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# train_temporal_ue_memory_v4 imports v3, whose parser runs at import time.
# Give that import only model-building arguments, then restore our argv.
_EVAL_ARGV = list(sys.argv)
_model_argv = [
    sys.argv[0],
    "--config", ARGS.config,
    "--gpu", str(ARGS.gpu),
    "--compression", ARGS.compression,
    "--pooling", ARGS.pooling,
    "--d-mem", str(ARGS.d_mem),
    "--num-it", str(ARGS.eval_num_it),
    "--batch-size", str(ARGS.batch_size),
    "--seq-len", str(ARGS.seq_len),
    "--ue-pool-size", str(ARGS.ue_pool_size),
    "--memory-expiry-slots", str(ARGS.memory_expiry_slots),
    "--schedule-switch-prob", str(ARGS.schedule_switch_prob),
    "--schedule-reorder-prob", str(ARGS.schedule_reorder_prob),
]
if not ARGS.dynamic_scheduling:
    _model_argv.append("--fixed-scheduling")
sys.argv[:] = _model_argv

import tensorflow as tf
import sionna as sn
import train_temporal_ue_memory_v4 as v4

sys.argv[:] = _EVAL_ARGV
v3 = v4.v3

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def build_eval_system(k: int):
    p = v3.Parameters(ARGS.config, training=False, num_tx_eval=2, system="nrx")
    e2e = v3.E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    # Build all configured NRX iteration blocks and load the shipped backbone.
    e2e(1, 1.0)
    v3.load_weights(e2e, f"../weights/{p.label}_weights")
    e2e._receiver._neural_rx._cgnn.num_it = int(k)
    return p, e2e


def build_temporal():
    set_seed(ARGS.seed)
    p, e2e = build_eval_system(ARGS.eval_num_it)
    base = e2e._receiver._neural_rx._cgnn
    model = v4.PooledTemporalUEMemoryCGNN(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        compression=ARGS.compression,
        name=(
            f"iteration_transfer_{ARGS.pooling}_{ARGS.compression}_"
            f"d{ARGS.d_mem}_traink{ARGS.train_num_it}_evalk{ARGS.eval_num_it}"
        ),
    )
    generator = v3.TemporalTrainingDataGenerator(
        p,
        e2e,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=ARGS.dynamic_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
    )
    manager = v3.DifferentiableUEMemoryManager(
        capacity=ARGS.ue_pool_size,
        d_mem=ARGS.d_mem,
        expiry_slots=ARGS.memory_expiry_slots,
    )

    # Build every variable before HDF5 restoration.
    warm = generator.sample_batch(1, ARGS.seq_len, 3.0)
    state = manager.zero_state(1, tf.float32)
    state, mem, gap, valid = manager.gather(state, warm["ue_ids"][:, 0], 0)
    v3.temporal_forward(
        e2e._receiver,
        model,
        warm["y"][:, 0],
        warm["ls"][:, 0],
        warm["active"][:, 0],
        mem,
        gap,
        valid,
        training=False,
    )

    ckpt = Path(ARGS.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model.load_weights(str(ckpt))
    return p, e2e, model, generator, manager, ckpt


def count_errors(bits, b_hat, active):
    err = tf.reduce_any(
        tf.not_equal(tf.cast(bits, tf.int32), tf.cast(tf.round(b_hat), tf.int32)),
        axis=-1,
    )
    mask = tf.cast(active, tf.bool)
    err = tf.logical_and(err, mask)
    return (
        int(tf.reduce_sum(tf.cast(err, tf.int64)).numpy()),
        int(tf.reduce_sum(tf.cast(mask, tf.int64)).numpy()),
    )


def counter():
    return {
        "all_e": 0,
        "all_n": 0,
        "warm_e": 0,
        "warm_n": 0,
        "per_tb_e": [0] * ARGS.seq_len,
        "per_tb_n": [0] * ARGS.seq_len,
    }


def add(c, t, bits, b_hat, active):
    e, n = count_errors(bits, b_hat, active)
    c["all_e"] += e
    c["all_n"] += n
    c["per_tb_e"][t] += e
    c["per_tb_n"][t] += n
    if t >= 1:
        c["warm_e"] += e
        c["warm_n"] += n


def finalize(c, snr):
    def ratio(a, b):
        return a / b if b else None

    return {
        "snr_db": float(snr),
        "batches": int(ARGS.batches_per_snr),
        "errors_all": c["all_e"],
        "blocks_all": c["all_n"],
        "bler_all": ratio(c["all_e"], c["all_n"]),
        "errors_tb2plus": c["warm_e"],
        "blocks_tb2plus": c["warm_n"],
        "bler_tb2plus": ratio(c["warm_e"], c["warm_n"]),
        "ci95_tb2plus": wilson_interval(c["warm_e"], c["warm_n"]),
        "low_error_warning": c["warm_e"] < ARGS.min_errors_warning,
        "per_tb": [
            {
                "tb": t + 1,
                "errors": e,
                "blocks": n,
                "bler": ratio(e, n),
                "ci95": wilson_interval(e, n),
            }
            for t, (e, n) in enumerate(zip(c["per_tb_e"], c["per_tb_n"]))
        ],
    }


def run():
    p, e2e, temporal, generator, manager, ckpt = build_temporal()
    if int(p.n_size_bwp) != 132:
        raise RuntimeError(f"Research evaluation must use 132 PRBs, got {p.n_size_bwp}")

    curve = []
    for si, snr in enumerate(SNR_GRID):
        # Re-seed independently at each SNR. Every architecture/K cell therefore
        # starts this SNR from the exact same random stream.
        snr_seed = int(ARGS.seed + 100003 * si)
        set_seed(snr_seed)
        counts = counter()

        for _ in range(ARGS.batches_per_snr):
            batch = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, snr)
            bsz = int(batch["y"].shape[0])
            state = manager.zero_state(bsz, tf.float32)

            for t in range(ARGS.seq_len):
                bits = batch["bits"][:, t]
                y = batch["y"][:, t]
                ls = batch["ls"][:, t]
                active = batch["active"][:, t]

                state, mem, gap, valid = manager.gather(
                    state, batch["ue_ids"][:, t], t
                )
                llr, _, next_mem, _, _ = v3.temporal_forward(
                    e2e._receiver,
                    temporal,
                    y,
                    ls,
                    active,
                    mem,
                    gap,
                    valid,
                    training=False,
                )
                b_hat, _ = e2e._receiver._tb_decoders[0](llr)
                add(counts, t, bits, b_hat, active)
                state = manager.scatter(
                    state,
                    batch["ue_ids"][:, t],
                    next_mem,
                    active,
                    t,
                )

        point = finalize(counts, snr)
        curve.append(point)
        print(
            "ITER_TRANSFER_POINT="
            + json.dumps(
                {
                    "train_k": ARGS.train_num_it,
                    "eval_k": ARGS.eval_num_it,
                    **point,
                }
            ),
            flush=True,
        )

    crossing = log_bler_crossing(curve, target=0.1)
    return {
        "experiment": "temporal_iteration_transfer_132prb_v1",
        "checkpoint": str(ckpt),
        "config": ARGS.config,
        "parameter_mode": "training=False",
        "n_size_bwp": int(p.n_size_bwp),
        "pooling": ARGS.pooling,
        "compression": ARGS.compression,
        "d_mem": int(ARGS.d_mem),
        "memory_bits_per_ue": int(ARGS.d_mem * 32),
        "train_num_it": int(ARGS.train_num_it),
        "eval_num_it": int(ARGS.eval_num_it),
        "seq_len": int(ARGS.seq_len),
        "primary_metric": "TB2+ TBLER",
        "target_bler": 0.1,
        "snr_grid_db": SNR_GRID,
        "snr_db_at_10pct_tbler": crossing,
        "crossing_method": (
            "log-BLER interpolation; zero-error points use "
            "(errors+0.5)/(blocks+1)"
        ),
        "common_random_numbers": {
            "enabled": True,
            "fixed_batches_per_snr": int(ARGS.batches_per_snr),
            "base_seed": int(ARGS.seed),
            "snr_seed_formula": "base_seed + 100003 * snr_index",
            "no_error_based_early_stopping": True,
        },
        "dynamic_scheduling": bool(ARGS.dynamic_scheduling),
        "ue_pool_size": int(ARGS.ue_pool_size),
        "memory_expiry_slots": int(ARGS.memory_expiry_slots),
        "schedule_switch_prob": float(ARGS.schedule_switch_prob),
        "schedule_reorder_prob": float(ARGS.schedule_reorder_prob),
        "batch_size": int(ARGS.batch_size),
        "batches_per_snr": int(ARGS.batches_per_snr),
        "min_errors_warning": int(ARGS.min_errors_warning),
        "curve": curve,
    }


def write(summary):
    out = Path(ARGS.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out / "curve.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "train_k",
                "eval_k",
                "snr_db",
                "bler_all",
                "bler_tb2plus",
                "errors_tb2plus",
                "blocks_tb2plus",
                "ci95_low",
                "ci95_high",
                "batches",
            ]
        )
        for point in summary["curve"]:
            writer.writerow(
                [
                    summary["train_num_it"],
                    summary["eval_num_it"],
                    point["snr_db"],
                    point["bler_all"],
                    point["bler_tb2plus"],
                    point["errors_tb2plus"],
                    point["blocks_tb2plus"],
                    point["ci95_tb2plus"][0],
                    point["ci95_tb2plus"][1],
                    point["batches"],
                ]
            )

    print("ITER_TRANSFER_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    write(run())
