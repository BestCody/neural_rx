#!/usr/bin/env python3
"""Evaluate trained compact temporal UE memory on the true 132-PRB NRX setup."""

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
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--compression", choices=["writer", "pca", "autoencoder"], required=True)
    p.add_argument("--pooling", choices=["mean", "attention", "cnn"], default="mean")
    p.add_argument("--d-mem", type=int, required=True)
    p.add_argument("--num-it", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--snr-min", type=float, default=1.5)
    p.add_argument("--snr-max", type=float, default=3.75)
    p.add_argument("--snr-step", type=float, default=0.25)
    p.add_argument("--target-errors", type=int, default=120)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--dynamic-scheduling", action="store_true")
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


ARGS = parse_args()
if ARGS.num_it != 2:
    raise ValueError("Primary temporal evaluation is K=2")
if ARGS.seq_len < 2:
    raise ValueError("seq-len must be >= 2")
if ARGS.batch_size <= 0 or ARGS.target_errors <= 0 or ARGS.max_batches <= 0:
    raise ValueError("batch-size, target-errors and max-batches must be positive")
if ARGS.d_mem <= 0:
    raise ValueError("d-mem must be positive")
for name in ("schedule_switch_prob", "schedule_reorder_prob"):
    value = float(getattr(ARGS, name))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
SNR_GRID = make_snr_grid(ARGS.snr_min, ARGS.snr_max, ARGS.snr_step)

os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# v4 imports v3, whose CLI is parsed at import time. Give it only model args.
_EVAL_ARGV = list(sys.argv)
_model_argv = [
    sys.argv[0],
    "--config", ARGS.config,
    "--gpu", str(ARGS.gpu),
    "--compression", ARGS.compression,
    "--pooling", ARGS.pooling,
    "--d-mem", str(ARGS.d_mem),
    "--num-it", "2",
    "--batch-size", str(ARGS.batch_size),
    "--seq-len", str(ARGS.seq_len),
    "--ue-pool-size", str(ARGS.ue_pool_size),
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
ColdTemporalClass = v3.TemporalUEMemoryCGNN

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


def build_eval_system(k):
    p = v3.Parameters(ARGS.config, training=False, num_tx_eval=2, system="nrx")
    e2e = v3.E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    e2e(1, 1.0)
    v3.load_weights(e2e, f"../weights/{p.label}_weights")
    e2e._receiver._neural_rx._cgnn.num_it = int(k)
    return p, e2e


def build_temporal():
    set_seed(ARGS.seed)
    p, e2e = build_eval_system(2)
    base = e2e._receiver._neural_rx._cgnn
    model = v4.PooledTemporalUEMemoryCGNN(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        compression=ARGS.compression,
        name=f"eval_temporal_{ARGS.pooling}_{ARGS.compression}_d{ARGS.d_mem}",
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
        capacity=ARGS.ue_pool_size, d_mem=ARGS.d_mem, expiry_slots=8
    )

    # Build every model variable before loading HDF5 weights.
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

    ckpt = Path(ARGS.checkpoint).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    model.load_weights(str(ckpt))
    return p, e2e, model, generator, manager, ckpt


def build_cold(k, warm):
    p, e2e = build_eval_system(k)
    base = e2e._receiver._neural_rx._cgnn
    model = ColdTemporalClass(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        compression="writer",
        name=f"cold_k{k}_eval_wrapper",
    )
    b = int(warm["y"].shape[0])
    u = int(warm["active"].shape[-1])
    z = tf.zeros([b, u, ARGS.d_mem], tf.float32)
    zg = tf.zeros([b, u], tf.int32)
    inv = tf.zeros([b, u], tf.bool)
    v3.temporal_forward(
        e2e._receiver,
        model,
        warm["y"][:, 0],
        warm["ls"][:, 0],
        warm["active"][:, 0],
        z,
        zg,
        inv,
        training=False,
    )
    return e2e._receiver, model


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


def finalize(c, snr, batches):
    def ratio(a, b):
        return a / b if b else None

    return {
        "snr_db": float(snr),
        "batches": int(batches),
        "errors_all": c["all_e"],
        "blocks_all": c["all_n"],
        "bler_all": ratio(c["all_e"], c["all_n"]),
        "errors_tb2plus": c["warm_e"],
        "blocks_tb2plus": c["warm_n"],
        "bler_tb2plus": ratio(c["warm_e"], c["warm_n"]),
        "ci95_tb2plus": wilson_interval(c["warm_e"], c["warm_n"]),
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

    warm = generator.sample_batch(1, ARGS.seq_len, 3.0)
    rx2, cold2 = build_cold(2, warm)
    rx8, cold8 = build_cold(8, warm)
    set_seed(ARGS.seed)

    methods = ["cold_k2", "cold_k8", "temporal_k2"]
    curves = {m: [] for m in methods}
    for snr in SNR_GRID:
        counts = {m: counter() for m in methods}
        batches_run = 0
        for bi in range(ARGS.max_batches):
            batch = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, snr)
            bsz = int(batch["y"].shape[0])
            num_ues = int(batch["active"].shape[-1])
            state = manager.zero_state(bsz, tf.float32)
            zero_mem = tf.zeros([bsz, num_ues, ARGS.d_mem], tf.float32)
            zero_gap = tf.zeros([bsz, num_ues], tf.int32)
            invalid = tf.zeros([bsz, num_ues], tf.bool)

            for t in range(ARGS.seq_len):
                bits = batch["bits"][:, t]
                y = batch["y"][:, t]
                ls = batch["ls"][:, t]
                active = batch["active"][:, t]

                l2, _, _, _, _ = v3.temporal_forward(
                    rx2, cold2, y, ls, active, zero_mem, zero_gap, invalid, training=False
                )
                b2, _ = rx2._tb_decoders[0](l2)
                add(counts["cold_k2"], t, bits, b2, active)

                l8, _, _, _, _ = v3.temporal_forward(
                    rx8, cold8, y, ls, active, zero_mem, zero_gap, invalid, training=False
                )
                b8, _ = rx8._tb_decoders[0](l8)
                add(counts["cold_k8"], t, bits, b8, active)

                state, mem, gap, valid = manager.gather(
                    state, batch["ue_ids"][:, t], t
                )
                lt, _, next_mem, _, _ = v3.temporal_forward(
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
                bt, _ = e2e._receiver._tb_decoders[0](lt)
                add(counts["temporal_k2"], t, bits, bt, active)
                state = manager.scatter(
                    state, batch["ue_ids"][:, t], next_mem, active, t
                )

            batches_run = bi + 1
            # All methods stop on the same batch so comparisons use identical samples.
            if all(counts[m]["warm_e"] >= ARGS.target_errors for m in methods):
                break

        for method in methods:
            point = finalize(counts[method], snr, batches_run)
            curves[method].append(point)
            print("EVAL_POINT=" + json.dumps({"method": method, **point}), flush=True)

    cross = {m: log_bler_crossing(curves[m], target=0.1) for m in methods}
    c2, c8, ct = cross["cold_k2"], cross["cold_k8"], cross["temporal_k2"]
    gap = c2 - c8 if c2 is not None and c8 is not None else None
    improvement = c2 - ct if c2 is not None and ct is not None else None
    recovered = (
        improvement / gap
        if gap is not None and gap > 0 and improvement is not None
        else None
    )

    return {
        "experiment": "temporal_ue_memory_132prb_evaluation_v2",
        "checkpoint": str(ckpt),
        "config": ARGS.config,
        "parameter_mode": "training=False",
        "n_size_bwp": int(p.n_size_bwp),
        "compression": ARGS.compression,
        "pooling": ARGS.pooling,
        "d_mem": ARGS.d_mem,
        "memory_bits_per_ue": ARGS.d_mem * 32,
        "seq_len": ARGS.seq_len,
        "primary_metric": "TB2+ TBLER",
        "target_bler": 0.1,
        "snr_grid_db": SNR_GRID,
        "crossing_method": "log-BLER interpolation; zero-error points use (errors+0.5)/(blocks+1)",
        "snr_db_at_10pct_tbler": cross,
        "cold_iteration_gap_db": gap,
        "temporal_improvement_over_cold_k2_db": improvement,
        "gap_recovered_fraction": recovered,
        "gap_recovered_percent": 100 * recovered if recovered is not None else None,
        "dynamic_scheduling": ARGS.dynamic_scheduling,
        "ue_pool_size": ARGS.ue_pool_size,
        "schedule_switch_prob": ARGS.schedule_switch_prob,
        "schedule_reorder_prob": ARGS.schedule_reorder_prob,
        "batch_size": ARGS.batch_size,
        "target_errors": ARGS.target_errors,
        "max_batches": ARGS.max_batches,
        "seed": ARGS.seed,
        "curves": curves,
    }


def write(summary):
    out = Path(ARGS.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "snr_db", "bler_all", "bler_tb2plus", "errors_tb2plus",
            "blocks_tb2plus", "ci95_low", "ci95_high", "batches",
        ])
        for method, points in summary["curves"].items():
            for point in points:
                writer.writerow([
                    method,
                    point["snr_db"],
                    point["bler_all"],
                    point["bler_tb2plus"],
                    point["errors_tb2plus"],
                    point["blocks_tb2plus"],
                    point["ci95_tb2plus"][0],
                    point["ci95_tb2plus"][1],
                    point["batches"],
                ])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for method, points in summary["curves"].items():
        ax.semilogy(
            [p["snr_db"] for p in points],
            [p["bler_tb2plus"] for p in points],
            marker="o",
            label=method,
        )
    ax.axhline(0.1, linestyle="--", linewidth=1)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("TB2+ TBLER")
    ax.set_title(
        f"132-PRB temporal UE memory: {ARGS.pooling}/{ARGS.compression}, d_mem={ARGS.d_mem}"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tbler_vs_snr.png", dpi=180)
    fig.savefig(out / "tbler_vs_snr.pdf")
    plt.close(fig)
    print("EVALUATION_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    write(run())
