#!/usr/bin/env python3
"""Evaluate trained temporal UE memory against cold K=2 and cold K=8.

The three receivers see the exact same temporally correlated TB sequences at
all SNR points. The primary temporal metric excludes TB1 because TB1 has no
previous memory by construction:

    cold K=2            -- same TB2+ samples, no temporal memory
    cold K=8            -- same TB2+ samples, no temporal memory
    temporal K=2        -- trained UE memory carried across the sequence

For each method this script reports TBLER over all TBs, TBLER over TB2+, and
per-TB TBLER. It estimates the SNR at 10% TBLER from the TB2+ curve and reports
how much of the cold K=2 -> cold K=8 gap the temporal K=2 model recovers.

The evaluator deliberately uses the same TemporalTrainingDataGenerator and
stable UE-ID memory manager as training so physical UE identity and temporal
channel continuity are preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--compression",
        choices=["writer", "pca", "autoencoder"],
        default="writer",
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "attention", "cnn"],
        default="mean",
    )
    p.add_argument("--d-mem", type=int, default=32)
    p.add_argument("--num-it", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
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
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


ARGS = parse_args()
if ARGS.num_it != 2:
    raise ValueError("Primary temporal evaluation is defined for K=2")
if ARGS.seq_len < 2:
    raise ValueError("seq-len must be >= 2 so TB2+ exists")
if ARGS.batch_size <= 0 or ARGS.target_errors <= 0 or ARGS.max_batches <= 0:
    raise ValueError("batch-size, target-errors and max-batches must be positive")
if ARGS.snr_step <= 0 or ARGS.snr_max < ARGS.snr_min:
    raise ValueError("invalid SNR range")

os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# train_temporal_ue_memory_v4 imports v3, whose established CLI is parsed at
# import time. Reconstruct only the architecture-relevant v3/v4 arguments,
# import the model implementation, and then restore this evaluator's argv.
_EVAL_ARGV = list(sys.argv)
_model_argv = [
    sys.argv[0],
    "--config", ARGS.config,
    "--gpu", str(ARGS.gpu),
    "--compression", ARGS.compression,
    "--pooling", ARGS.pooling,
    "--d-mem", str(ARGS.d_mem),
    "--num-it", str(ARGS.num_it),
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
import train_temporal_ue_memory_v4 as v4

sys.argv[:] = _EVAL_ARGV
v3 = v4.v3

# Preserve the original mean-pooling v3 class for cold baselines. The trained
# model itself must use v4 so attention/CNN checkpoints are also evaluable.
ColdTemporalClass = v3.TemporalUEMemoryCGNN
v3.TemporalUEMemoryCGNN = v4.PooledTemporalUEMemoryCGNN

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def set_seed(seed: int):
    v3.set_seed(int(seed))


def make_snrs(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step))
    values = [lo + i * step for i in range(n + 1)]
    if values[-1] < hi - 1e-9:
        values.append(hi)
    return [float(round(x, 10)) for x in values]


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return [None, None]
    p = errors / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total))
        / denom
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def snr_at_target(points, target=0.1):
    """Log-BLER interpolation of SNR at target; returns None if not bracketed."""
    clean = [
        (float(p["snr_db"]), float(p["bler_tb2plus"]))
        for p in points
        if p["bler_tb2plus"] is not None and p["bler_tb2plus"] > 0.0
    ]
    clean.sort()
    for (x0, y0), (x1, y1) in zip(clean[:-1], clean[1:]):
        if (y0 >= target >= y1) or (y0 <= target <= y1):
            if y0 == y1:
                return float((x0 + x1) / 2.0)
            ly0 = math.log10(y0)
            ly1 = math.log10(y1)
            lt = math.log10(target)
            frac = (lt - ly0) / (ly1 - ly0)
            return float(x0 + frac * (x1 - x0))
    return None


def build_trained_temporal():
    set_seed(ARGS.seed)
    p, e2e, model, generator, manager = v3.build()
    receiver = e2e._receiver

    # Build every weight before HDF5 restoration.
    warm = generator.sample_batch(1, ARGS.seq_len, 3.0)
    state = manager.zero_state(1, tf.float32)
    state, mem, gap, valid = manager.gather(state, warm["ue_ids"][:, 0], 0)
    v3.temporal_forward(
        receiver,
        model,
        warm["y"][:, 0],
        warm["ls"][:, 0],
        warm["active"][:, 0],
        mem,
        gap,
        valid,
        training=False,
    )

    checkpoint = Path(ARGS.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model.load_weights(str(checkpoint))
    if int(model.base._num_it) != 2:
        raise RuntimeError(f"loaded temporal model is K={model.base._num_it}, expected K=2")
    return p, e2e, model, generator, manager, checkpoint


def build_cold(k: int, warm_batch):
    """Fresh shipped NRX weights with no temporal injection, fixed to K."""
    set_seed(ARGS.seed + int(k))
    p = v3.Parameters(ARGS.config, training=True, system="nrx")
    e2e = v3.E2E_Model(p, training=True)
    e2e(1, 1.0)
    v3.load_weights(e2e, f"../weights/{p.label}_weights")
    base = e2e._receiver._neural_rx._cgnn
    base.num_it = int(k)
    model = ColdTemporalClass(
        base,
        d_mem=ARGS.d_mem,
        d_s=p.d_s,
        compression="writer",
        name=f"cold_k{k}_evaluation_wrapper",
    )

    b = int(warm_batch["y"].shape[0])
    u = int(warm_batch["active"].shape[-1])
    zero_mem = tf.zeros([b, u, ARGS.d_mem], tf.float32)
    zero_gap = tf.zeros([b, u], tf.int32)
    invalid = tf.zeros([b, u], tf.bool)
    v3.temporal_forward(
        e2e._receiver,
        model,
        warm_batch["y"][:, 0],
        warm_batch["ls"][:, 0],
        warm_batch["active"][:, 0],
        zero_mem,
        zero_gap,
        invalid,
        training=False,
    )
    return e2e._receiver, model


def decode(receiver, llr):
    b_hat, crc_ok = receiver._tb_decoders[0](llr)
    return b_hat, crc_ok


def block_errors(bits, b_hat, active):
    bits_i = tf.cast(bits, tf.int32)
    bhat_i = tf.cast(tf.round(b_hat), tf.int32)
    err = tf.reduce_any(tf.not_equal(bits_i, bhat_i), axis=-1)
    active_b = tf.cast(active, tf.bool)
    err = tf.logical_and(err, active_b)
    return int(tf.reduce_sum(tf.cast(err, tf.int64)).numpy()), int(
        tf.reduce_sum(tf.cast(active_b, tf.int64)).numpy()
    )


def new_counter():
    return {
        "all_errors": 0,
        "all_blocks": 0,
        "tb2plus_errors": 0,
        "tb2plus_blocks": 0,
        "per_tb_errors": [0 for _ in range(ARGS.seq_len)],
        "per_tb_blocks": [0 for _ in range(ARGS.seq_len)],
        "crc_failures": 0,
        "crc_blocks": 0,
    }


def add_observation(counter, t, bits, b_hat, crc_ok, active):
    errors, blocks = block_errors(bits, b_hat, active)
    counter["all_errors"] += errors
    counter["all_blocks"] += blocks
    counter["per_tb_errors"][t] += errors
    counter["per_tb_blocks"][t] += blocks
    if t >= 1:
        counter["tb2plus_errors"] += errors
        counter["tb2plus_blocks"] += blocks

    active_b = tf.cast(active, tf.bool)
    crc_fail = tf.logical_and(tf.logical_not(tf.cast(crc_ok, tf.bool)), active_b)
    counter["crc_failures"] += int(
        tf.reduce_sum(tf.cast(crc_fail, tf.int64)).numpy()
    )
    counter["crc_blocks"] += int(
        tf.reduce_sum(tf.cast(active_b, tf.int64)).numpy()
    )


def finalize_counter(counter, snr_db, batches):
    def ratio(a, b):
        return (a / b) if b else None

    per_tb = []
    for t, (e, n) in enumerate(
        zip(counter["per_tb_errors"], counter["per_tb_blocks"]), start=1
    ):
        per_tb.append(
            {
                "tb": t,
                "errors": int(e),
                "blocks": int(n),
                "bler": ratio(e, n),
                "ci95": wilson_interval(e, n),
            }
        )

    return {
        "snr_db": float(snr_db),
        "batches": int(batches),
        "errors_all": int(counter["all_errors"]),
        "blocks_all": int(counter["all_blocks"]),
        "bler_all": ratio(counter["all_errors"], counter["all_blocks"]),
        "errors_tb2plus": int(counter["tb2plus_errors"]),
        "blocks_tb2plus": int(counter["tb2plus_blocks"]),
        "bler_tb2plus": ratio(counter["tb2plus_errors"], counter["tb2plus_blocks"]),
        "ci95_tb2plus": wilson_interval(
            counter["tb2plus_errors"], counter["tb2plus_blocks"]
        ),
        "crc_bler_all": ratio(counter["crc_failures"], counter["crc_blocks"]),
        "per_tb": per_tb,
    }


def evaluate():
    p, e2e, temporal_model, generator, manager, checkpoint = build_trained_temporal()
    receiver_temporal = e2e._receiver

    # One common warm batch is only used to build the cold wrapper variables.
    warm = generator.sample_batch(1, ARGS.seq_len, 3.0)
    receiver_k2, cold_k2 = build_cold(2, warm)
    receiver_k8, cold_k8 = build_cold(8, warm)

    # Reset data RNG after all model/layer creation. From here onward the three
    # methods see exactly the same generated sequence batch at each SNR.
    set_seed(ARGS.seed)

    methods = ["cold_k2", "cold_k8", "temporal_k2"]
    curves = {m: [] for m in methods}
    snrs = make_snrs(ARGS.snr_min, ARGS.snr_max, ARGS.snr_step)

    for snr in snrs:
        counters = {m: new_counter() for m in methods}
        batches_run = 0

        for batch_idx in range(ARGS.max_batches):
            batch = generator.sample_batch(ARGS.batch_size, ARGS.seq_len, snr)
            bsz = int(batch["y"].shape[0])
            num_ues = int(batch["active"].shape[-1])

            state = manager.zero_state(bsz, tf.float32)
            zero_mem = tf.zeros([bsz, num_ues, ARGS.d_mem], tf.float32)
            zero_gap = tf.zeros([bsz, num_ues], tf.int32)
            invalid = tf.zeros([bsz, num_ues], tf.bool)

            for t in range(ARGS.seq_len):
                bits_t = batch["bits"][:, t]
                y_t = batch["y"][:, t]
                ls_t = batch["ls"][:, t]
                active_t = batch["active"][:, t]

                # Cold K=2 on exactly the same TB.
                llr2, _, _, _, _ = v3.temporal_forward(
                    receiver_k2,
                    cold_k2,
                    y_t,
                    ls_t,
                    active_t,
                    zero_mem,
                    zero_gap,
                    invalid,
                    training=False,
                )
                b2, crc2 = decode(receiver_k2, llr2)
                add_observation(counters["cold_k2"], t, bits_t, b2, crc2, active_t)

                # Cold K=8 on exactly the same TB.
                llr8, _, _, _, _ = v3.temporal_forward(
                    receiver_k8,
                    cold_k8,
                    y_t,
                    ls_t,
                    active_t,
                    zero_mem,
                    zero_gap,
                    invalid,
                    training=False,
                )
                b8, crc8 = decode(receiver_k8, llr8)
                add_observation(counters["cold_k8"], t, bits_t, b8, crc8, active_t)

                # Temporal K=2: gather and commit by stable physical UE identity.
                state, prev_memory, memory_gap, memory_valid = manager.gather(
                    state, batch["ue_ids"][:, t], t
                )
                llrt, _, next_memory, _, _ = v3.temporal_forward(
                    receiver_temporal,
                    temporal_model,
                    y_t,
                    ls_t,
                    active_t,
                    prev_memory,
                    memory_gap,
                    memory_valid,
                    training=False,
                )
                bt, crct = decode(receiver_temporal, llrt)
                add_observation(
                    counters["temporal_k2"], t, bits_t, bt, crct, active_t
                )
                state = manager.scatter(
                    state,
                    batch["ue_ids"][:, t],
                    next_memory,
                    active_t,
                    t,
                )

            batches_run = batch_idx + 1
            if all(
                counters[m]["tb2plus_errors"] >= ARGS.target_errors
                for m in methods
            ):
                break

        for method in methods:
            point = finalize_counter(counters[method], snr, batches_run)
            curves[method].append(point)
            print(
                "EVAL_POINT="
                + json.dumps({"method": method, **point}),
                flush=True,
            )

    crossings = {
        method: snr_at_target(curves[method], 0.1) for method in methods
    }
    c2 = crossings["cold_k2"]
    c8 = crossings["cold_k8"]
    ct = crossings["temporal_k2"]
    gap = (c2 - c8) if c2 is not None and c8 is not None else None
    recovered = (
        (c2 - ct) / gap
        if gap is not None and gap > 0.0 and ct is not None
        else None
    )

    summary = {
        "experiment": "temporal_ue_memory_fixed_budget_evaluation_v1",
        "checkpoint": str(checkpoint),
        "architecture": "ue_identity_aware_temporal_memory_v4_pooling",
        "config": ARGS.config,
        "compression": ARGS.compression,
        "pooling": ARGS.pooling,
        "d_mem": ARGS.d_mem,
        "memory_dtype": "float32",
        "memory_bytes_per_ue": ARGS.d_mem * 4,
        "memory_bits_per_ue": ARGS.d_mem * 32,
        "temporal_k": ARGS.num_it,
        "cold_k2": 2,
        "cold_k8": 8,
        "seq_len": ARGS.seq_len,
        "primary_metric": "TB2+ TBLER; TB1 excluded because no prior memory exists",
        "snr_crossing_interpolation": "linear in SNR vs log10(TBLER)",
        "target_bler": 0.1,
        "snr_db_at_10pct_tbler": crossings,
        "cold_iteration_gap_db": gap,
        "temporal_improvement_over_cold_k2_db": (
            c2 - ct if c2 is not None and ct is not None else None
        ),
        "gap_recovered_fraction": recovered,
        "gap_recovered_percent": (
            100.0 * recovered if recovered is not None else None
        ),
        "fixed_scheduling": not ARGS.dynamic_scheduling,
        "ue_pool_size": ARGS.ue_pool_size,
        "schedule_switch_prob": ARGS.schedule_switch_prob,
        "schedule_reorder_prob": ARGS.schedule_reorder_prob,
        "batch_size": ARGS.batch_size,
        "target_errors": ARGS.target_errors,
        "max_batches": ARGS.max_batches,
        "seed": ARGS.seed,
        "curves": curves,
    }
    return summary


def write_outputs(summary):
    out = Path(
        ARGS.output_dir
        or (
            Path.home()
            / "sionna-srsran"
            / "temporal_reuse"
            / "ue_memory"
            / "evaluation"
            / f"{ARGS.pooling}_{ARGS.compression}_d{ARGS.d_mem}"
        )
    ).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = out / "curves.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
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
        for method, points in summary["curves"].items():
            for point in points:
                writer.writerow(
                    [
                        method,
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

    # Plotting is useful for professor-facing inspection but is not required for
    # the numerical evaluation to succeed.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        labels = {
            "cold_k2": "Cold K=2",
            "cold_k8": "Cold K=8",
            "temporal_k2": f"Temporal K=2 ({ARGS.pooling}+{ARGS.compression}, d={ARGS.d_mem})",
        }
        for method in ["cold_k2", "cold_k8", "temporal_k2"]:
            pts = summary["curves"][method]
            ax.semilogy(
                [p["snr_db"] for p in pts],
                [max(p["bler_tb2plus"], 1e-5) for p in pts],
                marker="o",
                label=labels[method],
            )
        ax.axhline(0.1, linestyle="--", linewidth=1.0, label="10% TBLER")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("TBLER (TB2+)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "tbler_vs_snr.png", dpi=180)
        fig.savefig(out / "tbler_vs_snr.pdf")
        plt.close(fig)
    except Exception as exc:
        print(f"PLOT_WARNING={exc}", flush=True)

    print("EVALUATION_OUTPUT_DIR=" + str(out), flush=True)
    print("TEMPORAL_EVAL_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    result = evaluate()
    write_outputs(result)
