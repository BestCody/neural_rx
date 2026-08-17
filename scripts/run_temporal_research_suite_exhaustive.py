#!/usr/bin/env python3
"""Exhaustive temporal-memory study: pooling x compression x memory cap.

Base factorial (all fixed scheduling, all 6000-step runs by default):
    pooling      = mean | attention | cnn
    compression  = writer | pca | autoencoder
    d_mem        = 8 | 16 | 32 | 56

This gives 36 base learned configurations. The suite additionally evaluates a
raw full-state upper bound, retrains the best compressed configuration under
dynamic scheduling, and repeats the fixed winner with two additional seeds.

PCA fairness invariant
----------------------
Mean+PCA uses the ordinary v4 trainer because mean pooling is parameter-free.
Attention/CNN+PCA MUST use v5_pca_pooler_calibrated, which calibrates the
learned pooler, freezes it, then fits/fixes PCA. The exhaustive suite refuses to
accept an old v4 checkpoint for those cells.

The suite is resumable and schedules one independent job per configured GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import run_temporal_research_suite as base

A = base.A
ROOT = base.ROOT
SCRIPT_DIR = base.SCRIPT_DIR
PY = base.PY
CAPS = base.CAPS
GPUS = base.GPUS
EXTRA_SEEDS = base.EXTRA_SEEDS
COMPRESSORS = ["writer", "pca", "autoencoder"]
POOLINGS = ["mean", "attention", "cnn"]


def learned_pca_valid(out, pooling, d_mem, seed, dynamic):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_pca_idaware_d{d_mem}_k2.weights.h5"
    if not summary.exists() or not ckpt.exists():
        return False
    try:
        s = base.load_json(summary)
        protocol = s.get("pca_protocol", {})
        return all(
            [
                s.get("architecture")
                == "ue_identity_aware_temporal_memory_v5_pca_pooler_calibrated",
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
            ]
        )
    except Exception:
        return False


def train_factorial(gpu, compression, pooling, d_mem, seed, dynamic=False):
    # All non-problematic cells continue through the established v4 path.
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

    if learned_pca_valid(out, pooling, d_mem, seed, dynamic):
        base.safe_print("REUSE_CALIBRATED_PCA_TRAINING", out, flush=True)
        return ckpt

    # Intentionally do NOT reuse any pre-fix v4 Attention/CNN+PCA checkpoint.
    label = f"train-{mode}-{pooling}-pca-calibrated-d{d_mem}-seed{seed}"
    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_ue_memory_v5_pca_pooler_calibrated.py",
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
    if not learned_pca_valid(out, pooling, d_mem, seed, dynamic):
        raise RuntimeError(
            f"calibrated learned-pool PCA output failed validation: {out}"
        )
    return ckpt


def factorial_job(pooling, compression, d_mem):
    def job(gpu):
        ckpt = train_factorial(
            gpu, compression, pooling, d_mem, A.seed, dynamic=False
        )
        return base.eval_compressed(
            gpu,
            ckpt,
            compression,
            pooling,
            d_mem,
            A.seed,
            "fixed",
            tag=f"factorial_{pooling}_{compression}_d{d_mem}",
        )

    return job


def winner_checkpoint(pooling, compression, d_mem, seed):
    out = (
        ROOT
        / "trained"
        / "fixed"
        / f"seed_{seed}"
        / f"{pooling}_{compression}_d{d_mem}"
    )
    ckpt = out / f"ue_memory_{pooling}_{compression}_idaware_d{d_mem}_k2.weights.h5"
    if ckpt.exists():
        return ckpt
    legacy = base.legacy_checkpoint(compression, pooling, d_mem, seed, False)
    if legacy is not None:
        return legacy
    raise FileNotFoundError(ckpt)


def scheduling_job(pooling, compression, d_mem, scenario):
    def job(gpu):
        ckpt = winner_checkpoint(pooling, compression, d_mem, A.seed)
        return base.eval_compressed(
            gpu,
            ckpt,
            compression,
            pooling,
            d_mem,
            A.seed,
            scenario,
            tag=f"winner_{pooling}_{compression}_d{d_mem}",
        )

    return job


def dynamic_winner_job(pooling, compression, d_mem):
    def job(gpu):
        ckpt = train_factorial(
            gpu, compression, pooling, d_mem, A.seed, dynamic=True
        )
        return base.eval_compressed(
            gpu,
            ckpt,
            compression,
            pooling,
            d_mem,
            A.seed,
            "switch_reorder",
            tag=f"dynamic_trained_winner_{pooling}_{compression}_d{d_mem}",
        )

    return job


def seed_winner_job(pooling, compression, d_mem, seed):
    def job(gpu):
        ckpt = train_factorial(
            gpu, compression, pooling, d_mem, seed, dynamic=False
        )
        return base.eval_compressed(
            gpu,
            ckpt,
            compression,
            pooling,
            d_mem,
            seed,
            "fixed",
            tag=f"winner_{pooling}_{compression}_d{d_mem}",
        )

    return job


def best_factorial(results):
    choices = []
    for key, result in results.items():
        x = base.temporal_cross(result)
        if x is not None:
            choices.append((float(x), key))
    if not choices:
        raise RuntimeError("No exhaustive temporal curve bracketed 10% TBLER")
    choices.sort()
    return choices[0][1]


def aggregate(results, full_state, winner_key, scheduling, seeds):
    rows = []
    for (pool, comp, d_mem), s in sorted(results.items()):
        rows.append(
            {
                "group": "factorial",
                "pooling": pool,
                "compression": comp,
                "d_mem": d_mem,
                "scenario": "fixed",
                "seed": A.seed,
                "temporal_snr10": base.temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )

    if full_state is not None:
        rows.append(
            {
                "group": "full_state",
                "pooling": "none",
                "compression": "raw_full_state",
                "d_mem": None,
                "scenario": "fixed",
                "seed": A.seed,
                "temporal_snr10": base.temporal_cross(full_state),
                "cold_k2_snr10": full_state["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": full_state["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": full_state.get("gap_recovered_percent"),
                "memory_bits_per_ue": full_state.get("memory_bits_per_ue"),
            }
        )

    pool, comp, d_mem = winner_key
    for scenario, s in scheduling.items():
        rows.append(
            {
                "group": "scheduling",
                "pooling": pool,
                "compression": comp,
                "d_mem": d_mem,
                "scenario": scenario,
                "seed": A.seed,
                "temporal_snr10": base.temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )

    for seed, s in seeds.items():
        rows.append(
            {
                "group": "seed",
                "pooling": pool,
                "compression": comp,
                "d_mem": d_mem,
                "scenario": "fixed",
                "seed": seed,
                "temporal_snr10": base.temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )

    fields = [
        "group",
        "pooling",
        "compression",
        "d_mem",
        "scenario",
        "seed",
        "temporal_snr10",
        "cold_k2_snr10",
        "cold_k8_snr10",
        "gap_recovered_percent",
        "memory_bits_per_ue",
    ]
    with (ROOT / "all_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    graphs = ROOT / "graphs"
    graphs.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for pool in POOLINGS:
        for comp_name in COMPRESSORS:
            ys = [
                base.temporal_cross(results[(pool, comp_name, d)]) for d in CAPS
            ]
            ax.plot(CAPS, ys, marker="o", label=f"{pool} + {comp_name}")
    first = next(iter(results.values()))
    c2 = first["snr_db_at_10pct_tbler"].get("cold_k2")
    c8 = first["snr_db_at_10pct_tbler"].get("cold_k8")
    if c2 is not None:
        ax.axhline(c2, linestyle="--", label="cold K=2")
    if c8 is not None:
        ax.axhline(c8, linestyle=":", label="cold K=8")
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
    ax.set_title("Exhaustive pooling x compression x memory-cap study")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graphs / "factorial_snr10.png", dpi=180)
    fig.savefig(graphs / "factorial_snr10.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for pool in POOLINGS:
        for comp_name in COMPRESSORS:
            ys = [
                results[(pool, comp_name, d)].get("gap_recovered_percent")
                for d in CAPS
            ]
            ax.plot(CAPS, ys, marker="o", label=f"{pool} + {comp_name}")
    ax.axhline(0, linewidth=1)
    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("K2 -> K8 gap recovered (%)")
    ax.set_title("Exhaustive temporal-memory gap recovery")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(graphs / "factorial_gap_recovered.png", dpi=180)
    fig.savefig(graphs / "factorial_gap_recovered.pdf")
    plt.close(fig)

    summary = {
        "suite": "temporal_ue_memory_exhaustive_factorial_v1_calibrated_pca",
        "config": A.config,
        "gpus": GPUS,
        "parallel_gpu_count": len(GPUS),
        "poolings": POOLINGS,
        "compressors": COMPRESSORS,
        "capacities": CAPS,
        "base_factorial_training_count": len(POOLINGS)
        * len(COMPRESSORS)
        * len(CAPS),
        "train_steps_per_trained_configuration": A.train_steps,
        "pca_fairness_protocol": {
            "mean": "parameter-free mean pooled state; ordinary frozen PCA fit",
            "attention_cnn": "temporally calibrate learned pooler, freeze pooler, fit PCA once, freeze PCA during temporal training",
        },
        "winner": {
            "pooling": pool,
            "compression": comp,
            "d_mem": d_mem,
            "fixed_seed": A.seed,
            "snr10_db": base.temporal_cross(results[winner_key]),
            "gap_recovered_percent": results[winner_key].get(
                "gap_recovered_percent"
            ),
        },
        "full_state": None
        if full_state is None
        else {
            "snr10_db": base.temporal_cross(full_state),
            "memory_bits_per_ue": full_state.get("memory_bits_per_ue"),
            "state_shape_per_ue": full_state.get("state_shape_per_ue"),
        },
        "graphs": [p.name for p in sorted(graphs.glob("*.png"))],
        "rows": rows,
    }
    (ROOT / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    base.safe_print("SUITE_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


def main():
    stage_a = []
    for pool in POOLINGS:
        for comp in COMPRESSORS:
            for d_mem in CAPS:
                key = f"factorial:{pool}:{comp}:d{d_mem}"
                stage_a.append((key, factorial_job(pool, comp, d_mem)))
    if not A.skip_full_state:
        stage_a.append(("raw_full_state", base.full_state_job))

    a_results = base.run_parallel("STAGE_A_EXHAUSTIVE", stage_a)

    results = {}
    for pool in POOLINGS:
        for comp in COMPRESSORS:
            for d_mem in CAPS:
                results[(pool, comp, d_mem)] = a_results[
                    f"factorial:{pool}:{comp}:d{d_mem}"
                ]

    winner_key = best_factorial(results)
    pool, comp, d_mem = winner_key
    base.safe_print(
        "BEST_FIXED_COMPRESSED="
        + json.dumps(
            {
                "pooling": pool,
                "compression": comp,
                "d_mem": d_mem,
                "snr10_db": base.temporal_cross(results[winner_key]),
            }
        ),
        flush=True,
    )

    stage_b = [
        (
            "winner:reorder_only",
            scheduling_job(pool, comp, d_mem, "reorder_only"),
        ),
        (
            "winner:switch_reorder",
            scheduling_job(pool, comp, d_mem, "switch_reorder"),
        ),
    ]
    if not A.skip_dynamic_retrain:
        stage_b.append(
            ("winner:dynamic_train", dynamic_winner_job(pool, comp, d_mem))
        )
    if not A.skip_extra_seeds:
        for seed in EXTRA_SEEDS:
            stage_b.append(
                (
                    f"winner:seed:{seed}",
                    seed_winner_job(pool, comp, d_mem, seed),
                )
            )

    b_results = base.run_parallel("STAGE_B_WINNER", stage_b)

    scheduling = {
        "fixed-trained / fixed": results[winner_key],
        "fixed-trained / reorder": b_results["winner:reorder_only"],
        "fixed-trained / switch+reorder": b_results["winner:switch_reorder"],
    }
    if not A.skip_dynamic_retrain:
        scheduling["dynamic-trained / switch+reorder"] = b_results[
            "winner:dynamic_train"
        ]

    seed_results = {A.seed: results[winner_key]}
    if not A.skip_extra_seeds:
        for seed in EXTRA_SEEDS:
            seed_results[seed] = b_results[f"winner:seed:{seed}"]

    full_state = None if A.skip_full_state else a_results["raw_full_state"]
    aggregate(results, full_state, winner_key, scheduling, seed_results)


if __name__ == "__main__":
    main()
