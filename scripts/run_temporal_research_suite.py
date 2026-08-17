#!/usr/bin/env python3
"""Run the complete temporal UE-memory research suite on both atlas A40 GPUs.

The suite is resumable and GPU-aware. Independent train->evaluate experiments
are scheduled across GPU 0 and GPU 1 concurrently. A synchronization barrier is
used only when the fixed-scheduling winner must be known before the dependent
robustness, dynamic-training, and seed-repeat experiments can be launched.

Core matrix (6000 training steps per trained configuration by default)
-----------------------------------------------------------------------
Stage A -- independent, dual-GPU:
  1. Raw full-state K=2 upper bound.
  2. mean pooling x {writer,pca,autoencoder} x d_mem {8,16,32,56}.
  3. Pooling ablation extras: attention+writer+d32, cnn+writer+d32.
     (mean+writer+d32 is already in item 2.)

Stage B -- after selecting the best compressed fixed model, dual-GPU:
  4. Winner evaluated with reorder-only scheduling.
  5. Winner evaluated with UE switching + reordering.
  6. Winner retrained for dynamic scheduling, then evaluated dynamically.
  7. Winner trained/evaluated with two additional random seeds.

Cold K=2 and cold K=8 are evaluated inside every evaluation and never trained.
All reported research curves use the 132-PRB evaluation configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"),
    )
    # --gpu is retained only so older workflow invocations keep working. The
    # suite itself defaults to both A40s through --gpus.
    p.add_argument("--gpu", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--memory-only-steps", type=int, default=1000)
    p.add_argument("--train-batch", type=int, default=8)
    p.add_argument("--full-state-train-batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--full-state-eval-batch", type=int, default=4)
    p.add_argument("--target-errors", type=int, default=120)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--snr-min", type=float, default=1.5)
    p.add_argument("--snr-max", type=float, default=3.75)
    p.add_argument("--snr-step", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--extra-seeds", default="20260817,20260818")
    p.add_argument("--capacities", default="8,16,32,56")
    p.add_argument("--skip-full-state", action="store_true")
    p.add_argument("--skip-pooling", action="store_true")
    p.add_argument("--skip-dynamic-retrain", action="store_true")
    p.add_argument("--skip-extra-seeds", action="store_true")
    return p.parse_args()


A = parse_args()
ROOT = Path(A.root).expanduser().resolve()
ROOT.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR = Path(__file__).resolve().parent
PY = sys.executable
CAPS = [int(x) for x in A.capacities.split(",") if x.strip()]
EXTRA_SEEDS = [int(x) for x in A.extra_seeds.split(",") if x.strip()]
GPUS = [int(x) for x in A.gpus.split(",") if x.strip()]
COMPRESSORS = ["writer", "pca", "autoencoder"]
PRINT_LOCK = threading.Lock()

if not GPUS:
    raise ValueError("--gpus must contain at least one GPU index")
if len(set(GPUS)) != len(GPUS):
    raise ValueError("--gpus contains duplicate GPU indices")
if any(x <= 0 for x in CAPS):
    raise ValueError("capacities must be positive")
if max(CAPS) > 56:
    raise ValueError("Fair PCA comparison cannot exceed nrx_large d_s=56")
if A.train_steps != 6000:
    print(f"WARNING: requested train_steps={A.train_steps}; project default is 6000", flush=True)

print(f"RESEARCH_SUITE_GPUS={GPUS}", flush=True)
if A.gpu is not None:
    print(
        f"LEGACY_GPU_ARGUMENT={A.gpu} (ignored for suite scheduling; using --gpus={A.gpus})",
        flush=True,
    )


def load_json(path):
    return json.loads(Path(path).read_text())


def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)


def tee_run(cmd, log_path, gpu, label):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[GPU {gpu} | {label}]"
    safe_print(prefix, "RUN:", " ".join(map(str, cmd)), flush=True)
    with log_path.open("w", buffering=1) as f:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            # Keep the master Actions log readable even with two subprocesses.
            with PRINT_LOCK:
                print(prefix, line, end="", flush=True)
            f.write(line)
        rc = p.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def run_parallel(stage_name, jobs):
    """Run named callables across the configured GPUs, one job per GPU."""
    if not jobs:
        return {}
    gpu_pool = Queue()
    for gpu in GPUS:
        gpu_pool.put(gpu)

    def wrapped(name, fn):
        gpu = gpu_pool.get()
        try:
            safe_print(f"{stage_name}: START {name} on GPU {gpu}", flush=True)
            value = fn(gpu)
            safe_print(f"{stage_name}: DONE  {name} on GPU {gpu}", flush=True)
            return name, value
        finally:
            gpu_pool.put(gpu)

    results = {}
    with ThreadPoolExecutor(max_workers=len(GPUS)) as ex:
        futures = [ex.submit(wrapped, name, fn) for name, fn in jobs]
        for future in as_completed(futures):
            name, value = future.result()  # propagate failures immediately
            results[name] = value
    return results


def valid_training_dir(out, compression, pooling, d_mem, seed, dynamic):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_{compression}_idaware_d{d_mem}_k2.weights.h5"
    if not summary.exists() or not ckpt.exists():
        return False
    try:
        s = load_json(summary)
        return all(
            [
                s.get("architecture") == "ue_identity_aware_temporal_memory_v4_pooling",
                s.get("pooling") == pooling,
                s.get("compression") == compression,
                int(s.get("d_mem", -1)) == d_mem,
                int(s.get("num_it", -1)) == 2,
                int(s.get("train_steps", -1)) == A.train_steps,
                int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
                int(s.get("seq_len", -1)) == A.seq_len,
                int(s.get("seed", seed)) == seed,
                bool(s.get("dynamic_scheduling")) == bool(dynamic),
            ]
        )
    except Exception:
        return False


def legacy_checkpoint(compression, pooling, d_mem, seed, dynamic):
    """Reuse the already-running first writer/d32 experiment when compatible."""
    if dynamic or compression != "writer" or pooling != "mean" or d_mem != 32 or seed != A.seed:
        return None
    legacy = Path.home() / "sionna-srsran" / "temporal_reuse" / "ue_memory" / "mean" / "writer"
    if valid_training_dir(legacy, compression, pooling, d_mem, seed, dynamic=False):
        ckpt = legacy / "ue_memory_mean_writer_idaware_d32_k2.weights.h5"
        safe_print("REUSE_LEGACY_6000_STEP_CHECKPOINT", ckpt, flush=True)
        return ckpt
    return None


def train_compressed(gpu, compression, pooling, d_mem, seed, dynamic=False):
    mode = "dynamic" if dynamic else "fixed"
    out = ROOT / "trained" / mode / f"seed_{seed}" / f"{pooling}_{compression}_d{d_mem}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"ue_memory_{pooling}_{compression}_idaware_d{d_mem}_k2.weights.h5"
    if valid_training_dir(out, compression, pooling, d_mem, seed, dynamic):
        safe_print("REUSE_TRAINING", out, flush=True)
        return ckpt

    old = legacy_checkpoint(compression, pooling, d_mem, seed, dynamic)
    if old is not None:
        return old

    label = f"train-{mode}-{pooling}-{compression}-d{d_mem}-seed{seed}"
    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_ue_memory_v4.py",
        "--gpu",
        gpu,
        "--pooling",
        pooling,
        "--compression",
        compression,
        "--d-mem",
        d_mem,
        "--num-it",
        2,
        "--train-steps",
        A.train_steps,
        "--memory-only-steps",
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
    ]
    if not dynamic:
        cmd.append("--fixed-scheduling")
    tee_run(cmd, out / "train.log", gpu, label)
    if not valid_training_dir(out, compression, pooling, d_mem, seed, dynamic):
        raise RuntimeError(f"training output failed metadata validation: {out}")
    return ckpt


def eval_valid(out, compression=None, pooling=None, d_mem=None, full_state=False):
    p = Path(out) / "evaluation.json"
    if not p.exists():
        return False
    try:
        s = load_json(p)
        if int(s.get("n_size_bwp", -1)) != 132:
            return False
        if s.get("parameter_mode") != "training=False":
            return False
        if full_state:
            return s.get("experiment") == "temporal_raw_full_state_132prb_evaluation_v1"
        return all(
            [
                s.get("experiment") == "temporal_ue_memory_132prb_evaluation_v2",
                s.get("compression") == compression,
                s.get("pooling") == pooling,
                int(s.get("d_mem", -1)) == int(d_mem),
            ]
        )
    except Exception:
        return False


def eval_compressed(
    gpu,
    ckpt,
    compression,
    pooling,
    d_mem,
    seed,
    scenario="fixed",
    tag=None,
):
    tag = tag or f"{pooling}_{compression}_d{d_mem}"
    out = ROOT / "evaluations" / scenario / f"seed_{seed}" / tag
    out.mkdir(parents=True, exist_ok=True)
    if eval_valid(out, compression, pooling, d_mem):
        safe_print("REUSE_EVALUATION", out, flush=True)
        return load_json(out / "evaluation.json")

    label = f"eval-{scenario}-{tag}-seed{seed}"
    cmd = [
        PY,
        SCRIPT_DIR / "evaluate_temporal_ue_memory_v2.py",
        "--checkpoint",
        ckpt,
        "--config",
        A.config,
        "--gpu",
        gpu,
        "--compression",
        compression,
        "--pooling",
        pooling,
        "--d-mem",
        d_mem,
        "--num-it",
        2,
        "--seq-len",
        A.seq_len,
        "--batch-size",
        A.eval_batch,
        "--snr-min",
        A.snr_min,
        "--snr-max",
        A.snr_max,
        "--snr-step",
        A.snr_step,
        "--target-errors",
        A.target_errors,
        "--max-batches",
        A.max_batches,
        "--seed",
        seed,
        "--output-dir",
        out,
    ]
    if scenario == "fixed":
        cmd += ["--ue-pool-size", 4]
    elif scenario == "reorder_only":
        cmd += [
            "--dynamic-scheduling",
            "--ue-pool-size",
            2,
            "--schedule-switch-prob",
            0.0,
            "--schedule-reorder-prob",
            1.0,
        ]
    elif scenario == "switch_reorder":
        cmd += [
            "--dynamic-scheduling",
            "--ue-pool-size",
            4,
            "--schedule-switch-prob",
            0.65,
            "--schedule-reorder-prob",
            0.50,
        ]
    else:
        raise ValueError(scenario)

    tee_run(cmd, out / "eval.log", gpu, label)
    if not eval_valid(out, compression, pooling, d_mem):
        raise RuntimeError(f"evaluation output failed 132-PRB validation: {out}")
    return load_json(out / "evaluation.json")


def full_state_training_valid(out):
    out = Path(out)
    p = out / "training_summary.json"
    ckpt = out / "ue_memory_full_state_raw_k2.weights.h5"
    if not p.exists() or not ckpt.exists():
        return False
    try:
        s = load_json(p)
        return all(
            [
                s.get("architecture") == "ue_identity_aware_temporal_full_state_v1",
                int(s.get("train_steps", -1)) == A.train_steps,
                int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
                int(s.get("seed", A.seed)) == A.seed,
                bool(s.get("dynamic_scheduling")) is False,
            ]
        )
    except Exception:
        return False


def train_full_state(gpu):
    out = ROOT / "trained" / "fixed" / f"seed_{A.seed}" / "raw_full_state"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ue_memory_full_state_raw_k2.weights.h5"
    if full_state_training_valid(out):
        safe_print("REUSE_FULL_STATE_TRAINING", out, flush=True)
        return ckpt
    cmd = [
        PY,
        SCRIPT_DIR / "train_temporal_full_state.py",
        "--config",
        A.config,
        "--gpu",
        gpu,
        "--num-it",
        2,
        "--train-steps",
        A.train_steps,
        "--memory-only-steps",
        A.memory_only_steps,
        "--batch-size",
        A.full_state_train_batch,
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
        "--fixed-scheduling",
        "--seed",
        A.seed,
        "--output-dir",
        out,
        "--log-every",
        25,
    ]
    tee_run(cmd, out / "train.log", gpu, "train-raw-full-state")
    if not full_state_training_valid(out):
        raise RuntimeError("invalid full-state training output")
    return ckpt


def eval_full_state(gpu, ckpt):
    out = ROOT / "evaluations" / "fixed" / f"seed_{A.seed}" / "raw_full_state"
    out.mkdir(parents=True, exist_ok=True)
    if eval_valid(out, full_state=True):
        safe_print("REUSE_FULL_STATE_EVALUATION", out, flush=True)
        return load_json(out / "evaluation.json")
    cmd = [
        PY,
        SCRIPT_DIR / "evaluate_temporal_full_state.py",
        "--checkpoint",
        ckpt,
        "--config",
        A.config,
        "--gpu",
        gpu,
        "--seq-len",
        A.seq_len,
        "--batch-size",
        A.full_state_eval_batch,
        "--snr-min",
        A.snr_min,
        "--snr-max",
        A.snr_max,
        "--snr-step",
        A.snr_step,
        "--target-errors",
        A.target_errors,
        "--max-batches",
        A.max_batches,
        "--seed",
        A.seed,
        "--ue-pool-size",
        4,
        "--output-dir",
        out,
    ]
    tee_run(cmd, out / "eval.log", gpu, "eval-raw-full-state")
    if not eval_valid(out, full_state=True):
        raise RuntimeError("invalid full-state 132-PRB evaluation")
    return load_json(out / "evaluation.json")


def temporal_cross(s):
    c = s.get("snr_db_at_10pct_tbler", {})
    return c.get("temporal_k2", c.get("temporal_k2_full_state"))


def best_config(results):
    choices = []
    for key, s in results.items():
        x = temporal_cross(s)
        if x is not None:
            choices.append((float(x), key))
    if not choices:
        raise RuntimeError("No compressed temporal curve bracketed 10% TBLER")
    choices.sort()
    return choices[0][1]


def compressed_job(comp, d):
    def job(gpu):
        ckpt = train_compressed(gpu, comp, "mean", d, A.seed, dynamic=False)
        return eval_compressed(gpu, ckpt, comp, "mean", d, A.seed, "fixed")

    return job


def pooling_job(pool):
    def job(gpu):
        ckpt = train_compressed(gpu, "writer", pool, 32, A.seed, dynamic=False)
        return eval_compressed(
            gpu,
            ckpt,
            "writer",
            pool,
            32,
            A.seed,
            "fixed",
            tag=f"pooling_{pool}_writer_d32",
        )

    return job


def full_state_job(gpu):
    return eval_full_state(gpu, train_full_state(gpu))


def winner_scheduling_job(comp, d, scenario):
    def job(gpu):
        fixed_ckpt = (
            ROOT
            / "trained"
            / "fixed"
            / f"seed_{A.seed}"
            / f"mean_{comp}_d{d}"
            / f"ue_memory_mean_{comp}_idaware_d{d}_k2.weights.h5"
        )
        if not fixed_ckpt.exists():
            legacy = legacy_checkpoint(comp, "mean", d, A.seed, dynamic=False)
            if legacy is None:
                raise FileNotFoundError(fixed_ckpt)
            fixed_ckpt = legacy
        return eval_compressed(
            gpu,
            fixed_ckpt,
            comp,
            "mean",
            d,
            A.seed,
            scenario,
            tag=f"winner_{comp}_d{d}",
        )

    return job


def dynamic_winner_job(comp, d):
    def job(gpu):
        ckpt = train_compressed(gpu, comp, "mean", d, A.seed, dynamic=True)
        return eval_compressed(
            gpu,
            ckpt,
            comp,
            "mean",
            d,
            A.seed,
            "switch_reorder",
            tag=f"dynamic_trained_winner_{comp}_d{d}",
        )

    return job


def seed_winner_job(comp, d, seed):
    def job(gpu):
        ckpt = train_compressed(gpu, comp, "mean", d, seed, dynamic=False)
        return eval_compressed(
            gpu,
            ckpt,
            comp,
            "mean",
            d,
            seed,
            "fixed",
            tag=f"winner_{comp}_d{d}",
        )

    return job


def aggregate(compressed, full_state, winner_key, scheduling, pooling, seeds):
    rows = []
    for (comp, d), s in compressed.items():
        rows.append(
            {
                "group": "capacity",
                "compression": comp,
                "pooling": "mean",
                "d_mem": d,
                "scenario": "fixed",
                "seed": A.seed,
                "temporal_snr10": temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )
    if full_state:
        rows.append(
            {
                "group": "full_state",
                "compression": "raw_full_state",
                "pooling": "none",
                "d_mem": None,
                "scenario": "fixed",
                "seed": A.seed,
                "temporal_snr10": temporal_cross(full_state),
                "cold_k2_snr10": full_state["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": full_state["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": full_state.get("gap_recovered_percent"),
                "memory_bits_per_ue": full_state.get("memory_bits_per_ue"),
            }
        )
    for name, s in scheduling.items():
        rows.append(
            {
                "group": "scheduling",
                "compression": winner_key[0],
                "pooling": "mean",
                "d_mem": winner_key[1],
                "scenario": name,
                "seed": A.seed,
                "temporal_snr10": temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )
    for name, s in pooling.items():
        rows.append(
            {
                "group": "pooling",
                "compression": "writer",
                "pooling": name,
                "d_mem": 32,
                "scenario": "fixed",
                "seed": A.seed,
                "temporal_snr10": temporal_cross(s),
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
                "compression": winner_key[0],
                "pooling": "mean",
                "d_mem": winner_key[1],
                "scenario": "fixed",
                "seed": seed,
                "temporal_snr10": temporal_cross(s),
                "cold_k2_snr10": s["snr_db_at_10pct_tbler"].get("cold_k2"),
                "cold_k8_snr10": s["snr_db_at_10pct_tbler"].get("cold_k8"),
                "gap_recovered_percent": s.get("gap_recovered_percent"),
                "memory_bits_per_ue": s.get("memory_bits_per_ue"),
            }
        )

    csv_path = ROOT / "all_results.csv"
    fields = [
        "group",
        "compression",
        "pooling",
        "d_mem",
        "scenario",
        "seed",
        "temporal_snr10",
        "cold_k2_snr10",
        "cold_k8_snr10",
        "gap_recovered_percent",
        "memory_bits_per_ue",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    graphs = ROOT / "graphs"
    graphs.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for comp in COMPRESSORS:
        xs, ys = [], []
        for d in CAPS:
            y = temporal_cross(compressed[(comp, d)])
            if y is not None:
                xs.append(d)
                ys.append(y)
        ax.plot(xs, ys, marker="o", label=comp)
    first = next(iter(compressed.values()))
    c2 = first["snr_db_at_10pct_tbler"].get("cold_k2")
    c8 = first["snr_db_at_10pct_tbler"].get("cold_k8")
    if c2 is not None:
        ax.axhline(c2, linestyle="--", label="cold K=2")
    if c8 is not None:
        ax.axhline(c8, linestyle=":", label="cold K=8")
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
    ax.set_title("Compression and memory-capacity sweep (132 PRBs)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(graphs / "compression_capacity_snr10.png", dpi=180)
    fig.savefig(graphs / "compression_capacity_snr10.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for comp in COMPRESSORS:
        xs = CAPS
        ys = [compressed[(comp, d)].get("gap_recovered_percent") for d in CAPS]
        ax.plot(xs, ys, marker="o", label=comp)
    ax.axhline(0, linewidth=1)
    ax.axhline(100, linestyle="--", linewidth=1)
    ax.set_xlabel("Persistent memory floats / UE")
    ax.set_ylabel("K2->K8 gap recovered (%)")
    ax.set_title("Temporal gain vs memory capacity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(graphs / "gap_recovered_vs_capacity.png", dpi=180)
    fig.savefig(graphs / "gap_recovered_vs_capacity.pdf")
    plt.close(fig)

    if full_state:
        winner = compressed[winner_key]
        labels = [
            "Cold K=2",
            f"Best compressed\n{winner_key[0]} d={winner_key[1]}",
            "Raw full state",
            "Cold K=8",
        ]
        vals = [
            winner["snr_db_at_10pct_tbler"].get("cold_k2"),
            temporal_cross(winner),
            temporal_cross(full_state),
            winner["snr_db_at_10pct_tbler"].get("cold_k8"),
        ]
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.bar(labels, vals)
        ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
        ax.set_title("Full-state upper bound vs compressed temporal memory")
        fig.tight_layout()
        fig.savefig(graphs / "full_state_upper_bound.png", dpi=180)
        fig.savefig(graphs / "full_state_upper_bound.pdf")
        plt.close(fig)

    if scheduling:
        labels = list(scheduling)
        vals = [temporal_cross(scheduling[x]) for x in labels]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, vals)
        ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
        ax.set_title(f"Scheduling robustness: {winner_key[0]} d={winner_key[1]}")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(graphs / "scheduling_robustness.png", dpi=180)
        fig.savefig(graphs / "scheduling_robustness.pdf")
        plt.close(fig)

    if pooling:
        labels = list(pooling)
        vals = [temporal_cross(pooling[x]) for x in labels]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, vals)
        ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
        ax.set_title("Pooling ablation: writer, d_mem=32")
        fig.tight_layout()
        fig.savefig(graphs / "pooling_ablation.png", dpi=180)
        fig.savefig(graphs / "pooling_ablation.pdf")
        plt.close(fig)

    if seeds:
        labels = [str(x) for x in seeds]
        vals = [temporal_cross(seeds[x]) for x in seeds]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, vals)
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Eb/N0 at 10% TB2+ TBLER (dB)")
        ax.set_title(f"Winner seed stability: {winner_key[0]} d={winner_key[1]}")
        fig.tight_layout()
        fig.savefig(graphs / "winner_seed_stability.png", dpi=180)
        fig.savefig(graphs / "winner_seed_stability.pdf")
        plt.close(fig)

    summary = {
        "suite": "temporal_ue_memory_complete_research_suite_v2_dual_gpu",
        "config": A.config,
        "gpus": GPUS,
        "parallel_gpu_count": len(GPUS),
        "train_steps_per_trained_configuration": A.train_steps,
        "capacities": CAPS,
        "compressors": COMPRESSORS,
        "winner": {
            "compression": winner_key[0],
            "d_mem": winner_key[1],
            "fixed_seed": A.seed,
            "snr10_db": temporal_cross(compressed[winner_key]),
            "gap_recovered_percent": compressed[winner_key].get("gap_recovered_percent"),
        },
        "full_state": None
        if not full_state
        else {
            "snr10_db": temporal_cross(full_state),
            "memory_bits_per_ue": full_state.get("memory_bits_per_ue"),
            "state_shape_per_ue": full_state.get("state_shape_per_ue"),
        },
        "graphs": [p.name for p in sorted(graphs.glob("*.png"))],
        "rows": rows,
    }
    (ROOT / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    safe_print("SUITE_SUMMARY=" + json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    # Stage A: everything whose definition does not depend on knowing the winner.
    stage_a = []
    for comp in COMPRESSORS:
        for d in CAPS:
            stage_a.append((f"compressed:{comp}:d{d}", compressed_job(comp, d)))
    if not A.skip_full_state:
        stage_a.append(("raw_full_state", full_state_job))
    if not A.skip_pooling:
        stage_a.append(("pooling:attention", pooling_job("attention")))
        stage_a.append(("pooling:cnn", pooling_job("cnn")))

    a_results = run_parallel("STAGE_A", stage_a)

    compressed = {}
    for comp in COMPRESSORS:
        for d in CAPS:
            compressed[(comp, d)] = a_results[f"compressed:{comp}:d{d}"]

    winner_key = best_config(compressed)
    comp, d = winner_key
    safe_print(
        "BEST_FIXED_COMPRESSED="
        + json.dumps(
            {
                "compression": comp,
                "d_mem": d,
                "snr10_db": temporal_cross(compressed[winner_key]),
            }
        ),
        flush=True,
    )

    full_state = None if A.skip_full_state else a_results["raw_full_state"]
    pooling = {}
    if not A.skip_pooling:
        # Mean/writer/d32 is already one of the core compressed runs.
        pooling["mean"] = compressed[("writer", 32)]
        pooling["attention"] = a_results["pooling:attention"]
        pooling["cnn"] = a_results["pooling:cnn"]

    # Stage B: winner-dependent work. These are again independent of each other
    # once winner_key is known, so both GPUs stay occupied.
    stage_b = [
        ("winner:reorder_only", winner_scheduling_job(comp, d, "reorder_only")),
        ("winner:switch_reorder", winner_scheduling_job(comp, d, "switch_reorder")),
    ]
    if not A.skip_dynamic_retrain:
        stage_b.append(("winner:dynamic_train", dynamic_winner_job(comp, d)))
    if not A.skip_extra_seeds:
        for seed in EXTRA_SEEDS:
            stage_b.append((f"winner:seed:{seed}", seed_winner_job(comp, d, seed)))

    b_results = run_parallel("STAGE_B", stage_b)

    scheduling = {
        "fixed-trained / fixed": compressed[winner_key],
        "fixed-trained / reorder": b_results["winner:reorder_only"],
        "fixed-trained / switch+reorder": b_results["winner:switch_reorder"],
    }
    if not A.skip_dynamic_retrain:
        scheduling["dynamic-trained / switch+reorder"] = b_results["winner:dynamic_train"]

    seed_results = {A.seed: compressed[winner_key]}
    if not A.skip_extra_seeds:
        for seed in EXTRA_SEEDS:
            seed_results[seed] = b_results[f"winner:seed:{seed}"]

    aggregate(compressed, full_state, winner_key, scheduling, pooling, seed_results)


if __name__ == "__main__":
    main()
