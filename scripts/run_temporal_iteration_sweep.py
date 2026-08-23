#!/usr/bin/env python3
"""Run the approved temporal-memory iteration sweep on two GPUs.

Experiment
----------
Finalists:
  1. mean + PCA, d_mem=56
  2. CNN + PCA, d_mem=16
  3. CNN + autoencoder, d_mem=56
  4. mean + writer, d_mem=32

Training:
  reuse the existing K=2 checkpoint for every finalist;
  train fresh K=3..8 checkpoints with the same protocol used by the screening.

Evaluation:
  train K=2 -> evaluate K=2..8
  train K=3 -> evaluate K=3..8
  ...
  train K=8 -> evaluate K=8

Cold NRX is never run by this script. Existing cold K=1..8 results are meant
for later overlay/analysis.

The runner is resumable and launches one process per GPU in fail-fast waves.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FINALISTS = (
    {
        "slug": "mean_pca_d56",
        "label": "Mean + PCA d56",
        "pooling": "mean",
        "compression": "pca",
        "d_mem": 56,
    },
    {
        "slug": "cnn_pca_d16",
        "label": "CNN + PCA d16",
        "pooling": "cnn",
        "compression": "pca",
        "d_mem": 16,
    },
    {
        "slug": "cnn_autoencoder_d56",
        "label": "CNN + AE d56",
        "pooling": "cnn",
        "compression": "autoencoder",
        "d_mem": 56,
    },
    {
        "slug": "mean_writer_d32",
        "label": "Best Writer (Mean d32)",
        "pooling": "mean",
        "compression": "writer",
        "d_mem": 32,
    },
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=(
            "/home/h3lou/sionna-srsran/temporal_reuse/"
            "iteration_sweep_v1"
        ),
    )
    p.add_argument(
        "--k2-root",
        default=(
            "/home/h3lou/sionna-srsran/temporal_reuse/"
            "research_suite/autoencoder_v2"
        ),
        help="Existing finalist K=2 artifacts. K=2 is never retrained.",
    )
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--python", default=sys.executable)
    p.add_argument(
        "--phase",
        choices=["all", "train", "eval", "plot"],
        default="all",
    )
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--seed", type=int, default=20260816)

    # Training protocol: intentionally matches the 36-cell screening.
    p.add_argument("--train-steps", type=int, default=6000)
    p.add_argument("--memory-only-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--min-ebno-db", type=float, default=1.0)
    p.add_argument("--max-ebno-db", type=float, default=5.0)
    p.add_argument("--memory-lr", type=float, default=1e-3)
    p.add_argument("--joint-lr", type=float, default=2e-5)
    p.add_argument("--chest-weight", type=float, default=0.01)
    p.add_argument("--ae-reconstruction-weight", type=float, default=0.1)
    p.add_argument("--pca-fit-batches", type=int, default=16)
    p.add_argument("--pca-fit-batch-size", type=int, default=8)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--memory-expiry-slots", type=int, default=8)
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)

    # Evaluation protocol.
    p.add_argument("--snr-min", type=float, default=1.5)
    p.add_argument("--snr-max", type=float, default=3.75)
    p.add_argument("--snr-step", type=float, default=0.25)
    p.add_argument("--batches-per-snr", type=int, default=32)
    p.add_argument("--min-errors-warning", type=int, default=120)

    p.add_argument(
        "--cold-csv",
        default="/home/h3lou/sionna-srsran/nrx_iter_study/measurements.csv",
        help="Existing cold results for plotting only. Never executed here.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


A = parse_args()
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(A.root).expanduser().resolve()
K2_ROOT = Path(A.k2_root).expanduser().resolve()
GPUS = [int(x) for x in A.gpus.split(",") if x.strip()]
if not GPUS:
    raise ValueError("--gpus must contain at least one GPU id")
if A.seq_len < 2:
    raise ValueError("--seq-len must be >= 2")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def checkpoint_name(f, k: int) -> str:
    return (
        f"ue_memory_{f['pooling']}_{f['compression']}_idaware_"
        f"d{f['d_mem']}_k{k}.weights.h5"
    )


def training_dir(f, k: int) -> Path:
    return ROOT / "trained" / f["slug"] / f"train_k{k}"


def evaluation_dir(f, train_k: int, eval_k: int) -> Path:
    return (
        ROOT
        / "evaluations"
        / f["slug"]
        / f"train_k{train_k}"
        / f"eval_k{eval_k}"
    )


def preferred_k2_checkpoint(f) -> Path:
    return (
        K2_ROOT
        / "trained"
        / "fixed"
        / f"seed_{A.seed}"
        / f"{f['pooling']}_{f['compression']}_d{f['d_mem']}"
        / checkpoint_name(f, 2)
    )


def find_k2_checkpoint(f) -> Path:
    preferred = preferred_k2_checkpoint(f)
    if preferred.is_file():
        return preferred.resolve()

    if not K2_ROOT.exists():
        raise FileNotFoundError(
            f"K=2 artifact root does not exist: {K2_ROOT}. "
            "This runner will not retrain K=2."
        )

    matches = list(K2_ROOT.rglob(checkpoint_name(f, 2)))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Existing K=2 checkpoint not found for {f['label']}. "
            f"Expected {checkpoint_name(f, 2)} under {K2_ROOT}. "
            "K=2 will not be retrained automatically."
        )
    raise RuntimeError(
        f"Ambiguous K=2 checkpoint for {f['label']}: "
        + ", ".join(str(x) for x in matches)
    )


def checkpoint_for(f, k: int) -> Path:
    if k == 2:
        return find_k2_checkpoint(f)
    return training_dir(f, k) / checkpoint_name(f, k)


def training_valid(f, k: int) -> bool:
    if k == 2:
        return checkpoint_for(f, 2).is_file()
    out = training_dir(f, k)
    summary_file = out / "training_summary.json"
    ckpt = checkpoint_for(f, k)
    if not summary_file.is_file() or not ckpt.is_file():
        return False
    try:
        s = load_json(summary_file)
        return bool(
            s.get("architecture") == "ue_identity_aware_temporal_memory_v4_pooling"
            and s.get("pooling") == f["pooling"]
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
    except Exception:
        return False


def evaluation_valid(f, train_k: int, eval_k: int) -> bool:
    path = evaluation_dir(f, train_k, eval_k) / "evaluation.json"
    if not path.is_file():
        return False
    try:
        s = load_json(path)
        return bool(
            s.get("experiment") in {
                "temporal_iteration_transfer_132prb_v1",
                "temporal_iteration_transfer_132prb_v1_imported_k2",
            }
            and s.get("pooling") == f["pooling"]
            and s.get("compression") == f["compression"]
            and int(s.get("d_mem", -1)) == f["d_mem"]
            and int(s.get("train_num_it", -1)) == train_k
            and int(s.get("eval_num_it", -1)) == eval_k
            and s.get("snr_db_at_10pct_tbler") is not None
        )
    except Exception:
        return False


def tee_run(cmd, log_path: Path, gpu: int, label: str) -> None:
    cmd = [str(x) for x in cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "RUN="
        + json.dumps(
            {"label": label, "gpu": gpu, "cmd": cmd, "log": str(log_path)}
        ),
        flush=True,
    )
    if A.dry_run:
        return

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with log_path.open("w") as log:
        for line in proc.stdout:
            print(f"[{label}] {line}", end="", flush=True)
            log.write(line)
            log.flush()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def train_one(gpu: int, f, k: int):
    if k == 2:
        ckpt = checkpoint_for(f, 2)
        print(f"REUSE_EXISTING_K2={f['label']}::{ckpt}", flush=True)
        return str(ckpt)

    if training_valid(f, k):
        ckpt = checkpoint_for(f, k)
        print(f"REUSE_TRAINING={f['label']} K={k}::{ckpt}", flush=True)
        return str(ckpt)

    out = training_dir(f, k)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        A.python,
        SCRIPT_DIR / "train_temporal_ue_memory_v4.py",
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
    ]
    tee_run(cmd, out / "train.log", gpu, f"train-{f['slug']}-k{k}")
    if A.dry_run:
        return str(checkpoint_for(f, k))
    if not training_valid(f, k):
        raise RuntimeError(f"Training artifact failed validation: {f['label']} K={k}")
    return str(checkpoint_for(f, k))


def find_old_k2_evaluation(f) -> Path | None:
    preferred = (
        K2_ROOT
        / "evaluations"
        / "fixed"
        / f"seed_{A.seed}"
        / f"factorial_{f['pooling']}_{f['compression']}_d{f['d_mem']}"
        / "evaluation.json"
    )
    candidates = [preferred] if preferred.is_file() else []
    if not candidates and K2_ROOT.exists():
        candidates = list(K2_ROOT.rglob("evaluation.json"))

    valid = []
    for path in candidates:
        try:
            s = load_json(path)
            curves = s.get("curves") or {}
            cross = s.get("snr_db_at_10pct_tbler") or {}
            if (
                s.get("pooling") == f["pooling"]
                and s.get("compression") == f["compression"]
                and int(s.get("d_mem", -1)) == f["d_mem"]
                and "temporal_k2" in curves
                and cross.get("temporal_k2") is not None
            ):
                valid.append(path)
        except Exception:
            continue
    if len(valid) == 1:
        return valid[0]
    if preferred in valid:
        return preferred
    return None


def import_existing_k2_evaluation(f) -> bool:
    out = evaluation_dir(f, 2, 2)
    if evaluation_valid(f, 2, 2):
        print(f"REUSE_IMPORTED_K2_EVAL={f['label']}", flush=True)
        return True

    old_path = find_old_k2_evaluation(f)
    if old_path is None:
        return False
    old = load_json(old_path)
    curve = (old.get("curves") or {})["temporal_k2"]
    crossing = (old.get("snr_db_at_10pct_tbler") or {})["temporal_k2"]
    summary = {
        "experiment": "temporal_iteration_transfer_132prb_v1_imported_k2",
        "checkpoint": str(checkpoint_for(f, 2)),
        "config": old.get("config", A.config),
        "parameter_mode": old.get("parameter_mode", "training=False"),
        "n_size_bwp": old.get("n_size_bwp", 132),
        "pooling": f["pooling"],
        "compression": f["compression"],
        "d_mem": f["d_mem"],
        "memory_bits_per_ue": f["d_mem"] * 32,
        "train_num_it": 2,
        "eval_num_it": 2,
        "seq_len": old.get("seq_len", A.seq_len),
        "primary_metric": "TB2+ TBLER",
        "target_bler": 0.1,
        "snr_grid_db": old.get("snr_grid_db"),
        "snr_db_at_10pct_tbler": crossing,
        "crossing_method": old.get("crossing_method"),
        "curve": curve,
        "provenance": {
            "reused_existing_k2_evaluation": True,
            "source": str(old_path.resolve()),
        },
    }
    write_json(out / "evaluation.json", summary)
    print(f"IMPORTED_EXISTING_K2_EVAL={f['label']}::{old_path}", flush=True)
    return True


def eval_one(gpu: int, f, train_k: int, eval_k: int):
    if evaluation_valid(f, train_k, eval_k):
        print(f"REUSE_EVAL={f['label']} trainK={train_k} evalK={eval_k}", flush=True)
        return str(evaluation_dir(f, train_k, eval_k) / "evaluation.json")

    if train_k == 2 and eval_k == 2 and import_existing_k2_evaluation(f):
        return str(evaluation_dir(f, 2, 2) / "evaluation.json")

    ckpt = checkpoint_for(f, train_k)
    if not ckpt.is_file() and not A.dry_run:
        raise FileNotFoundError(ckpt)

    out = evaluation_dir(f, train_k, eval_k)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        A.python,
        SCRIPT_DIR / "evaluate_temporal_iteration_transfer.py",
        "--checkpoint", ckpt,
        "--train-num-it", train_k,
        "--eval-num-it", eval_k,
        "--config", A.config,
        "--gpu", gpu,
        "--pooling", f["pooling"],
        "--compression", f["compression"],
        "--d-mem", f["d_mem"],
        "--seq-len", A.seq_len,
        "--batch-size", A.batch_size,
        "--snr-min", A.snr_min,
        "--snr-max", A.snr_max,
        "--snr-step", A.snr_step,
        "--batches-per-snr", A.batches_per_snr,
        "--min-errors-warning", A.min_errors_warning,
        "--seed", A.seed,
        "--ue-pool-size", A.ue_pool_size,
        "--memory-expiry-slots", A.memory_expiry_slots,
        "--schedule-switch-prob", A.schedule_switch_prob,
        "--schedule-reorder-prob", A.schedule_reorder_prob,
        "--output-dir", out,
    ]
    tee_run(
        cmd,
        out / "eval.log",
        gpu,
        f"eval-{f['slug']}-traink{train_k}-evalk{eval_k}",
    )
    if A.dry_run:
        return str(out / "evaluation.json")
    if not evaluation_valid(f, train_k, eval_k):
        raise RuntimeError(
            f"Evaluation artifact failed validation: {f['label']} "
            f"trainK={train_k} evalK={eval_k}"
        )
    return str(out / "evaluation.json")


def run_waves(jobs, worker, phase: str):
    """Fail-fast wave scheduler: at most one active process per GPU."""
    if not jobs:
        print(f"{phase.upper()}_NOTHING_TO_DO", flush=True)
        return
    for start in range(0, len(jobs), len(GPUS)):
        wave = jobs[start : start + len(GPUS)]
        print(
            f"{phase.upper()}_WAVE="
            + json.dumps([job[0] for job in wave]),
            flush=True,
        )
        failures = []
        with ThreadPoolExecutor(max_workers=len(wave)) as ex:
            futures = []
            for gpu, job in zip(GPUS, wave):
                futures.append((job[0], ex.submit(worker, gpu, *job[1:])))
            for name, future in futures:
                try:
                    future.result()
                    print(f"{phase.upper()}_DONE={name}", flush=True)
                except Exception as exc:
                    failures.append((name, repr(exc)))
        if failures:
            raise RuntimeError(
                f"{phase} wave failed; no next wave started: " + json.dumps(failures)
            )


def run_training():
    # Verify all existing K=2 finalists first. This is a hard guard against
    # accidentally spending time retraining a baseline we already have.
    for f in FINALISTS:
        train_one(GPUS[0], f, 2)

    jobs = []
    for k in range(3, 9):
        for f in FINALISTS:
            if not training_valid(f, k):
                jobs.append((f"{f['slug']}-k{k}", f, k))
    run_waves(jobs, train_one, "train")


def run_evaluation():
    # Import the four already-computed K2/K2 temporal results when possible.
    for f in FINALISTS:
        import_existing_k2_evaluation(f)

    jobs = []
    for train_k in range(2, 9):
        for eval_k in range(train_k, 9):
            for f in FINALISTS:
                if not evaluation_valid(f, train_k, eval_k):
                    jobs.append(
                        (
                            f"{f['slug']}-traink{train_k}-evalk{eval_k}",
                            f,
                            train_k,
                            eval_k,
                        )
                    )
    run_waves(jobs, eval_one, "eval")


def run_plotting():
    cmd = [
        A.python,
        SCRIPT_DIR / "plot_temporal_iteration_sweep.py",
        "--root", ROOT,
        "--cold-csv", A.cold_csv,
    ]
    print("PLOT_CMD=" + json.dumps([str(x) for x in cmd]), flush=True)
    if not A.dry_run:
        subprocess.run([str(x) for x in cmd], cwd=str(SCRIPT_DIR), check=True)


def write_manifest():
    manifest = {
        "experiment": "temporal_iteration_sweep_v1",
        "finalists": list(FINALISTS),
        "train_k": list(range(2, 9)),
        "new_training_k": list(range(3, 9)),
        "evaluation_rule": "for each train K, evaluate every K from train K through 8",
        "cold_nrx_policy": "reuse existing cold K=1..8 results; never run cold in this sweep",
        "training": {
            "config": A.config,
            "train_steps": A.train_steps,
            "memory_only_steps": A.memory_only_steps,
            "batch_size": A.batch_size,
            "seq_len": A.seq_len,
            "min_ebno_db": A.min_ebno_db,
            "max_ebno_db": A.max_ebno_db,
            "memory_lr": A.memory_lr,
            "joint_lr": A.joint_lr,
            "chest_weight": A.chest_weight,
            "ae_reconstruction_weight": A.ae_reconstruction_weight,
            "fixed_scheduling": True,
            "seed": A.seed,
        },
        "evaluation": {
            "snr_min": A.snr_min,
            "snr_max": A.snr_max,
            "snr_step": A.snr_step,
            "batches_per_snr": A.batches_per_snr,
            "common_random_numbers": True,
            "fixed_batches_no_early_stop": True,
            "primary_metric": "SNR at 10% TB2+ TBLER",
        },
        "gpus": GPUS,
        "root": str(ROOT),
        "k2_root": str(K2_ROOT),
        "cold_csv": A.cold_csv,
    }
    write_json(ROOT / "experiment_manifest.json", manifest)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest()
    if A.phase in {"all", "train"}:
        run_training()
    if A.phase in {"all", "eval"}:
        run_evaluation()
    if A.phase in {"all", "plot"}:
        run_plotting()
    print(f"ITERATION_SWEEP_COMPLETE={ROOT}", flush=True)


if __name__ == "__main__":
    main()
