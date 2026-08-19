#!/usr/bin/env python3
"""Active exhaustive suite with strict provenance and fail-fast scheduling.

Builds on v3 (capacity-tuned learned pooling for PCA), rejects stale or
underspecified artifacts, and cancels queued experiment jobs after the first
failure so a broken cell cannot waste the rest of the GPU sweep.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from temporal_eval_metrics import make_snr_grid
import run_temporal_research_suite_exhaustive_v3 as v3suite

suite = v3suite.suite
base = v3suite.base
A = v3suite.A
ROOT = v3suite.ROOT
GPUS = base.GPUS
_ORIGINAL_CAPACITY_PCA_VALID = v3suite.capacity_pca_valid


def _autoencoder_protocol_valid(summary):
    protocol = summary.get("autoencoder_protocol")
    if not isinstance(protocol, dict):
        return False
    return all(
        [
            int(protocol.get("version", -1)) == 2,
            protocol.get("bounded_tanh_bottleneck") is True,
            protocol.get("reconstruction_upstream_state_detached") is True,
            protocol.get("scale_normalized_reconstruction_aux_loss") is True,
            protocol.get("raw_reconstruction_mse_retained_for_diagnostics") is True,
        ]
    )


def strict_training_valid(out, compression, pooling, d_mem, seed, dynamic):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_{compression}_idaware_d{d_mem}_k2.weights.h5"
    if not summary.is_file() or not ckpt.is_file():
        return False
    try:
        s = base.load_json(summary)
        checks = [
            s.get("architecture") == "ue_identity_aware_temporal_memory_v4_pooling",
            s.get("config") == A.config,
            s.get("pooling") == pooling,
            s.get("compression") == compression,
            int(s.get("d_mem", -1)) == int(d_mem),
            int(s.get("num_it", -1)) == 2,
            int(s.get("train_steps", -1)) == A.train_steps,
            int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
            int(s.get("batch_size", -1)) == A.train_batch,
            int(s.get("seq_len", -1)) == A.seq_len,
            int(s.get("seed", -1)) == int(seed),
            int(s.get("ue_pool_size", -1)) == 4,
            bool(s.get("dynamic_scheduling")) == bool(dynamic),
        ]
        if compression == "autoencoder":
            checks.append(_autoencoder_protocol_valid(s))
        if dynamic:
            checks += [
                abs(float(s.get("schedule_switch_prob", -1.0)) - 0.65) < 1e-12,
                abs(float(s.get("schedule_reorder_prob", -1.0)) - 0.50) < 1e-12,
            ]
        return all(checks)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def strict_capacity_pca_valid(out, pooling, d_mem, seed, dynamic):
    if not _ORIGINAL_CAPACITY_PCA_VALID(out, pooling, d_mem, seed, dynamic):
        return False
    try:
        s = base.load_json(Path(out) / "training_summary.json")
        calibration = s.get("pooler_calibration", {})
        checks = [
            s.get("config") == A.config,
            int(s.get("batch_size", -1)) == A.train_batch,
            int(s.get("ue_pool_size", -1)) == 4,
            int(calibration.get("steps", -1)) == A.memory_only_steps,
            int(calibration.get("target_pca_d_mem", -1)) == int(d_mem),
        ]
        if dynamic:
            checks += [
                abs(float(s.get("schedule_switch_prob", -1.0)) - 0.65) < 1e-12,
                abs(float(s.get("schedule_reorder_prob", -1.0)) - 0.50) < 1e-12,
            ]
        return all(checks)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def strict_full_state_training_valid(out):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / "ue_memory_full_state_raw_k2.weights.h5"
    if not summary.is_file() or not ckpt.is_file():
        return False
    try:
        s = base.load_json(summary)
        return all(
            [
                s.get("architecture") == "ue_identity_aware_temporal_full_state_v1",
                s.get("config") == A.config,
                int(s.get("num_it", -1)) == 2,
                int(s.get("train_steps", -1)) == A.train_steps,
                int(s.get("memory_only_steps", -1)) == A.memory_only_steps,
                int(s.get("batch_size", -1)) == A.full_state_train_batch,
                int(s.get("seq_len", -1)) == A.seq_len,
                int(s.get("seed", -1)) == A.seed,
                int(s.get("ue_pool_size", -1)) == 4,
                int(s.get("memory_expiry_slots", -1)) == 8,
                bool(s.get("dynamic_scheduling")) is False,
            ]
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _path_seed_and_scenario(out):
    seed = None
    scenario = None
    for part in Path(out).parts:
        if part.startswith("seed_"):
            try:
                seed = int(part.split("_", 1)[1])
            except ValueError:
                pass
        if part in {"fixed", "reorder_only", "switch_reorder"}:
            scenario = part
    return seed, scenario


def strict_eval_valid(out, compression=None, pooling=None, d_mem=None, full_state=False):
    path = Path(out) / "evaluation.json"
    if not path.is_file():
        return False
    try:
        s = base.load_json(path)
        expected_batch = A.full_state_eval_batch if full_state else A.eval_batch
        expected_grid = make_snr_grid(A.snr_min, A.snr_max, A.snr_step)
        if int(s.get("n_size_bwp", -1)) != 132:
            return False
        if s.get("parameter_mode") != "training=False":
            return False
        if s.get("config") != A.config:
            return False
        if int(s.get("seq_len", -1)) != A.seq_len:
            return False
        if int(s.get("batch_size", -1)) != expected_batch:
            return False
        if int(s.get("target_errors", -1)) != A.target_errors:
            return False
        if int(s.get("max_batches", -1)) != A.max_batches:
            return False
        if s.get("snr_grid_db") != expected_grid:
            return False
        if not str(s.get("crossing_method", "")).startswith("log-BLER interpolation"):
            return False

        expected_seed, scenario = _path_seed_and_scenario(out)
        if expected_seed is not None and int(s.get("seed", -1)) != expected_seed:
            return False
        if scenario == "fixed" and bool(s.get("dynamic_scheduling")):
            return False
        if scenario == "reorder_only":
            if not bool(s.get("dynamic_scheduling")):
                return False
            if int(s.get("ue_pool_size", -1)) != 2:
                return False
            if float(s.get("schedule_switch_prob", -1.0)) != 0.0:
                return False
            if float(s.get("schedule_reorder_prob", -1.0)) != 1.0:
                return False
        if scenario == "switch_reorder":
            if not bool(s.get("dynamic_scheduling")):
                return False
            if int(s.get("ue_pool_size", -1)) != 4:
                return False
            if abs(float(s.get("schedule_switch_prob", -1.0)) - 0.65) > 1e-12:
                return False
            if abs(float(s.get("schedule_reorder_prob", -1.0)) - 0.50) > 1e-12:
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
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def fail_fast_parallel(stage_name, jobs):
    """One job/GPU; after first failure cancel every not-yet-started job."""
    if not jobs:
        return {}
    gpu_pool = Queue()
    for gpu in GPUS:
        gpu_pool.put(gpu)

    def wrapped(name, fn):
        gpu = gpu_pool.get()
        try:
            base.safe_print(f"{stage_name}: START {name} on GPU {gpu}", flush=True)
            value = fn(gpu)
            base.safe_print(f"{stage_name}: DONE  {name} on GPU {gpu}", flush=True)
            return name, value
        finally:
            gpu_pool.put(gpu)

    executor = ThreadPoolExecutor(max_workers=len(GPUS))
    futures = [executor.submit(wrapped, name, fn) for name, fn in jobs]
    results = {}
    try:
        for future in as_completed(futures):
            name, value = future.result()
            results[name] = value
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


# Existing functions resolve these module globals at execution time.
base.valid_training_dir = strict_training_valid
base.eval_valid = strict_eval_valid
base.full_state_training_valid = strict_full_state_training_valid
base.run_parallel = fail_fast_parallel
v3suite.capacity_pca_valid = strict_capacity_pca_valid


def main():
    v3suite.main()
    summary_path = ROOT / "suite_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["suite"] = "temporal_ue_memory_exhaustive_factorial_v4_strict_resume"
    summary["strict_artifact_validation"] = {
        "training_seed_required": True,
        "training_config_batch_and_schedule_checked": True,
        "autoencoder_protocol_v2_required": True,
        "pca_calibration_capacity_and_steps_checked": True,
        "full_state_provenance_checked": True,
        "evaluation_seed_scenario_grid_and_crossing_method_checked": True,
        "evaluation_132prb_required": True,
    }
    summary["fail_fast_parallelism"] = {
        "cancel_pending_jobs_after_first_failure": True,
        "max_simultaneous_jobs": len(GPUS),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("STRICT_RESUME_EXHAUSTIVE_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
