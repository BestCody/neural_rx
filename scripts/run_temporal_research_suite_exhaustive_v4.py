#!/usr/bin/env python3
"""Active exhaustive suite with strict resumability metadata validation.

Builds on v3 (capacity-tuned learned pooling for PCA) and prevents stale or
underspecified training/evaluation artifacts from being silently reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_temporal_research_suite_exhaustive_v3 as v3suite

suite = v3suite.suite
base = v3suite.base
A = v3suite.A
ROOT = v3suite.ROOT


def strict_training_valid(out, compression, pooling, d_mem, seed, dynamic):
    out = Path(out)
    summary = out / "training_summary.json"
    ckpt = out / f"ue_memory_{pooling}_{compression}_idaware_d{d_mem}_k2.weights.h5"
    if not summary.is_file() or not ckpt.is_file():
        return False
    try:
        s = base.load_json(summary)
        return all(
            [
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
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _path_seed_and_scenario(out):
    parts = Path(out).parts
    seed = None
    scenario = None
    for part in parts:
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
        if int(s.get("n_size_bwp", -1)) != 132:
            return False
        if s.get("parameter_mode") != "training=False":
            return False
        if s.get("config") not in (None, A.config):
            return False
        if int(s.get("seq_len", -1)) != A.seq_len:
            return False
        if int(s.get("batch_size", -1)) not in {
            A.eval_batch,
            A.full_state_eval_batch if full_state else A.eval_batch,
        }:
            return False
        if int(s.get("target_errors", -1)) != A.target_errors:
            return False
        if int(s.get("max_batches", -1)) != A.max_batches:
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


# Existing train/eval functions resolve these globals when called, so replacing
# them here hardens every base-factorial and winner-dependent reuse path.
base.valid_training_dir = strict_training_valid
base.eval_valid = strict_eval_valid


def main():
    v3suite.main()
    summary_path = ROOT / "suite_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["suite"] = "temporal_ue_memory_exhaustive_factorial_v4_strict_resume"
    summary["strict_artifact_validation"] = {
        "training_seed_required": True,
        "training_config_batch_and_schedule_checked": True,
        "evaluation_seed_and_scenario_checked_from_output_path": True,
        "evaluation_132prb_required": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("STRICT_RESUME_EXHAUSTIVE_SUMMARY=" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
