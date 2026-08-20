#!/usr/bin/env python3
"""Run only the corrected protocol-v2 autoencoder factorial cells.

This runner intentionally excludes writer, PCA, full-state, scheduling-robustness,
and seed-repeat jobs. It is meant to be launched only after
run_autoencoder_stability_smoke.py passes.

Use a fresh --root so pre-stabilization AE checkpoints/evaluations cannot be
mistaken for corrected results. The normal suite defaults are preserved:
6000 training steps, seed 20260816, 132-PRB evaluation, and GPUs 0,1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import run_temporal_research_suite_exhaustive_v4 as v4

base = v4.base
v3suite = v4.v3suite
A = v4.A
ROOT = v4.ROOT
POOLINGS = ("mean", "attention", "cnn")
CAPACITIES = tuple(base.CAPS)


def _require_protocol_v2(training_dir: Path) -> dict:
    summary_path = training_dir / "training_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing training summary: {summary_path}")
    summary = base.load_json(summary_path)
    if not v4._autoencoder_protocol_valid(summary):
        raise RuntimeError(
            "autoencoder training artifact is not stabilized protocol v2: "
            f"{training_dir}"
        )
    return summary


def _repair_missing_seed_metadata() -> list[str]:
    """Repair only protocol-v2 artifacts from this fresh corrected-AE run.

    The pre-repair v4 trainer used ARGS.seed during training but inherited v3's
    historical omission of that seed from training_summary.json. The strict
    resume validator correctly rejects such summaries. For already-completed
    protocol-v2 artifacts under this runner's exact seed-scoped output tree, we
    can deterministically persist the seed that the runner supplied.

    This does not make old pre-fix AE artifacts reusable: protocol-v2 and every
    other training invariant must already match before the repair is applied.
    """
    repaired = []
    for pooling in POOLINGS:
        for d_mem in CAPACITIES:
            out = (
                ROOT
                / "trained"
                / "fixed"
                / f"seed_{A.seed}"
                / f"{pooling}_autoencoder_d{d_mem}"
            )
            summary_path = out / "training_summary.json"
            ckpt = out / (
                f"ue_memory_{pooling}_autoencoder_idaware_"
                f"d{d_mem}_k2.weights.h5"
            )
            if not summary_path.is_file() or not ckpt.is_file():
                continue

            summary = base.load_json(summary_path)
            if "seed" in summary:
                continue

            checks = [
                summary.get("architecture")
                == "ue_identity_aware_temporal_memory_v4_pooling",
                summary.get("config") == A.config,
                summary.get("pooling") == pooling,
                summary.get("compression") == "autoencoder",
                int(summary.get("d_mem", -1)) == int(d_mem),
                int(summary.get("num_it", -1)) == 2,
                int(summary.get("train_steps", -1)) == A.train_steps,
                int(summary.get("memory_only_steps", -1))
                == A.memory_only_steps,
                int(summary.get("batch_size", -1)) == A.train_batch,
                int(summary.get("seq_len", -1)) == A.seq_len,
                int(summary.get("ue_pool_size", -1)) == 4,
                bool(summary.get("dynamic_scheduling")) is False,
                v4._autoencoder_protocol_valid(summary),
            ]
            if not all(checks):
                raise RuntimeError(
                    "refusing to repair seed metadata on incompatible artifact: "
                    f"{out}"
                )

            summary["seed"] = int(A.seed)
            summary["seed_provenance_repair"] = {
                "reason": (
                    "v4 protocol-v2 trainer used the runner seed but omitted "
                    "the seed field from training_summary.json"
                ),
                "source": "corrected_autoencoder_factorial_resume",
                "seed_scoped_output_directory": f"seed_{A.seed}",
                "protocol_version_required": 2,
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            repaired.append(str(out))
            print(
                "REPAIRED_PROTOCOL_V2_SEED_METADATA=" + str(out),
                flush=True,
            )

    return repaired


def _job(pooling: str, d_mem: int):
    def run(gpu: int):
        ckpt = v3suite.train_factorial(
            gpu,
            "autoencoder",
            pooling,
            d_mem,
            A.seed,
            dynamic=False,
        )
        training_dir = Path(ckpt).parent
        train_summary = _require_protocol_v2(training_dir)

        evaluation = base.eval_compressed(
            gpu,
            ckpt,
            "autoencoder",
            pooling,
            d_mem,
            A.seed,
            "fixed",
            tag=f"factorial_{pooling}_autoencoder_d{d_mem}",
        )
        return {
            "pooling": pooling,
            "d_mem": int(d_mem),
            "checkpoint": str(ckpt),
            "training_dir": str(training_dir),
            "autoencoder_protocol": train_summary.get("autoencoder_protocol"),
            "evaluation": evaluation,
        }

    return run


def _row(item: dict) -> dict:
    evaluation = item["evaluation"]
    crossings = evaluation.get("snr_db_at_10pct_tbler", {})
    temporal = base.temporal_cross(evaluation)
    return {
        "pooling": item["pooling"],
        "compression": "autoencoder",
        "d_mem": item["d_mem"],
        "seed": A.seed,
        "temporal_snr10": temporal,
        "cold_k2_snr10": crossings.get("cold_k2"),
        "cold_k8_snr10": crossings.get("cold_k8"),
        "gap_recovered_percent": evaluation.get("gap_recovered_percent"),
        "memory_bits_per_ue": evaluation.get("memory_bits_per_ue"),
        "checkpoint": item["checkpoint"],
        "training_dir": item["training_dir"],
    }


def main():
    if A.train_steps != 6000:
        raise SystemExit(
            f"corrected AE factorial requires 6000 steps; got {A.train_steps}"
        )
    if A.seed != 20260816:
        raise SystemExit(
            f"corrected AE factorial requires seed 20260816; got {A.seed}"
        )
    if len(CAPACITIES) != 4 or set(CAPACITIES) != {8, 16, 32, 56}:
        raise SystemExit(
            "corrected AE factorial requires capacities 8,16,32,56; "
            f"got {CAPACITIES}"
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    repaired_seed_metadata = _repair_missing_seed_metadata()

    jobs = []
    for pooling in POOLINGS:
        for d_mem in CAPACITIES:
            name = f"{pooling}_autoencoder_d{d_mem}"
            jobs.append((name, _job(pooling, d_mem)))

    raw_results = v4.fail_fast_parallel("corrected-ae-factorial", jobs)
    items = [raw_results[name] for name, _ in jobs]
    rows = [_row(item) for item in items]
    rows.sort(key=lambda r: (r["pooling"], int(r["d_mem"])))

    csv_path = ROOT / "corrected_autoencoder_results.csv"
    fields = [
        "pooling",
        "compression",
        "d_mem",
        "seed",
        "temporal_snr10",
        "cold_k2_snr10",
        "cold_k8_snr10",
        "gap_recovered_percent",
        "memory_bits_per_ue",
        "checkpoint",
        "training_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    finite = [r for r in rows if r["temporal_snr10"] is not None]
    best = None
    if finite:
        best = min(
            finite,
            key=lambda r: (
                float(r["temporal_snr10"]),
                r["pooling"],
                int(r["d_mem"]),
            ),
        )

    summary = {
        "suite": "corrected_autoencoder_factorial_protocol_v2",
        "purpose": "corrected_ae_only_research_rerun",
        "root": str(ROOT),
        "config": A.config,
        "gpus": list(base.GPUS),
        "poolings": list(POOLINGS),
        "capacities": list(CAPACITIES),
        "compression": "autoencoder",
        "seed": A.seed,
        "train_steps": A.train_steps,
        "memory_only_steps": A.memory_only_steps,
        "repaired_seed_metadata": repaired_seed_metadata,
        "evaluation": {
            "n_size_bwp": 132,
            "snr_min": A.snr_min,
            "snr_max": A.snr_max,
            "snr_step": A.snr_step,
            "target_errors": A.target_errors,
            "max_batches": A.max_batches,
        },
        "cell_count": len(rows),
        "finite_crossing_count": len(finite),
        "null_crossing_count": len(rows) - len(finite),
        "best_corrected_autoencoder": best,
        "rows": rows,
    }
    summary_path = ROOT / "corrected_autoencoder_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        "CORRECTED_AUTOENCODER_FACTORIAL_SUMMARY="
        + json.dumps(summary, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
