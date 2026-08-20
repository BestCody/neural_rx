#!/usr/bin/env python3
"""Final graph-merge audit: bind historical 24 rows to their evaluated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import merge_corrected_autoencoder_factorial_strict as strict

_ORIGINAL_AUDIT_HISTORICAL = strict.audit_historical


def _num(x):
    if x in (None, "", "None", "null"):
        return None
    return float(x)


def audit_historical_artifacts(base_root: Path) -> dict:
    report = _ORIGINAL_AUDIT_HISTORICAL(base_root)
    failures = []
    artifacts = []

    import csv

    with (base_root / "all_results.csv").open(newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r.get("group") == "factorial"
            and r.get("compression") in strict.BASE_COMPRESSIONS
        ]

    for row in rows:
        pool, comp, d_mem = strict.key(row)
        seed = int(float(row["seed"]))
        train_dir = (
            base_root / "trained" / "fixed" / f"seed_{seed}"
            / f"{pool}_{comp}_d{d_mem}"
        )
        ckpt = train_dir / f"ue_memory_{pool}_{comp}_idaware_d{d_mem}_k2.weights.h5"
        train_summary = train_dir / "training_summary.json"
        eval_dir = (
            base_root / "evaluations" / "fixed" / f"seed_{seed}"
            / f"factorial_{pool}_{comp}_d{d_mem}"
        )
        evaluation_path = eval_dir / "evaluation.json"
        k = (pool, comp, d_mem)

        if not ckpt.is_file():
            failures.append(f"{k}:checkpoint_missing")
        if not train_summary.is_file():
            failures.append(f"{k}:training_summary_missing")
        if not evaluation_path.is_file():
            failures.append(f"{k}:evaluation_missing")
            continue

        evaluation = json.loads(evaluation_path.read_text())
        if evaluation.get("experiment") != "temporal_ue_memory_132prb_evaluation_v2":
            failures.append(f"{k}:experiment")
        if int(evaluation.get("n_size_bwp", -1)) != 132:
            failures.append(f"{k}:n_size_bwp")
        if evaluation.get("parameter_mode") != "training=False":
            failures.append(f"{k}:parameter_mode")
        if evaluation.get("pooling") != pool:
            failures.append(f"{k}:pooling")
        if evaluation.get("compression") != comp:
            failures.append(f"{k}:compression")
        if int(evaluation.get("d_mem", -1)) != d_mem:
            failures.append(f"{k}:d_mem")
        if int(evaluation.get("seed", -1)) != seed:
            failures.append(f"{k}:evaluation_seed")
        if bool(evaluation.get("dynamic_scheduling")):
            failures.append(f"{k}:dynamic_scheduling")

        if ckpt.is_file():
            recorded = Path(evaluation.get("checkpoint", "")).expanduser().resolve()
            if recorded != ckpt.resolve():
                failures.append(f"{k}:evaluation_checkpoint_path")

        crossing = evaluation.get("snr_db_at_10pct_tbler") or {}
        pairs = (
            ("temporal_snr10", crossing.get("temporal_k2")),
            ("cold_k2_snr10", crossing.get("cold_k2")),
            ("cold_k8_snr10", crossing.get("cold_k8")),
            ("gap_recovered_percent", evaluation.get("gap_recovered_percent")),
        )
        for field, actual in pairs:
            expected = _num(row.get(field))
            if expected is None or actual is None:
                if expected is not None or actual is not None:
                    failures.append(f"{k}:{field}:null_mismatch")
            elif abs(float(expected) - float(actual)) > 1e-9:
                failures.append(f"{k}:{field}:evaluation_csv_mismatch")

        artifacts.append(
            {
                "pooling": pool,
                "compression": comp,
                "d_mem": d_mem,
                "checkpoint": str(ckpt),
                "evaluation": str(evaluation_path),
                "evaluation_sha256": strict.sha256(evaluation_path),
            }
        )

    if failures:
        raise RuntimeError(
            "historical artifact audit failed: " + "; ".join(failures[:40])
        )
    report["artifact_bindings"] = artifacts
    report["artifact_binding_count"] = len(artifacts)
    report["evaluations_bound_to_rows"] = len(artifacts) == 24
    return report


strict.audit_historical = audit_historical_artifacts


def main():
    strict.main()


if __name__ == "__main__":
    main()
