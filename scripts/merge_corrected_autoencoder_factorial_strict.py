#!/usr/bin/env python3
"""Audit provenance, then merge/graph the 24 valid writer/PCA + 12 corrected AE cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

POOLINGS = ("mean", "attention", "cnn")
CAPACITIES = (8, 16, 32, 56)
BASE_COMPRESSIONS = ("writer", "pca")
SEED = 20260816


def parse_args():
    p = argparse.ArgumentParser()
    base = Path.home() / "sionna-srsran" / "temporal_reuse" / "research_suite"
    p.add_argument("--base-root", default=str(base))
    p.add_argument("--ae-root", default=str(base / "autoencoder_v2"))
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def key(row):
    return (
        row.get("pooling"),
        row.get("compression"),
        int(float(row.get("d_mem"))),
    )


def expected_keys(compressions):
    return {
        (pool, comp, d)
        for pool in POOLINGS
        for comp in compressions
        for d in CAPACITIES
    }


def normalize_number(x):
    if x in (None, "", "None", "null"):
        return None
    return float(x)


def audit_historical(base_root: Path) -> dict:
    csv_path = base_root / "all_results.csv"
    summary_path = base_root / "suite_summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise RuntimeError("historical suite CSV/summary missing")

    with csv_path.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [
        r for r in all_rows
        if r.get("group") == "factorial"
        and r.get("compression") in BASE_COMPRESSIONS
    ]
    keys = {key(r) for r in rows}
    expected = expected_keys(BASE_COMPRESSIONS)
    failures = []
    if len(rows) != 24 or keys != expected:
        failures.append(
            f"historical matrix mismatch rows={len(rows)} missing={sorted(expected-keys)} extra={sorted(keys-expected)}"
        )

    for r in rows:
        k = key(r)
        try:
            if int(float(r.get("seed"))) != SEED:
                failures.append(f"{k}:seed")
            if (r.get("scenario") or "fixed") != "fixed":
                failures.append(f"{k}:scenario")
            expected_bits = int(k[2]) * 32
            if int(float(r.get("memory_bits_per_ue"))) != expected_bits:
                failures.append(f"{k}:memory_bits")
            c2 = normalize_number(r.get("cold_k2_snr10"))
            c8 = normalize_number(r.get("cold_k8_snr10"))
            if c2 is not None and c8 is not None and (not finite(c2) or not finite(c8) or c2 <= c8):
                failures.append(f"{k}:cold_gap")
        except Exception as exc:
            failures.append(f"{k}:parse:{type(exc).__name__}")

    summary = json.loads(summary_path.read_text())
    summary_rows = [
        r for r in (summary.get("rows") or [])
        if r.get("group") == "factorial"
        and r.get("compression") in BASE_COMPRESSIONS
    ]
    by_key = {key(r): r for r in summary_rows}
    if set(by_key) != expected:
        failures.append("suite_summary historical 24-row matrix mismatch")
    else:
        for r in rows:
            k = key(r)
            sr = by_key[k]
            for field in (
                "temporal_snr10",
                "cold_k2_snr10",
                "cold_k8_snr10",
                "gap_recovered_percent",
            ):
                a = normalize_number(r.get(field))
                b = normalize_number(sr.get(field))
                if a is None or b is None:
                    if a is not None or b is not None:
                        failures.append(f"{k}:{field}:csv_summary_null_mismatch")
                elif abs(a - b) > 1e-9:
                    failures.append(f"{k}:{field}:csv_summary_mismatch")

    if failures:
        raise RuntimeError("historical writer/PCA audit failed: " + "; ".join(failures[:30]))
    return {
        "rows": 24,
        "csv_sha256": sha256(csv_path),
        "suite_summary_sha256": sha256(summary_path),
        "suite_name": summary.get("suite"),
        "passed": True,
    }


def audit_corrected_ae(ae_root: Path) -> dict:
    csv_path = ae_root / "corrected_autoencoder_results.csv"
    summary_path = ae_root / "corrected_autoencoder_summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise RuntimeError("corrected AE run is incomplete")

    summary = json.loads(summary_path.read_text())
    failures = []
    if summary.get("suite") != "corrected_autoencoder_factorial_protocol_v2":
        failures.append("suite_name")
    if summary.get("status") != "complete":
        failures.append("status")
    if int(summary.get("cell_count", -1)) != 12:
        failures.append("cell_count")
    semantics = summary.get("training_semantics") or {}
    if int(semantics.get("seed", -1)) != SEED:
        failures.append("training_seed")
    if float(semantics.get("ae_reconstruction_weight", -1.0)) != 0.1:
        failures.append("ae_reconstruction_weight")
    if int(semantics.get("memory_expiry_slots", -1)) != 8:
        failures.append("memory_expiry_slots")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    keys = {key(r) for r in rows}
    expected = expected_keys(("autoencoder",))
    if len(rows) != 12 or keys != expected:
        failures.append("corrected_AE_matrix")

    artifacts = []
    for pool in POOLINGS:
        for d_mem in CAPACITIES:
            k = (pool, "autoencoder", d_mem)
            train_dir = (
                ae_root / "trained" / "fixed" / f"seed_{SEED}"
                / f"{pool}_autoencoder_d{d_mem}"
            )
            ckpt = train_dir / f"ue_memory_{pool}_autoencoder_idaware_d{d_mem}_k2.weights.h5"
            manifest_path = train_dir / "corrected_ae_training_manifest.json"
            eval_dir = (
                ae_root / "evaluations" / "fixed" / f"seed_{SEED}"
                / f"factorial_{pool}_autoencoder_d{d_mem}"
            )
            evaluation_path = eval_dir / "evaluation.json"
            provenance_path = eval_dir / "corrected_ae_evaluation_provenance.json"
            for path, label in (
                (ckpt, "checkpoint"),
                (manifest_path, "manifest"),
                (evaluation_path, "evaluation"),
                (provenance_path, "evaluation_provenance"),
            ):
                if not path.is_file():
                    failures.append(f"{k}:{label}_missing")
            if not all(p.is_file() for p in (ckpt, manifest_path, evaluation_path, provenance_path)):
                continue

            manifest = json.loads(manifest_path.read_text())
            protocol = manifest.get("autoencoder_protocol") or {}
            if int(manifest.get("manifest_version", -1)) != 1:
                failures.append(f"{k}:manifest_version")
            if protocol.get("version") != 2:
                failures.append(f"{k}:protocol_version")
            if protocol.get("bounded_tanh_bottleneck") is not True:
                failures.append(f"{k}:bounded_tanh")
            if (manifest.get("stability") or {}).get("passed") is not True:
                failures.append(f"{k}:stability")
            ckpt_sha = sha256(ckpt)
            if manifest.get("checkpoint_sha256") != ckpt_sha:
                failures.append(f"{k}:checkpoint_hash")

            provenance = json.loads(provenance_path.read_text())
            if provenance.get("checkpoint_sha256") != ckpt_sha:
                failures.append(f"{k}:evaluation_checkpoint_hash")
            eval_sha = sha256(evaluation_path)
            if provenance.get("evaluation_json_sha256") != eval_sha:
                failures.append(f"{k}:evaluation_json_hash")
            evaluation = json.loads(evaluation_path.read_text())
            stamp = evaluation.get("corrected_ae_evaluation_stamp") or {}
            if stamp.get("checkpoint_sha256") != ckpt_sha:
                failures.append(f"{k}:evaluation_stamp_hash")
            artifacts.append(
                {
                    "pooling": pool,
                    "d_mem": d_mem,
                    "checkpoint_sha256": ckpt_sha,
                    "evaluation_sha256": eval_sha,
                }
            )

    for r in rows:
        k = key(r)
        try:
            if int(float(r.get("seed"))) != SEED:
                failures.append(f"{k}:row_seed")
            if int(float(r.get("memory_bits_per_ue"))) != int(k[2]) * 32:
                failures.append(f"{k}:row_memory_bits")
        except Exception as exc:
            failures.append(f"{k}:row_parse:{type(exc).__name__}")

    if failures:
        raise RuntimeError("corrected AE audit failed: " + "; ".join(failures[:40]))
    return {
        "rows": 12,
        "csv_sha256": sha256(csv_path),
        "summary_sha256": sha256(summary_path),
        "artifacts": artifacts,
        "passed": True,
    }


def main():
    a = parse_args()
    base_root = Path(a.base_root).expanduser().resolve()
    ae_root = Path(a.ae_root).expanduser().resolve()
    historical = audit_historical(base_root)
    corrected = audit_corrected_ae(ae_root)

    core = Path(__file__).resolve().parent / "merge_corrected_autoencoder_factorial.py"
    subprocess.run(
        [
            sys.executable,
            str(core),
            "--base-root",
            str(base_root),
            "--ae-root",
            str(ae_root),
        ],
        check=True,
        cwd=str(core.parent),
    )

    merged_summary_path = ae_root / "merged_valid_factorial_summary.json"
    merged = json.loads(merged_summary_path.read_text())
    merged["strict_provenance_audit"] = {
        "historical_writer_pca": historical,
        "corrected_autoencoder": corrected,
        "interpretation": (
            "36-cell factorial is the low-stat screening experiment; final method "
            "selection requires high-stat common-random-number evaluation with confidence intervals"
        ),
        "passed": True,
    }
    merged_summary_path.write_text(json.dumps(merged, indent=2) + "\n")
    print("STRICT_MERGED_VALID_FACTORIAL=" + json.dumps(merged, indent=2), flush=True)


if __name__ == "__main__":
    main()
