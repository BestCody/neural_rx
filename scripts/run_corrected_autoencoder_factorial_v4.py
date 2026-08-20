#!/usr/bin/env python3
"""Final integrity layer for the corrected-AE factorial runner.

Adds evaluation-file hash verification and records the Atlas runtime environment
without changing any research/training/evaluation hyperparameters.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import run_corrected_autoencoder_factorial_v3 as v3

v2 = v3.v2
_PREVIOUS_PREFLIGHT = v2.preflight
_PREVIOUS_EVALUATE_CELL = v2.evaluate_cell


def _gpu_inventory() -> list[dict]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",", 3)]
        if len(parts) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_mib": int(parts[3]),
            }
        )
    return rows


def _python_environment() -> dict:
    code = (
        "import json,sys,tensorflow as tf,sionna; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'tensorflow':tf.__version__,'sionna':getattr(sionna,'__version__','unknown')}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(v2.SCRIPT_DIR),
        text=True,
        capture_output=True,
        check=True,
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def preflight_final(current_commit: str) -> dict:
    report = _PREVIOUS_PREFLIGHT(current_commit)
    gpus = _gpu_inventory()
    by_index = {g["index"]: g for g in gpus}
    requested = list(v2.base.GPUS)
    failures = []
    if requested != [0, 1]:
        failures.append(f"expected requested GPU IDs [0, 1], got {requested}")
    for idx in requested:
        item = by_index.get(idx)
        if item is None:
            failures.append(f"GPU {idx} is missing")
        elif "A40" not in item["name"]:
            failures.append(f"GPU {idx} is not an A40: {item['name']}")
        elif item["memory_mib"] < 45000:
            failures.append(
                f"GPU {idx} reports unexpectedly low memory: {item['memory_mib']} MiB"
            )
    if failures:
        raise RuntimeError("Atlas GPU preflight failed: " + "; ".join(failures))

    report["atlas_environment"] = {
        "requested_gpu_ids": requested,
        "gpus": [by_index[i] for i in requested],
        "python_stack": _python_environment(),
    }
    report["final_integrity_pass"] = True
    v2.write_json(v2.ROOT / "corrected_autoencoder_preflight.json", report)
    print("CORRECTED_AE_FINAL_PREFLIGHT=" + json.dumps(report, indent=2), flush=True)
    return report


def evaluate_cell_final(
    gpu: int,
    ckpt: Path,
    pooling: str,
    d_mem: int,
    current_commit: str,
) -> dict:
    sidecar_path = v3._eval_sidecar(pooling, d_mem)
    evaluation_file = v2.evaluation_dir(pooling, d_mem) / "evaluation.json"

    # Detect byte-level modification/corruption before the v3 provenance check.
    if sidecar_path.is_file() and evaluation_file.is_file():
        try:
            sidecar = v2.load_json(sidecar_path)
            saved = sidecar.get("evaluation_json_sha256")
            actual = v3._sha256(evaluation_file)
            if not saved or saved != actual:
                existing = v2.load_json(evaluation_file)
                existing.pop("corrected_ae_evaluation_stamp", None)
                v2.write_json(evaluation_file, existing)
                print(
                    f"INVALIDATE_MODIFIED_CORRECTED_AE_EVALUATION={evaluation_file.parent}",
                    flush=True,
                )
        except Exception:
            try:
                existing = v2.load_json(evaluation_file)
                existing.pop("corrected_ae_evaluation_stamp", None)
                v2.write_json(evaluation_file, existing)
            except Exception:
                pass

    evaluation = _PREVIOUS_EVALUATE_CELL(
        gpu, ckpt, pooling, d_mem, current_commit
    )

    # v3 writes the sidecar after successful strict validation. Verify the hash
    # immediately so a write/serialization issue cannot pass unnoticed.
    sidecar = v2.load_json(sidecar_path)
    actual = v3._sha256(evaluation_file)
    if sidecar.get("evaluation_json_sha256") != actual:
        raise RuntimeError(
            f"evaluation provenance hash self-check failed: {evaluation_file}"
        )
    return evaluation


v2.preflight = preflight_final
v2.evaluate_cell = evaluate_cell_final


def main():
    v2.main()


if __name__ == "__main__":
    main()
