#!/usr/bin/env python3
"""PCA trainer that loads one pre-calibrated learned pooler and freezes it.

This wrapper reuses the corrected v5 PCA training protocol but replaces its
per-run pooler calibration with a shared NPZ produced by
calibrate_temporal_pooler.py. Therefore PCA d_mem=8/16/32/56 all see the exact
same Attention/CNN representation for a given seed/scheduling regime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _extract_pooler_file(argv):
    path = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--pooler-calibration-file":
            path = argv[i + 1]; i += 2; continue
        if arg.startswith("--pooler-calibration-file="):
            path = arg.split("=", 1)[1]; i += 1; continue
        cleaned.append(arg); i += 1
    if not path:
        raise SystemExit("--pooler-calibration-file is required")
    return Path(path).expanduser(), cleaned


POOLER_FILE, _CLEAN = _extract_pooler_file(sys.argv)
sys.argv[:] = _CLEAN

import train_temporal_ue_memory_v5_pca_pooler_calibrated as core

np = core.np
v3 = core.v3
v4 = core.v4


def _load_shared_pooler(p, e2e, actual_model, generator):
    if not POOLER_FILE.exists():
        raise FileNotFoundError(POOLER_FILE)

    # Build the pooler variables before assigning the stored arrays.
    batch = generator.sample_batch(1, v3.ARGS.seq_len, 3.0)
    inputs = v3.prepare_cgnn_inputs(
        e2e._receiver,
        batch["y"][:, 0],
        batch["ls"][:, 0],
        batch["active"][:, 0],
    )
    _ = actual_model.cold_pooled_final(inputs)

    z = np.load(str(POOLER_FILE))
    keys = sorted(z.files, key=lambda x: int(x[1:]))
    weights = [z[k] for k in keys]
    expected = len(actual_model.pooler.get_weights())
    if len(weights) != expected:
        raise RuntimeError(
            f"Shared pooler has {len(weights)} arrays, model expects {expected}"
        )
    actual_model.pooler.set_weights(weights)
    actual_model.pooler.trainable = False

    sidecar = POOLER_FILE.with_suffix(POOLER_FILE.suffix + ".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if metadata:
        if metadata.get("pooling") != v4.POOLING:
            raise RuntimeError(
                f"Calibration pooling={metadata.get('pooling')} does not match {v4.POOLING}"
            )
        if int(metadata.get("d_s", -1)) != int(p.d_s):
            raise RuntimeError("Shared pooler d_s mismatch")
        if int(metadata.get("seed", -1)) != int(v3.ARGS.seed):
            raise RuntimeError("Shared pooler seed mismatch")
        if bool(metadata.get("dynamic_scheduling")) != bool(
            not v3.ARGS.fixed_scheduling
        ):
            raise RuntimeError("Shared pooler scheduling-regime mismatch")

    return {
        "method": "load_shared_temporally_calibrated_pooler_then_freeze",
        "pooling": v4.POOLING,
        "weights_file": str(POOLER_FILE),
        "metadata_file": str(sidecar) if sidecar.exists() else None,
        "shared_across_pca_capacities": True,
        "pooler_frozen_before_pca_fit": True,
        "source_metadata": metadata,
    }


def main():
    core._calibrate_pooler = _load_shared_pooler
    core.main()

    out = Path(v3.ARGS.output_dir)
    summary_path = out / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["architecture"] = (
            "ue_identity_aware_temporal_memory_v6_pca_shared_pooler"
        )
        summary["pca_protocol"]["shared_pooler_across_capacities"] = True
        summary["pca_protocol"]["pooler_calibration_file"] = str(POOLER_FILE)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("V6_SHARED_POOLER_SUMMARY=" + json.dumps({
            "pooling": summary.get("pooling"),
            "d_mem": summary.get("d_mem"),
            "pooler_calibration_file": str(POOLER_FILE),
            "checkpoint": summary.get("checkpoint"),
        }), flush=True)


if __name__ == "__main__":
    main()
