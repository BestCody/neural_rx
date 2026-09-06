#!/usr/bin/env python3
"""Seal matched-Cold-K8 benchmark operating points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_benchmark_candidate import (
    file_sha256,
    validate_candidate_manifest,
)
from temporal_benchmark_plan import build_calibration_artifact
from temporal_benchmark_protocol import CONDITIONS, load_benchmark_protocol
from temporal_generalization_protocol import verify_payload


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def validate_baseline_result(result: dict, protocol: dict) -> dict:
    """Validate calibration counts from matched Cold K=8."""
    verify_payload(result, field="result_sha256")
    if result.get("result") != "temporal_benchmark_calibration_counts_v1":
        raise ValueError("calibration result type differs")
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("calibration result protocol differs")
    if result.get("method") != "matched_cold_k8":
        raise ValueError("calibration method must be matched Cold K=8")
    if result.get("episodes_per_condition") != protocol["calibration"][
        "episodes_per_condition"
    ]:
        raise ValueError("calibration episode count differs")
    if result.get("num_tbs") != protocol["calibration"]["num_tbs"]:
        raise ValueError("calibration TB horizon differs")
    if result.get("condition_names") != list(CONDITIONS):
        raise ValueError("calibration condition names differ")
    manifest = result.get("baseline_manifest", {})
    validate_candidate_manifest(manifest, protocol)
    if manifest["role"] != "matched_cold_k8":
        raise ValueError("calibration baseline role differs")
    expected_training = {
        "updates": protocol["calibration"]["training_updates"],
        "examples": protocol["calibration"]["training_examples"],
        "seeds": [protocol["calibration"]["training_recipe_seed"]],
    }
    for name, value in expected_training.items():
        if manifest["training"][name] != value:
            raise ValueError(f"calibration baseline differs in {name}")
    if not result.get("preflight") or not all(
        value is True for value in result["preflight"].values()
    ):
        raise ValueError("calibration baseline preflight failed")
    curves = result.get("curves", {})
    if list(curves) != list(CONDITIONS):
        raise ValueError("calibration condition order differs")
    return curves


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    result_path = args.baseline_result.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    curves = validate_baseline_result(result, protocol)
    artifact = build_calibration_artifact(
        protocol,
        file_sha256(result_path),
        result["baseline_manifest"]["manifest_sha256"],
        curves,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
