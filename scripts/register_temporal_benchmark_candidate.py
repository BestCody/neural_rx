#!/usr/bin/env python3
"""Register one receiver or baseline for benchmark v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_benchmark_candidate import (
    build_candidate_manifest,
    file_sha256,
    resolve_manifest_artifacts,
    validate_candidate_manifest,
)
from temporal_benchmark_protocol import load_benchmark_protocol


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--training-recipe", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument(
        "--support-artifact", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-map-output", type=Path)
    parser.add_argument("--base-artifact-map", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def _required(mapping: dict, key: str):
    """Read one required registration field."""
    if key not in mapping:
        raise ValueError(f"candidate spec lacks {key}")
    return mapping[key]


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    training = _required(spec, "training")
    inference = _required(spec, "inference")
    registration = _required(spec, "registration")
    manifest = build_candidate_manifest(
        candidate_id=_required(spec, "candidate_id"),
        role=_required(spec, "role"),
        architecture=_required(spec, "architecture"),
        adapter_factory=_required(spec, "adapter_factory"),
        benchmark_protocol_sha256=protocol["protocol_sha256"],
        radio=protocol["radio"],
        training_data_sha256=file_sha256(args.training_data),
        training_recipe_sha256=file_sha256(args.training_recipe),
        checkpoint_selection_sha256=file_sha256(
            args.checkpoint_selection
        ),
        training_seeds=_required(training, "seeds"),
        training_examples=_required(training, "examples"),
        training_updates=_required(training, "updates"),
        inference_iterations=_required(inference, "iterations"),
        persistent_memory=_required(inference, "persistent_memory"),
        memory_horizon_tbs=inference.get("memory_horizon_tbs"),
        checkpoint_artifacts=args.checkpoint,
        support_artifacts=args.support_artifact,
        structural_change=_required(registration, "structural_change"),
        hypothesis=_required(registration, "hypothesis"),
        expected_metric=_required(registration, "expected_metric"),
        minimum_useful_improvement=_required(
            registration, "minimum_useful_improvement"
        ),
        maximum_gpu_hours=_required(registration, "maximum_gpu_hours"),
        stop_rule=_required(registration, "stop_rule"),
        parent_candidate_id=registration.get("parent_candidate_id"),
    )
    validate_candidate_manifest(manifest, protocol)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.artifact_map_output is not None:
        artifact_map = {}
        if args.base_artifact_map is not None:
            artifact_map = json.loads(
                args.base_artifact_map.read_text(encoding="utf-8")
            )
            if not isinstance(artifact_map, dict):
                raise ValueError("base artifact map must be an object")
        for path in args.checkpoint + args.support_artifact:
            resolved = path.expanduser().resolve()
            artifact_map[file_sha256(resolved)] = str(resolved)
        resolve_manifest_artifacts(manifest, artifact_map)
        map_output = args.artifact_map_output.expanduser().resolve()
        map_output.parent.mkdir(parents=True, exist_ok=True)
        map_output.write_text(
            json.dumps(artifact_map, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {map_output}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
