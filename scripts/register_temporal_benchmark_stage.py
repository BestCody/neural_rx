#!/usr/bin/env python3
"""Register trained artifacts for one benchmark stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_benchmark_candidate import (
    build_candidate_manifest,
    file_sha256,
    manifest_artifacts,
    validate_candidate_manifest,
)
from temporal_benchmark_protocol import load_benchmark_protocol
from train_temporal_benchmark import training_source_checksums


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("calibration", "screen", "development", "confirmation"),
        required=True,
    )
    parser.add_argument(
        "--candidate-training-dir", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--matched-k8-training-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=(
            ROOT / "config" / "temporal_benchmark_training_data_v3.json"
        ),
    )
    parser.add_argument(
        "--training-recipe",
        type=Path,
        default=(
            ROOT / "config" / "temporal_benchmark_training_recipe_v3.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        type=Path,
        default=(
            ROOT
            / "config"
            / "temporal_benchmark_checkpoint_selection_v1.json"
        ),
    )
    return parser


def stage_contract(protocol: dict, stage: str) -> tuple[int, list[int]]:
    """Return the expected updates and seeds."""
    if stage == "calibration":
        value = protocol["calibration"]
        return int(value["training_updates"]), [
            int(value["training_recipe_seed"])
        ]
    value = protocol["stages"][stage]
    return int(value["training_updates"]), [
        int(seed) for seed in value["training_recipe_seeds"]
    ]


def load_summaries(
    directories: list[Path],
    role: str,
    stage: str,
    updates: int,
    seeds: list[int],
    training_type: str,
) -> list[tuple[dict, Path]]:
    """Load summaries in the required seed order."""
    found = {}
    for directory in directories:
        path = directory.expanduser().resolve() / "training_summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("training") != training_type:
            raise ValueError("training summary type differs")
        if value.get("role") != role or value.get("stage") != stage:
            raise ValueError("training summary role or stage differs")
        if value.get("smoke_test"):
            raise ValueError("smoke-test weights cannot be registered")
        if int(value.get("training_updates", -1)) != int(updates):
            raise ValueError("training summary update count differs")
        if int(value.get("training_examples", -1)) != int(updates):
            raise ValueError("training summary example count differs")
        seed = int(value["training_seed"])
        if seed in found:
            raise ValueError("duplicate trained seed")
        found[seed] = (value, path)
    if set(found) != set(seeds):
        raise ValueError("trained seeds differ from the stage")
    return [found[seed] for seed in seeds]


def checkpoint_path(summary: dict, selected: bool) -> Path:
    """Resolve and verify one trained checkpoint."""
    key = "selected_checkpoint" if selected else "full_checkpoint"
    digest_key = f"{key}_sha256"
    path = Path(summary[key]).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != summary[digest_key]:
        raise ValueError("trained checkpoint differs from its summary")
    return path


def episode_manifest_path(summary: dict) -> Path:
    """Resolve and verify one sealed training episode manifest."""
    path = Path(summary["training_episode_manifest"]).expanduser().resolve()
    expected = summary["training_episode_manifest_sha256"]
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError("training episode manifest differs from its summary")
    return path


def validate_training_pair(
    candidate: dict,
    baseline: dict,
    baseline_summary_path: Path,
) -> None:
    """Require paired data and the exact evaluated K=8 teacher."""
    if candidate["data_fingerprint"] != baseline["data_fingerprint"]:
        raise ValueError("candidate and K=8 training data differ")
    if candidate.get("teacher_checkpoint_sha256") != baseline.get(
        "full_checkpoint_sha256"
    ):
        raise ValueError("candidate teacher is not matched K=8")
    if candidate.get("teacher_value_checksum") != baseline.get(
        "final_value_checksum"
    ):
        raise ValueError("candidate teacher values differ")
    if candidate.get("teacher_weight_signature") != baseline.get(
        "final_weight_signature"
    ):
        raise ValueError("candidate teacher signature differs")
    if candidate.get("teacher_summary_sha256") != file_sha256(
        baseline_summary_path
    ):
        raise ValueError("candidate teacher summary is not sealed")


def validate_training_contract(
    summary: dict,
    protocol: dict,
    data: Path,
    recipe: Path,
    selection: Path,
) -> None:
    """Bind one training summary to the requested frozen contracts."""
    expected = {
        "benchmark_protocol_sha256": protocol["protocol_sha256"],
        "training_data_sha256": file_sha256(data),
        "training_recipe_sha256": file_sha256(recipe),
        "checkpoint_selection_sha256": file_sha256(selection),
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"training summary has invalid {name}")
    training_name = summary.get("training", "")
    if training_name == "temporal_benchmark_training_v3":
        version = int(training_name.rsplit("v", 1)[1])
        if summary.get("training_source_sha256") != (
            training_source_checksums(version)
        ):
            raise ValueError("training source changed")


def build_manifest(
    *,
    protocol: dict,
    stage: str,
    role: str,
    seeds: list[int],
    updates: int,
    checkpoints: list[Path],
    support: list[Path],
    training_data: Path,
    training_recipe: Path,
    selection: Path,
    training_version: int,
) -> dict:
    """Build one stage registration."""
    suffix = "_".join(str(seed) for seed in seeds)
    labels = {2: "aligned", 3: "power_aware"}
    label = labels[training_version]
    candidate_settings = {
        "candidate_id": f"v18_{label}_{stage}_{suffix}",
        "architecture": f"segment_safe_v18_{label}_training",
        "adapter": (
            "temporal_benchmark_adapters:make_v18_segment_safe_adapter"
        ),
        "change": (
            "Train V18 on anonymous weak-user power imbalance."
            if training_version == 3
            else "Align V18 teacher, memory, and transition coverage."
        ),
        "hypothesis": (
            "Power-aware training closes weak-user K=8 errors."
            if training_version == 3
            else "Aligned training closes schedule recovery errors."
        ),
    }
    settings = {
        "candidate": {
            **candidate_settings,
            "iterations": 2,
            "persistent": True,
            "horizon": 8,
            "metric": "worst_core_k8_gap",
        },
        "matched_cold_k8": {
            "candidate_id": f"matched_k8_{stage}_{suffix}",
            "architecture": "training_matched_neural_rx_k8",
            "adapter": "temporal_benchmark_adapters:make_cold_nrx_adapter",
            "iterations": 8,
            "persistent": False,
            "horizon": None,
            "change": "Train the matched Cold K=8 accuracy reference.",
            "hypothesis": "Matched K=8 defines the stage accuracy target.",
            "metric": "mean_k8_gap",
        },
    }[role]
    return build_candidate_manifest(
        candidate_id=settings["candidate_id"],
        role=role,
        architecture=settings["architecture"],
        adapter_factory=settings["adapter"],
        benchmark_protocol_sha256=protocol["protocol_sha256"],
        radio=protocol["radio"],
        training_data_sha256=file_sha256(training_data),
        training_recipe_sha256=file_sha256(training_recipe),
        checkpoint_selection_sha256=file_sha256(selection),
        training_seeds=seeds,
        training_examples=updates,
        training_updates=updates,
        inference_iterations=settings["iterations"],
        persistent_memory=settings["persistent"],
        memory_horizon_tbs=settings["horizon"],
        checkpoint_artifacts=checkpoints,
        support_artifacts=support,
        structural_change=settings["change"],
        hypothesis=settings["hypothesis"],
        expected_metric=settings["metric"],
        minimum_useful_improvement=0.005,
        maximum_gpu_hours=48.0,
        stop_rule="Stop after the selected benchmark stage.",
    )


def build_latency_manifest(protocol: dict) -> dict:
    """Register the fixed Cold K=3 latency reference."""
    config = ROOT / "config" / "nrx_large.cfg"
    weights = ROOT / "weights" / "nrx_large_weights"
    digest = file_sha256(config)
    return build_candidate_manifest(
        candidate_id="frozen_nvidia_cold_k3_latency",
        role="matched_cold_k3",
        architecture="frozen_nvidia_neural_rx_k3",
        adapter_factory="temporal_benchmark_adapters:make_cold_nrx_adapter",
        benchmark_protocol_sha256=protocol["protocol_sha256"],
        radio=protocol["radio"],
        training_data_sha256=digest,
        training_recipe_sha256=digest,
        checkpoint_selection_sha256=digest,
        training_seeds=[42],
        training_examples=10_000_000,
        training_updates=10_000_000,
        inference_iterations=3,
        persistent_memory=False,
        memory_horizon_tbs=None,
        checkpoint_artifacts=[weights],
        support_artifacts=[config],
        structural_change="Use frozen NVIDIA weights for K=3 latency only.",
        hypothesis="Cold K=3 provides the fixed runtime denominator.",
        expected_metric="latency_p95",
        minimum_useful_improvement=0.005,
        maximum_gpu_hours=1.0,
        stop_rule="Do not run Cold K=3 on accuracy episodes.",
    )


def add_artifacts(mapping: dict, manifest: dict, paths: list[Path]) -> None:
    """Add verified manifest artifacts to a path map."""
    by_digest = {file_sha256(path): path.resolve() for path in paths}
    for record in manifest_artifacts(manifest):
        digest = record["sha256"]
        if digest not in by_digest:
            raise ValueError("artifact map cannot resolve a manifest file")
        mapping[digest] = str(by_digest[digest])


def write_json(path: Path, value: dict) -> None:
    """Write one deterministic JSON artifact."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Register one calibration or benchmark stage."""
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    updates, seeds = stage_contract(protocol, args.stage)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = args.training_data.expanduser().resolve()
    recipe = args.training_recipe.expanduser().resolve()
    selection = args.checkpoint_selection.expanduser().resolve()
    data_contract = json.loads(data.read_text(encoding="utf-8"))
    protocol_name = data_contract.get("protocol")
    recipe_contract = json.loads(recipe.read_text(encoding="utf-8"))
    versions = {
        (
            "temporal_benchmark_training_data_v2",
            "temporal_benchmark_training_recipe_v2",
        ): 2,
        (
            "temporal_benchmark_training_data_v3",
            "temporal_benchmark_training_recipe_v3",
        ): 3,
    }
    key = (protocol_name, recipe_contract.get("protocol"))
    if key not in versions:
        raise ValueError("training contract versions differ")
    training_version = versions[key]
    training_type = f"temporal_benchmark_training_v{training_version}"
    k8_rows = load_summaries(
        args.matched_k8_training_dir,
        "matched_cold_k8",
        args.stage,
        updates,
        seeds,
        training_type,
    )
    for row, _ in k8_rows:
        validate_training_contract(
            row, protocol, data, recipe, selection
        )
    k8_checkpoints = [checkpoint_path(row, False) for row, _ in k8_rows]
    k8_support = [ROOT / "config" / "nrx_large.cfg"] + [
        path for _, path in k8_rows
    ]
    k8_support += [episode_manifest_path(row) for row, _ in k8_rows]
    if training_version >= 3:
        k8_support += [
            ROOT / "scripts" / name
            for name in training_source_checksums(training_version)
        ]
    k8 = build_manifest(
        protocol=protocol,
        stage=args.stage,
        role="matched_cold_k8",
        seeds=seeds,
        updates=updates,
        checkpoints=k8_checkpoints,
        support=k8_support,
        training_data=data,
        training_recipe=recipe,
        selection=selection,
        training_version=training_version,
    )
    validate_candidate_manifest(k8, protocol)
    manifests = {"matched_cold_k8": k8}
    paths = {"matched_cold_k8": k8_checkpoints + k8_support}
    if args.stage != "calibration":
        candidate_rows = load_summaries(
            args.candidate_training_dir,
            "candidate",
            args.stage,
            updates,
            seeds,
            training_type,
        )
        for row, _ in candidate_rows:
            validate_training_contract(
                row, protocol, data, recipe, selection
            )
        for (candidate, _), (baseline, baseline_path) in zip(
            candidate_rows, k8_rows
        ):
            validate_training_pair(
                candidate,
                baseline,
                baseline_path,
            )
        candidate_checkpoints = [
            checkpoint_path(row, True) for row, _ in candidate_rows
        ]
        candidate_support = [
            ROOT / "config" / "nrx_large.cfg",
            ROOT / "config" / "temporal_receiver_v18_segment_safe.json",
            ROOT / "config" / "segment_safe_pd_l8_threshold.json",
        ] + [path for _, path in candidate_rows]
        candidate_support += [
            episode_manifest_path(row) for row, _ in candidate_rows
        ]
        if training_version >= 3:
            candidate_support += [
                ROOT / "scripts" / name
                for name in training_source_checksums(training_version)
            ]
        candidate = build_manifest(
            protocol=protocol,
            stage=args.stage,
            role="candidate",
            seeds=seeds,
            updates=updates,
            checkpoints=candidate_checkpoints,
            support=candidate_support,
            training_data=data,
            training_recipe=recipe,
            selection=selection,
            training_version=training_version,
        )
        latency = build_latency_manifest(protocol)
        for manifest in (candidate, latency):
            validate_candidate_manifest(manifest, protocol)
        manifests.update({
            "candidate": candidate,
            "matched_cold_k3": latency,
        })
        paths.update({
            "candidate": candidate_checkpoints + candidate_support,
            "matched_cold_k3": [
                ROOT / "weights" / "nrx_large_weights",
                ROOT / "config" / "nrx_large.cfg",
            ],
        })
    artifact_map = {}
    for name, manifest in manifests.items():
        write_json(output / f"{name}.json", manifest)
        add_artifacts(artifact_map, manifest, paths[name])
    write_json(output / "artifact_map.json", artifact_map)
    write_json(output / "registration.json", {
        "schema_version": 2,
        "stage": args.stage,
        "training_updates": updates,
        "training_seeds": seeds,
        "training_protocol": protocol_name,
        "training_data_sha256": file_sha256(data),
        "training_recipe_sha256": file_sha256(recipe),
        "checkpoint_selection_sha256": file_sha256(selection),
        "manifest_sha256": {
            name: value["manifest_sha256"]
            for name, value in manifests.items()
        },
    })
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
