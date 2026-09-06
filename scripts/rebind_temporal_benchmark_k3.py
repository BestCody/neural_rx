#!/usr/bin/env python3
"""Rebind a K=2-latency run to the K=3 protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from calibrate_temporal_benchmark import validate_baseline_result
from register_temporal_benchmark_stage import (
    add_artifacts,
    build_latency_manifest,
    write_json,
)
from temporal_benchmark_candidate import (
    artifact_record,
    file_sha256,
    resolve_manifest_artifacts,
    validate_candidate_manifest,
    validate_matched_baseline,
)
from temporal_benchmark_plan import (
    build_calibration_artifact,
    build_stage_plan,
    validate_calibration_artifact,
    validate_stage_plan,
)
from temporal_benchmark_protocol import load_benchmark_protocol
from temporal_generalization_protocol import seal_payload


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MIGRATED_FIELDS = (
    "methods",
    "training.fixed_latency_checkpoint_required",
    "ranking.latency_reference",
    "execution.latency_only_reference_methods",
    "execution.latency.reference_methods",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registration", type=Path, required=True)
    parser.add_argument("--source-calibration-result", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen",), default="screen")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def _replace_one(values: list, old: str, new: str, label: str) -> None:
    """Replace one required list value."""
    if values.count(old) != 1 or new in values:
        raise ValueError(f"{label} does not contain one {old}")
    values[values.index(old)] = new


def k2_latency_predecessor(protocol: dict) -> dict:
    """Reconstruct the exact K=2-latency predecessor."""
    value = copy.deepcopy(protocol)
    value.pop("protocol_sha256", None)
    _replace_one(
        value["methods"],
        "matched_cold_k3",
        "matched_cold_k2",
        "methods",
    )
    training = value["training"]
    flag = "fixed_latency_k3_checkpoint_required"
    if training.pop(flag, None) is not True:
        raise ValueError("current K=3 training flag differs")
    training["fixed_latency_k2_checkpoint_required"] = True
    ranking = value["ranking"]
    if ranking.get("latency_reference") != "matched_cold_k3":
        raise ValueError("current ranking latency reference differs")
    ranking["latency_reference"] = "matched_cold_k2"
    execution = value["execution"]
    _replace_one(
        execution["latency_only_reference_methods"],
        "matched_cold_k3",
        "matched_cold_k2",
        "latency-only methods",
    )
    _replace_one(
        execution["latency"]["reference_methods"],
        "matched_cold_k3",
        "matched_cold_k2",
        "latency methods",
    )
    return seal_payload(value, field="protocol_sha256")


def _read_json(path: Path) -> dict:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _verified_path(record: dict, artifact_map: dict) -> Path | None:
    """Resolve one artifact when its bytes remain available."""
    path = artifact_map.get(record["sha256"])
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return None
    if path.stat().st_size != int(record["bytes"]):
        return None
    if file_sha256(path) != record["sha256"]:
        return None
    return path


def _source_code_records(manifest: dict, artifact_map: dict) -> set[tuple]:
    """Read the training-source checksums from its summary."""
    for record in manifest["support_artifacts"]:
        if record["name"] != "training_summary.json":
            continue
        path = _verified_path(record, artifact_map)
        if path is None:
            raise ValueError("the source training summary is unavailable")
        summary = _read_json(path)
        return set(summary.get("training_source_sha256", {}).items())
    raise ValueError("the source manifest lacks a training summary")


def _rebind_manifest(
    source: dict,
    source_path: Path,
    source_map: dict,
    protocol: dict,
) -> tuple[dict, list[Path], list[dict]]:
    """Rebind one trained manifest without changing weights."""
    source_code = _source_code_records(source, source_map)
    paths = []
    support = []
    omitted = []
    for record in source["checkpoints"]:
        path = _verified_path(record, source_map)
        if path is None:
            raise ValueError(f"checkpoint is unavailable: {record['name']}")
        paths.append(path)
    for record in source["support_artifacts"]:
        path = _verified_path(record, source_map)
        if path is not None:
            support.append(artifact_record(path))
            paths.append(path)
            continue
        identity = (record["name"], record["sha256"])
        if identity not in source_code:
            raise ValueError(f"support artifact changed: {record['name']}")
        omitted.append(copy.deepcopy(record))
    support.append(artifact_record(source_path))
    paths.append(source_path)
    rebound = copy.deepcopy(source)
    rebound.pop("manifest_sha256", None)
    rebound["benchmark_protocol_sha256"] = protocol["protocol_sha256"]
    rebound["support_artifacts"] = support
    rebound = seal_payload(rebound, field="manifest_sha256")
    validate_candidate_manifest(rebound, protocol)
    source_hashes = [row["sha256"] for row in source["checkpoints"]]
    rebound_hashes = [row["sha256"] for row in rebound["checkpoints"]]
    if source_hashes != rebound_hashes:
        raise ValueError("checkpoint hashes changed during rebinding")
    return rebound, paths, omitted


def _rebind_calibration_result(
    source: dict, protocol: dict
) -> dict:
    """Rebind latency-independent calibration counts."""
    rebound = copy.deepcopy(source)
    baseline = copy.deepcopy(rebound["baseline_manifest"])
    baseline.pop("manifest_sha256", None)
    baseline["benchmark_protocol_sha256"] = protocol["protocol_sha256"]
    baseline = seal_payload(baseline, field="manifest_sha256")
    validate_candidate_manifest(baseline, protocol)
    rebound.pop("result_sha256", None)
    rebound["protocol_sha256"] = protocol["protocol_sha256"]
    rebound["baseline_manifest"] = baseline
    return seal_payload(rebound, field="result_sha256")


def main() -> None:
    """Create the K=3-bound V3 benchmark artifacts."""
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    predecessor = k2_latency_predecessor(protocol)
    source_dir = args.source_registration.expanduser().resolve()
    source_map = _read_json(source_dir / "artifact_map.json")
    source_registration = _read_json(source_dir / "registration.json")
    source_paths = {
        name: source_dir / f"{name}.json"
        for name in ("candidate", "matched_cold_k8")
    }
    sources = {name: _read_json(path) for name, path in source_paths.items()}
    for name, manifest in sources.items():
        validate_candidate_manifest(manifest, predecessor)
        recorded = source_registration["manifest_sha256"].get(name)
        if recorded != manifest["manifest_sha256"]:
            raise ValueError(f"source registration differs for {name}")
    validate_matched_baseline(sources["candidate"], sources["matched_cold_k8"])
    output = args.output_dir.expanduser().resolve()
    registration_dir = output / "registration"
    registration_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    paths = {}
    omitted = {}
    for name in ("candidate", "matched_cold_k8"):
        manifests[name], paths[name], omitted[name] = _rebind_manifest(
            sources[name], source_paths[name], source_map, protocol
        )
    validate_matched_baseline(manifests["candidate"], manifests["matched_cold_k8"])
    manifests["matched_cold_k3"] = build_latency_manifest(protocol)
    validate_candidate_manifest(manifests["matched_cold_k3"], protocol)
    paths["matched_cold_k3"] = [
        ROOT / "weights" / "nrx_large_weights",
        ROOT / "config" / "nrx_large.cfg",
    ]
    artifact_map = {}
    for name, manifest in manifests.items():
        write_json(registration_dir / f"{name}.json", manifest)
        add_artifacts(artifact_map, manifest, paths[name])
        resolve_manifest_artifacts(manifest, artifact_map)
    write_json(registration_dir / "artifact_map.json", artifact_map)
    source_result_path = args.source_calibration_result.expanduser().resolve()
    source_result = _read_json(source_result_path)
    source_calibration = _read_json(
        args.source_calibration.expanduser().resolve()
    )
    validate_baseline_result(source_result, predecessor)
    validate_calibration_artifact(source_calibration, predecessor)
    if source_calibration["baseline_result_sha256"] != file_sha256(
        source_result_path
    ):
        raise ValueError("source calibration result differs")
    result = _rebind_calibration_result(source_result, protocol)
    validate_baseline_result(result, protocol)
    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    result_path = calibration_dir / "calibration_counts.json"
    write_json(result_path, result)
    calibration = build_calibration_artifact(
        protocol,
        file_sha256(result_path),
        result["baseline_manifest"]["manifest_sha256"],
        result["curves"],
    )
    validate_calibration_artifact(calibration, protocol)
    calibration_path = calibration_dir / "calibration.json"
    write_json(calibration_path, calibration)
    plan = build_stage_plan(
        protocol,
        calibration,
        args.stage,
        manifests["candidate"]["manifest_sha256"],
        manifests["matched_cold_k3"]["manifest_sha256"],
        manifests["matched_cold_k8"]["manifest_sha256"],
    )
    validate_stage_plan(plan, protocol, calibration)
    group_dir = output / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    write_json(group_dir / "stage_plan.json", plan)
    migration = seal_payload({
        "schema_version": 1,
        "artifact": "temporal_benchmark_k3_latency_rebind_v1",
        "source_protocol_sha256": predecessor["protocol_sha256"],
        "target_protocol_sha256": protocol["protocol_sha256"],
        "changed_fields": list(MIGRATED_FIELDS),
        "checkpoint_retrained": False,
        "source_manifest_sha256": {
            name: value["manifest_sha256"] for name, value in sources.items()
        },
        "target_manifest_sha256": {
            name: value["manifest_sha256"] for name, value in manifests.items()
        },
        "checkpoint_sha256": {
            name: [row["sha256"] for row in value["checkpoints"]]
            for name, value in manifests.items()
        },
        "omitted_unavailable_training_sources": omitted,
        "source_calibration_sha256": source_calibration[
            "calibration_sha256"
        ],
        "target_calibration_sha256": calibration["calibration_sha256"],
        "stage_plan_sha256": plan["plan_sha256"],
    }, field="migration_sha256")
    write_json(output / "protocol_migration.json", migration)
    registration = copy.deepcopy(source_registration)
    registration["manifest_sha256"] = {
        name: value["manifest_sha256"] for name, value in manifests.items()
    }
    registration["protocol_migration_sha256"] = migration[
        "migration_sha256"
    ]
    write_json(registration_dir / "registration.json", registration)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
