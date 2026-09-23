#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from temporal_generalization_protocol import seal_payload, verify_payload


ROLES = (
    "candidate",
    "matched_cold_k3",
    "matched_cold_k8",
    "historical_cold_k8",
    "perfect_csi_kbest",
)
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path | str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "name": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def manifest_artifacts(manifest: dict) -> list[dict]:
    return list(manifest.get("checkpoints", [])) + list(
        manifest.get("support_artifacts", [])
    )


def resolve_manifest_artifacts(
    manifest: dict, artifact_map: dict[str, str]
) -> dict[str, Path]:
    resolved = {}
    for record in manifest_artifacts(manifest):
        digest = record["sha256"]
        if digest not in artifact_map:
            raise FileNotFoundError(f"artifact map lacks {digest}")
        path = Path(artifact_map[digest]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"artifact size differs for {path.name}")
        if file_sha256(path) != digest:
            raise ValueError(f"artifact checksum differs for {path.name}")
        resolved[digest] = path
    return resolved


def build_candidate_manifest(
    *,
    candidate_id: str,
    role: str,
    architecture: str,
    adapter_factory: str,
    benchmark_protocol_sha256: str,
    radio: dict,
    training_data_sha256: str,
    training_recipe_sha256: str,
    checkpoint_selection_sha256: str,
    training_seeds: list[int],
    training_examples: int,
    training_updates: int,
    inference_iterations: int,
    persistent_memory: bool,
    memory_horizon_tbs: int | None,
    checkpoint_artifacts: list[Path | str],
    support_artifacts: list[Path | str],
    structural_change: str,
    hypothesis: str,
    expected_metric: str,
    minimum_useful_improvement: float,
    maximum_gpu_hours: float,
    stop_rule: str,
    parent_candidate_id: str | None = None,
) -> dict:
    return seal_payload({
        "schema_version": 1,
        "manifest": "temporal_benchmark_candidate_v1",
        "candidate_id": candidate_id,
        "role": role,
        "architecture": architecture,
        "adapter_factory": adapter_factory,
        "benchmark_protocol_sha256": benchmark_protocol_sha256,
        "radio": dict(radio),
        "training": {
            "data_sha256": training_data_sha256,
            "recipe_sha256": training_recipe_sha256,
            "checkpoint_selection_sha256": checkpoint_selection_sha256,
            "seeds": [int(value) for value in training_seeds],
            "examples": int(training_examples),
            "updates": int(training_updates),
        },
        "inference": {
            "iterations": int(inference_iterations),
            "persistent_memory": bool(persistent_memory),
            "memory_horizon_tbs": (
                None if memory_horizon_tbs is None else int(memory_horizon_tbs)
            ),
        },
        "checkpoints": [
            {
                "training_seed": int(seed),
                **artifact_record(path),
            }
            for seed, path in zip(training_seeds, checkpoint_artifacts)
        ],
        "support_artifacts": [
            artifact_record(path) for path in support_artifacts
        ],
        "registration": {
            "parent_candidate_id": parent_candidate_id,
            "structural_change": structural_change,
            "hypothesis": hypothesis,
            "expected_metric": expected_metric,
            "minimum_useful_improvement": float(
                minimum_useful_improvement
            ),
            "maximum_gpu_hours": float(maximum_gpu_hours),
            "stop_rule": stop_rule,
        },
    }, field="manifest_sha256")


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def validate_candidate_manifest(manifest: dict, protocol: dict) -> None:
    verify_payload(manifest, field="manifest_sha256")
    if manifest.get("manifest") != "temporal_benchmark_candidate_v1":
        raise ValueError("candidate manifest type differs")
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not IDENTIFIER.fullmatch(
        candidate_id
    ):
        raise ValueError("candidate_id is invalid")
    if manifest.get("role") not in ROLES:
        raise ValueError("candidate role is invalid")
    role = manifest["role"]
    if manifest.get("benchmark_protocol_sha256") != protocol[
        "protocol_sha256"
    ]:
        raise ValueError("candidate benchmark protocol differs")
    if manifest.get("radio") != protocol["radio"]:
        raise ValueError("candidate radio contract differs")
    adapter = manifest.get("adapter_factory")
    if not isinstance(adapter, str) or adapter.count(":") != 1:
        raise ValueError("adapter_factory must be module:function")
    training = manifest.get("training", {})
    for name in (
        "data_sha256",
        "recipe_sha256",
        "checkpoint_selection_sha256",
    ):
        _digest(training.get(name), f"training.{name}")
    seeds = training.get("seeds")
    if not isinstance(seeds, list):
        raise ValueError("training seeds must be a list")
    if role == "perfect_csi_kbest" and seeds:
        raise ValueError("perfect-CSI K-Best cannot have training seeds")
    if role != "perfect_csi_kbest" and not seeds:
        raise ValueError("at least one training seed is required")
    if len(set(int(value) for value in seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    examples = int(training.get("examples", -1))
    updates = int(training.get("updates", -1))
    if role == "perfect_csi_kbest":
        if examples != 0 or updates != 0:
            raise ValueError("perfect-CSI K-Best cannot have training")
    elif examples <= 0 or updates <= 0:
        raise ValueError("training examples and updates must be positive")
    inference = manifest.get("inference", {})
    iterations = int(inference.get("iterations", 0))
    expected_iterations = {
        "candidate": int(protocol["objective"]["candidate_iterations"]),
        "matched_cold_k3": 3,
        "matched_cold_k8": 8,
        "historical_cold_k8": 8,
        "perfect_csi_kbest": 0,
    }[role]
    if iterations != expected_iterations:
        raise ValueError(f"{role} must use K={expected_iterations}")
    if role == "candidate" and not inference.get("persistent_memory"):
        raise ValueError("the candidate must expose persistent memory")
    if role != "candidate" and inference.get("persistent_memory"):
        raise ValueError("cold baselines cannot persist memory")
    horizon = inference.get("memory_horizon_tbs")
    if inference.get("persistent_memory"):
        if horizon is not None and int(horizon) <= 0:
            raise ValueError("memory horizon must be positive")
    elif horizon is not None:
        raise ValueError("stateless candidates cannot have a memory horizon")
    checkpoints = manifest.get("checkpoints")
    support = manifest.get("support_artifacts")
    if not isinstance(checkpoints, list) or not isinstance(support, list):
        raise ValueError("candidate artifact lists are required")
    if [value.get("training_seed") for value in checkpoints] != seeds:
        raise ValueError("checkpoint artifacts do not map to training seeds")
    artifacts = checkpoints + support
    if not artifacts:
        raise ValueError("at least one candidate artifact is required")
    digests = set()
    for artifact in artifacts:
        _digest(artifact.get("sha256"), "artifact.sha256")
        name = artifact.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("candidate artifact name is invalid")
        if int(artifact.get("bytes", 0)) <= 0:
            raise ValueError("candidate artifact cannot be empty")
        digests.add(artifact["sha256"])
    if len(digests) != len(artifacts):
        raise ValueError("candidate artifacts must be unique")
    registration = manifest.get("registration", {})
    required = protocol["registration"]["required_fields"]
    for name in required:
        value = registration.get(name)
        if value is None or value == "":
            raise ValueError(f"registration.{name} is required")
    change = registration.get("structural_change")
    if not isinstance(change, str) or len(change.splitlines()) != 1:
        raise ValueError("structural_change must be one line")
    parent = registration.get("parent_candidate_id")
    if parent is not None:
        if not isinstance(parent, str) or not IDENTIFIER.fullmatch(parent):
            raise ValueError("parent_candidate_id is invalid")
        if parent == candidate_id:
            raise ValueError("a candidate cannot be its own parent")
    if registration.get("expected_metric") not in protocol[
        "registration"
    ]["allowed_metrics"]:
        raise ValueError("expected metric is not approved")
    if float(registration.get("minimum_useful_improvement", 0.0)) <= 0.0:
        raise ValueError("minimum useful improvement must be positive")
    if float(registration.get("maximum_gpu_hours", 0.0)) <= 0.0:
        raise ValueError("maximum GPU hours must be positive")


def load_candidate_manifest(path: Path | str, protocol: dict) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_manifest(manifest, protocol)
    return manifest


def validate_matched_baseline(candidate: dict, baseline: dict) -> None:
    if candidate["role"] != "candidate":
        raise ValueError("the first manifest must be a candidate")
    if baseline["role"] != "matched_cold_k8":
        raise ValueError("accuracy baseline must be matched Cold K=8")
    if candidate["benchmark_protocol_sha256"] != baseline[
        "benchmark_protocol_sha256"
    ]:
        raise ValueError("candidate and baseline benchmark versions differ")
    if candidate["radio"] != baseline["radio"]:
        raise ValueError("candidate and baseline radio settings differ")
    for name in (
        "data_sha256",
        "checkpoint_selection_sha256",
        "seeds",
        "examples",
        "updates",
    ):
        if candidate["training"][name] != baseline["training"][name]:
            raise ValueError(f"training-matched baseline differs in {name}")
    if baseline["inference"]["iterations"] != 8:
        raise ValueError("matched baseline iteration count differs")
    if baseline["inference"]["persistent_memory"]:
        raise ValueError("matched cold baseline cannot persist memory")


def validate_stage_training(manifest: dict, protocol: dict, stage: str) -> None:
    if stage not in protocol["stages"]:
        raise ValueError("unknown benchmark stage")
    expected = protocol["stages"][stage]
    if manifest["training"]["updates"] != expected["training_updates"]:
        raise ValueError("candidate training updates differ from the stage")
    if manifest["training"]["examples"] != expected["training_examples"]:
        raise ValueError("candidate training examples differ from the stage")
    if len(manifest["training"]["seeds"]) != expected[
        "required_training_seeds"
    ]:
        raise ValueError("candidate training-seed count differs from the stage")
    if manifest["training"]["seeds"] != expected["training_recipe_seeds"]:
        raise ValueError("candidate training seeds differ from the stage")


@runtime_checkable
class BenchmarkDecoderAdapter(Protocol):

    manifest: dict

    def start_episode(self) -> None:
        ...

    def start_episode_batch(self, batch_size: int) -> None:
        ...

    def reset_memory(self) -> None:
        ...

    def state_is_empty(self) -> bool:
        ...

    def decode_tb(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...

    def decode_tb_batch(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...

    def synchronize(self) -> None:
        ...

    def checksums_unchanged(self) -> bool:
        ...


def load_adapter(factory: str, **kwargs) -> BenchmarkDecoderAdapter:
    module_name, function_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError("adapter factory is not callable")
    adapter = function(**kwargs)
    if not isinstance(adapter, BenchmarkDecoderAdapter):
        raise TypeError("factory did not return a benchmark adapter")
    return adapter


def validate_adapter_output(output: dict, bits_shape: tuple[int, ...]) -> None:
    required = {"hard_bits", "crc_status", "logits"}
    if set(output) != required:
        raise ValueError("adapter output fields differ")
    expected_crc = bits_shape[:-1]
    hard = np.asarray(output["hard_bits"])
    crc = np.asarray(output["crc_status"])
    logits = np.asarray(output["logits"])
    if hard.shape != bits_shape:
        raise ValueError("adapter hard-bit shape differs")
    if crc.shape != expected_crc:
        raise ValueError("adapter CRC shape differs")
    if logits.shape != bits_shape:
        raise ValueError("adapter logits shape differs")
    if not np.all(np.isfinite(logits)):
        raise ValueError("adapter logits are not finite")
