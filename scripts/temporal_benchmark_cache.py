#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporal_benchmark_candidate import (
    file_sha256,
    validate_candidate_manifest,
    validate_matched_baseline,
    validate_stage_training,
)
from temporal_benchmark_io import (
    COMMON_ARRAYS,
    validate_common_arrays,
    validate_latency_arrays,
    validate_method_arrays,
    write_run_bundle,
)
from temporal_generalization_protocol import seal_payload, verify_payload


def _descriptors(arrays: dict[str, np.ndarray]) -> dict:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _validate_finite(arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")


def _validate_preflight(actual: dict, expected: list[str]) -> None:
    if set(actual) != set(expected):
        raise ValueError("benchmark cache preflight checks differ")
    if not all(value is True for value in actual.values()):
        raise ValueError("benchmark cache preflight did not pass")


def _write_npz_bundle(
    destination: Path,
    filename: str,
    arrays: dict[str, np.ndarray],
    payload: dict,
    digest_field: str,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / filename
    np.savez_compressed(raw_path, **arrays)
    payload["raw_metrics"] = {
        "file": raw_path.name,
        "sha256": file_sha256(raw_path),
        "arrays": _descriptors(arrays),
    }
    return_payload = seal_payload(payload, field=digest_field)
    json_path = destination / payload["json_file"]
    json_path.write_text(
        json.dumps(return_payload, indent=2) + "\n", encoding="utf-8"
    )
    return json_path


def _load_npz_bundle(
    path: Path | str,
    type_field: str,
    type_value: str,
    digest_field: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_payload(payload, field=digest_field)
    if payload.get(type_field) != type_value:
        raise ValueError("benchmark cache type differs")
    raw_path = path.parent / payload["raw_metrics"]["file"]
    if file_sha256(raw_path) != payload["raw_metrics"]["sha256"]:
        raise ValueError("benchmark cache raw checksum differs")
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if payload["raw_metrics"].get("arrays") != _descriptors(arrays):
        raise ValueError("benchmark cache array descriptors differ")
    return payload, arrays


def _validate_plan_binding(payload: dict, protocol: dict, plan: dict) -> None:
    expected = {
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "stage": plan["stage"],
        "scoring_window": plan["scoring_window"],
        "jobs": plan["jobs"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"benchmark cache differs in {name}")


def _validate_reference_manifests(
    baselines: dict[str, dict], protocol: dict, plan: dict
) -> None:
    execution = protocol["execution"]
    expected = set(
        execution["latency_only_reference_methods"]
        + execution["accuracy_reference_methods"]
    )
    if set(baselines) != expected:
        raise ValueError("reference baseline manifests differ")
    for name, manifest in baselines.items():
        validate_candidate_manifest(manifest, protocol)
        if manifest["manifest_sha256"] != plan["baseline_manifests"][name]:
            raise ValueError(f"{name} differs from the stage plan")
    k3 = baselines["matched_cold_k3"]
    k8 = baselines["matched_cold_k8"]
    validate_stage_training(k8, protocol, plan["stage"])
    if k3["role"] != "matched_cold_k3" or k8["role"] != "matched_cold_k8":
        raise ValueError("matched reference role differs")


def _validate_static_reference_manifests(
    baselines: dict[str, dict], protocol: dict, plan: dict
) -> None:
    expected = set(protocol["execution"]["static_reference_methods"])
    if set(baselines) != expected:
        raise ValueError("static reference manifests differ")
    if not protocol["stages"][plan["stage"]]["include_static_references"]:
        raise ValueError("stage does not include static references")
    for name, manifest in baselines.items():
        validate_candidate_manifest(manifest, protocol)
        if manifest["manifest_sha256"] != plan["baseline_manifests"][name]:
            raise ValueError(f"{name} differs from the stage plan")
        if manifest["role"] != name:
            raise ValueError(f"{name} role differs")


def validate_candidate_arrays(
    arrays: dict[str, np.ndarray], protocol: dict, plan: dict
) -> None:
    validate_common_arrays(arrays, protocol, plan)
    methods = protocol["execution"]["candidate_methods"]
    validate_method_arrays(arrays, methods, protocol, plan)
    validate_latency_arrays(
        arrays,
        protocol["execution"]["latency"]["candidate_methods"],
        protocol,
        plan,
    )
    _validate_finite(arrays)


def validate_reference_arrays(
    arrays: dict[str, np.ndarray], protocol: dict, plan: dict
) -> None:
    validate_common_arrays(arrays, protocol, plan)
    methods = protocol["execution"]["accuracy_reference_methods"]
    validate_method_arrays(arrays, methods, protocol, plan)
    validate_latency_arrays(
        arrays,
        protocol["execution"]["latency"]["reference_methods"],
        protocol,
        plan,
    )
    _validate_finite(arrays)


def validate_static_reference_arrays(
    arrays: dict[str, np.ndarray], protocol: dict, plan: dict
) -> None:
    validate_common_arrays(arrays, protocol, plan)
    methods = protocol["execution"]["static_reference_methods"]
    validate_method_arrays(arrays, methods, protocol, plan)
    _validate_finite(arrays)


def write_candidate_output(
    output_dir: Path | str,
    *,
    protocol: dict,
    plan: dict,
    candidate: dict,
    training_seed: int,
    arrays: dict[str, np.ndarray],
    preflight: dict[str, bool],
    elapsed_seconds: float,
) -> Path:
    validate_candidate_manifest(candidate, protocol)
    validate_stage_training(candidate, protocol, plan["stage"])
    if candidate["manifest_sha256"] != plan["candidate_manifest_sha256"]:
        raise ValueError("candidate differs from the stage plan")
    if int(training_seed) not in candidate["training"]["seeds"]:
        raise ValueError("candidate training seed differs")
    validate_candidate_arrays(arrays, protocol, plan)
    _validate_preflight(
        preflight, protocol["execution"]["candidate_preflight"]
    )
    if not np.isfinite(elapsed_seconds) or float(elapsed_seconds) <= 0.0:
        raise ValueError("candidate elapsed time must be positive")
    payload = {
        "schema_version": 1,
        "output": "temporal_benchmark_candidate_output_v1",
        "json_file": "candidate_output.json",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "stage": plan["stage"],
        "scoring_window": plan["scoring_window"],
        "jobs": plan["jobs"],
        "candidate": candidate,
        "training_seed": int(training_seed),
        "preflight": preflight,
        "elapsed_seconds": float(elapsed_seconds),
    }
    return _write_npz_bundle(
        Path(output_dir).expanduser().resolve(),
        "candidate_metrics.npz",
        arrays,
        payload,
        "output_sha256",
    )


def load_candidate_output(
    path: Path | str, protocol: dict, plan: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    payload, arrays = _load_npz_bundle(
        path,
        "output",
        "temporal_benchmark_candidate_output_v1",
        "output_sha256",
    )
    _validate_plan_binding(payload, protocol, plan)
    candidate = payload["candidate"]
    validate_candidate_manifest(candidate, protocol)
    validate_stage_training(candidate, protocol, plan["stage"])
    if candidate["manifest_sha256"] != plan["candidate_manifest_sha256"]:
        raise ValueError("candidate output differs from the stage plan")
    if payload["training_seed"] not in candidate["training"]["seeds"]:
        raise ValueError("candidate output training seed differs")
    _validate_preflight(
        payload["preflight"], protocol["execution"]["candidate_preflight"]
    )
    validate_candidate_arrays(arrays, protocol, plan)
    return payload, arrays


def write_reference_cache(
    output_dir: Path | str,
    *,
    protocol: dict,
    plan: dict,
    baselines: dict[str, dict],
    training_seed: int,
    arrays: dict[str, np.ndarray],
    preflight: dict[str, bool],
    elapsed_seconds: float,
) -> Path:
    _validate_reference_manifests(baselines, protocol, plan)
    if int(training_seed) not in baselines["matched_cold_k8"]["training"][
        "seeds"
    ]:
        raise ValueError("reference training seed differs")
    validate_reference_arrays(arrays, protocol, plan)
    _validate_preflight(
        preflight, protocol["execution"]["matched_reference_preflight"]
    )
    if not np.isfinite(elapsed_seconds) or float(elapsed_seconds) <= 0.0:
        raise ValueError("reference elapsed time must be positive")
    payload = {
        "schema_version": 1,
        "cache": "temporal_benchmark_reference_cache_v1",
        "json_file": "reference_cache.json",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "stage": plan["stage"],
        "scoring_window": plan["scoring_window"],
        "jobs": plan["jobs"],
        "baselines": baselines,
        "training_seed": int(training_seed),
        "preflight": preflight,
        "elapsed_seconds": float(elapsed_seconds),
    }
    return _write_npz_bundle(
        Path(output_dir).expanduser().resolve(),
        "reference_metrics.npz",
        arrays,
        payload,
        "cache_sha256",
    )


def load_reference_cache(
    path: Path | str, protocol: dict, plan: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    payload, arrays = _load_npz_bundle(
        path,
        "cache",
        "temporal_benchmark_reference_cache_v1",
        "cache_sha256",
    )
    _validate_plan_binding(payload, protocol, plan)
    _validate_reference_manifests(payload["baselines"], protocol, plan)
    if payload["training_seed"] not in payload["baselines"][
        "matched_cold_k8"
    ]["training"]["seeds"]:
        raise ValueError("reference cache training seed differs")
    _validate_preflight(
        payload["preflight"],
        protocol["execution"]["matched_reference_preflight"],
    )
    validate_reference_arrays(arrays, protocol, plan)
    return payload, arrays


def write_static_reference_cache(
    output_dir: Path | str,
    *,
    protocol: dict,
    plan: dict,
    baselines: dict[str, dict],
    arrays: dict[str, np.ndarray],
    preflight: dict[str, bool],
    elapsed_seconds: float,
) -> Path:
    _validate_static_reference_manifests(baselines, protocol, plan)
    validate_static_reference_arrays(arrays, protocol, plan)
    _validate_preflight(
        preflight, protocol["execution"]["static_reference_preflight"]
    )
    if not np.isfinite(elapsed_seconds) or float(elapsed_seconds) <= 0.0:
        raise ValueError("static reference elapsed time must be positive")
    payload = {
        "schema_version": 1,
        "cache": "temporal_benchmark_static_reference_cache_v1",
        "json_file": "static_reference_cache.json",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "stage": plan["stage"],
        "scoring_window": plan["scoring_window"],
        "jobs": plan["jobs"],
        "baselines": baselines,
        "preflight": preflight,
        "elapsed_seconds": float(elapsed_seconds),
    }
    return _write_npz_bundle(
        Path(output_dir).expanduser().resolve(),
        "static_reference_metrics.npz",
        arrays,
        payload,
        "cache_sha256",
    )


def load_static_reference_cache(
    path: Path | str, protocol: dict, plan: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    payload, arrays = _load_npz_bundle(
        path,
        "cache",
        "temporal_benchmark_static_reference_cache_v1",
        "cache_sha256",
    )
    _validate_plan_binding(payload, protocol, plan)
    _validate_static_reference_manifests(
        payload["baselines"], protocol, plan
    )
    _validate_preflight(
        payload["preflight"],
        protocol["execution"]["static_reference_preflight"],
    )
    validate_static_reference_arrays(arrays, protocol, plan)
    return payload, arrays


def compose_run_arrays(
    candidate: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    protocol: dict,
    plan: dict,
    static_references: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    validate_candidate_arrays(candidate, protocol, plan)
    validate_reference_arrays(references, protocol, plan)
    required_static = protocol["stages"][plan["stage"]][
        "include_static_references"
    ]
    if required_static:
        if static_references is None:
            raise ValueError("stage requires static references")
        validate_static_reference_arrays(static_references, protocol, plan)
    elif static_references is not None:
        raise ValueError("stage does not accept static references")
    groups = [references]
    if static_references is not None:
        groups.append(static_references)
    for values in groups:
        for name in COMMON_ARRAYS:
            if not np.array_equal(candidate[name], values[name]):
                raise ValueError(f"candidate and references differ in {name}")
    combined = {name: candidate[name] for name in COMMON_ARRAYS}
    for method in protocol["execution"]["candidate_methods"]:
        combined[f"errors_{method}"] = candidate[f"errors_{method}"]
        combined[f"crc_disagreement_{method}"] = candidate[
            f"crc_disagreement_{method}"
        ]
    for method in protocol["execution"]["accuracy_reference_methods"]:
        combined[f"errors_{method}"] = references[f"errors_{method}"]
        combined[f"crc_disagreement_{method}"] = references[
            f"crc_disagreement_{method}"
        ]
    if static_references is not None:
        for method in protocol["execution"]["static_reference_methods"]:
            combined[f"errors_{method}"] = static_references[
                f"errors_{method}"
            ]
            combined[f"crc_disagreement_{method}"] = static_references[
                f"crc_disagreement_{method}"
            ]
    latency = protocol["execution"]["latency"]
    for method in latency["candidate_methods"]:
        combined[f"latency_ms_{method}"] = candidate[
            f"latency_ms_{method}"
        ]
    for method in latency["reference_methods"]:
        combined[f"latency_ms_{method}"] = references[
            f"latency_ms_{method}"
        ]
    return combined


def compose_run_bundle(
    output_dir: Path | str,
    *,
    protocol: dict,
    plan: dict,
    candidate_output_path: Path | str,
    reference_cache_path: Path | str,
    static_reference_cache_path: Path | str | None = None,
) -> Path:
    candidate_payload, candidate_arrays = load_candidate_output(
        candidate_output_path, protocol, plan
    )
    reference_payload, reference_arrays = load_reference_cache(
        reference_cache_path, protocol, plan
    )
    if candidate_payload["training_seed"] != reference_payload["training_seed"]:
        raise ValueError("candidate and reference training seeds differ")
    static_payload = None
    static_arrays = None
    if static_reference_cache_path is not None:
        static_payload, static_arrays = load_static_reference_cache(
            static_reference_cache_path, protocol, plan
        )
    required_static = protocol["stages"][plan["stage"]][
        "include_static_references"
    ]
    if required_static != (static_payload is not None):
        raise ValueError("static reference cache requirement differs")
    candidate = candidate_payload["candidate"]
    baselines = dict(reference_payload["baselines"])
    if static_payload is not None:
        baselines.update(static_payload["baselines"])
    validate_matched_baseline(candidate, baselines["matched_cold_k8"])
    arrays = compose_run_arrays(
        candidate_arrays,
        reference_arrays,
        protocol,
        plan,
        static_arrays,
    )
    preflight = {
        **candidate_payload["preflight"],
        **reference_payload["preflight"],
    }
    if static_payload is not None:
        preflight.update(static_payload["preflight"])
    sources = {
        "candidate_output": {"sha256": file_sha256(candidate_output_path)},
        "reference_cache": {"sha256": file_sha256(reference_cache_path)},
    }
    if static_reference_cache_path is not None:
        sources["static_reference_cache"] = {
            "sha256": file_sha256(static_reference_cache_path)
        }
    return write_run_bundle(
        output_dir,
        protocol=protocol,
        plan=plan,
        candidate=candidate,
        baselines=baselines,
        training_seed=candidate_payload["training_seed"],
        arrays=arrays,
        preflight=preflight,
        elapsed_seconds=candidate_payload["elapsed_seconds"],
        source_artifacts=sources,
    )
