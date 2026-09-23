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
from temporal_benchmark_protocol import (
    METHODS,
    stage_method_names,
    stage_reference_method_names,
)
from temporal_generalization_protocol import seal_payload, verify_payload


COMMON_ARRAYS = (
    "snr_db",
    "episode_index",
    "episode_fingerprint",
    "active",
    "ue_ids",
    "channel_change_event",
)


def validate_latency_arrays(
    arrays: dict[str, np.ndarray], methods: list[str] | tuple[str, ...],
    protocol: dict, plan: dict,
) -> None:
    timing = protocol["execution"]["latency"]
    expected = int(timing["sample_count"])
    for method in methods:
        name = f"latency_ms_{method}"
        values = arrays.get(name)
        if values is None or values.ndim != 1 or len(values) != expected:
            raise ValueError(f"{method} latency sample count differs")
        if np.any(np.asarray(values) <= 0.0):
            raise ValueError(f"{method} latency must be positive")


def required_run_preflight(protocol: dict, stage: str) -> set[str]:
    execution = protocol["execution"]
    required = (
        execution["candidate_preflight"]
        + execution["matched_reference_preflight"]
    )
    if protocol["stages"][stage]["include_static_references"]:
        required += execution["static_reference_preflight"]
    return set(required)


def expected_array_shapes(protocol: dict, plan: dict) -> dict[str, tuple]:
    stage = plan["stage"]
    settings = protocol["stages"][stage]
    conditions = len(plan["jobs"])
    episodes = int(plan["episodes_per_condition"])
    tbs = int(settings["num_tbs"])
    users = int(protocol["radio"]["maximum_active_ues"])
    snrs = len(plan["snr_point_names"])
    error_shape = (conditions, snrs, episodes, tbs, users)
    shapes = {
        "snr_db": (conditions, snrs),
        "episode_index": (conditions, episodes),
        "episode_fingerprint": (conditions, episodes, 32),
        "active": (conditions, episodes, tbs, users),
        "ue_ids": (conditions, episodes, tbs, users),
        "channel_change_event": (conditions, episodes, tbs, users),
    }
    for method in stage_method_names(protocol, stage):
        shapes[f"errors_{method}"] = error_shape
        shapes[f"crc_disagreement_{method}"] = error_shape
    return shapes


def validate_common_arrays(
    arrays: dict[str, np.ndarray], protocol: dict, plan: dict
) -> None:
    shapes = expected_array_shapes(protocol, plan)
    for name in COMMON_ARRAYS:
        if name not in arrays or arrays[name].shape != shapes[name]:
            raise ValueError(f"{name} shape differs")
    expected_snr = np.asarray(
        [job["snr_db"] for job in plan["jobs"]], np.float64
    )
    if not np.array_equal(
        np.asarray(arrays["snr_db"], np.float64), expected_snr
    ):
        raise ValueError("run SNR values differ from the stage plan")
    episodes = int(plan["episodes_per_condition"])
    expected_episode = np.tile(
        np.arange(episodes, dtype=np.int64), (len(plan["jobs"]), 1)
    )
    if not np.array_equal(arrays["episode_index"], expected_episode):
        raise ValueError("episode indices must be contiguous and zero based")
    if arrays["episode_fingerprint"].dtype != np.uint8:
        raise ValueError("episode fingerprints must be uint8")
    fingerprints = [
        row.tobytes()
        for condition in arrays["episode_fingerprint"]
        for row in condition
    ]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("episode fingerprints must be globally unique")
    if arrays["active"].dtype != np.bool_:
        raise ValueError("activity masks must be boolean")
    if not np.all(np.any(arrays["active"], axis=-1)):
        raise ValueError("every TB must have an active UE")
    if not np.issubdtype(arrays["ue_ids"].dtype, np.integer):
        raise ValueError("UE identities must be integers")
    if np.any(arrays["ue_ids"] < 0):
        raise ValueError("UE identities cannot be negative")
    if np.any(np.diff(np.sort(arrays["ue_ids"], axis=-1), axis=-1) == 0):
        raise ValueError("scheduled UE identities must be unique")
    if arrays["channel_change_event"].dtype != np.bool_:
        raise ValueError("channel-change events must be boolean")
    validate_scenario_metadata(
        arrays["active"],
        arrays["ue_ids"],
        arrays["channel_change_event"],
        plan["jobs"],
    )


def validate_scenario_metadata(
    active: np.ndarray,
    ue_ids: np.ndarray,
    change_event: np.ndarray,
    jobs: list[dict],
) -> None:
    for condition, job in enumerate(jobs):
        provider = job["provider"]
        point_active = active[condition]
        point_ids = ue_ids[condition]
        point_change = change_event[condition]
        if provider == "shared_geometry_change":
            expected = np.zeros_like(point_change)
            expected[:, int(job["provider_settings"]["change_tb"]) - 1] = True
            if not np.array_equal(point_change, expected):
                raise ValueError("controlled channel-change events differ")
        elif np.any(point_change):
            raise ValueError("unexpected channel-change event")
        if provider == "shared_geometry_schedule":
            if np.all(point_active):
                raise ValueError("schedule condition contains no gap")
            if not np.any(point_ids[:, 1:] != point_ids[:, :-1]):
                raise ValueError("schedule condition contains no UE transition")
        else:
            if not np.all(point_active):
                raise ValueError("non-scheduling condition contains a gap")
            expected_ids = np.arange(point_ids.shape[-1], dtype=point_ids.dtype)
            if not np.all(point_ids == expected_ids):
                raise ValueError("fixed-UE condition changed identities")


def validate_method_arrays(
    arrays: dict[str, np.ndarray],
    methods: list[str] | tuple[str, ...],
    protocol: dict,
    plan: dict,
) -> None:
    shapes = expected_array_shapes(protocol, plan)
    active = arrays["active"]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown benchmark method {method}")
        error_name = f"errors_{method}"
        crc_name = f"crc_disagreement_{method}"
        for name in (error_name, crc_name):
            if name not in arrays or arrays[name].shape != shapes[name]:
                raise ValueError(f"{name} shape differs")
            if arrays[name].dtype != np.bool_:
                raise ValueError(f"{name} must be boolean")
        inactive = np.logical_not(active[:, None])
        if np.any(arrays[error_name] & inactive):
            raise ValueError(f"{method} records errors for inactive UEs")
        if np.any(arrays[crc_name]):
            raise ValueError(f"{method} CRC disagrees with payload")


def validate_run_manifests(
    candidate: dict,
    baselines: dict[str, dict],
    protocol: dict,
    stage: str,
) -> None:
    validate_candidate_manifest(candidate, protocol)
    validate_stage_training(candidate, protocol, stage)
    expected = set(stage_reference_method_names(protocol, stage))
    if set(baselines) != expected:
        raise ValueError("benchmark baseline manifests differ")
    for manifest in baselines.values():
        validate_candidate_manifest(manifest, protocol)
    validate_stage_training(baselines["matched_cold_k8"], protocol, stage)
    if baselines["matched_cold_k3"]["role"] != "matched_cold_k3":
        raise ValueError("matched Cold K=3 role differs")
    if baselines["matched_cold_k8"]["role"] != "matched_cold_k8":
        raise ValueError("matched Cold K=8 role differs")
    if "historical_cold_k8" in baselines and baselines[
        "historical_cold_k8"
    ]["role"] != "historical_cold_k8":
        raise ValueError("historical Cold K=8 role differs")
    if "perfect_csi_kbest" in baselines and baselines[
        "perfect_csi_kbest"
    ]["role"] != "perfect_csi_kbest":
        raise ValueError("perfect-CSI K-Best role differs")
    validate_matched_baseline(candidate, baselines["matched_cold_k8"])


def validate_run_arrays(
    arrays: dict[str, np.ndarray],
    protocol: dict,
    plan: dict,
) -> None:
    condition_names = [job["condition"] for job in plan["jobs"]]
    if condition_names != list(
        protocol["stages"][plan["stage"]]["condition_names"]
    ):
        raise ValueError("run condition order differs")
    validate_common_arrays(arrays, protocol, plan)
    validate_method_arrays(
        arrays, stage_method_names(protocol, plan["stage"]), protocol, plan
    )
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    latency = protocol["execution"]["latency"]
    validate_latency_arrays(
        arrays,
        tuple(latency["candidate_methods"] + latency["reference_methods"]),
        protocol,
        plan,
    )


def write_run_bundle(
    output_dir: Path | str,
    *,
    protocol: dict,
    plan: dict,
    candidate: dict,
    baselines: dict[str, dict],
    training_seed: int,
    arrays: dict[str, np.ndarray],
    preflight: dict[str, bool],
    elapsed_seconds: float,
    source_artifacts: dict[str, dict] | None = None,
) -> Path:
    stage = plan["stage"]
    validate_run_manifests(candidate, baselines, protocol, stage)
    if candidate["manifest_sha256"] != plan["candidate_manifest_sha256"]:
        raise ValueError("candidate manifest differs from the stage plan")
    for name, digest in plan["baseline_manifests"].items():
        if baselines[name]["manifest_sha256"] != digest:
            raise ValueError(f"{name} manifest differs from the stage plan")
    if int(training_seed) not in candidate["training"]["seeds"]:
        raise ValueError("run seed is absent from the candidate manifest")
    required_preflight = required_run_preflight(protocol, stage)
    if not required_preflight.issubset(preflight):
        raise ValueError("benchmark preflight is incomplete")
    if not preflight or not all(value is True for value in preflight.values()):
        raise ValueError("benchmark preflight did not pass")
    if not np.isfinite(elapsed_seconds) or float(elapsed_seconds) <= 0.0:
        raise ValueError("benchmark elapsed time must be positive")
    sources = {} if source_artifacts is None else source_artifacts
    for name, record in sources.items():
        digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("benchmark source artifact is invalid")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("benchmark source artifact digest is invalid")
    conditions = [job["condition"] for job in plan["jobs"]]
    validate_run_arrays(arrays, protocol, plan)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "benchmark_metrics.npz"
    np.savez_compressed(raw_path, **arrays)
    payload = seal_payload({
        "schema_version": 1,
        "run": "temporal_benchmark_run_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "stage": stage,
        "training_seed": int(training_seed),
        "condition_names": conditions,
        "candidate": candidate,
        "baselines": baselines,
        "preflight": preflight,
        "elapsed_seconds": float(elapsed_seconds),
        "source_artifacts": sources,
        "raw_metrics": {
            "file": raw_path.name,
            "sha256": file_sha256(raw_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        },
    }, field="run_sha256")
    path = destination / "benchmark_run.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_run_bundle(
    path: Path | str,
    protocol: dict,
    plan: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_payload(payload, field="run_sha256")
    if payload.get("run") != "temporal_benchmark_run_v1":
        raise ValueError("benchmark run type differs")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("benchmark run protocol differs")
    if payload.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("benchmark run plan differs")
    if payload.get("stage") != plan["stage"]:
        raise ValueError("benchmark run stage differs")
    if not np.isfinite(payload.get("elapsed_seconds", np.nan)) or float(
        payload["elapsed_seconds"]
    ) <= 0.0:
        raise ValueError("benchmark run elapsed time differs")
    validate_run_manifests(
        payload["candidate"], payload["baselines"], protocol, plan["stage"]
    )
    if payload["candidate"]["manifest_sha256"] != plan[
        "candidate_manifest_sha256"
    ]:
        raise ValueError("run candidate differs from the plan")
    for name, digest in plan["baseline_manifests"].items():
        if payload["baselines"][name]["manifest_sha256"] != digest:
            raise ValueError(f"run {name} differs from the plan")
    if payload["training_seed"] not in payload["candidate"]["training"][
        "seeds"
    ]:
        raise ValueError("run training seed differs")
    required_preflight = required_run_preflight(protocol, plan["stage"])
    if not required_preflight.issubset(payload.get("preflight", {})):
        raise ValueError("benchmark run preflight is incomplete")
    if not payload.get("preflight") or not all(
        value is True for value in payload["preflight"].values()
    ):
        raise ValueError("benchmark run preflight differs")
    raw_path = path.parent / payload["raw_metrics"]["file"]
    if file_sha256(raw_path) != payload["raw_metrics"]["sha256"]:
        raise ValueError("benchmark raw metrics checksum differs")
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    descriptors = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    if payload["raw_metrics"].get("arrays") != descriptors:
        raise ValueError("benchmark raw array descriptors differ")
    if payload["condition_names"] != [
        job["condition"] for job in plan["jobs"]
    ]:
        raise ValueError("benchmark run condition names differ")
    validate_run_arrays(arrays, protocol, plan)
    return payload, arrays
