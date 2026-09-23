#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import numpy as np

from temporal_benchmark_protocol import (
    CONDITIONS,
    STAGES,
    resolved_condition_settings,
    stage_condition_names,
    stage_reference_method_names,
    stage_seed_bundle,
)
from temporal_generalization_protocol import seal_payload, verify_payload


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 digest")


def corrected_rate(errors: int, blocks: int) -> float:
    if int(blocks) <= 0 or not 0 <= int(errors) <= int(blocks):
        raise ValueError("invalid error count")
    return (float(errors) + 0.5) / (float(blocks) + 1.0)


def decreasing_isotonic(values, weights) -> np.ndarray:
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("isotonic inputs must be matching vectors")
    if len(values) == 0 or np.any(weights <= 0.0):
        raise ValueError("isotonic weights must be positive")
    levels = []
    masses = []
    lengths = []
    for value, weight in zip(values, weights):
        levels.append(float(value))
        masses.append(float(weight))
        lengths.append(1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            mass = masses[-2] + masses[-1]
            level = (
                levels[-2] * masses[-2] + levels[-1] * masses[-1]
            ) / mass
            length = lengths[-2] + lengths[-1]
            levels[-2:] = [level]
            masses[-2:] = [mass]
            lengths[-2:] = [length]
    return np.concatenate([
        np.full(length, level, np.float64)
        for level, length in zip(levels, lengths)
    ])


def calibrate_condition(points: list[dict], target: float) -> dict:
    if len(points) < 3:
        raise ValueError("calibration requires at least three SNR points")
    points = sorted(points, key=lambda value: float(value["snr_db"]))
    snrs = np.asarray([float(value["snr_db"]) for value in points])
    if len(np.unique(snrs)) != len(snrs):
        raise ValueError("calibration SNR values must be unique")
    errors = np.asarray([int(value["errors"]) for value in points])
    blocks = np.asarray([int(value["blocks"]) for value in points])
    raw = np.asarray([
        corrected_rate(error, block)
        for error, block in zip(errors, blocks)
    ])
    fitted = decreasing_isotonic(raw, blocks)
    eligible = np.arange(1, len(snrs) - 1)
    distance = np.abs(np.log(np.maximum(fitted[eligible], 1e-12) / target))
    anchor_index = int(eligible[np.argmin(distance)])
    crossing = None
    for index in range(len(snrs) - 1):
        high = fitted[index]
        low = fitted[index + 1]
        if high >= target >= low:
            if high == low:
                crossing = float((snrs[index] + snrs[index + 1]) / 2.0)
            else:
                fraction = (
                    math.log(target) - math.log(high)
                ) / (math.log(low) - math.log(high))
                crossing = float(
                    snrs[index] + fraction * (snrs[index + 1] - snrs[index])
                )
            break
    return {
        "lower_snr_db": float(snrs[anchor_index - 1]),
        "anchor_snr_db": float(snrs[anchor_index]),
        "upper_snr_db": float(snrs[anchor_index + 1]),
        "anchor_index": anchor_index,
        "target_bracketed": crossing is not None,
        "snr_at_target_db": crossing,
        "raw_tbler": raw.tolist(),
        "isotonic_tbler": fitted.tolist(),
        "points": copy.deepcopy(points),
    }


def build_calibration_artifact(
    protocol: dict,
    baseline_result_sha256: str,
    baseline_manifest_sha256: str,
    curves: dict[str, list[dict]],
) -> dict:
    _require_digest(baseline_result_sha256, "baseline result")
    _require_digest(baseline_manifest_sha256, "baseline manifest")
    if set(curves) != set(CONDITIONS):
        raise ValueError("calibration condition coverage differs")
    expected_grid = protocol["calibration"]["snr_grid_db"]
    conditions = {}
    minimum_blocks = (
        int(protocol["calibration"]["episodes_per_condition"])
        * int(protocol["calibration"]["num_tbs"])
    )
    maximum_blocks = (
        minimum_blocks * int(protocol["radio"]["maximum_active_ues"])
    )
    for name in CONDITIONS:
        for point in curves[name]:
            blocks = int(point["blocks"])
            errors = int(point["errors"])
            if not minimum_blocks <= blocks <= maximum_blocks:
                raise ValueError(f"{name} calibration block count differs")
            if not 0 <= errors <= blocks:
                raise ValueError(f"{name} calibration errors are invalid")
        actual_grid = sorted(float(point["snr_db"]) for point in curves[name])
        if actual_grid != expected_grid:
            raise ValueError(f"{name} calibration grid differs")
        conditions[name] = calibrate_condition(
            curves[name], float(protocol["calibration"]["target_tbler"])
        )
        anchor = conditions[name]["anchor_index"]
        offset = float(protocol["calibration"]["neighbor_offset_db"])
        if not math.isclose(
            actual_grid[anchor] - actual_grid[anchor - 1], offset
        ) or not math.isclose(
            actual_grid[anchor + 1] - actual_grid[anchor], offset
        ):
            raise ValueError(f"{name} calibration neighbors differ")
        role = protocol["conditions"][name]["role"]
        if role != "stress" and not conditions[name]["target_bracketed"]:
            raise ValueError(f"{name} does not bracket target TBLER")
    return seal_payload({
        "schema_version": 1,
        "artifact": "temporal_benchmark_snr_calibration_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "baseline_method": protocol["calibration"]["baseline"],
        "baseline_result_sha256": baseline_result_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "target_tbler": protocol["calibration"]["target_tbler"],
        "conditions": conditions,
    }, field="calibration_sha256")


def validate_calibration_artifact(artifact: dict, protocol: dict) -> None:
    verify_payload(artifact, field="calibration_sha256")
    if artifact.get("artifact") != "temporal_benchmark_snr_calibration_v1":
        raise ValueError("calibration artifact type differs")
    if artifact.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("calibration benchmark version differs")
    if artifact.get("baseline_method") != "matched_cold_k8":
        raise ValueError("calibration did not use matched Cold K=8")
    curves = {
        name: value["points"]
        for name, value in artifact.get("conditions", {}).items()
    }
    expected = build_calibration_artifact(
        protocol,
        artifact["baseline_result_sha256"],
        artifact["baseline_manifest_sha256"],
        curves,
    )
    if artifact != expected:
        raise ValueError("calibration artifact differs from its recipe")


def load_calibration_artifact(path: Path | str, protocol: dict) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_calibration_artifact(artifact, protocol)
    return artifact


def build_confirmation_authorization(
    protocol: dict,
    calibration: dict,
    candidate_manifest_sha256: str,
    design_freeze_sha256: str,
) -> dict:
    _require_digest(candidate_manifest_sha256, "candidate manifest")
    _require_digest(design_freeze_sha256, "design freeze")
    return seal_payload({
        "schema_version": 1,
        "artifact": "temporal_benchmark_confirmation_authorization_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "design_freeze_sha256": design_freeze_sha256,
        "authorized": True,
    }, field="authorization_sha256")


def validate_confirmation_authorization(
    artifact: dict,
    protocol: dict,
    calibration: dict,
    candidate_manifest_sha256: str,
) -> None:
    verify_payload(artifact, field="authorization_sha256")
    _require_digest(artifact.get("design_freeze_sha256"), "design freeze")
    expected = {
        "artifact": "temporal_benchmark_confirmation_authorization_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "authorized": True,
    }
    for name, value in expected.items():
        if artifact.get(name) != value:
            raise ValueError(f"confirmation authorization differs in {name}")


def build_stage_plan(
    protocol: dict,
    calibration: dict,
    stage: str,
    candidate_manifest_sha256: str,
    matched_k3_manifest_sha256: str,
    matched_k8_manifest_sha256: str,
    historical_k8_manifest_sha256: str | None = None,
    perfect_csi_kbest_manifest_sha256: str | None = None,
    authorization: dict | None = None,
    episodes_per_condition: int | None = None,
) -> dict:
    validate_calibration_artifact(calibration, protocol)
    if stage not in STAGES:
        raise ValueError("unknown benchmark stage")
    baseline_digests = {
        "matched_cold_k3": matched_k3_manifest_sha256,
        "matched_cold_k8": matched_k8_manifest_sha256,
        "historical_cold_k8": historical_k8_manifest_sha256,
        "perfect_csi_kbest": perfect_csi_kbest_manifest_sha256,
    }
    labels = {
        "matched_cold_k3": "matched Cold K=3 manifest",
        "matched_cold_k8": "matched Cold K=8 manifest",
        "historical_cold_k8": "historical Cold K=8 manifest",
        "perfect_csi_kbest": "perfect-CSI K-Best manifest",
    }
    for name in stage_reference_method_names(protocol, stage):
        _require_digest(baseline_digests[name], labels[name])
    for label, digest in (
        ("candidate manifest", candidate_manifest_sha256),
    ):
        _require_digest(digest, label)
    settings = protocol["stages"][stage]
    allowed_episodes = [int(value) for value in settings["episode_schedule"]]
    selected_episodes = (
        int(settings["episodes_per_condition"])
        if episodes_per_condition is None
        else int(episodes_per_condition)
    )
    if selected_episodes not in allowed_episodes:
        raise ValueError("episode count is outside the stage schedule")
    if settings["confirmation"]:
        if authorization is None:
            raise PermissionError("confirmation remains locked")
        validate_confirmation_authorization(
            authorization,
            protocol,
            calibration,
            candidate_manifest_sha256,
        )
    point_names = settings["snr_points"]
    jobs = []
    for condition in stage_condition_names(protocol, stage):
        calibrated = calibration["conditions"][condition]
        provider_settings = resolved_condition_settings(
            protocol, condition, int(settings["num_tbs"])
        )
        snr_lookup = {
            "lower": calibrated["lower_snr_db"],
            "anchor": calibrated["anchor_snr_db"],
            "upper": calibrated["upper_snr_db"],
        }
        jobs.append({
            "condition": condition,
            "provider": protocol["conditions"][condition]["provider"],
            "provider_settings": provider_settings,
            "episode_count": selected_episodes,
            "num_tbs": settings["num_tbs"],
            "snr_db": [snr_lookup[name] for name in point_names],
            "seed_streams": stage_seed_bundle(
                protocol, stage, condition, 0
            ),
        })
    return seal_payload({
        "schema_version": 1,
        "plan": "temporal_benchmark_stage_plan_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "stage": stage,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "baseline_manifests": {
            name: baseline_digests[name]
            for name in stage_reference_method_names(protocol, stage)
        },
        "episodes_per_condition": selected_episodes,
        "snr_point_names": list(settings["snr_points"]),
        "scoring_window": settings["scoring_window"],
        "jobs": jobs,
        "confirmation_authorization_sha256": (
            None if authorization is None else authorization[
                "authorization_sha256"
            ]
        ),
    }, field="plan_sha256")


def validate_stage_plan(plan: dict, protocol: dict, calibration: dict) -> None:
    verify_payload(plan, field="plan_sha256")
    validate_calibration_artifact(calibration, protocol)
    if plan.get("plan") != "temporal_benchmark_stage_plan_v1":
        raise ValueError("benchmark plan type differs")
    if plan.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("benchmark plan protocol differs")
    if plan.get("calibration_sha256") != calibration["calibration_sha256"]:
        raise ValueError("benchmark plan calibration differs")
    _require_digest(plan.get("candidate_manifest_sha256"), "candidate manifest")
    for name, digest in plan.get("baseline_manifests", {}).items():
        _require_digest(digest, f"{name} manifest")
    stage = plan.get("stage")
    if stage not in STAGES:
        raise ValueError("benchmark plan stage differs")
    expected_references = set(stage_reference_method_names(protocol, stage))
    if set(plan.get("baseline_manifests", {})) != expected_references:
        raise ValueError("benchmark plan baseline manifests differ")
    settings = protocol["stages"][stage]
    episodes = plan.get("episodes_per_condition")
    if episodes not in settings["episode_schedule"]:
        raise ValueError("benchmark plan episode count differs")
    if plan.get("snr_point_names") != settings["snr_points"]:
        raise ValueError("benchmark plan SNR names differ")
    if plan.get("scoring_window") != settings["scoring_window"]:
        raise ValueError("benchmark plan scoring window differs")
    if settings["confirmation"]:
        if not plan.get("confirmation_authorization_sha256"):
            raise ValueError("confirmation plan lacks authorization")
    elif plan.get("confirmation_authorization_sha256") is not None:
        raise ValueError("development plan has confirmation authorization")
    jobs = plan.get("jobs", [])
    if [job.get("condition") for job in jobs] != list(
        stage_condition_names(protocol, stage)
    ):
        raise ValueError("benchmark plan condition order differs")
    for job in jobs:
        name = job["condition"]
        expected_condition = resolved_condition_settings(
            protocol, name, int(settings["num_tbs"])
        )
        if job.get("provider") != expected_condition["provider"]:
            raise ValueError(f"{name} provider differs")
        if job.get("provider_settings") != expected_condition:
            raise ValueError(f"{name} provider settings differ")
        if job.get("episode_count") != episodes:
            raise ValueError(f"{name} episode count differs")
        if job.get("num_tbs") != settings["num_tbs"]:
            raise ValueError(f"{name} TB horizon differs")
        calibrated = calibration["conditions"][name]
        lookup = {
            "lower": calibrated["lower_snr_db"],
            "anchor": calibrated["anchor_snr_db"],
            "upper": calibrated["upper_snr_db"],
        }
        expected_snr = [lookup[key] for key in settings["snr_points"]]
        if job.get("snr_db") != expected_snr:
            raise ValueError(f"{name} SNR points differ")
        if job.get("seed_streams") != stage_seed_bundle(
            protocol, stage, name, 0
        ):
            raise ValueError(f"{name} seed streams differ")


def load_stage_plan(
    path: Path | str,
    protocol: dict,
    calibration: dict,
) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_stage_plan(plan, protocol, calibration)
    return plan
