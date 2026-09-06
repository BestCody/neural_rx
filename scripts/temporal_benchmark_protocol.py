#!/usr/bin/env python3
"""Define the immutable Temporal Neural RX benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from temporal_generalization_protocol import seal_payload, verify_payload


METHODS = (
    "candidate_persistent",
    "candidate_reset",
    "matched_cold_k3",
    "matched_cold_k8",
    "historical_cold_k8",
    "perfect_csi_kbest",
)
CANDIDATE_METHODS = (
    "candidate_persistent",
    "candidate_reset",
)
ACCURACY_REFERENCE_METHODS = ("matched_cold_k8",)
LATENCY_ONLY_REFERENCE_METHODS = ("matched_cold_k3",)
STATIC_REFERENCE_METHODS = (
    "historical_cold_k8",
    "perfect_csi_kbest",
)
CORE_CONDITIONS = (
    "natural_umi",
    "natural_uma",
    "controlled_topology_change",
    "schedule_gap_reentry",
)
REPORT_CONDITIONS = ("historical_tdl_b100", "historical_tdl_c300")
STRESS_CONDITIONS = ("close_los", "near_far")
CONDITIONS = REPORT_CONDITIONS + CORE_CONDITIONS + STRESS_CONDITIONS
STAGES = ("screen", "development", "confirmation")
METRICS = (
    "mean_k8_gap",
    "worst_core_k8_gap",
    "memory_gain",
    "late_drift",
    "adaptation_excess_area",
    "episode_outage",
    "latency_p95",
)


def _condition(provider, role, offset, **settings):
    """Build one condition entry."""
    return {
        "provider": provider,
        "role": role,
        "seed_offset": int(offset),
        "receiver_inputs": [],
        **settings,
    }


def build_benchmark_protocol() -> dict:
    """Build benchmark v1."""
    conditions = {
        "historical_tdl_b100": _condition(
            "independent_tdl",
            "historical_report",
            0,
            profile="B100",
            doppler_hz=[0.0, 400.0],
            fixed_ues=True,
            pathloss=False,
            shadow_fading=False,
        ),
        "historical_tdl_c300": _condition(
            "independent_tdl",
            "historical_report",
            500,
            profile="C300",
            doppler_hz=[0.0, 100.0],
            fixed_ues=True,
            pathloss=False,
            shadow_fading=False,
        ),
        "natural_umi": _condition(
            "shared_geometry",
            "core",
            1_000,
            scenario="umi",
            force_los=None,
            angular_separation_deg=[0.0, 180.0],
            doppler_strata_hz=[[0.0, 50.0], [50.0, 175.0], [175.0, 400.0]],
            pathloss=False,
            shadow_fading=False,
        ),
        "natural_uma": _condition(
            "shared_geometry",
            "core",
            2_000,
            scenario="uma",
            force_los=None,
            angular_separation_deg=[0.0, 180.0],
            doppler_strata_hz=[[0.0, 50.0], [50.0, 175.0], [175.0, 400.0]],
            pathloss=False,
            shadow_fading=False,
        ),
        "controlled_topology_change": _condition(
            "shared_geometry_change",
            "core",
            3_000,
            scenario="umi",
            change_fraction=0.5,
            force_los=None,
            pathloss=False,
            shadow_fading=False,
        ),
        "schedule_gap_reentry": _condition(
            "shared_geometry_schedule",
            "core",
            4_000,
            scenario="umi",
            gap_start_probability=0.05,
            maximum_gap_tbs=4,
            ue_pool_size=4,
            schedule_switch_probability=0.25,
            schedule_reorder_probability=0.25,
            pathloss=False,
            shadow_fading=False,
        ),
        "close_los": _condition(
            "shared_geometry",
            "stress",
            5_000,
            scenario="umi",
            force_los=True,
            angular_separation_deg=[0.0, 5.0],
            doppler_strata_hz=[[0.0, 400.0]],
            pathloss=False,
            shadow_fading=False,
        ),
        "near_far": _condition(
            "shared_geometry_power",
            "stress",
            6_000,
            scenario="umi",
            force_los=None,
            received_power_gap_db=[6.0, 12.0],
            pathloss=False,
            shadow_fading=False,
            power_control="controlled_residual_gap",
        ),
    }
    stages = {
        "screen": {
            "seed_base": 41_000_000,
            "condition_names": list(CORE_CONDITIONS),
            "episodes_per_condition": 16,
            "episode_schedule": [16],
            "num_tbs": 16,
            "snr_points": ["anchor"],
            "training_updates": 500,
            "training_examples": 500,
            "training_recipe_seeds": [20262001],
            "required_training_seeds": 1,
            "scoring_window": [9, 16],
            "confirmation": False,
            "include_static_references": False,
        },
        "development": {
            "seed_base": 42_000_000,
            "condition_names": list(CORE_CONDITIONS),
            "episodes_per_condition": 64,
            "episode_schedule": [64],
            "num_tbs": 64,
            "snr_points": ["lower", "anchor", "upper"],
            "training_updates": 6_000,
            "training_examples": 6_000,
            "training_recipe_seeds": [20262002],
            "required_training_seeds": 1,
            "scoring_window": [17, 64],
            "confirmation": False,
            "include_static_references": False,
        },
        "confirmation": {
            "seed_base": 44_000_000,
            "condition_names": list(CONDITIONS),
            "episodes_per_condition": 64,
            "episode_schedule": [64, 128, 256],
            "num_tbs": 64,
            "snr_points": ["anchor"],
            "training_updates": 6_000,
            "training_examples": 6_000,
            "training_recipe_seeds": [20261001, 20261002, 20261003],
            "required_training_seeds": 3,
            "scoring_window": [17, 64],
            "confirmation": True,
            "locked": True,
            "include_static_references": True,
        },
    }
    return seal_payload({
        "schema_version": 1,
        "protocol": "temporal_nrx_benchmark_v1",
        "objective": {
            "candidate_iterations": 2,
            "target_iterations": 8,
            "target_tbler": 0.10,
            "claims": [
                "matched_k2_approaches_matched_k8",
                "persistent_memory_improves_same_checkpoint_reset",
                "memory_is_stable_and_recovers",
                "quality_justifies_runtime_and_storage",
            ],
        },
        "radio": {
            "receive_antennas": 4,
            "maximum_active_ues": 2,
            "bits_per_symbol": 4,
            "mcs_index": 14,
            "n_size_bwp": 132,
            "noise_axis": "ebno_db",
            "advancing_slot_pilots": True,
        },
        "methods": list(METHODS),
        "conditions": conditions,
        "condition_order": list(CONDITIONS),
        "core_conditions": list(CORE_CONDITIONS),
        "stationary_conditions": ["natural_umi", "natural_uma"],
        "report_conditions": list(REPORT_CONDITIONS),
        "stress_conditions": list(STRESS_CONDITIONS),
        "calibration": {
            "seed_base": 40_000_000,
            "episodes_per_condition": 64,
            "num_tbs": 16,
            "snr_grid_db": [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0,
            ],
            "baseline": "matched_cold_k8",
            "target_tbler": 0.10,
            "neighbor_offset_db": 1.0,
            "artifact_required": True,
            "training_updates": 6_000,
            "training_examples": 6_000,
            "training_recipe_seed": 20262000,
        },
        "stages": stages,
        "seed_domains": {
            "channel": 0,
            "payload": 100_000,
            "noise": 200_000,
            "schedule": 300_000,
        },
        "training": {
            "equal_examples_required": True,
            "matched_distribution_required": True,
            "matched_recipe_seeds_required": True,
            "fixed_checkpoint_selection_required": True,
            "matched_k8_control_required": True,
            "fixed_latency_k3_checkpoint_required": True,
        },
        "windows": {
            "warmup": [1, 8],
            "mature": [17, 32],
            "late": [49, 64],
            "pre_change": [25, 32],
            "immediate_post_change": [33, 40],
            "late_post_change": [57, 64],
        },
        "statistics": {
            "independent_unit": "episode",
            "bootstrap_samples": 10_000,
            "confidence_level": 0.95,
            "confirmation_resamples_training_seeds": True,
            "episode_outage_threshold": 0.50,
            "equal_core_condition_weight": True,
            "primary_snr_point": "anchor",
            "confirmation_ci_half_width": 0.01,
        },
        "ranking": {
            "primary_metric": "worst_core_k8_gap",
            "tie_break_order": [
                "mean_k8_gap",
                "memory_gain_descending",
                "latency_p95",
            ],
            "use_pareto_frontier": True,
            "stress_conditions_affect_ranking": False,
            "historical_condition_affects_ranking": False,
            "latency_reference": "matched_cold_k3",
        },
        "registration": {
            "one_structural_change_required": True,
            "required_fields": [
                "hypothesis",
                "expected_metric",
                "minimum_useful_improvement",
                "maximum_gpu_hours",
                "stop_rule",
            ],
            "allowed_metrics": list(METRICS),
        },
        "execution": {
            "observable_receiver_inputs": [
                "active",
                "ls_channel",
                "noise_variance",
                "received_grid",
                "slot_number",
                "ue_ids",
            ],
            "oracle_only_inputs": ["perfect_channel"],
            "candidate_methods": list(CANDIDATE_METHODS),
            "accuracy_reference_methods": list(ACCURACY_REFERENCE_METHODS),
            "latency_only_reference_methods": list(
                LATENCY_ONLY_REFERENCE_METHODS
            ),
            "static_reference_methods": list(STATIC_REFERENCE_METHODS),
            "paired_reference_cache_required": True,
            "accuracy_episode_batch_size": 8,
            "latency": {
                "candidate_methods": ["candidate_persistent"],
                "reference_methods": ["matched_cold_k3"],
                "condition": "natural_umi",
                "snr_point": "anchor",
                "warmup_samples": 16,
                "sample_count": 128,
                "batch_size": 1,
                "includes_synchronization": True,
            },
            "candidate_preflight": [
                "candidate_artifacts_unchanged",
                "candidate_crc_matches_payload_error",
                "candidate_deterministic_replay",
                "candidate_finite_outputs",
                "no_future_inputs",
                "persistent_and_reset_paired",
                "receiver_inputs_observable",
                "reset_state_is_empty",
            ],
            "matched_reference_preflight": [
                "matched_reference_artifacts_unchanged",
                "matched_reference_crc_matches_payload_error",
                "matched_reference_deterministic_replay",
                "matched_reference_finite_outputs",
                "matched_references_paired",
            ],
            "static_reference_preflight": [
                "kbest_oracle_isolated",
                "static_reference_artifacts_unchanged",
                "static_reference_crc_matches_payload_error",
                "static_reference_deterministic_replay",
                "static_reference_finite_outputs",
                "static_references_paired",
            ],
        },
        "versioning": {
            "silent_protocol_changes_forbidden": True,
            "cross_version_ranking_forbidden": True,
            "confirmation_reuse_after_redesign_forbidden": True,
        },
    }, field="protocol_sha256")


def _seed_intervals(protocol: dict):
    """Return all episode seed intervals."""
    intervals = []
    conditions = protocol["condition_order"]
    domains = protocol["seed_domains"]
    calibration = protocol["calibration"]
    phases = [("calibration", calibration)] + [
        (name, protocol["stages"][name]) for name in STAGES
    ]
    for phase, settings in phases:
        episodes = max(
            int(value)
            for value in settings.get(
                "episode_schedule",
                [settings["episodes_per_condition"]],
            )
        )
        phase_conditions = settings.get("condition_names", conditions)
        for condition in phase_conditions:
            offset = int(protocol["conditions"][condition]["seed_offset"])
            for domain, domain_offset in domains.items():
                start = int(settings["seed_base"]) + offset + int(domain_offset)
                intervals.append((start, start + episodes, phase, condition, domain))
    return intervals


def validate_benchmark_protocol(protocol: dict) -> None:
    """Validate benchmark v1."""
    verify_payload(protocol, field="protocol_sha256")
    if protocol != build_benchmark_protocol():
        raise ValueError("benchmark protocol differs from its recipe")
    if tuple(protocol["methods"]) != METHODS:
        raise ValueError("benchmark methods differ")
    if tuple(protocol["condition_order"]) != CONDITIONS:
        raise ValueError("benchmark condition order differs")
    if set(protocol["core_conditions"]) & set(protocol["stress_conditions"]):
        raise ValueError("core and stress conditions overlap")
    if not set(protocol["stationary_conditions"]).issubset(
        protocol["core_conditions"]
    ):
        raise ValueError("stationary conditions must be core conditions")
    roles = (
        set(protocol["core_conditions"])
        | set(protocol["report_conditions"])
        | set(protocol["stress_conditions"])
    )
    if roles != set(CONDITIONS):
        raise ValueError("condition roles do not cover the scenario bank")
    if protocol["stages"]["confirmation"].get("locked") is not True:
        raise ValueError("confirmation must remain locked")
    for stage in STAGES:
        settings = protocol["stages"][stage]
        points = settings["snr_points"]
        if points.count("anchor") != 1:
            raise ValueError(f"{stage} must contain one anchor SNR")
        names = settings.get("condition_names")
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{stage} condition names differ")
        if not set(names).issubset(CONDITIONS):
            raise ValueError(f"{stage} contains an unknown condition")
        schedule = settings.get("episode_schedule")
        if not schedule or schedule != sorted(set(schedule)):
            raise ValueError(f"{stage} episode schedule differs")
        if any(int(value) <= 0 for value in schedule):
            raise ValueError(f"{stage} episode schedule is invalid")
        if settings["episodes_per_condition"] != schedule[0]:
            raise ValueError(f"{stage} must start at its smallest episode count")
    for stage in ("screen", "development"):
        settings = protocol["stages"][stage]
        if settings["condition_names"] != list(CORE_CONDITIONS):
            raise ValueError(f"{stage} must contain only core conditions")
        if settings["include_static_references"]:
            raise ValueError(f"{stage} cannot include static references")
    confirmation = protocol["stages"]["confirmation"]
    if confirmation["condition_names"] != list(CONDITIONS):
        raise ValueError("confirmation must cover every condition")
    if not confirmation["include_static_references"]:
        raise ValueError("confirmation must include report references")
    execution = protocol["execution"]
    methods = (
        execution["candidate_methods"]
        + execution["latency_only_reference_methods"]
        + execution["accuracy_reference_methods"]
        + execution["static_reference_methods"]
    )
    if methods != list(METHODS):
        raise ValueError("execution methods differ from the method contract")
    timing = execution["latency"]
    if not set(timing["candidate_methods"]).issubset(
        execution["candidate_methods"]
    ):
        raise ValueError("candidate latency methods differ")
    if not set(timing["reference_methods"]).issubset(
        execution["latency_only_reference_methods"]
    ):
        raise ValueError("reference latency methods differ")
    if protocol["ranking"]["latency_reference"] not in timing[
        "reference_methods"
    ]:
        raise ValueError("ranking latency reference is not measured")
    if timing["condition"] not in CORE_CONDITIONS:
        raise ValueError("latency condition must be a core condition")
    if timing["snr_point"] != "anchor":
        raise ValueError("latency must use the anchor SNR")
    if int(timing["sample_count"]) < 10:
        raise ValueError("latency sample count is too small")
    if int(timing["warmup_samples"]) < 1:
        raise ValueError("latency warmup is too small")
    if int(timing["batch_size"]) != 1:
        raise ValueError("latency must measure one episode at a time")
    if int(execution["accuracy_episode_batch_size"]) < 1:
        raise ValueError("accuracy batch size must be positive")
    required_preflight = (
        execution["candidate_preflight"]
        + execution["matched_reference_preflight"]
        + execution["static_reference_preflight"]
    )
    if len(required_preflight) != len(set(required_preflight)):
        raise ValueError("required preflight checks must be unique")
    intervals = sorted(_seed_intervals(protocol))
    for first, second in zip(intervals, intervals[1:]):
        if second[0] < first[1]:
            raise ValueError(
                f"seed intervals overlap: {first[2:]} and {second[2:]}"
            )
    for condition in protocol["conditions"].values():
        if condition.get("receiver_inputs") != []:
            raise ValueError("scenario labels cannot enter the receiver")


def load_benchmark_protocol(path: Path | str) -> dict:
    """Load benchmark v1."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_benchmark_protocol(protocol)
    return protocol


def resolved_condition_settings(
    protocol: dict, condition: str, num_tbs: int
) -> dict:
    """Resolve horizon-dependent scenario settings."""
    if condition not in CONDITIONS:
        raise ValueError("unknown benchmark condition")
    settings = dict(protocol["conditions"][condition])
    if settings["provider"] == "shared_geometry_change":
        fraction = float(settings["change_fraction"])
        if not 0.0 < fraction < 1.0:
            raise ValueError("controlled change fraction is invalid")
        settings["change_tb"] = int(int(num_tbs) * fraction) + 1
    return settings


def stage_seed_bundle(
    protocol: dict,
    stage: str,
    condition: str,
    episode_offset: int = 0,
) -> dict[str, int]:
    """Return one stage's deterministic seed domains."""
    if stage not in STAGES:
        raise ValueError("unknown benchmark stage")
    if condition not in CONDITIONS:
        raise ValueError("unknown benchmark condition")
    settings = protocol["stages"][stage]
    count = max(int(value) for value in settings["episode_schedule"])
    offset = int(episode_offset)
    if not 0 <= offset < count:
        raise ValueError("episode offset is outside the stage")
    base = (
        int(settings["seed_base"])
        + int(protocol["conditions"][condition]["seed_offset"])
        + offset
    )
    return {
        name: base + int(value)
        for name, value in protocol["seed_domains"].items()
    }


def stage_condition_names(protocol: dict, stage: str) -> tuple[str, ...]:
    """Return the conditions evaluated by one stage."""
    if stage not in STAGES:
        raise ValueError("unknown benchmark stage")
    return tuple(protocol["stages"][stage]["condition_names"])


def stage_method_names(protocol: dict, stage: str) -> tuple[str, ...]:
    """Return every method required by one stage."""
    if stage not in STAGES:
        raise ValueError("unknown benchmark stage")
    execution = protocol["execution"]
    methods = (
        list(execution["candidate_methods"])
        + list(execution["accuracy_reference_methods"])
    )
    if protocol["stages"][stage]["include_static_references"]:
        methods += list(execution["static_reference_methods"])
    return tuple(methods)


def stage_reference_method_names(
    protocol: dict, stage: str
) -> tuple[str, ...]:
    """Return reference methods required by one stage."""
    execution = protocol["execution"]
    methods = (
        list(execution["latency_only_reference_methods"])
        + list(execution["accuracy_reference_methods"])
    )
    if protocol["stages"][stage]["include_static_references"]:
        methods += list(protocol["execution"]["static_reference_methods"])
    return tuple(methods)


def stage_accuracy_reference_method_names(
    protocol: dict, stage: str
) -> tuple[str, ...]:
    """Return reference methods with accuracy measurements."""
    methods = list(protocol["execution"]["accuracy_reference_methods"])
    if protocol["stages"][stage]["include_static_references"]:
        methods += list(protocol["execution"]["static_reference_methods"])
    return tuple(methods)


def calibration_seed_bundle(
    protocol: dict,
    condition: str,
    episode_offset: int = 0,
) -> dict[str, int]:
    """Return deterministic calibration seed domains."""
    if condition not in CONDITIONS:
        raise ValueError("unknown benchmark condition")
    settings = protocol["calibration"]
    offset = int(episode_offset)
    count = int(settings["episodes_per_condition"])
    if not 0 <= offset < count:
        raise ValueError("episode offset is outside calibration")
    base = (
        int(settings["seed_base"])
        + int(protocol["conditions"][condition]["seed_offset"])
        + offset
    )
    return {
        name: base + int(value)
        for name, value in protocol["seed_domains"].items()
    }


if __name__ == "__main__":
    print(json.dumps(build_benchmark_protocol(), indent=2))
