from __future__ import annotations

import numpy as np

from temporal_benchmark_protocol import benchmark_method_names


COMMON_ARRAYS = (
    "snr_db",
    "episode_index",
    "episode_fingerprint",
    "active",
    "ue_ids",
    "channel_change_event",
)

DECODED_LATENCY_DEFINITION = {
    "scope": "prepared_full_grid_ls_to_decoded_payload",
    "timer": "host_wall_clock_with_completion_synchronization",
    "primary_statistic": "p95",
    "includes": ["adapter_input_transfers", "candidate_memory", "detection",
                 "ldpc", "crc", "output_transfers"],
    "excludes": ["initial_ls_estimation", "initial_channel_interpolation"],
    "original_paper_detector_timing": False,
}


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
    for method in benchmark_method_names(protocol):
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
        allowed = protocol["methods"]
        if method not in allowed:
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
        if np.any(arrays[crc_name] & inactive):
            raise ValueError(f"{method} records CRC disagreements for inactive UEs")
