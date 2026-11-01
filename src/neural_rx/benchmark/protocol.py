from __future__ import annotations

import json
import math
from pathlib import Path

from neural_rx.benchmark.generalization_protocol import verify_payload
from neural_rx.benchmark.scoring import TBLER_DEFINITION


METHOD_ROLES = {
    "candidate_persistent": "candidate",
    "candidate_reset": "candidate",
    "frozen_cold_k2": "frozen_cold_k2",
    "frozen_cold_k3": "frozen_cold_k3",
    "frozen_cold_k8": "frozen_cold_k8",
}


def _only_keys(value, allowed, section):
    unknown = set(value) - set(allowed.split())
    if unknown:
        raise ValueError(f"Unsupported {section} fields: {sorted(unknown)}")


def validate_benchmark_protocol(protocol):
    verify_payload(protocol, field="protocol_sha256")
    if protocol.get("protocol") != "pretrained_nrx_benchmark":
        raise ValueError("Expected the pretrained-reference benchmark")
    names = protocol["condition_order"]
    if len(names) != len(set(names)) or set(names) != set(protocol["conditions"]):
        raise ValueError("Condition order differs from scenario definitions")
    groups = [set(protocol[key]) for key in ("core_conditions", "report_conditions", "stress_conditions")]
    if set.union(*groups) != set(names) or sum(map(len, groups)) != len(names):
        raise ValueError("Condition roles must cover the scenario bank without overlap")
    if not set(protocol["stationary_conditions"]).issubset(protocol["core_conditions"]):
        raise ValueError("Stationary conditions must be core conditions")
    development = protocol["development_conditions"]
    if not development or len(development) != len(set(development)) or not set(development).issubset(names):
        raise ValueError("Development conditions must be unique declared scenarios")
    if not set(protocol["core_conditions"]).issubset(development):
        raise ValueError("Development must include every primary confirmation condition")
    phases = protocol["evaluation_phases"]
    if set(phases) != {"smoke", "latency", "calibration", "development", "confirmation"}:
        raise ValueError("Incomplete evaluation phase settings")
    for phase, settings in phases.items():
        _only_keys(settings, "episodes num_tbs scoring_start_tb seed snr_db" +
                   (" snr_bounds_db" if phase == "calibration" else ""), "evaluation phase")
        if any(type(settings[k]) is not int or settings[k] < 1
               for k in ("episodes", "num_tbs", "scoring_start_tb", "seed")):
            raise ValueError("Evaluation sizes and seeds must be positive integers")
        if settings["scoring_start_tb"] > settings["num_tbs"]:
            raise ValueError("Scoring starts outside the episode")
        if phase in ("development", "confirmation") and settings["num_tbs"] < max(
                bounds[1] for bounds in protocol["windows"].values()):
            raise ValueError("Accuracy episodes must cover every declared reporting window")
        if phase in ("smoke", "latency", "calibration") and not settings.get("snr_db"):
            raise ValueError("Explicit SNR settings are required for this phase")
        if phase == "calibration":
            bounds = settings["snr_bounds_db"]
            snrs = settings["snr_db"]
            if (len(bounds) != 2 or not all(math.isfinite(value) for value in [*bounds, *snrs])
                    or len(snrs) < 3 or sorted(set(snrs)) != snrs
                    or not bounds[0] <= snrs[0] < snrs[-1] <= bounds[1]):
                raise ValueError("Invalid calibration SNR grid or search bounds")
    seeds = [settings["seed"] for settings in phases.values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Calibration, development, and confirmation need independent seeds")
    for name, condition in protocol["conditions"].items():
        _only_keys(condition, "provider receiver_inputs profile doppler_hz "
                   "scenario force_los angular_separation_deg doppler_strata_hz change_fraction "
                   "gap_start_probability maximum_gap_tbs ue_pool_size schedule_switch_probability "
                   "schedule_reorder_probability received_power_gap_db power_control channel_banks", "condition")
        if condition.get("receiver_inputs") != []:
            raise ValueError("Scenario labels cannot enter the receiver")
        if condition["provider"] == "ray_traced":
            _validate_ray_traced(name, condition, protocol)
    execution = protocol["execution"]
    _only_keys(execution, "observable_receiver_inputs "
               "accuracy_episode_batch_size latency", "execution")
    batch_size = execution["accuracy_episode_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Accuracy episode batch size must be a positive integer")
    timing = execution["latency"]
    if timing["condition"] not in names or timing["batch_size"] != 1:
        raise ValueError("Latency requires a declared condition and batch size one")
    if timing["sample_count"] < 1 or timing["warmup_samples"] < 1:
        raise ValueError("Latency requires warmup and measured samples")
    _only_keys(protocol["statistics"], "bootstrap_samples bootstrap_seed "
               "primary_snr_point tbler_definition", "statistics")
    if protocol["statistics"].get("tbler_definition") != TBLER_DEFINITION:
        raise ValueError("TBLER must pool failed and evaluated active user blocks")


def _validate_ray_traced(name, condition, protocol):
    banks = condition.get("channel_banks", {})
    required = {"calibration", "confirmation"}
    if name in protocol["development_conditions"]:
        required.add("development")
    if not required <= set(banks) <= {"calibration", "development", "confirmation"}:
        raise ValueError("Ray-traced conditions need a declared channel bank for every accuracy phase")
    if any(set(bank) != {"path", "sha256"} for bank in banks.values()):
        raise ValueError("Declare each ray-traced channel bank by path and sha256")
    if len({bank["sha256"] for bank in banks.values()}) != len(banks):
        raise ValueError("Each phase needs an independently exported ray-traced channel bank")
    if name in protocol["stationary_conditions"] or name == protocol["execution"]["latency"]["condition"]:
        raise ValueError("Ray-traced conditions cannot be stationary or the latency condition")


def load_benchmark_protocol(path):
    protocol = json.loads(Path(path).read_text())
    validate_benchmark_protocol(protocol)
    return protocol


def resolved_condition_settings(protocol, condition, num_tbs):
    if condition not in protocol["conditions"]:
        raise ValueError("Unknown benchmark condition")
    settings = dict(protocol["conditions"][condition])
    if settings["provider"] == "shared_geometry_change":
        fraction = float(settings["change_fraction"])
        if not 0 < fraction < 1:
            raise ValueError("Controlled change fraction is invalid")
        settings["change_tb"] = int(int(num_tbs) * fraction) + 1
    return settings


def benchmark_method_names(protocol):
    execution = protocol["execution"]
    methods = list(execution["candidate_methods"]) + list(execution["accuracy_reference_methods"])
    return tuple(methods)
