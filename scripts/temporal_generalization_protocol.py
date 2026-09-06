#!/usr/bin/env python3
"""Define temporal generalization splits and calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SPLIT_ROLES = (
    "training",
    "calibration",
    "known_comparison",
    "ood",
)
SEED_DOMAINS = ("channel", "payload", "noise", "schedule")
TDL_MODELS = ("A", "B", "C", "D", "E", "A30", "B100", "C300")
FIXED_TDL_DELAYS = {
    "A30": 30e-9,
    "B100": 100e-9,
    "C300": 300e-9,
}
FORBIDDEN_ADAPTER_FEATURES = frozenset({
    "absolute_tb_position",
    "channel_profile",
    "channel_truth",
    "crc_status",
    "decoded_bits",
    "delay_spread",
    "doppler",
    "future_samples",
    "random_seed",
    "scenario_label",
    "split_role",
    "ue_id",
})
ALLOWED_ADAPTER_FEATURES = frozenset({
    "coupling_delta",
    "log_symbol_variance_ratio",
    "pairwise_channel_coupling",
    "posterior_delta",
    "posterior_log_precision",
    "posterior_reliability",
    "posterior_uncertainty_ratio",
    "soft_symbol_delta",
})


@dataclass(frozen=True)
class ChannelProfileSpec:
    """Describe one channel-family range."""

    name: str
    family: str
    model: str
    min_delay_spread_s: float | None
    max_delay_spread_s: float | None
    min_doppler_hz: float
    max_doppler_hz: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("channel profile name cannot be empty")
        if self.family not in ("tdl", "umi"):
            raise ValueError(f"unsupported channel family: {self.family}")
        if self.family == "tdl":
            if self.model not in TDL_MODELS:
                raise ValueError(f"unsupported TDL model: {self.model}")
            if self.min_delay_spread_s is None:
                raise ValueError("TDL delay spread is required")
            if self.max_delay_spread_s is None:
                raise ValueError("TDL delay spread is required")
            if self.min_delay_spread_s <= 0.0:
                raise ValueError("TDL delay spread must be positive")
            if self.max_delay_spread_s < self.min_delay_spread_s:
                raise ValueError("invalid TDL delay-spread range")
            fixed = FIXED_TDL_DELAYS.get(self.model)
            if fixed is not None and not (
                math.isclose(
                    self.min_delay_spread_s, fixed, rel_tol=0.0, abs_tol=1e-18
                )
                and math.isclose(
                    self.max_delay_spread_s, fixed, rel_tol=0.0, abs_tol=1e-18
                )
            ):
                raise ValueError(f"{self.model} has a fixed delay spread")
        else:
            if self.model != "UMi":
                raise ValueError("UMi family requires model UMi")
            if self.min_delay_spread_s is not None:
                raise ValueError("UMi delay spread comes from topology")
            if self.max_delay_spread_s is not None:
                raise ValueError("UMi delay spread comes from topology")
        if self.min_doppler_hz < 0.0:
            raise ValueError("Doppler cannot be negative")
        if self.max_doppler_hz < self.min_doppler_hz:
            raise ValueError("invalid Doppler range")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ChannelProfileSpec":
        profile = cls(**value)
        profile.validate()
        return profile


def _tdl(
    model: str,
    min_delay_ns: float,
    max_delay_ns: float,
    min_doppler_hz: float,
    max_doppler_hz: float,
) -> ChannelProfileSpec:
    return ChannelProfileSpec(
        name=f"TDL-{model}",
        family="tdl",
        model=model,
        min_delay_spread_s=float(min_delay_ns) / 1e9,
        max_delay_spread_s=float(max_delay_ns) / 1e9,
        min_doppler_hz=float(min_doppler_hz),
        max_doppler_hz=float(max_doppler_hz),
    )


def unique_profiles_for_role(role: str) -> tuple[ChannelProfileSpec, ...]:
    """Return the fixed profile pool for one split role."""
    if role in ("training", "calibration"):
        return tuple(
            _tdl(model, 30.0, 500.0, 0.0, 400.0)
            for model in ("A", "B", "C")
        )
    if role == "known_comparison":
        return (
            _tdl("B100", 100.0, 100.0, 0.0, 400.0),
            _tdl("C300", 300.0, 300.0, 0.0, 100.0),
        )
    if role == "ood":
        return (
            _tdl("D", 30.0, 500.0, 0.0, 400.0),
            _tdl("E", 30.0, 500.0, 0.0, 400.0),
            ChannelProfileSpec(
                name="UMi",
                family="umi",
                model="UMi",
                min_delay_spread_s=None,
                max_delay_spread_s=None,
                min_doppler_hz=0.0,
                max_doppler_hz=400.0,
            ),
        )
    raise ValueError(f"unknown split role: {role}")


def rotated_event_profiles(
    profiles: tuple[ChannelProfileSpec, ...],
) -> tuple[ChannelProfileSpec, ...]:
    """Create a different target family for each source."""
    if not profiles:
        raise ValueError("profiles cannot be empty")
    if len({profile.name for profile in profiles}) < 2:
        return profiles
    return profiles[1:] + profiles[:1]


def _canonical_json(value: dict) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_sha256(value: dict, field: str = "manifest_sha256") -> str:
    """Hash a JSON payload without its digest field."""
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def seal_payload(value: dict, field: str = "manifest_sha256") -> dict:
    """Return a copied payload with a deterministic digest."""
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    payload[field] = payload_sha256(payload, field)
    return payload


def verify_payload(value: dict, field: str = "manifest_sha256") -> None:
    """Verify a sealed JSON payload."""
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing or invalid {field}")
    actual = payload_sha256(value, field)
    if actual != expected:
        raise ValueError(f"{field} does not match payload")


def build_split_manifest() -> dict:
    """Build the frozen temporal data split manifest."""
    role_settings = {
        "training": {"episodes": 6000, "seed_base": 1_000_000},
        "calibration": {"episodes": 128, "seed_base": 2_000_000},
        "known_comparison": {"episodes": 256, "seed_base": 3_000_000},
        "ood": {"episodes": 128, "seed_base": 5_000_000},
    }
    domain_offsets = {
        "channel": 0,
        "payload": 100_000,
        "noise": 200_000,
        "schedule": 300_000,
    }
    roles = {}
    for role in SPLIT_ROLES:
        roles[role] = {
            **role_settings[role],
            "domain_offsets": domain_offsets,
            "profiles": [
                profile.to_dict()
                for profile in unique_profiles_for_role(role)
            ],
        }
    return seal_payload({
        "schema_version": 1,
        "protocol": "temporal_generalization_v1",
        "training_recipe_seeds": [20260911, 20260912, 20260913],
        "roles": roles,
        "forbidden_adapter_features": sorted(
            FORBIDDEN_ADAPTER_FEATURES
        ),
    })


def _seed_intervals(manifest: dict) -> list[tuple[int, int, str]]:
    intervals = []
    for role in SPLIT_ROLES:
        config = manifest["roles"][role]
        count = int(config["episodes"])
        if count <= 0:
            raise ValueError(f"{role} episode count must be positive")
        base = int(config["seed_base"])
        offsets = config["domain_offsets"]
        if set(offsets) != set(SEED_DOMAINS):
            raise ValueError(f"{role} has invalid seed domains")
        for domain in SEED_DOMAINS:
            start = base + int(offsets[domain])
            intervals.append((start, start + count - 1, f"{role}:{domain}"))
    return intervals


def validate_split_manifest(manifest: dict) -> None:
    """Validate hashes, families, and seed separation."""
    verify_payload(manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema")
    if tuple(manifest.get("roles", {}).keys()) != SPLIT_ROLES:
        raise ValueError("split manifest roles are missing or reordered")
    expected_features = sorted(FORBIDDEN_ADAPTER_FEATURES)
    if manifest.get("forbidden_adapter_features") != expected_features:
        raise ValueError("forbidden adapter features changed")
    recipe_seeds = manifest.get("training_recipe_seeds")
    if (
        not isinstance(recipe_seeds, list)
        or len(recipe_seeds) != 3
        or len(set(recipe_seeds)) != 3
        or any(not isinstance(seed, int) or seed < 0 for seed in recipe_seeds)
    ):
        raise ValueError("exactly three unique training seeds are required")

    model_sets = {}
    for role in SPLIT_ROLES:
        profiles = tuple(
            ChannelProfileSpec.from_dict(value)
            for value in manifest["roles"][role]["profiles"]
        )
        if len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError(f"{role} contains duplicate profile names")
        model_sets[role] = {profile.model for profile in profiles}
    if model_sets["training"] != {"A", "B", "C"}:
        raise ValueError("training families must be TDL-A/B/C")
    if model_sets["calibration"] != model_sets["training"]:
        raise ValueError("calibration families must match training")
    if model_sets["known_comparison"] != {"B100", "C300"}:
        raise ValueError("known comparison families changed")
    if model_sets["ood"] != {"D", "E", "UMi"}:
        raise ValueError("OOD families must be TDL-D/E and UMi")
    if model_sets["training"] & model_sets["ood"]:
        raise ValueError("training and OOD models overlap")

    intervals = sorted(_seed_intervals(manifest))
    for index, (start, stop, label) in enumerate(intervals):
        for other_start, other_stop, other_label in intervals[index + 1:]:
            if other_start > stop:
                break
            if start <= other_stop and other_start <= stop:
                raise ValueError(
                    f"seed ranges overlap: {label} and {other_label}"
                )


def load_split_manifest(path: Path) -> dict:
    """Load and validate a split manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def role_profile_specs(
    manifest: dict, role: str
) -> tuple[ChannelProfileSpec, ...]:
    """Read one role's unique profile specs."""
    validate_split_manifest(manifest)
    if role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {role}")
    return tuple(
        ChannelProfileSpec.from_dict(value)
        for value in manifest["roles"][role]["profiles"]
    )


def episode_seed(
    manifest: dict, role: str, domain: str, episode: int
) -> int:
    """Resolve one deterministic split seed."""
    validate_split_manifest(manifest)
    if role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {role}")
    if domain not in SEED_DOMAINS:
        raise ValueError(f"unknown seed domain: {domain}")
    config = manifest["roles"][role]
    episode = int(episode)
    if not 0 <= episode < int(config["episodes"]):
        raise ValueError("episode is outside the split")
    return (
        int(config["seed_base"])
        + int(config["domain_offsets"][domain])
        + episode
    )


def seed_bundle(
    manifest: dict, role: str, episode: int = 0
) -> dict[str, int]:
    """Resolve all independent seed domains for one split offset."""
    return {
        domain: episode_seed(manifest, role, domain, episode)
        for domain in SEED_DOMAINS
    }


def validate_split_slice(
    manifest: dict, role: str, start_episode: int, episode_count: int
) -> None:
    """Ensure a run stays inside its immutable split allocation."""
    validate_split_manifest(manifest)
    if role not in SPLIT_ROLES:
        raise ValueError(f"unknown split role: {role}")
    start = int(start_episode)
    count = int(episode_count)
    if start < 0:
        raise ValueError("split offset cannot be negative")
    if count <= 0:
        raise ValueError("split episode count must be positive")
    available = int(manifest["roles"][role]["episodes"])
    if start + count > available:
        raise ValueError(
            f"{role} split has {available} episodes, but the run requests "
            f"[{start}, {start + count})"
        )


def assert_adapter_features(feature_names) -> None:
    """Reject benchmark metadata in adapter tensors."""
    names = {str(name) for name in feature_names}
    forbidden = sorted(names & FORBIDDEN_ADAPTER_FEATURES)
    if forbidden:
        raise ValueError(f"forbidden adapter features: {forbidden}")
    unknown = sorted(names - ALLOWED_ADAPTER_FEATURES)
    if unknown:
        raise ValueError(f"unapproved adapter features: {unknown}")


def normalized_dmrs_innovation(
    pilots: np.ndarray, error_variance: np.ndarray
) -> np.ndarray:
    """Calculate current-to-previous DMRS disagreement."""
    pilots = np.asarray(pilots)
    error_variance = np.asarray(error_variance, dtype=np.float64)
    if pilots.shape != error_variance.shape or pilots.ndim != 6:
        raise ValueError("pilot and error tensors must be matching rank six")
    if not np.all(error_variance > 0.0):
        raise ValueError("pilot error variances must be positive")
    innovation = np.zeros(pilots.shape[:3], dtype=np.float64)
    difference = np.square(np.abs(pilots[:, 1:] - pilots[:, :-1]))
    noise = error_variance[:, 1:] + error_variance[:, :-1]
    innovation[:, 1:] = np.mean(
        difference / np.maximum(noise, 1e-12), axis=(3, 4, 5)
    )
    if not np.all(np.isfinite(innovation)):
        raise FloatingPointError("DMRS innovation is not finite")
    return innovation


def innovation_jump_score(
    innovation: np.ndarray, history_window: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a causal median-normalized innovation score."""
    innovation = np.asarray(innovation, dtype=np.float64)
    if innovation.ndim != 3:
        raise ValueError("innovation must have shape [episode, TB, UE]")
    if int(history_window) < 2:
        raise ValueError("history_window must be at least two")
    score = np.zeros_like(innovation)
    eligible = np.zeros(innovation.shape, dtype=bool)
    first_eligible = int(history_window) + 1
    for tb in range(first_eligible, innovation.shape[1]):
        baseline = np.median(
            innovation[:, tb - int(history_window):tb], axis=1
        )
        score[:, tb] = innovation[:, tb] / np.maximum(baseline, 1e-12)
        eligible[:, tb] = True
    if not np.all(np.isfinite(score[eligible])):
        raise FloatingPointError("innovation score is not finite")
    return score, eligible


def conformal_false_reset_threshold(
    score: np.ndarray,
    eligible: np.ndarray,
    alpha: float = 0.01,
) -> dict:
    """Calibrate an episode-level no-change threshold."""
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if score.shape != eligible.shape or score.ndim != 3:
        raise ValueError("score and eligible tensors must match")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    maxima = []
    for episode in range(score.shape[0]):
        values = score[episode][eligible[episode]]
        if values.size == 0:
            raise ValueError("every calibration episode needs eligible scores")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("eligible score is not finite")
        maxima.append(float(np.max(values)))
    maxima = np.sort(np.asarray(maxima, dtype=np.float64))
    rank = int(math.ceil((len(maxima) + 1) * (1.0 - float(alpha))))
    if rank > len(maxima):
        raise ValueError("more calibration episodes are required for alpha")
    threshold = float(maxima[rank - 1])
    return {
        "method": "episode_max_split_conformal",
        "alpha": float(alpha),
        "calibration_episodes": int(len(maxima)),
        "order_statistic_rank": rank,
        "threshold": threshold,
        "minimum_episode_max": float(maxima[0]),
        "maximum_episode_max": float(maxima[-1]),
        "trigger_rule": "score_strictly_greater_than_threshold",
    }


SEGMENT_SAFE_STATE_MACHINE = {
    "history_window": 8,
    "minimum_prior_innovations": 1,
    "probation_rebase": True,
    "refractory_tbs": 0,
    "boundary_innovation_accepted": False,
}


def build_segment_safe_threshold_artifact(
    calibration: dict,
    split_manifest_sha256: str,
    source_sha256: str,
) -> dict:
    """Build a segment-safe threshold artifact."""
    if calibration.get("method") != "episode_max_split_conformal":
        raise ValueError("unsupported threshold calibration method")
    if len(str(split_manifest_sha256)) != 64:
        raise ValueError("invalid split manifest digest")
    if len(str(source_sha256)) != 64:
        raise ValueError("invalid calibration source digest")
    _validate_calibration_payload(calibration)
    return seal_payload({
        "schema_version": 1,
        "artifact": "segment_safe_pd_l8_threshold",
        "split_role": "calibration",
        "split_manifest_sha256": str(split_manifest_sha256),
        "source_sha256": str(source_sha256),
        "state_machine": dict(SEGMENT_SAFE_STATE_MACHINE),
        "calibration": copy.deepcopy(calibration),
    }, field="artifact_sha256")


def load_segment_safe_threshold_artifact(
    path: Path, split_manifest: dict | None = None
) -> dict:
    """Load a segment-safe threshold artifact."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_payload(artifact, field="artifact_sha256")
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported threshold artifact schema")
    if artifact.get("artifact") != "segment_safe_pd_l8_threshold":
        raise ValueError("wrong threshold artifact type")
    if artifact.get("split_role") != "calibration":
        raise ValueError("threshold did not use the calibration split")
    if artifact.get("state_machine") != SEGMENT_SAFE_STATE_MACHINE:
        raise ValueError("segment-safe state machine differs")
    _validate_calibration_payload(artifact.get("calibration", {}))
    source_sha256 = artifact.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("invalid calibration source digest")
    if split_manifest is not None:
        validate_split_manifest(split_manifest)
        if artifact.get("split_manifest_sha256") != split_manifest[
            "manifest_sha256"
        ]:
            raise ValueError("threshold and split manifest do not match")
    return artifact


def _validate_calibration_payload(calibration: dict) -> None:
    """Validate the fixed false-reset calibration protocol."""
    if calibration.get("method") != "episode_max_split_conformal":
        raise ValueError("threshold is not false-reset calibrated")
    if calibration.get("alpha") != 0.01:
        raise ValueError("reset calibration alpha must be 0.01")
    episodes = calibration.get("calibration_episodes")
    if not isinstance(episodes, int) or episodes < 128:
        raise ValueError("reset calibration requires at least 128 episodes")
    expected_rank = int(math.ceil((episodes + 1) * 0.99))
    if calibration.get("order_statistic_rank") != expected_rank:
        raise ValueError("reset calibration rank is invalid")
    threshold = calibration.get("threshold")
    if not isinstance(threshold, (int, float)) or threshold <= 0.0:
        raise ValueError("invalid reset threshold")
    if calibration.get("trigger_rule") != (
        "score_strictly_greater_than_threshold"
    ):
        raise ValueError("reset trigger rule changed")
