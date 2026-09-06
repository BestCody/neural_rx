#!/usr/bin/env python3
"""Train one receiver under the temporal benchmark v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

BASE_DOMAINS = ("channel", "payload", "noise", "schedule")


def file_sha256(path: Path | str) -> str:
    """Hash one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variable_checksum(variables) -> str:
    """Hash an ordered variable collection."""
    digest = hashlib.sha256()
    for variable in variables:
        value = np.asarray(variable.numpy())
        digest.update(variable.name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def variable_value_checksum(variables) -> str:
    """Hash ordered variable values without generated names."""
    digest = hashlib.sha256()
    for index, variable in enumerate(variables):
        value = np.asarray(variable.numpy())
        digest.update(str(index).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def derived_seed(seed: int, domain: str) -> int:
    """Derive one independent positive seed."""
    raw = hashlib.sha256(f"{int(seed)}:{domain}".encode()).digest()
    return int.from_bytes(raw[:4], "little") % (2**31 - 1)


def seed_streams(seed: int, include_power=False) -> dict[str, int]:
    """Build deterministic independent data streams."""
    domains = BASE_DOMAINS + (("power",) if include_power else ())
    values = {name: derived_seed(seed, name) for name in domains}
    if len(set(values.values())) != len(values):
        raise RuntimeError("derived training seeds collided")
    return values


def contract_version(data: dict, recipe: dict) -> int:
    """Return one matching training-contract version."""
    pairs = {
        (
            "temporal_benchmark_training_data_v2",
            "temporal_benchmark_training_recipe_v2",
        ): 2,
        (
            "temporal_benchmark_training_data_v3",
            "temporal_benchmark_training_recipe_v3",
        ): 3,
    }
    key = (data.get("protocol"), recipe.get("protocol"))
    if key not in pairs:
        raise ValueError("training data and recipe versions differ")
    return pairs[key]


def training_type(version: int) -> str:
    """Return the summary type for one contract."""
    return f"temporal_benchmark_training_v{int(version)}"


def episode_manifest_type(version: int) -> str:
    """Return the episode-manifest type."""
    return f"temporal_training_episode_manifest_v{int(version)}"


def training_source_checksums(version: int) -> dict[str, str]:
    """Seal the code that defines training examples."""
    names = [
        "train_temporal_benchmark.py",
        "temporal_training_data.py",
        "temporal_training_curriculum.py",
        "pilot_domain_memory.py",
    ]
    if int(version) >= 3:
        names.append("temporal_power_training_data.py")
    return {name: file_sha256(HERE / name) for name in names}


def stage_contract(protocol: dict, stage: str) -> tuple[int, list[int]]:
    """Return the frozen update count and seeds."""
    if stage == "calibration":
        value = protocol["calibration"]
        return int(value["training_updates"]), [
            int(value["training_recipe_seed"])
        ]
    value = protocol["stages"][stage]
    return int(value["training_updates"]), [
        int(seed) for seed in value["training_recipe_seeds"]
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("candidate", "matched_cold_k8"), required=True
    )
    parser.add_argument(
        "--stage",
        choices=("calibration", "screen", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="nrx_large.cfg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-summary",
        type=Path,
        help="Sealed matched-K=8 summary required by v2 candidates.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--benchmark-protocol",
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
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "config" / "temporal_generalization_protocol.json",
    )
    parser.add_argument(
        "--threshold-artifact",
        type=Path,
        default=None,
    )
    return parser


def load_contract(args) -> tuple[dict, dict, dict, dict, int, int]:
    """Load and validate all training contracts."""
    from temporal_benchmark_protocol import load_benchmark_protocol
    from temporal_generalization_protocol import load_split_manifest

    benchmark = load_benchmark_protocol(args.benchmark_protocol)
    data = json.loads(args.training_data.read_text(encoding="utf-8"))
    recipe = json.loads(args.training_recipe.read_text(encoding="utf-8"))
    selection = json.loads(
        args.checkpoint_selection.read_text(encoding="utf-8")
    )
    split = load_split_manifest(args.split_manifest)
    version = contract_version(data, recipe)
    if selection != {
        "schema_version": 1,
        "protocol": "temporal_benchmark_checkpoint_selection_v1",
        "rule": "last_update",
        "validation_data_used": False,
    }:
        raise ValueError("checkpoint-selection protocol differs")
    updates, seeds = stage_contract(benchmark, args.stage)
    if int(args.training_seed) not in seeds:
        raise ValueError("training seed is outside the selected stage")
    if args.stage == "calibration" and args.role != "matched_cold_k8":
        raise ValueError("calibration trains only matched Cold K=8")
    if args.smoke_test:
        updates = int(data["updates_per_sequence"])
    from temporal_training_curriculum import STRATA

    if tuple(data["target_strata"]) != STRATA:
        raise ValueError("event-relative training strata changed")
    if int(data["updates_per_sequence"]) != len(STRATA):
        raise ValueError("training must perform one update per stratum")
    teacher = recipe.get("candidate_teacher", {})
    if teacher.get("source") != "matched_stage_checkpoint":
        raise ValueError("teacher source changed")
    if not teacher.get("exact_checkpoint_required"):
        raise ValueError("training requires exact teacher binding")
    if int(data["stream_tbs"]) != 16:
        raise ValueError("training stream length changed")
    if version >= 3:
        expected_target = "event_relative_unique_near_far_v3"
        if data.get("target_selection") != expected_target:
            raise ValueError(f"v{version} target selection differs")
        augmentation = data.get("received_power_augmentation", {})
        if augmentation.get("mode") != "anonymous_episode_span_v1":
            raise ValueError("received-power mode changed")
        if augmentation.get("scaling") != "centered_raw_channel":
            raise ValueError("received-power scaling changed")
        span = [float(value) for value in augmentation["episode_span_db"]]
        if span[0] < 9.0 or span[1] < span[0] or span[1] > 15.0:
            raise ValueError("received-power span is invalid")
        if not 0.0 < float(augmentation["probability"]) <= 1.0:
            raise ValueError("power augmentation probability is invalid")
        bins = [float(value) for value in data["power_audit_bins_db"]]
        if len(bins) < 2 or any(
            right <= left for left, right in zip(bins, bins[1:])
        ):
            raise ValueError("power-audit bins are invalid")
        initialization = recipe.get("candidate_initialization", {})
        if initialization.get("source") != (
            "matched_teacher_executed_k2_path"
        ):
            raise ValueError("candidate initialization changed")
        schedule = recipe.get("learning_rate_schedule", {})
        if schedule.get("type") != "warmup_cosine_v1":
            raise ValueError("learning-rate schedule changed")
        rates = [
            float(schedule[name]) for name in ("initial", "peak", "final")
        ]
        if any(rate <= 0.0 for rate in rates):
            raise ValueError("learning rates must be positive")
        if rates[1] < max(rates[0], rates[2]):
            raise ValueError("peak learning rate is invalid")
        if not 0.0 < float(schedule["warmup_fraction"]) < 1.0:
            raise ValueError("warmup fraction is invalid")
        priority = recipe.get("candidate_loss", {}).get(
            "user_priority", {}
        )
        for name in (
            "power_max_multiplier",
            "teacher_advantage_max_multiplier",
            "combined_max_multiplier",
        ):
            if float(priority.get(name, 0.0)) < 1.0:
                raise ValueError(f"{name} is invalid")
        if float(priority.get("teacher_advantage_scale", 0.0)) <= 0.0:
            raise ValueError("teacher-advantage scale is invalid")
        if float(priority.get("combined_max_multiplier", 0.0)) < 1.0:
            raise ValueError("user-priority cap is invalid")
        if float(recipe["matched_cold_k8_loss"].get(
            "power_max_multiplier", 0.0
        )) < 1.0:
            raise ValueError("K=8 power multiplier is invalid")
    elif "received_power_augmentation" in data:
        raise ValueError("v2 cannot enable power augmentation")
    initial = ROOT / recipe["initialization"]["checkpoint"]
    if file_sha256(initial) != recipe["initialization"]["sha256"]:
        raise ValueError("initial Neural RX checkpoint changed")
    return benchmark, data, recipe, split, updates, version


def active_mean(values, active, tf):
    """Average a loss over active UEs."""
    values = per_user_mean(values, tf)
    return weighted_active_mean(
        values, active, tf.ones_like(values), tf
    )


def per_user_mean(values, tf):
    """Reduce all non-batch and non-UE loss axes."""
    values = tf.convert_to_tensor(values)
    if values.shape.rank > 2:
        values = tf.reduce_mean(
            values, axis=list(range(2, values.shape.rank))
        )
    return values


def weighted_active_mean(values, active, weights, tf):
    """Average per-UE values with normalized weights."""
    values = tf.convert_to_tensor(values)
    active = tf.cast(active, values.dtype)
    weights = tf.cast(weights, values.dtype) * active
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * weights), tf.reduce_sum(weights)
    )


def power_user_weights(
    difficulty_db, maximum_db, maximum_multiplier, tf
):
    """Prioritize weak UEs without changing inputs."""
    difficulty = tf.convert_to_tensor(difficulty_db, tf.float32)
    fraction = tf.clip_by_value(
        tf.math.divide_no_nan(difficulty, float(maximum_db)),
        0.0,
        1.0,
    )
    return 1.0 + (float(maximum_multiplier) - 1.0) * fraction


def candidate_user_weights(
    difficulty_db,
    student_truth,
    teacher_truth,
    maximum_db,
    settings,
    tf,
):
    """Prioritize weak UEs where K=8 has an advantage."""
    power = power_user_weights(
        difficulty_db,
        maximum_db,
        settings["power_max_multiplier"],
        tf,
    )
    advantage = tf.nn.relu(student_truth - teacher_truth)
    fraction = tf.clip_by_value(
        tf.math.divide_no_nan(
            advantage, float(settings["teacher_advantage_scale"])
        ),
        0.0,
        1.0,
    )
    teacher = 1.0 + (
        float(settings["teacher_advantage_max_multiplier"]) - 1.0
    ) * fraction
    combined = tf.minimum(
        power * teacher,
        float(settings["combined_max_multiplier"]),
    )
    return tf.stop_gradient(combined)


def learning_rate_at_step(recipe, step: int, updates: int) -> float:
    """Evaluate a scalar training learning rate."""
    if "learning_rate_schedule" not in recipe:
        return float(recipe["learning_rate"])
    settings = recipe["learning_rate_schedule"]
    if settings.get("type") != "warmup_cosine_v1":
        raise ValueError("unknown learning-rate schedule")
    initial = float(settings["initial"])
    peak = float(settings["peak"])
    final = float(settings["final"])
    fraction = float(settings["warmup_fraction"])
    warmup = max(2, int(round(int(updates) * fraction)))
    if int(step) < warmup:
        progress = float(step) / float(warmup - 1)
        return initial + progress * (peak - initial)
    remaining = max(1, int(updates) - warmup - 1)
    progress = min(1.0, float(step - warmup) / remaining)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return final + cosine * (peak - final)


def copy_variable_values(target, source) -> None:
    """Copy one shape-checked variable sequence."""
    if len(target) != len(source):
        raise ValueError("warm-start variable counts differ")
    for destination, origin in zip(target, source):
        if tuple(destination.shape) != tuple(origin.shape):
            raise ValueError("warm-start variable shapes differ")
        destination.assign(origin)


def physical_channel_power_db(channel, tf):
    """Measure effective power for each scheduled UE."""
    channel = tf.convert_to_tensor(channel)
    if channel.shape.rank != 7:
        raise ValueError("physical channel must have rank seven")
    power = tf.reduce_mean(
        tf.abs(channel) ** 2, axis=(1, 2, 4, 5, 6)
    )
    floor = tf.constant(1e-12, power.dtype)
    return 10.0 * tf.math.log(tf.maximum(power, floor)) / tf.math.log(
        tf.constant(10.0, power.dtype)
    )


def effective_user_difficulty_db(
    channel, active, attenuation_db, tf
):
    """Measure absolute and relative weak-user difficulty."""
    power = physical_channel_power_db(channel, tf)
    active = tf.cast(active, tf.bool)
    floor = tf.fill(tf.shape(power), tf.constant(-1e9, power.dtype))
    strongest = tf.reduce_max(
        tf.where(active, power, floor), axis=1, keepdims=True
    )
    relative = tf.where(active, strongest - power, 0.0)
    absolute = tf.where(active, tf.nn.relu(-power), 0.0)
    assigned = tf.where(
        active, tf.cast(attenuation_db, power.dtype), 0.0
    )
    return tf.maximum(tf.maximum(relative, absolute), assigned), power


def episode_difficulty_scores(batch, tf) -> np.ndarray:
    """Score each TB by active near-far separation."""
    active = batch["active"]
    scores = []
    for tb in range(int(active.shape[1])):
        power = physical_channel_power_db(batch["h"][:, tb], tf)
        mask = tf.cast(active[:, tb], tf.bool)
        high = tf.reduce_max(
            tf.where(mask, power, tf.fill(tf.shape(power), -1e9)),
            axis=1,
        )
        low = tf.reduce_min(
            tf.where(mask, power, tf.fill(tf.shape(power), 1e9)),
            axis=1,
        )
        count = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        gap = tf.where(count >= 2, high - low, 0.0)
        weakness = tf.nn.relu(-low)
        scores.append(float(tf.reduce_max(1000.0 * gap + weakness).numpy()))
    return np.asarray(scores, np.float64)


def executed_variables(backbone, iterations: int) -> list:
    """Collect variables used by a fixed iteration count."""
    layers = list(backbone._s_init)
    for iteration in backbone._iterations[:int(iterations)]:
        layers.extend([iteration._state_aggreg, iteration._state_update])
    layers.extend(backbone._readout_llrs)
    output = []
    seen = set()
    for layer in layers:
        for variable in layer.trainable_variables:
            if id(variable) not in seen:
                output.append(variable)
                seen.add(id(variable))
    return output


def run_iterations(backbone, inputs, count, helpers):
    """Run the requested CGNN depth."""
    state = helpers["frozen_state"](backbone, inputs)
    for index in range(int(count)):
        state = backbone._iterations[index]([
            state, inputs[1], inputs[3]
        ])
    return helpers["readout"](backbone, state)


def update_fingerprint(digest, batch, target_tbs) -> None:
    """Record a sparse audit fingerprint of one episode."""
    for name in ("bits", "active", "ue_ids", "slot_number", "ebno_db"):
        digest.update(np.asarray(batch[name].numpy()).tobytes(order="C"))
    if "ue_power_attenuation_db" in batch:
        digest.update(np.asarray(
            batch["ue_power_attenuation_db"].numpy()
        ).tobytes(order="C"))
    target_tbs = tuple(int(value) for value in target_tbs)
    digest.update(np.asarray(target_tbs, np.int16).tobytes(order="C"))
    y = np.asarray(batch["y"].numpy())[:, target_tbs]
    ls = np.asarray(batch["ls"].numpy())[:, target_tbs]
    digest.update(y[..., ::127].tobytes(order="C"))
    digest.update(ls[..., ::127, :].tobytes(order="C"))


def episode_fingerprint(batch, target_tbs) -> str:
    """Hash one accepted episode and its selected TBs."""
    digest = hashlib.sha256()
    update_fingerprint(digest, batch, target_tbs)
    return digest.hexdigest()


def load_teacher_summary(
    args, updates: int, version: int
) -> tuple[dict, Path]:
    """Resolve an exact matched-K=8 teacher from its sealed summary."""
    if args.teacher_summary is None:
        raise ValueError("candidate requires --teacher-summary")
    path = args.teacher_summary.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "training": training_type(version),
        "role": "matched_cold_k8",
        "stage": args.stage,
        "training_seed": int(args.training_seed),
        "training_updates": int(updates),
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"teacher summary has invalid {name}")
    if bool(value.get("smoke_test")) != bool(args.smoke_test):
        raise ValueError("teacher and candidate smoke settings differ")
    checkpoint = Path(value["full_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if file_sha256(checkpoint) != value["full_checkpoint_sha256"]:
        raise ValueError("teacher checkpoint differs from its summary")
    manifest = Path(
        value["training_episode_manifest"]
    ).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if file_sha256(manifest) != value["training_episode_manifest_sha256"]:
        raise ValueError("teacher episode manifest differs from its summary")
    return value, checkpoint


def main() -> None:
    """Train one stage-specific receiver."""
    args = build_parser().parse_args()
    benchmark, data, recipe, split, updates, version = load_contract(args)
    source_checksums = training_source_checksums(version)
    updates_per_episode = int(data["updates_per_sequence"])
    episodes = updates // updates_per_episode
    if episodes * updates_per_episode != updates:
        raise ValueError("training updates must divide the TB schedule")
    if args.role == "matched_cold_k8":
        if args.teacher_summary is not None:
            raise ValueError("matched K=8 cannot consume a teacher summary")
        sealed_teacher = None
        teacher_checkpoint = None
        expected_episode_manifest_path = None
        expected_episode_manifest = None
    else:
        sealed_teacher, teacher_checkpoint = load_teacher_summary(
            args, updates, version
        )
        expected_episode_manifest_path = Path(
            sealed_teacher["training_episode_manifest"]
        ).expanduser().resolve()
        expected_episode_manifest = json.loads(
            expected_episode_manifest_path.read_text(encoding="utf-8")
        )
    if expected_episode_manifest is not None:
        expected_contract = {
            "protocol": episode_manifest_type(version),
            "stage": args.stage,
            "training_seed": int(args.training_seed),
            "training_updates": int(updates),
            "smoke_test": bool(args.smoke_test),
            "training_data_sha256": file_sha256(args.training_data),
            "training_recipe_sha256": file_sha256(args.training_recipe),
            "training_source_sha256": source_checksums,
        }
        for name, expected_value in expected_contract.items():
            if expected_episode_manifest.get(name) != expected_value:
                raise ValueError(f"teacher episode manifest has invalid {name}")
        if len(expected_episode_manifest.get("episodes", [])) != episodes:
            raise ValueError("teacher episode manifest count differs")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.chdir(HERE)

    import sionna as sn
    import tensorflow as tf

    from pilot_domain_memory import build_segment_safe_pd_l8_sequence
    from temporal_generalization_protocol import (
        load_segment_safe_threshold_artifact,
        role_profile_specs,
        rotated_event_profiles,
    )
    from temporal_receiver_core import (
        build_backbone,
        demap_llr,
        frozen_cold_state,
        model_checksum,
        model_value_checksum,
        model_weight_signature,
        prepare_cgnn_inputs,
        readout_llr_grid,
        select_channel_input,
    )
    from temporal_training_curriculum import (
        STRATA,
        event_relative_candidates,
        select_event_relative_targets,
        select_unique_event_relative_targets,
    )
    from temporal_training_data import TemporalTrainingDataGenerator
    from temporal_power_training_data import (
        PowerAwareTemporalTrainingDataGenerator,
    )

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    seed = int(args.training_seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sn.config.seed = seed
    iteration_count = 2 if args.role == "candidate" else 8
    parameters, e2e = build_backbone(
        args.config,
        num_it=iteration_count,
        training=False,
        num_tx_eval=int(benchmark["radio"]["maximum_active_ues"]),
        n_size_bwp_override=int(benchmark["radio"]["n_size_bwp"]),
    )
    receiver = e2e._receiver
    student = receiver._neural_rx._cgnn
    selected = executed_variables(student, iteration_count)
    selected_ids = {id(value) for value in selected}
    frozen = [
        value for value in student.variables if id(value) not in selected_ids
    ]
    frozen_before = variable_checksum(frozen)
    initial_checksum = model_checksum(student)
    optimization_initial_checksum = initial_checksum
    warm_start_source_checksum = None
    warm_start_target_checksum = None
    teacher = None
    teacher_checksum = None
    teacher_value_checksum = None
    teacher_weight_signature = None
    teacher_checkpoint_sha256 = None
    teacher_summary_sha256 = None
    if args.role == "candidate":
        _, teacher_e2e = build_backbone(
            args.config,
            num_it=8,
            training=False,
            num_tx_eval=int(benchmark["radio"]["maximum_active_ues"]),
            n_size_bwp_override=int(benchmark["radio"]["n_size_bwp"]),
        )
        teacher = teacher_e2e._receiver._neural_rx._cgnn
        teacher.load_weights(str(teacher_checkpoint))
        teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
        teacher_summary_sha256 = file_sha256(args.teacher_summary)
        teacher_weight_signature = model_weight_signature(teacher)
        if teacher_weight_signature != sealed_teacher.get(
            "final_weight_signature"
        ):
            raise ValueError("loaded teacher weight signature differs")
        teacher_value_checksum = model_value_checksum(teacher)
        if teacher_value_checksum != sealed_teacher.get(
            "final_value_checksum"
        ):
            raise ValueError("loaded teacher weight values differ")
        if version >= 3:
            teacher_k2 = executed_variables(teacher, 2)
            warm_start_source_checksum = variable_value_checksum(
                teacher_k2
            )
            copy_variable_values(selected, teacher_k2)
            warm_start_target_checksum = variable_value_checksum(selected)
            if warm_start_target_checksum != warm_start_source_checksum:
                raise RuntimeError("candidate warm start differs")
            optimization_initial_checksum = model_checksum(student)
        teacher.trainable = False
        teacher_checksum = model_checksum(teacher)
    threshold_artifact = args.threshold_artifact
    if threshold_artifact is None:
        threshold_artifact = (
            ROOT / "config" / "segment_safe_pd_l8_threshold.json"
        )
    threshold_artifact = threshold_artifact.expanduser().resolve()
    threshold = load_segment_safe_threshold_artifact(
        threshold_artifact, split
    )["calibration"]["threshold"]
    profiles = role_profile_specs(split, "training")
    streams = seed_streams(
        seed,
        include_power=version >= 3,
    )
    generation_limit = episodes * int(
        data["max_generation_attempts_per_sequence"]
    )
    generator_options = {
        "effective_ue_pool_size_range": data[
            "effective_ue_pool_size_range"
        ],
        "schedule_switch_prob_range": data[
            "schedule_switch_prob_range"
        ],
        "schedule_reorder_prob_range": data[
            "schedule_reorder_prob_range"
        ],
        "gap_start_prob": float(data["gap_start_prob"]),
        "max_gap_slots": int(data["max_gap_slots"]),
        "hard_reset_prob": float(data["hard_reset_prob"]),
    }
    for name in ("channel_change_prob", "ebno_drop_prob"):
        if name in data:
            generator_options[name] = float(data[name])
    if version >= 3:
        generator_options["received_power_augmentation"] = data[
            "received_power_augmentation"
        ]
    if version == 3:
        generator_class = PowerAwareTemporalTrainingDataGenerator
    else:
        generator_class = TemporalTrainingDataGenerator
    generator = generator_class(
        parameters,
        e2e,
        ue_pool_size=int(data["ue_pool_size"]),
        dynamic_scheduling=bool(data["dynamic_scheduling"]),
        scenario_training=bool(data["scenario_training"]),
        random_slot_offset=bool(data["random_slot_offset"]),
        advance_slot_pilots=bool(data["advancing_slot_pilots"]),
        channel_profiles=profiles,
        event_channel_profiles=rotated_event_profiles(profiles),
        randomize_profile_assignment=bool(
            data["randomize_profile_assignment"]
        ),
        channel_role="training",
        seed_streams=streams,
        seed_stream_limit=generation_limit,
        **generator_options,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate_at_step(recipe, 0, updates)
    )
    helpers = {
        "frozen_state": frozen_cold_state,
        "readout": readout_llr_grid,
    }
    history = []
    coverage = {
        "channel_change_cells": 0,
        "ebno_change_slots": 0,
        "inactive_cells": 0,
        "hard_reset_cells": 0,
        "reordered_boundaries": 0,
    }
    stratum_coverage = {name: 0 for name in STRATA}
    target_tb_histogram = {
        name: [0] * int(data["stream_tbs"]) for name in STRATA
    }
    rejected_for_missing = {name: 0 for name in STRATA}
    rejected_for_nonunique_targets = 0
    duplicate_target_updates = 0
    teacher_loss_by_stratum = {
        name: {"sum": 0.0, "count": 0} for name in STRATA
    }
    teacher_loss_by_profile = {}
    power_bins = [
        float(value) for value in data.get(
            "power_audit_bins_db", [0.0, 1000.0]
        )
    ]
    power_coverage = {
        "accepted_augmented_episodes": 0,
        "active_user_cells": 0,
        "observed_pool_user_cells": 0,
        "pool_attenuation_histogram": [0] * (len(power_bins) - 1),
        "assigned_attenuation_histogram": [0] * (len(power_bins) - 1),
        "actual_pair_gap_histogram": [0] * (len(power_bins) - 1),
        "maximum_assigned_attenuation_db": 0.0,
        "maximum_actual_pair_gap_db": 0.0,
        "minimum_actual_user_power_db": None,
    }
    priority_stats = {"sum": 0.0, "count": 0, "maximum": 0.0}
    accepted_identity_counts = []
    fingerprint = hashlib.sha256()
    ebno_rng = np.random.RandomState(derived_seed(seed, "ebno"))
    target_rng = np.random.RandomState(derived_seed(seed, "target"))
    ebno_limits = [float(value) for value in data["ebno_db"]]
    started = time.time()
    step = 0
    episode = 0
    generated_episodes = 0
    episode_manifest_rows = []
    while episode < episodes:
        if generated_episodes >= generation_limit:
            raise RuntimeError("event-relative episode generation was exhausted")
        ebno = float(ebno_rng.uniform(*ebno_limits))
        batch = generator.sample_batch(
            1,
            int(data["stream_tbs"]),
            np.asarray([ebno], np.float32),
            scenario_training=True,
        )
        generated_episodes += 1
        ids = np.asarray(batch["ue_ids"].numpy())
        active_values = np.asarray(batch["active"].numpy(), bool)
        sequence = build_segment_safe_pd_l8_sequence(
            receiver, generator, batch, float(threshold), tf
        )
        candidates = event_relative_candidates(
            ids[0],
            active_values[0],
            sequence.reset_mask[0],
            sequence.history_valid[0],
        )
        if version == 3:
            selected_targets = select_unique_event_relative_targets(
                ids[0],
                active_values[0],
                sequence.reset_mask[0],
                sequence.history_valid[0],
                target_rng,
                difficulty_scores=episode_difficulty_scores(batch, tf),
            )
        else:
            selected_targets = select_event_relative_targets(
                ids[0],
                active_values[0],
                sequence.reset_mask[0],
                sequence.history_valid[0],
                target_rng,
            )
        if selected_targets is None:
            for name in STRATA:
                if not candidates[name]:
                    rejected_for_missing[name] += 1
            if version >= 3 and all(candidates[name] for name in STRATA):
                rejected_for_nonunique_targets += 1
            continue
        targets = tuple(
            (target.stratum, target.tb) for target in selected_targets
        )
        duplicate_target_updates += len(targets) - len({
            tb for _, tb in targets
        })
        if version >= 3 and len({tb for _, tb in targets}) != len(targets):
            raise RuntimeError("training selected duplicate target TBs")
        observed = {
            int(identity)
            for row, mask in zip(ids[0], active_values[0])
            for identity, enabled in zip(row, mask)
            if enabled
        }
        accepted_identity_counts.append(len(observed))
        selected_tbs = [target[1] for target in targets]
        point_fingerprint = episode_fingerprint(batch, selected_tbs)
        fingerprint.update(bytes.fromhex(point_fingerprint))
        manifest_row = {
            "episode": episode,
            "generation_attempt": generated_episodes,
            "base_ebno_db": ebno,
            "targets": [
                {"stratum": stratum, "tb_one_indexed": tb + 1}
                for stratum, tb in targets
            ],
            "episode_fingerprint": point_fingerprint,
        }
        if version >= 3:
            manifest_row["power_augmented"] = bool(
                batch["power_augmented"].numpy()[0]
            )
            manifest_row["ue_power_attenuation_db"] = (
                batch["ue_power_attenuation_db"].numpy()[0].tolist()
            )
        if expected_episode_manifest is not None:
            expected_row = expected_episode_manifest["episodes"][episode]
            if manifest_row != expected_row:
                raise RuntimeError("candidate episode replay differs")
        episode_manifest_rows.append(manifest_row)
        if version >= 3 and manifest_row["power_augmented"]:
            power_coverage["accepted_augmented_episodes"] += 1
        if version >= 3:
            pool_values = np.asarray(
                batch["ue_power_attenuation_db"].numpy()[0]
            )[sorted(observed)]
            power_coverage["observed_pool_user_cells"] += int(
                pool_values.size
            )
            pool_indices = np.searchsorted(
                power_bins[1:], pool_values, side="right"
            )
            for index in np.minimum(
                pool_indices, len(power_bins) - 2
            ):
                power_coverage[
                    "pool_attenuation_histogram"
                ][int(index)] += 1
        counts = {
            "channel_change_cells": int(np.sum(batch["channel_change_event"])),
            "ebno_change_slots": int(np.sum(batch["ebno_change_event"])),
            "inactive_cells": int(np.sum(~active_values)),
            "hard_reset_cells": int(np.sum(batch["hard_reset"])),
            "reordered_boundaries": int(np.sum(np.any(
                ids[:, 1:] != ids[:, :-1], axis=-1
            ))),
        }
        for name, count in counts.items():
            coverage[name] += count
        for stratum, tb in targets:
            stratum_coverage[stratum] += 1
            target_tb_histogram[stratum][tb] += 1
            active = batch["active"][:, tb]
            attenuation = tf.zeros_like(active, tf.float32)
            difficulty = attenuation
            if version >= 3:
                attenuation = batch[
                    "scheduled_power_attenuation_db"
                ][:, tb]
                difficulty, actual_power = effective_user_difficulty_db(
                    batch["h"][:, tb], active, attenuation, tf
                )
                active_mask = active_values[0, tb]
                assigned = np.asarray(attenuation.numpy())[0][active_mask]
                actual = np.asarray(actual_power.numpy())[0][active_mask]
                power_coverage["active_user_cells"] += int(assigned.size)
                if assigned.size:
                    power_coverage["maximum_assigned_attenuation_db"] = max(
                        power_coverage[
                            "maximum_assigned_attenuation_db"
                        ],
                        float(np.max(assigned)),
                    )
                    minimum = float(np.min(actual))
                    previous = power_coverage[
                        "minimum_actual_user_power_db"
                    ]
                    power_coverage["minimum_actual_user_power_db"] = (
                        minimum if previous is None else min(previous, minimum)
                    )
                    indices = np.searchsorted(
                        power_bins[1:], assigned, side="right"
                    )
                    for index in np.minimum(
                        indices, len(power_bins) - 2
                    ):
                        power_coverage[
                            "assigned_attenuation_histogram"
                        ][int(index)] += 1
                if actual.size >= 2:
                    pair_gap = float(np.max(actual) - np.min(actual))
                    power_coverage["maximum_actual_pair_gap_db"] = max(
                        power_coverage["maximum_actual_pair_gap_db"],
                        pair_gap,
                    )
                    index = min(
                        int(np.searchsorted(
                            power_bins[1:], pair_gap, side="right"
                        )),
                        len(power_bins) - 2,
                    )
                    power_coverage["actual_pair_gap_histogram"][index] += 1
            coded = receiver._tb_encoders[0](batch["bits"][:, tb])
            inputs = prepare_cgnn_inputs(
                receiver, batch["y"][:, tb], batch["ls"][:, tb], active
            )
            teacher_llr = None
            teacher_truth_by_user = None
            if teacher is not None:
                teacher_grid = tf.stop_gradient(
                    run_iterations(teacher, inputs, 8, helpers)
                )
                teacher_llr = tf.stop_gradient(demap_llr(
                    receiver._neural_rx,
                    teacher_grid,
                    tf.shape(active)[1],
                    0,
                ))
                teacher_truth_by_user = per_user_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(coded, tf.float32),
                        logits=teacher_llr,
                    ),
                    tf,
                )
                selected_channel, _ = select_channel_input(
                    inputs[2],
                    tf.convert_to_tensor(
                        sequence.posterior_channel[:, tb], tf.float32
                    ),
                    tf.convert_to_tensor(
                        sequence.history_valid[:, tb], tf.bool
                    ),
                    reset_mask=tf.convert_to_tensor(
                        sequence.reset_mask[:, tb], tf.bool
                    ),
                )
                inputs = list(inputs)
                inputs[2] = selected_channel
            with tf.GradientTape() as tape:
                grid = run_iterations(
                    student, inputs, iteration_count, helpers
                )
                logits = demap_llr(
                    receiver._neural_rx,
                    grid,
                    tf.shape(active)[1],
                    0,
                )
                truth_values = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(coded, tf.float32), logits=logits
                )
                truth_by_user = per_user_mean(truth_values, tf)
                user_weights = tf.ones_like(truth_by_user)
                if version >= 3:
                    maximum_db = float(
                        data["received_power_augmentation"][
                            "episode_span_db"
                        ][1]
                    )
                    if teacher_truth_by_user is None:
                        user_weights = power_user_weights(
                            difficulty,
                            maximum_db,
                            recipe["matched_cold_k8_loss"][
                                "power_max_multiplier"
                            ],
                            tf,
                        )
                    else:
                        user_weights = candidate_user_weights(
                            difficulty,
                            truth_by_user,
                            teacher_truth_by_user,
                            maximum_db,
                            recipe["candidate_loss"]["user_priority"],
                            tf,
                        )
                truth = weighted_active_mean(
                    truth_by_user, active, user_weights, tf
                )
                distill = tf.constant(0.0, tf.float32)
                if teacher_llr is not None:
                    distill_values = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.math.sigmoid(teacher_llr),
                        logits=logits,
                    )
                    distill_by_user = per_user_mean(distill_values, tf)
                    distill = weighted_active_mean(
                        distill_by_user, active, user_weights, tf
                    )
                    weights = recipe["candidate_loss"]
                    loss = (
                        float(weights["payload"]) * truth
                        + float(weights["cold_k8"]) * distill
                    )
                else:
                    loss = truth
            gradients = tape.gradient(loss, selected)
            if any(value is None for value in gradients):
                raise RuntimeError("a deployed variable has no gradient")
            tf.debugging.assert_all_finite(loss, "training loss is not finite")
            for gradient in gradients:
                tf.debugging.assert_all_finite(
                    gradient, "training gradient is not finite"
                )
            current_lr = learning_rate_at_step(recipe, step, updates)
            optimizer.learning_rate.assign(current_lr)
            optimizer.apply_gradients(zip(gradients, selected))
            active_weights = np.asarray(user_weights.numpy())[0][
                active_values[0, tb]
            ]
            if active_weights.size:
                priority_stats["sum"] += float(np.sum(active_weights))
                priority_stats["count"] += int(active_weights.size)
                priority_stats["maximum"] = max(
                    priority_stats["maximum"],
                    float(np.max(active_weights)),
                )
            if step % int(args.log_every) == 0 or step == updates - 1:
                row = {
                    "step": step,
                    "episode": episode,
                    "tb": tb + 1,
                    "stratum": stratum,
                    "loss": float(loss.numpy()),
                    "truth_loss": float(truth.numpy()),
                    "distill_loss": float(distill.numpy()),
                    "gradient_norm": float(
                        tf.linalg.global_norm(gradients).numpy()
                    ),
                    "learning_rate": current_lr,
                    "user_weight_mean": float(
                        np.mean(active_weights)
                    ),
                    "seconds": time.time() - started,
                }
                history.append(row)
                print("BENCHMARK_TRAIN=" + json.dumps(row), flush=True)
            if teacher_truth_by_user is not None:
                values = np.asarray(teacher_truth_by_user.numpy())[0]
                teacher_loss_by_stratum[stratum]["sum"] += float(
                    np.sum(values * active_values[0, tb])
                )
                teacher_loss_by_stratum[stratum]["count"] += int(
                    np.sum(active_values[0, tb])
                )
                channel_regime = np.asarray(
                    batch["channel_regime"].numpy(), bool
                )
                for position, enabled in enumerate(active_values[0, tb]):
                    if not enabled:
                        continue
                    identity = int(ids[0, tb, position])
                    metadata = generator.ue_profiles[identity]
                    prefix = "harsh" if channel_regime[0, tb, identity] else "steady"
                    key = f"{prefix}:{metadata[prefix + '_model']}"
                    record = teacher_loss_by_profile.setdefault(
                        key, {"sum": 0.0, "count": 0}
                    )
                    record["sum"] += float(values[position])
                    record["count"] += 1
            step += 1
        episode += 1
    if step != updates:
        raise RuntimeError("training update count differs")
    if model_checksum(student) == optimization_initial_checksum:
        raise RuntimeError("receiver weights did not change")
    if variable_checksum(frozen) != frozen_before:
        raise RuntimeError("training changed an unselected variable")
    if teacher is not None and model_checksum(teacher) != teacher_checksum:
        raise RuntimeError("training changed the frozen teacher")
    if teacher is not None and model_value_checksum(
        teacher
    ) != teacher_value_checksum:
        raise RuntimeError("training changed frozen teacher values")
    data_fingerprint = fingerprint.hexdigest()
    if set(stratum_coverage.values()) != {episodes}:
        raise RuntimeError("training stratum coverage is not balanced")
    if not accepted_identity_counts or min(accepted_identity_counts) < 3:
        raise RuntimeError("training accepted fewer than three UEs")
    if version >= 3:
        if (
            not args.smoke_test
            and power_coverage["accepted_augmented_episodes"] < 1
        ):
            raise RuntimeError("training accepted no power augmentation")
        if (
            not args.smoke_test
            and power_coverage["maximum_assigned_attenuation_db"] < 9.0
        ):
            raise RuntimeError("training omitted the weak-user range")
        if (
            not args.smoke_test
            and power_coverage["maximum_actual_pair_gap_db"] < 9.0
        ):
            raise RuntimeError("training omitted severe near-far pairs")
        if not args.smoke_test and any(
            count == 0
            for count in power_coverage["pool_attenuation_histogram"]
        ):
            raise RuntimeError("training power-bin coverage is incomplete")
    if sealed_teacher is not None:
        if file_sha256(teacher_checkpoint) != teacher_checkpoint_sha256:
            raise RuntimeError("teacher checkpoint changed during training")
        if sealed_teacher["data_fingerprint"] != data_fingerprint:
            raise RuntimeError("candidate and teacher training data differ")
        if expected_episode_manifest["generated_episodes"] != generated_episodes:
            raise RuntimeError("candidate generation attempts differ")
    full_checkpoint = output_dir / f"{args.role}.weights.h5"
    student.save_weights(str(full_checkpoint))
    selected_checkpoint = None
    if args.role == "candidate":
        selected_checkpoint = output_dir / "candidate_selected.npz"
        np.savez_compressed(
            selected_checkpoint,
            **{
                f"value_{index:03d}": variable.numpy()
                for index, variable in enumerate(selected)
            },
        )
    training_episode_manifest_path = None
    training_episode_manifest_sha256 = None
    if args.role == "matched_cold_k8":
        training_episode_manifest_path = (
            output_dir / "training_episode_manifest.json"
        )
        training_episode_manifest = {
            "schema_version": version,
            "protocol": episode_manifest_type(version),
            "stage": args.stage,
            "training_seed": seed,
            "training_updates": updates,
            "smoke_test": bool(args.smoke_test),
            "training_data_sha256": file_sha256(args.training_data),
            "training_recipe_sha256": file_sha256(args.training_recipe),
            "benchmark_protocol_sha256": benchmark["protocol_sha256"],
            "split_manifest_sha256": split["manifest_sha256"],
            "threshold_artifact_sha256": file_sha256(threshold_artifact),
            "training_source_sha256": source_checksums,
            "accepted_episodes": episodes,
            "generated_episodes": generated_episodes,
            "data_fingerprint": data_fingerprint,
            "episodes": episode_manifest_rows,
        }
        training_episode_manifest_path.write_text(
            json.dumps(training_episode_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        training_episode_manifest_sha256 = file_sha256(
            training_episode_manifest_path
        )
    else:
        training_episode_manifest_path = expected_episode_manifest_path
        training_episode_manifest_sha256 = file_sha256(
            training_episode_manifest_path
        )
    def metric_summary(records):
        return {
            name: {
                "mean": (
                    None
                    if int(record["count"]) == 0
                    else float(record["sum"]) / int(record["count"])
                ),
                "count": int(record["count"]),
            }
            for name, record in sorted(records.items())
        }

    power_coverage["bins_db"] = power_bins
    priority_summary = {
        "mean": (
            None if priority_stats["count"] == 0
            else priority_stats["sum"] / priority_stats["count"]
        ),
        "count": priority_stats["count"],
        "maximum": priority_stats["maximum"],
    }
    summary = {
        "schema_version": version,
        "training": training_type(version),
        "role": args.role,
        "stage": args.stage,
        "smoke_test": bool(args.smoke_test),
        "training_seed": seed,
        "training_updates": updates,
        "training_examples": updates,
        "episodes": episodes,
        "generated_episodes": generated_episodes,
        "rejected_episodes": generated_episodes - episodes,
        "data_seed_streams": streams,
        "target_seed": derived_seed(seed, "target"),
        "data_fingerprint": data_fingerprint,
        "scenario_coverage": coverage,
        "stratum_coverage": stratum_coverage,
        "target_tb_histogram": target_tb_histogram,
        "rejected_for_missing_stratum": rejected_for_missing,
        "rejected_for_nonunique_targets": rejected_for_nonunique_targets,
        "duplicate_target_updates": duplicate_target_updates,
        "power_coverage": power_coverage if version >= 3 else None,
        "user_priority": priority_summary,
        "accepted_identity_count_minimum": (
            min(accepted_identity_counts)
            if accepted_identity_counts else None
        ),
        "teacher_loss_by_stratum": metric_summary(
            teacher_loss_by_stratum
        ) if teacher is not None else None,
        "teacher_loss_by_profile": metric_summary(
            teacher_loss_by_profile
        ) if teacher is not None else None,
        "selected_variable_count": len(selected),
        "selected_scalar_count": int(sum(
            np.prod(tuple(value.shape)) for value in selected
        )),
        "initial_checksum": initial_checksum,
        "optimization_initial_checksum": optimization_initial_checksum,
        "warm_start_source_checksum": warm_start_source_checksum,
        "warm_start_target_checksum": warm_start_target_checksum,
        "final_checksum": model_checksum(student),
        "final_value_checksum": model_value_checksum(student),
        "final_weight_signature": model_weight_signature(student),
        "teacher_checksum": teacher_checksum,
        "teacher_value_checksum": teacher_value_checksum,
        "teacher_weight_signature": teacher_weight_signature,
        "teacher_checkpoint": (
            None if teacher_checkpoint is None else str(teacher_checkpoint)
        ),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_summary": (
            None
            if args.teacher_summary is None
            else str(args.teacher_summary.expanduser().resolve())
        ),
        "teacher_summary_sha256": teacher_summary_sha256,
        "training_episode_manifest": (
            None
            if training_episode_manifest_path is None
            else str(training_episode_manifest_path)
        ),
        "training_episode_manifest_sha256": (
            training_episode_manifest_sha256
        ),
        "benchmark_protocol_sha256": benchmark["protocol_sha256"],
        "training_data_sha256": file_sha256(args.training_data),
        "training_recipe_sha256": file_sha256(args.training_recipe),
        "training_source_sha256": source_checksums,
        "checkpoint_selection_sha256": file_sha256(
            args.checkpoint_selection
        ),
        "split_manifest_sha256": split["manifest_sha256"],
        "threshold_artifact_sha256": file_sha256(threshold_artifact),
        "checkpoint_selection": json.loads(
            args.checkpoint_selection.read_text(encoding="utf-8")
        ),
        "full_checkpoint": str(full_checkpoint),
        "full_checkpoint_sha256": file_sha256(full_checkpoint),
        "selected_checkpoint": (
            None if selected_checkpoint is None else str(selected_checkpoint)
        ),
        "selected_checkpoint_sha256": (
            None
            if selected_checkpoint is None
            else file_sha256(selected_checkpoint)
        ),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    output = output_dir / "training_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
