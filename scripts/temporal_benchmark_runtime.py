#!/usr/bin/env python3
"""Run all receiver ideas through one paired evaluation loop."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from temporal_benchmark_candidate import (
    BenchmarkDecoderAdapter,
    validate_adapter_output,
    validate_candidate_manifest,
)
from temporal_benchmark_protocol import (
    CONDITIONS,
    METHODS,
    calibration_seed_bundle,
    resolved_condition_settings,
)
from temporal_generalization_protocol import seal_payload


@dataclass(frozen=True)
class BenchmarkEpisode:
    """Hold one paired channel, payload, and noise realization."""

    bits: np.ndarray
    active: np.ndarray
    ue_ids: np.ndarray
    channel_change_event: np.ndarray
    receiver_inputs: tuple[tuple[dict[str, Any], ...], ...]
    oracle_inputs: tuple[tuple[dict[str, Any], ...], ...]


@runtime_checkable
class BenchmarkScenarioBank(Protocol):
    """Generate an episode solely from its frozen job and index."""

    def generate(self, job: dict, episode_index: int) -> BenchmarkEpisode:
        """Generate one deterministic episode at every planned SNR."""


def _array(value: Any) -> np.ndarray:
    """Convert eager tensors and arrays without changing values."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _hash_array(digest, value: Any) -> None:
    """Add one typed array to an episode digest."""
    array = np.ascontiguousarray(_array(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def episode_fingerprint(episode: BenchmarkEpisode) -> np.ndarray:
    """Hash every paired input and the ground-truth payload."""
    digest = hashlib.sha256()
    _hash_array(digest, episode.bits)
    _hash_array(digest, episode.active)
    _hash_array(digest, episode.ue_ids)
    _hash_array(digest, episode.channel_change_event)
    for family in (episode.receiver_inputs, episode.oracle_inputs):
        for snr_inputs in family:
            for tb_inputs in snr_inputs:
                for name in sorted(tb_inputs):
                    digest.update(name.encode("utf-8"))
                    _hash_array(digest, tb_inputs[name])
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def validate_episode(
    episode: BenchmarkEpisode, protocol: dict, job: dict
) -> None:
    """Validate one provider episode before any receiver sees it."""
    tbs = int(job["num_tbs"])
    users = int(protocol["radio"]["maximum_active_ues"])
    snrs = len(job["snr_db"])
    bits = _array(episode.bits)
    active = _array(episode.active)
    ue_ids = _array(episode.ue_ids)
    change_event = _array(episode.channel_change_event)
    if bits.ndim != 3 or bits.shape[:2] != (tbs, users):
        raise ValueError("episode bits must have shape [TB,UE,bits]")
    if bits.shape[-1] < 1:
        raise ValueError("episode payload cannot be empty")
    if active.shape != (tbs, users) or active.dtype != np.bool_:
        raise ValueError("episode activity shape or dtype differs")
    if not np.all(np.any(active, axis=1)):
        raise ValueError("every episode TB requires an active UE")
    if ue_ids.shape != (tbs, users) or not np.issubdtype(
        ue_ids.dtype, np.integer
    ):
        raise ValueError("episode UE identities differ")
    if np.any(np.diff(np.sort(ue_ids, axis=-1), axis=-1) == 0):
        raise ValueError("episode UE identities must be unique")
    if change_event.shape != (tbs, users) or change_event.dtype != np.bool_:
        raise ValueError("episode channel-change events differ")
    observable = set(protocol["execution"]["observable_receiver_inputs"])
    oracle = observable | set(protocol["execution"]["oracle_only_inputs"])
    for family, expected, label in (
        (episode.receiver_inputs, observable, "receiver"),
        (episode.oracle_inputs, oracle, "oracle"),
    ):
        if len(family) != snrs:
            raise ValueError(f"{label} SNR count differs")
        for snr_inputs in family:
            if len(snr_inputs) != tbs:
                raise ValueError(f"{label} TB count differs")
            for tb, inputs in enumerate(snr_inputs):
                if set(inputs) != expected:
                    raise ValueError(f"{label} input fields differ")
                if not np.array_equal(_array(inputs["active"]), active[tb]):
                    raise ValueError(f"{label} activity input differs")
                point_ids = _array(inputs["ue_ids"])
                if not np.array_equal(point_ids, ue_ids[tb]):
                    raise ValueError(f"{label} UE identities differ")
                slot = _array(inputs["slot_number"])
                if slot.ndim != 0 or not np.issubdtype(
                    slot.dtype, np.integer
                ):
                    raise ValueError(f"{label} slot number differs")
                for name, value in inputs.items():
                    array = _array(value)
                    if array.dtype.kind in "fc" and not np.all(
                        np.isfinite(array)
                    ):
                        raise ValueError(f"{label} input {name} is not finite")
    for snr_index in range(snrs):
        for tb in range(tbs):
            for name in observable:
                left = _array(episode.receiver_inputs[snr_index][tb][name])
                right = _array(episode.oracle_inputs[snr_index][tb][name])
                if not np.array_equal(left, right):
                    raise ValueError("oracle and receiver observations differ")


def _outputs_equal(left: dict, right: dict) -> bool:
    """Compare two decoder outputs exactly."""
    return set(left) == set(right) and all(
        np.array_equal(_array(left[name]), _array(right[name]))
        for name in left
    )


def _deterministic_first_tb(
    adapter: BenchmarkDecoderAdapter,
    inputs: dict,
    bits_shape: tuple[int, ...],
) -> bool:
    """Replay a cold first TB through one adapter."""
    adapter.start_episode()
    first = adapter.decode_tb(inputs)
    adapter.synchronize()
    validate_adapter_output(first, bits_shape)
    adapter.start_episode()
    second = adapter.decode_tb(inputs)
    adapter.synchronize()
    validate_adapter_output(second, bits_shape)
    adapter.start_episode()
    return _outputs_equal(first, second)


def _adapter_preflight(
    adapters: dict[str, BenchmarkDecoderAdapter],
    methods: tuple[str, ...],
    episode: BenchmarkEpisode,
    oracle_method: str | None,
) -> bool:
    """Replay each method once before timed evaluation."""
    bits_shape = tuple(episode.bits[0].shape)
    passed = True
    for method in methods:
        inputs = (
            episode.oracle_inputs[0][0]
            if method == oracle_method
            else episode.receiver_inputs[0][0]
        )
        passed &= _deterministic_first_tb(
            adapters[method], inputs, bits_shape
        )
    return bool(passed)


def _method_group(protocol: dict, methods: tuple[str, ...]) -> str:
    execution = protocol["execution"]
    groups = {
        "candidate": tuple(execution["candidate_methods"]),
        "matched": tuple(execution["accuracy_reference_methods"]),
        "static": tuple(execution["static_reference_methods"]),
    }
    matches = [name for name, values in groups.items() if methods == values]
    if len(matches) != 1:
        raise ValueError("method set differs from the benchmark groups")
    return matches[0]


def _latency_inputs(protocol, plan, provider):
    timing = protocol["execution"]["latency"]
    names = [job["condition"] for job in plan["jobs"]]
    if timing["condition"] not in names:
        raise ValueError("latency condition is absent from the stage")
    job = plan["jobs"][names.index(timing["condition"])]
    snr_index = plan["snr_point_names"].index(timing["snr_point"])
    needed = int(timing["warmup_samples"]) + int(timing["sample_count"])
    episodes = []
    available = 0
    for index in range(int(job["episode_count"])):
        episode = provider.generate(job, index)
        validate_episode(episode, protocol, job)
        episodes.append(episode)
        available += int(job["num_tbs"])
        if available >= needed:
            break
    if available < needed:
        raise ValueError("stage has too few TBs for latency measurement")
    return episodes, snr_index


def _measure_latency(
    protocol, plan, provider, adapters, methods
) -> dict[str, np.ndarray]:
    if not methods:
        return {}
    timing = protocol["execution"]["latency"]
    warmup = int(timing["warmup_samples"])
    count = int(timing["sample_count"])
    episodes, snr_index = _latency_inputs(protocol, plan, provider)
    result = {}
    for method in methods:
        adapter = adapters[method]
        values = []
        seen = 0
        for episode in episodes:
            adapter.start_episode()
            for tb in range(episode.bits.shape[0]):
                if method != "candidate_persistent":
                    adapter.reset_memory()
                inputs = episode.receiver_inputs[snr_index][tb]
                adapter.synchronize()
                started = time.perf_counter()
                output = adapter.decode_tb(inputs)
                adapter.synchronize()
                elapsed = 1e3 * (time.perf_counter() - started)
                validate_adapter_output(output, tuple(episode.bits[tb].shape))
                hard = _array(output["hard_bits"]).astype(bool)
                crc = _array(output["crc_status"]).astype(bool)
                error = np.any(
                    hard != episode.bits[tb].astype(bool), axis=-1
                )
                disagreement = np.logical_xor(
                    error, np.logical_not(crc)
                ) & episode.active[tb]
                if np.any(disagreement):
                    raise RuntimeError(f"{method} latency CRC disagrees")
                if seen >= warmup:
                    values.append(elapsed)
                seen += 1
                if len(values) == count:
                    break
            if len(values) == count:
                break
        if len(values) != count:
            raise ValueError(f"{method} latency sample count differs")
        result[f"latency_ms_{method}"] = np.asarray(values, np.float64)
    return result


def evaluate_method_set(
    protocol: dict,
    plan: dict,
    provider: BenchmarkScenarioBank,
    adapters: dict[str, BenchmarkDecoderAdapter],
    methods: tuple[str, ...],
    *,
    latency_methods: tuple[str, ...] = (),
    oracle_method: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, bool], float]:
    """Evaluate one method set on the immutable paired episode bank."""
    required_adapters = tuple(dict.fromkeys(methods + latency_methods))
    if tuple(adapters) != required_adapters:
        raise ValueError("adapter order differs from the method contract")
    if any(method not in METHODS for method in required_adapters):
        raise ValueError("unknown method in evaluator")
    group = _method_group(protocol, methods)
    if group == "static" and not protocol["stages"][plan["stage"]][
        "include_static_references"
    ]:
        raise ValueError("static references are absent from this stage")
    timing_key = {
        "candidate": "candidate_methods",
        "matched": "reference_methods",
        "static": None,
    }[group]
    expected_latency = (
        ()
        if timing_key is None
        else tuple(protocol["execution"]["latency"][timing_key])
    )
    if latency_methods != expected_latency:
        raise ValueError("latency methods differ from the protocol")
    expected_oracle = "perfect_csi_kbest" if group == "static" else None
    if oracle_method != expected_oracle:
        raise ValueError("oracle method differs from the protocol")
    expected_roles = {
        "candidate_persistent": "candidate",
        "candidate_reset": "candidate",
        "matched_cold_k3": "matched_cold_k3",
        "matched_cold_k8": "matched_cold_k8",
        "historical_cold_k8": "historical_cold_k8",
        "perfect_csi_kbest": "perfect_csi_kbest",
    }
    for method, adapter in adapters.items():
        if not isinstance(adapter, BenchmarkDecoderAdapter):
            raise TypeError(f"{method} does not implement the adapter contract")
        validate_candidate_manifest(adapter.manifest, protocol)
        if adapter.manifest["role"] != expected_roles[method]:
            raise ValueError(f"{method} adapter role differs")
    if group == "candidate":
        hashes = {
            adapter.manifest["manifest_sha256"]
            for adapter in adapters.values()
        }
        if len(hashes) != 1:
            raise ValueError("persistent and reset adapters differ")
    jobs = plan["jobs"]
    stage = protocol["stages"][plan["stage"]]
    conditions = len(jobs)
    snrs = len(plan["snr_point_names"])
    episodes = int(plan["episodes_per_condition"])
    tbs = int(stage["num_tbs"])
    users = int(protocol["radio"]["maximum_active_ues"])
    shape = (conditions, snrs, episodes, tbs, users)
    arrays = {
        "snr_db": np.asarray([job["snr_db"] for job in jobs], np.float64),
        "episode_index": np.tile(
            np.arange(episodes, dtype=np.int32), (conditions, 1)
        ),
        "episode_fingerprint": np.zeros(
            (conditions, episodes, 32), np.uint8
        ),
        "active": np.zeros((conditions, episodes, tbs, users), bool),
        "ue_ids": np.zeros((conditions, episodes, tbs, users), np.int32),
        "channel_change_event": np.zeros(
            (conditions, episodes, tbs, users), bool
        ),
    }
    for method in methods:
        arrays[f"errors_{method}"] = np.zeros(shape, bool)
        arrays[f"crc_disagreement_{method}"] = np.zeros(shape, bool)
    deterministic = True
    finite_outputs = True
    crc_matches = True
    state_empty = True
    started = time.perf_counter()
    configured_batch = int(
        protocol["execution"]["accuracy_episode_batch_size"]
    )
    for condition_index, job in enumerate(jobs):
        for batch_start in range(0, episodes, configured_batch):
            batch_stop = min(batch_start + configured_batch, episodes)
            episode_batch = []
            for episode_index in range(batch_start, batch_stop):
                episode = provider.generate(job, episode_index)
                validate_episode(episode, protocol, job)
                fingerprint = episode_fingerprint(episode)
                if episode_index == 0:
                    replay = provider.generate(job, episode_index)
                    validate_episode(replay, protocol, job)
                    deterministic &= np.array_equal(
                        fingerprint, episode_fingerprint(replay)
                    )
                    deterministic &= _adapter_preflight(
                        adapters,
                        required_adapters,
                        episode,
                        oracle_method,
                    )
                arrays["episode_fingerprint"][
                    condition_index, episode_index
                ] = fingerprint
                arrays["active"][condition_index, episode_index] = episode.active
                arrays["ue_ids"][condition_index, episode_index] = episode.ue_ids
                arrays["channel_change_event"][
                    condition_index, episode_index
                ] = episode.channel_change_event
                episode_batch.append(episode)
            batch_size = len(episode_batch)
            for snr_index in range(snrs):
                for adapter in adapters.values():
                    adapter.start_episode_batch(batch_size)
                    state_empty &= bool(adapter.state_is_empty())
                for tb in range(tbs):
                    truth = np.stack([
                        episode.bits[tb] for episode in episode_batch
                    ])
                    active = np.stack([
                        episode.active[tb] for episode in episode_batch
                    ])
                    for method in methods:
                        adapter = adapters[method]
                        if method != "candidate_persistent":
                            adapter.reset_memory()
                            state_empty &= bool(adapter.state_is_empty())
                        family = (
                            "oracle_inputs"
                            if method == oracle_method
                            else "receiver_inputs"
                        )
                        inputs = [
                            getattr(episode, family)[snr_index][tb]
                            for episode in episode_batch
                        ]
                        output = adapter.decode_tb_batch(inputs)
                        adapter.synchronize()
                        validate_adapter_output(output, tuple(truth.shape))
                        hard = _array(output["hard_bits"]).astype(bool)
                        crc = _array(output["crc_status"]).astype(bool)
                        logits = _array(output["logits"])
                        finite_outputs &= bool(np.all(np.isfinite(logits)))
                        errors = np.any(hard != truth.astype(bool), axis=-1)
                        errors &= active
                        disagreement = np.logical_xor(
                            errors, np.logical_not(crc)
                        )
                        disagreement &= active
                        index = (
                            condition_index,
                            snr_index,
                            slice(batch_start, batch_stop),
                            tb,
                        )
                        arrays[f"errors_{method}"][index] = errors
                        arrays[f"crc_disagreement_{method}"][index] = disagreement
                        crc_matches &= not bool(np.any(disagreement))
    arrays.update(_measure_latency(
        protocol, plan, provider, adapters, latency_methods
    ))
    for adapter in adapters.values():
        if not adapter.checksums_unchanged():
            raise RuntimeError("receiver artifact changed during evaluation")
    if group == "candidate":
        preflight = {
            "candidate_artifacts_unchanged": True,
            "candidate_crc_matches_payload_error": bool(crc_matches),
            "candidate_deterministic_replay": bool(deterministic),
            "candidate_finite_outputs": bool(finite_outputs),
            "no_future_inputs": True,
            "persistent_and_reset_paired": True,
            "receiver_inputs_observable": True,
            "reset_state_is_empty": bool(state_empty),
        }
    elif group == "matched":
        preflight = {
            "matched_reference_artifacts_unchanged": True,
            "matched_reference_crc_matches_payload_error": bool(crc_matches),
            "matched_reference_deterministic_replay": bool(deterministic),
            "matched_reference_finite_outputs": bool(finite_outputs),
            "matched_references_paired": True,
        }
    else:
        preflight = {
            "kbest_oracle_isolated": oracle_method == "perfect_csi_kbest",
            "static_reference_artifacts_unchanged": True,
            "static_reference_crc_matches_payload_error": bool(crc_matches),
            "static_reference_deterministic_replay": bool(deterministic),
            "static_reference_finite_outputs": bool(finite_outputs),
            "static_references_paired": True,
        }
    from temporal_benchmark_io import validate_common_arrays, validate_method_arrays

    validate_common_arrays(arrays, protocol, plan)
    validate_method_arrays(arrays, methods, protocol, plan)
    return arrays, preflight, time.perf_counter() - started


def evaluate_calibration_counts(
    protocol: dict,
    provider: BenchmarkScenarioBank,
    adapter: BenchmarkDecoderAdapter,
) -> dict:
    """Measure the matched Cold K=8 calibration curves."""
    validate_candidate_manifest(adapter.manifest, protocol)
    if adapter.manifest["role"] != "matched_cold_k8":
        raise ValueError("calibration requires matched Cold K=8")
    training = adapter.manifest["training"]
    calibration = protocol["calibration"]
    expected_training = {
        "updates": calibration["training_updates"],
        "examples": calibration["training_examples"],
        "seeds": [calibration["training_recipe_seed"]],
    }
    for name, value in expected_training.items():
        if training[name] != value:
            raise ValueError(f"calibration baseline differs in {name}")
    jobs = []
    for condition in CONDITIONS:
        provider_settings = resolved_condition_settings(
            protocol, condition, int(calibration["num_tbs"])
        )
        jobs.append({
            "condition": condition,
            "provider": provider_settings["provider"],
            "provider_settings": provider_settings,
            "episode_count": calibration["episodes_per_condition"],
            "num_tbs": calibration["num_tbs"],
            "snr_db": calibration["snr_grid_db"],
            "seed_streams": calibration_seed_bundle(
                protocol, condition, 0
            ),
        })
    curves = {}
    fingerprints = []
    active_bank = np.zeros(
        (
            len(jobs),
            int(calibration["episodes_per_condition"]),
            int(calibration["num_tbs"]),
            int(protocol["radio"]["maximum_active_ues"]),
        ),
        bool,
    )
    ue_bank = np.zeros_like(active_bank, dtype=np.int32)
    change_bank = np.zeros_like(active_bank, dtype=bool)
    deterministic = True
    crc_matches = True
    finite_outputs = True
    state_empty = True
    batch_size = int(protocol["execution"]["accuracy_episode_batch_size"])
    for condition_index, job in enumerate(jobs):
        snrs = job["snr_db"]
        errors = np.zeros(len(snrs), np.int64)
        blocks = np.zeros(len(snrs), np.int64)
        episode_count = int(job["episode_count"])
        for batch_start in range(0, episode_count, batch_size):
            batch_stop = min(batch_start + batch_size, episode_count)
            episode_batch = []
            for episode_index in range(batch_start, batch_stop):
                episode = provider.generate(job, episode_index)
                validate_episode(episode, protocol, job)
                fingerprint = episode_fingerprint(episode)
                fingerprints.append(fingerprint.tobytes())
                active_bank[condition_index, episode_index] = episode.active
                ue_bank[condition_index, episode_index] = episode.ue_ids
                change_bank[
                    condition_index, episode_index
                ] = episode.channel_change_event
                if episode_index == 0:
                    replay = provider.generate(job, episode_index)
                    validate_episode(replay, protocol, job)
                    deterministic &= np.array_equal(
                        fingerprint, episode_fingerprint(replay)
                    )
                    deterministic &= _deterministic_first_tb(
                        adapter,
                        episode.receiver_inputs[0][0],
                        tuple(episode.bits[0].shape),
                    )
                episode_batch.append(episode)
            for snr_index in range(len(snrs)):
                adapter.start_episode_batch(len(episode_batch))
                for tb in range(int(job["num_tbs"])):
                    adapter.reset_memory()
                    state_empty &= bool(adapter.state_is_empty())
                    inputs = [
                        episode.receiver_inputs[snr_index][tb]
                        for episode in episode_batch
                    ]
                    output = adapter.decode_tb_batch(inputs)
                    adapter.synchronize()
                    truth = np.stack([
                        episode.bits[tb] for episode in episode_batch
                    ])
                    active = np.stack([
                        episode.active[tb] for episode in episode_batch
                    ])
                    validate_adapter_output(output, tuple(truth.shape))
                    hard = _array(output["hard_bits"]).astype(bool)
                    crc = _array(output["crc_status"]).astype(bool)
                    point = np.any(hard != truth.astype(bool), axis=-1)
                    point &= active
                    disagreement = np.logical_xor(
                        point, np.logical_not(crc)
                    )
                    disagreement &= active
                    errors[snr_index] += int(np.sum(point))
                    blocks[snr_index] += int(np.sum(active))
                    crc_matches &= not bool(np.any(disagreement))
                    finite_outputs &= bool(np.all(np.isfinite(
                        _array(output["logits"])
                    )))
        curves[job["condition"]] = [
            {
                "snr_db": float(snr),
                "errors": int(error),
                "blocks": int(block),
            }
            for snr, error, block in zip(snrs, errors, blocks)
        ]
    from temporal_benchmark_io import validate_scenario_metadata
    validate_scenario_metadata(active_bank, ue_bank, change_bank, jobs)
    unchanged = bool(adapter.checksums_unchanged())
    preflight = {
        "baseline_artifacts_unchanged": unchanged,
        "crc_matches_payload_error": bool(crc_matches),
        "deterministic_replay": bool(deterministic),
        "finite_outputs": bool(finite_outputs),
        "globally_unique_episodes": len(fingerprints) == len(set(fingerprints)),
        "reset_state_is_empty": bool(state_empty),
    }
    if not all(preflight.values()):
        raise RuntimeError("calibration preflight failed")
    return seal_payload({
        "schema_version": 1,
        "result": "temporal_benchmark_calibration_counts_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "method": "matched_cold_k8",
        "baseline_manifest": adapter.manifest,
        "episodes_per_condition": calibration["episodes_per_condition"],
        "num_tbs": calibration["num_tbs"],
        "condition_names": list(CONDITIONS),
        "preflight": preflight,
        "curves": curves,
    }, field="result_sha256")
