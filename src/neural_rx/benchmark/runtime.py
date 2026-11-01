#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from temporal_benchmark_candidate import (
    BenchmarkDecoderAdapter,
    validate_adapter_output,
)
from temporal_benchmark_candidate import validate_manifest
from temporal_benchmark_scoring import payload_error_flags
from temporal_benchmark_protocol import METHOD_ROLES


@dataclass(frozen=True)
class BenchmarkEpisode:

    bits: np.ndarray
    active: np.ndarray
    ue_ids: np.ndarray
    channel_change_event: np.ndarray
    receiver_inputs: tuple[tuple[dict[str, Any], ...], ...]
    fingerprint: np.ndarray | None = field(default=None, repr=False, compare=False)


@runtime_checkable
class BenchmarkScenarioBank(Protocol):

    def generate(self, job: dict, episode_index: int) -> BenchmarkEpisode:
        ...


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _hash_array(digest, value: Any) -> None:
    array = np.ascontiguousarray(_array(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def episode_fingerprint(episode: BenchmarkEpisode) -> np.ndarray:
    digest = hashlib.sha256()
    _hash_array(digest, episode.bits)
    _hash_array(digest, episode.active)
    _hash_array(digest, episode.ue_ids)
    _hash_array(digest, episode.channel_change_event)
    for snr_inputs in episode.receiver_inputs:
        for tb_inputs in snr_inputs:
            for name in sorted(tb_inputs):
                digest.update(name.encode("utf-8"))
                _hash_array(digest, tb_inputs[name])
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def validate_episode(
    episode: BenchmarkEpisode, protocol: dict, job: dict
) -> BenchmarkEpisode:
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
    if len(episode.receiver_inputs) != snrs:
        raise ValueError("receiver SNR count differs")
    for snr_inputs in episode.receiver_inputs:
        if len(snr_inputs) != tbs:
            raise ValueError("receiver TB count differs")
        for tb, inputs in enumerate(snr_inputs):
            if set(inputs) != observable:
                raise ValueError("receiver input fields differ")
            if not np.array_equal(_array(inputs["active"]), active[tb]):
                raise ValueError("receiver activity input differs")
            if not np.array_equal(_array(inputs["ue_ids"]), ue_ids[tb]):
                raise ValueError("receiver UE identities differ")
            slot = _array(inputs["slot_number"])
            if slot.ndim != 0 or not np.issubdtype(slot.dtype, np.integer):
                raise ValueError("receiver slot number differs")
            for name, value in inputs.items():
                array = _array(value)
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise ValueError(f"receiver input {name} is not finite")
                array.setflags(write=False)
    for array in (bits, active, ue_ids, change_event):
        array.setflags(write=False)
    fingerprint = episode_fingerprint(episode)
    fingerprint.setflags(write=False)
    return replace(episode, fingerprint=fingerprint)


def _outputs_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(_array(left[name]), _array(right[name]))
        for name in left
    )


def _deterministic_first_tb(
    adapter: BenchmarkDecoderAdapter,
    inputs: dict,
    bits_shape: tuple[int, ...],
) -> bool:
    adapter.start_episode()
    first = adapter.decode_tb(inputs)
    validate_adapter_output(first, bits_shape)
    adapter.start_episode()
    second = adapter.decode_tb(inputs)
    validate_adapter_output(second, bits_shape)
    adapter.start_episode()
    return _outputs_equal(first, second)


def _adapter_preflight(
    adapters: dict[str, BenchmarkDecoderAdapter],
    methods: tuple[str, ...],
    episode: BenchmarkEpisode,
) -> bool:
    bits_shape = tuple(episode.bits[0].shape)
    passed = True
    for method in methods:
        inputs = episode.receiver_inputs[0][0]
        passed &= _deterministic_first_tb(
            adapters[method], inputs, bits_shape
        )
    return bool(passed)


def _method_group(protocol: dict, methods: tuple[str, ...]) -> str:
    execution = protocol["execution"]
    groups = {
        "candidate": tuple(execution["candidate_methods"]),
        "frozen": tuple(execution["accuracy_reference_methods"]),
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
        if episode.fingerprint is None:
            episode = validate_episode(episode, protocol, job)
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
    timing = protocol["execution"]["latency"]
    warmup = int(timing["warmup_samples"])
    count = int(timing["sample_count"])
    episodes, snr_index = _latency_inputs(protocol, plan, provider)
    result = {}
    for method in methods:
        adapter = adapters[method]
        values = []
        payload_errors, crc_disagreements = [], []
        seen = 0
        for episode in episodes:
            adapter.start_episode()
            for tb in range(episode.bits.shape[0]):
                if method not in protocol["execution"].get("persistent_methods", ["candidate_persistent"]):
                    adapter.reset_memory()
                inputs = episode.receiver_inputs[snr_index][tb]
                started = time.perf_counter()
                output = adapter.decode_tb(inputs)
                elapsed = 1e3 * (time.perf_counter() - started)
                validate_adapter_output(output, tuple(episode.bits[tb].shape))
                errors, disagreement = payload_error_flags(
                    _array(output["hard_bits"]), _array(output["crc_status"]),
                    episode.bits[tb], episode.active[tb],
                )
                if seen >= warmup:
                    values.append(elapsed)
                    payload_errors.append(errors)
                    crc_disagreements.append(disagreement)
                seen += 1
                if len(values) == count:
                    break
            if len(values) == count:
                break
        if len(values) != count:
            raise ValueError(f"{method} latency sample count differs")
        result[f"latency_ms_{method}"] = np.asarray(values, np.float64)
        result[f"latency_errors_{method}"] = np.asarray(payload_errors, bool)
        result[f"latency_crc_disagreement_{method}"] = np.asarray(crc_disagreements, bool)
    return result


def evaluate_method_set(
    protocol: dict,
    plan: dict,
    provider: BenchmarkScenarioBank,
    adapters: dict[str, BenchmarkDecoderAdapter],
    methods: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, bool], float]:
    if tuple(adapters) != methods:
        raise ValueError("adapter order differs from the method contract")
    allowed = protocol["methods"]
    if any(method not in allowed for method in methods):
        raise ValueError("unknown method in evaluator")
    group = _method_group(protocol, methods)
    for method, adapter in adapters.items():
        if not isinstance(adapter, BenchmarkDecoderAdapter):
            raise TypeError(f"{method} does not implement the adapter contract")
        validate_manifest(adapter.manifest, protocol)
        if adapter.manifest["role"] != METHOD_ROLES[method]:
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
                if episode.fingerprint is None:
                    episode = validate_episode(episode, protocol, job)
                fingerprint = episode.fingerprint
                if episode_index == 0:
                    deterministic &= _adapter_preflight(
                        adapters,
                        methods,
                        episode,
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
                        if method not in protocol["execution"].get("persistent_methods", ["candidate_persistent"]):
                            adapter.reset_memory()
                            state_empty &= bool(adapter.state_is_empty())
                        inputs = [
                            episode.receiver_inputs[snr_index][tb]
                            for episode in episode_batch
                        ]
                        output = adapter.decode_tb_batch(inputs)
                        validate_adapter_output(output, tuple(truth.shape))
                        errors, disagreement = payload_error_flags(
                            _array(output["hard_bits"]), _array(output["crc_status"]),
                            truth, active,
                        )
                        index = (
                            condition_index,
                            snr_index,
                            slice(batch_start, batch_stop),
                            tb,
                        )
                        arrays[f"errors_{method}"][index] = errors
                        arrays[f"crc_disagreement_{method}"][index] = disagreement
    for adapter in adapters.values():
        if not adapter.checksums_unchanged():
            raise RuntimeError("receiver artifact changed during evaluation")
    if group == "candidate":
        preflight = {
            "candidate_artifacts_unchanged": True,
            "candidate_deterministic_replay": bool(deterministic),
            "candidate_finite_outputs": True,
            "no_future_inputs": True,
            "persistent_and_reset_paired": True,
            "receiver_inputs_observable": True,
            "reset_state_is_empty": bool(state_empty),
        }
    else:
        preflight = {
            "frozen_reference_artifacts_unchanged": True,
            "frozen_reference_deterministic_replay": bool(deterministic),
            "frozen_reference_finite_outputs": True,
            "frozen_references_paired": True,
        }
    from temporal_benchmark_io import validate_common_arrays, validate_method_arrays

    validate_common_arrays(arrays, protocol, plan)
    validate_method_arrays(arrays, methods, protocol, plan)
    return arrays, preflight, time.perf_counter() - started
