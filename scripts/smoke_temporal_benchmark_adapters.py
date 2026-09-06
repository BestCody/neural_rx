#!/usr/bin/env python3
"""Smoke-test real Temporal Benchmark v1 adapters."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=2.75)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "temporal_benchmark_adapter_smoke_20260904"
            / "adapter_smoke.json"
        ),
    )
    return parser


def _manifest(
    protocol,
    role,
    factory,
    checkpoint,
    support,
    seed=20260914,
):
    from temporal_benchmark_candidate import build_candidate_manifest

    k = {
        "candidate": 2,
        "matched_cold_k3": 3,
        "matched_cold_k8": 8,
        "historical_cold_k8": 8,
        "perfect_csi_kbest": 0,
    }[role]
    oracle = role == "perfect_csi_kbest"
    historical = role == "historical_cold_k8"
    seeds = [] if oracle else [1999 if historical else int(seed)]
    return build_candidate_manifest(
        candidate_id=f"smoke_{role}",
        role=role,
        architecture=f"Smoke fixture for {role}",
        adapter_factory=factory,
        benchmark_protocol_sha256=protocol["protocol_sha256"],
        radio=protocol["radio"],
        training_data_sha256="1" * 64,
        training_recipe_sha256="2" * 64,
        checkpoint_selection_sha256="3" * 64,
        training_seeds=seeds,
        training_examples=0 if oracle else 500,
        training_updates=0 if oracle else 500,
        inference_iterations=k,
        persistent_memory=role == "candidate",
        memory_horizon_tbs=8 if role == "candidate" else None,
        checkpoint_artifacts=[] if oracle else [checkpoint],
        support_artifacts=support,
        structural_change="Smoke-only adapter fixture",
        hypothesis="The production adapter obeys its runtime contract.",
        expected_metric="worst_core_k8_gap",
        minimum_useful_improvement=0.005,
        maximum_gpu_hours=1.0,
        stop_rule="Stop after three transport blocks.",
    )


def _artifact_map(paths):
    from temporal_benchmark_candidate import file_sha256

    return {file_sha256(path): Path(path).resolve() for path in paths}


def _crc_matches(output, truth, active):
    hard = np.asarray(output["hard_bits"], bool)
    crc = np.asarray(output["crc_status"], bool)
    error = np.any(hard != np.asarray(truth, bool), axis=-1)
    return not np.any(
        np.logical_xor(error, np.logical_not(crc)) & active
    )


def main() -> None:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import sionna as sn
    import tensorflow as tf
    from temporal_benchmark_candidate import (
        load_adapter,
        resolve_manifest_artifacts,
        validate_adapter_output,
    )
    from temporal_benchmark_protocol import (
        load_benchmark_protocol,
        resolved_condition_settings,
        stage_seed_bundle,
    )
    from temporal_benchmark_runtime import (
        episode_fingerprint,
        validate_episode,
    )
    from temporal_benchmark_scenarios import TemporalBenchmarkScenarioBank
    from utils import E2E_Model, Parameters

    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    seed = 20_262_100
    tf.random.set_seed(seed)
    np.random.seed(seed)
    sn.config.seed = seed

    protocol = load_benchmark_protocol(
        ROOT / "config" / "temporal_benchmark_v1.json"
    )
    checkpoint = (
        ROOT
        / "outputs"
        / "v18_aligned_screen_v3_20260905"
        / "train"
        / "candidate"
        / "candidate_selected.npz"
    )
    base_weights = ROOT / "weights" / "nrx_large_weights"
    matched_k8_weights = (
        ROOT
        / "outputs"
        / "v18_aligned_screen_v3_20260905"
        / "train"
        / "matched_cold_k8"
        / "matched_cold_k8.weights.h5"
    )
    candidate_config = (
        ROOT / "config" / "temporal_receiver_v18_segment_safe.json"
    )
    candidate_factory = (
        "temporal_benchmark_adapters:make_v18_segment_safe_adapter"
    )
    threshold = ROOT / "config" / "segment_safe_pd_l8_threshold.json"
    radio_config = ROOT / "config" / "nrx_large.cfg"
    candidate_checkpoint = checkpoint
    candidate_support = [candidate_config, threshold, radio_config]
    manifests = {
        "candidate": _manifest(
            protocol,
            "candidate",
            candidate_factory,
            candidate_checkpoint,
            candidate_support,
        ),
        "matched_cold_k3": _manifest(
            protocol,
            "matched_cold_k3",
            "temporal_benchmark_adapters:make_cold_nrx_adapter",
            base_weights,
            [radio_config],
        ),
        "matched_cold_k8": _manifest(
            protocol,
            "matched_cold_k8",
            "temporal_benchmark_adapters:make_cold_nrx_adapter",
            matched_k8_weights,
            [radio_config],
        ),
        "historical_cold_k8": _manifest(
            protocol,
            "historical_cold_k8",
            "temporal_benchmark_adapters:make_cold_nrx_adapter",
            base_weights,
            [radio_config],
        ),
        "perfect_csi_kbest": _manifest(
            protocol,
            "perfect_csi_kbest",
            "temporal_benchmark_adapters:make_perfect_csi_kbest_adapter",
            None,
            [radio_config],
        ),
    }
    paths = [
        candidate_checkpoint,
        base_weights,
        matched_k8_weights,
        candidate_config,
        threshold,
        radio_config,
    ]
    artifact_map = _artifact_map(paths)

    parameters = Parameters(
        "nrx_large.cfg",
        training=False,
        num_tx_eval=2,
        system="nrx",
        n_size_bwp_override=132,
    )
    e2e_model = E2E_Model(
        parameters, training=False, mcs_arr_eval_idx=0
    )
    provider = TemporalBenchmarkScenarioBank(parameters, e2e_model, tf)
    condition = "historical_tdl_b100"
    job = {
        "condition": condition,
        "provider": protocol["conditions"][condition]["provider"],
        "provider_settings": resolved_condition_settings(
            protocol, condition, 2
        ),
        "episode_count": 2,
        "num_tbs": 3,
        "snr_db": [float(args.snr_db)],
        "seed_streams": stage_seed_bundle(
            protocol, "screen", condition, 0
        ),
    }
    episode = provider.generate(job, 0)
    replay = provider.generate(job, 0)
    second_episode = provider.generate(job, 1)
    validate_episode(episode, protocol, job)
    validate_episode(replay, protocol, job)
    first_fingerprint = episode_fingerprint(episode)
    replay_fingerprint = episode_fingerprint(replay)
    if not np.array_equal(first_fingerprint, replay_fingerprint):
        raise RuntimeError("scenario replay changed its episode")

    context = {
        "config_name": "nrx_large.cfg",
        "parameters": parameters,
        "e2e_model": e2e_model,
        "tf_module": tf,
    }

    def make(method, role):
        manifest = manifests[role]
        artifacts = resolve_manifest_artifacts(manifest, artifact_map)
        return load_adapter(
            manifest["adapter_factory"],
            manifest=manifest,
            method=method,
            training_seed=20260914,
            artifacts=artifacts,
            **context,
        )

    adapters = {
        "candidate_persistent": make("candidate_persistent", "candidate"),
        "candidate_reset": make("candidate_reset", "candidate"),
        "matched_cold_k3": make("matched_cold_k3", "matched_cold_k3"),
        "matched_cold_k8": make("matched_cold_k8", "matched_cold_k8"),
        "historical_cold_k8": make(
            "historical_cold_k8", "historical_cold_k8"
        ),
        "perfect_csi_kbest": make(
            "perfect_csi_kbest", "perfect_csi_kbest"
        ),
    }
    for adapter in adapters.values():
        adapter.start_episode()
        if not adapter.state_is_empty():
            raise RuntimeError("adapter did not start empty")

    outputs = {}
    latency_ms = {}
    crc_matches = True
    observable = episode.receiver_inputs[0]
    oracle = episode.oracle_inputs[0]
    truth = episode.bits
    active = episode.active

    persistent = adapters["candidate_persistent"]
    num_tbs = 3
    warm_tb = 2
    outputs[("candidate_persistent", 0)] = persistent.decode_tb(observable[0])
    if persistent.state_is_empty():
        raise RuntimeError("persistent candidate did not retain UE state")
    outputs[("candidate_persistent", 1)] = persistent.decode_tb(
        observable[1]
    )
    started = time.perf_counter()
    outputs[("candidate_persistent", warm_tb)] = persistent.decode_tb(
        observable[warm_tb]
    )
    persistent.synchronize()
    latency_ms["candidate_persistent"] = 1e3 * (
        time.perf_counter() - started
    )
    persistent_valid = persistent.memory.last_point.history_valid[0]
    if not np.any(persistent_valid[active[warm_tb]]):
        raise RuntimeError("persistent v18 did not expose warm history")

    reset = adapters["candidate_reset"]
    for tb in range(num_tbs):
        reset.reset_memory()
        if not reset.state_is_empty():
            raise RuntimeError("reset candidate retained memory before decode")
        outputs[("candidate_reset", tb)] = reset.decode_tb(observable[tb])
        if np.any(reset.memory.last_point.history_valid):
            raise RuntimeError("reset candidate consumed earlier history")

    for method in (
        "matched_cold_k3",
        "matched_cold_k8",
        "historical_cold_k8",
    ):
        adapter = adapters[method]
        for tb in range(num_tbs):
            if tb == warm_tb:
                started = time.perf_counter()
            outputs[(method, tb)] = adapter.decode_tb(observable[tb])
            if tb == warm_tb and method == "matched_cold_k3":
                adapter.synchronize()
                latency_ms[method] = 1e3 * (
                    time.perf_counter() - started
                )

    oracle_adapter = adapters["perfect_csi_kbest"]
    outputs[("perfect_csi_kbest", 0)] = oracle_adapter.decode_tb(oracle[0])

    for (method, tb), output in outputs.items():
        validate_adapter_output(output, tuple(truth[tb].shape))
        crc_matches &= _crc_matches(output, truth[tb], active[tb])
    first_equal = all(
        np.array_equal(
            outputs[("candidate_persistent", 0)][name],
            outputs[("candidate_reset", 0)][name],
        )
        for name in ("hard_bits", "crc_status", "logits")
    )
    if not first_equal:
        raise RuntimeError("persistent and reset candidates differ when cold")
    if not crc_matches:
        raise RuntimeError("CRC status disagrees with payload error")
    if not all(
        np.isfinite(value) and value > 0.0
        for value in latency_ms.values()
    ):
        raise RuntimeError("latency measurement is invalid")
    if not all(
        adapter.checksums_unchanged() for adapter in adapters.values()
    ):
        raise RuntimeError("an adapter changed a frozen artifact")

    def sequential_outputs(adapter, method, episode_value):
        adapter.start_episode()
        values = []
        for tb in range(num_tbs):
            if method != "candidate_persistent":
                adapter.reset_memory()
            inputs = (
                episode_value.oracle_inputs[0][tb]
                if method == "perfect_csi_kbest"
                else episode_value.receiver_inputs[0][tb]
            )
            values.append(adapter.decode_tb(inputs))
        return values

    batch_decision_mismatches = []
    batch_logit_drift = {}
    for method, adapter in adapters.items():
        sequential = [
            sequential_outputs(adapter, method, value)
            for value in (episode, second_episode)
        ]
        adapter.start_episode_batch(2)
        for tb in range(num_tbs):
            if method != "candidate_persistent":
                adapter.reset_memory()
            family = (
                "oracle_inputs"
                if method == "perfect_csi_kbest"
                else "receiver_inputs"
            )
            point = adapter.decode_tb_batch([
                getattr(value, family)[0][tb]
                for value in (episode, second_episode)
            ])
            for name in point:
                expected = np.stack([
                    sequential[index][tb][name] for index in range(2)
                ])
                if name == "logits":
                    left = np.asarray(point[name])
                    right = np.asarray(expected)
                    batch_logit_drift[method] = max(
                        batch_logit_drift.get(method, 0.0),
                        float(np.max(np.abs(left - right))),
                    )
                    continue
                if not np.array_equal(point[name], expected):
                    batch_decision_mismatches.append({
                        "method": method,
                        "tb": tb,
                        "field": name,
                        "different_values": int(np.sum(
                            np.asarray(point[name]) != np.asarray(expected)
                        )),
                    })
    if batch_decision_mismatches:
        raise RuntimeError(
            "batched and sequential decisions differ: "
            + json.dumps(batch_decision_mismatches, sort_keys=True)
        )
    if not all(
        adapter.checksums_unchanged() for adapter in adapters.values()
    ):
        raise RuntimeError("batched decoding changed a frozen artifact")

    result = {
        "schema_version": 1,
        "result": "temporal_benchmark_adapter_smoke_v1",
        "not_benchmark_evidence": True,
        "condition": condition,
        "snr_db": float(args.snr_db),
        "candidate_version": args.candidate_version,
        "num_tbs": num_tbs,
        "episode_fingerprint": first_fingerprint.tobytes().hex(),
        "checks": {
            "identical_episode_replay": True,
            "persistent_state_retained": True,
            "persistent_history_used_on_third_tb": True,
            "reset_state_empty_before_each_tb": True,
            "reset_history_unavailable_on_each_tb": True,
            "persistent_reset_cold_outputs_equal": True,
            "crc_matches_payload_error": True,
            "finite_outputs": True,
            "frozen_artifacts_unchanged": True,
            "latency_measured_after_warmup": True,
            "batched_sequential_decisions_equal": True,
        },
        "batch_logit_max_absolute_difference": batch_logit_drift,
        "latency_ms": latency_ms,
        "methods": list(adapters),
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
