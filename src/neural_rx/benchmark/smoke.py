#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Smoke-test receivers using pretrained neural_rx references.')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=2.75)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True,
                        help="Memory-receiver checkpoint export directory or its receiver.weights.h5")
    parser.add_argument("--episode-cache", type=Path,
                        default=ROOT / "datasets/smoke")
    parser.add_argument("--prbs", type=int, default=132)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/historical"
            / "temporal_benchmark_adapter_smoke"
            / "adapter_smoke.json"
        ),
    )
    return parser


def _crc_diagnostics(output, truth, active):
    from temporal_benchmark_scoring import crc_diagnostics, payload_error_flags
    return crc_diagnostics(*payload_error_flags(
        output["hard_bits"], output["crc_status"], truth, active,
    ))


def main() -> None:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import sionna as sn
    import tensorflow as tf
    from temporal_benchmark_candidate import (
        validate_adapter_output,
    )
    from temporal_benchmark_protocol import (
        resolved_condition_settings,
    )
    from temporal_benchmark_scenarios import TemporalBenchmarkScenarioBank
    from temporal_benchmark_episodes import CachedScenarioBank, scenario_identity
    from utils import E2E_Model, Parameters

    tf.config.experimental.enable_op_determinism()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    seed = 20_262_100
    tf.random.set_seed(seed)
    np.random.seed(seed)
    sn.config.seed = seed

    from temporal_benchmark_frozen import make_benchmark_adapter, make_manifest, protocol_and_plan
    if args.prbs < 1 or args.prbs > 273:
        raise ValueError("Invalid smoke-test PRB count")
    protocol, plan = protocol_and_plan("smoke", args.prbs)
    radio_config = ROOT / "config" / "nrx_large.cfg"
    registrations = {
        role: make_manifest(protocol, role,
            args.candidate_checkpoint if role == "candidate" else None)
        for role in ("candidate", "frozen_cold_k3", "frozen_cold_k8")
    }

    parameters = Parameters(
        "nrx_large.cfg",
        training=False,
        num_tx_eval=2,
        n_size_bwp_override=args.prbs,
    )
    e2e_model = E2E_Model(
        parameters, training=False, mcs_arr_eval_idx=0
    )
    provider = CachedScenarioBank(
        TemporalBenchmarkScenarioBank(parameters, e2e_model, tf),
        args.episode_cache, protocol, scenario_identity(radio_config, protocol),
    )
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
        "seed_streams": plan["jobs"][0]["seed_streams"],
    }
    episode = provider.generate(job, 0)
    replay = provider.generate(job, 0)
    second_episode = provider.generate(job, 1)
    first_fingerprint = episode.fingerprint
    replay_fingerprint = replay.fingerprint
    if not np.array_equal(first_fingerprint, replay_fingerprint):
        raise RuntimeError("scenario replay changed its episode")

    context = {
        "config_name": "nrx_large.cfg",
        "tf_module": tf,
    }

    def make(method, role):
        manifest, artifact_map, training_seed = registrations[role]
        return make_benchmark_adapter(
            manifest=manifest,
            method=method,
            training_seed=training_seed,
            artifact_map=artifact_map,
            **context,
        )

    adapters = {
        "candidate_persistent": make("candidate_persistent", "candidate"),
        "candidate_reset": make("candidate_reset", "candidate"),
        "frozen_cold_k3": make("frozen_cold_k3", "frozen_cold_k3"),
        "frozen_cold_k8": make("frozen_cold_k8", "frozen_cold_k8"),
    }
    for adapter in adapters.values():
        adapter.include_logits = True
        adapter.start_episode()
        if not adapter.state_is_empty():
            raise RuntimeError("adapter did not start empty")

    outputs = {}
    latency_ms = {}
    crc_counts = {}
    observable = episode.receiver_inputs[0]
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
    latency_ms["candidate_persistent"] = 1e3 * (
        time.perf_counter() - started
    )
    if not np.any(persistent._state.valid.numpy()):
        raise RuntimeError("persistent channel memory did not retain memory")

    reset = adapters["candidate_reset"]
    for tb in range(num_tbs):
        reset.reset_memory()
        if not reset.state_is_empty():
            raise RuntimeError("reset candidate retained memory before decode")
        outputs[("candidate_reset", tb)] = reset.decode_tb(observable[tb])

    from verification.pilot_history import ReferenceMemoryManager, make_decode_step
    manager = persistent.manager
    reference_manager = ReferenceMemoryManager(manager, persistent.layout.count, persistent._channel_shape[-1]+1)
    reference_decode = make_decode_step(persistent.receiver, persistent.model, reference_manager, persistent.decoder)
    memory_drift = {}
    for method, adapter in (("candidate_persistent", persistent), ("candidate_reset", reset)):
        state = reference_manager.zero_state(1)
        for tb, point in enumerate(observable):
            if method == "candidate_reset":
                state = reference_manager.zero_state(1)
            hard, crc, state = reference_decode(
                tf.constant(point["received_grid"][None]),
                tf.constant(point["ls_channel"][None]),
                tf.constant([point["noise_variance"]], tf.float32),
                tf.constant(point["active"][None], tf.float32),
                tf.constant(point["ue_ids"][None], tf.int32),
                tf.constant(tb, tf.int32), state,
            )
            np.testing.assert_array_equal(hard.numpy()[0].astype(bool), outputs[(method, tb)]["hard_bits"])
            np.testing.assert_array_equal(crc.numpy()[0], outputs[(method, tb)]["crc_status"])
        identities = tf.range(manager.capacity)[None]
        active_ids = tf.ones_like(identities, tf.bool)
        actual = adapter._buffer.read(adapter._committed.head, adapter._committed.count,
                                      identities, active_ids, tf.constant(num_tbs))
        expected = reference_manager.gather(state, identities, num_tbs, active_ids)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left.numpy(), right.numpy())
        memory_drift[method] = {"max_absolute": 0., "rms": 0.,
                                "reference_rms": float(tf.sqrt(tf.reduce_mean(expected.observations**2)))}
    standalone_equal = True

    for method in (
        "frozen_cold_k3",
        "frozen_cold_k8",
    ):
        adapter = adapters[method]
        for tb in range(num_tbs):
            if tb == warm_tb:
                started = time.perf_counter()
            outputs[(method, tb)] = adapter.decode_tb(observable[tb])
            if tb == warm_tb and method == "frozen_cold_k3":
                latency_ms[method] = 1e3 * (
                    time.perf_counter() - started
                )


    for (method, tb), output in outputs.items():
        validate_adapter_output(output, tuple(truth[tb].shape))
        counts = _crc_diagnostics(output, truth[tb], active[tb])
        totals = crc_counts.setdefault(method, {name: 0 for name in counts})
        for name, value in counts.items():
            totals[name] += value
    first_equal = all(
        np.array_equal(
            outputs[("candidate_persistent", 0)][name],
            outputs[("candidate_reset", 0)][name],
        )
        for name in ("hard_bits", "crc_status", "logits")
    )
    if not first_equal:
        raise RuntimeError("persistent and reset candidates differ when cold")
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
            inputs = episode_value.receiver_inputs[0][tb]
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
            point = adapter.decode_tb_batch([
                value.receiver_inputs[0][tb]
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
        "receiver": registrations["candidate"][0]["training"]["binding"]["architecture"]["memory"],
        "prbs": args.prbs,
        "standalone_decoder_and_memory_equivalence": standalone_equal,
        "episode_cache": str(provider.directory),
        "num_tbs": num_tbs,
        "episode_fingerprint": first_fingerprint.tobytes().hex(),
        "crc_diagnostics": crc_counts,
        "checks": {
            "identical_episode_replay": True,
            "persistent_state_retained": True,
            "persistent_history_available_on_third_tb": True,
            "reset_state_empty_before_each_tb": True,
            "reset_history_unavailable_on_each_tb": True,
            "persistent_reset_cold_outputs_equal": True,
            "finite_outputs": True,
            "frozen_artifacts_unchanged": True,
            "latency_measured_after_warmup": True,
            "batched_sequential_decisions_equal": True,
        },
        "memory_reference_drift": memory_drift,
        "memory_equivalence": "numerical (TF32); payload and CRC comparisons are exact",
        "batch_logit_max_absolute_difference": batch_logit_drift,
        "latency_ms": latency_ms,
        "methods": list(adapters),
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    from artifact_io import atomic_json
    atomic_json(destination, result)
    print(json.dumps(result, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
