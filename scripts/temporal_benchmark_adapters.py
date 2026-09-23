#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np

from temporal_benchmark_candidate import file_sha256


METHOD_ROLES = {
    "candidate_persistent": "candidate",
    "candidate_reset": "candidate",
    "matched_cold_k3": "matched_cold_k3",
    "matched_cold_k8": "matched_cold_k8",
    "historical_cold_k8": "historical_cold_k8",
    "perfect_csi_kbest": "perfect_csi_kbest",
}
_TEMPORAL_RUNTIME_CACHE = {}


def _verify_method(manifest: dict, method: str) -> None:
    expected = METHOD_ROLES.get(method)
    if expected is None or manifest["role"] != expected:
        raise ValueError("benchmark method and manifest role differ")


def _record_path(record: dict, artifacts: dict[str, Path]) -> Path:
    digest = record["sha256"]
    if digest not in artifacts:
        raise FileNotFoundError(f"resolved artifact lacks {digest}")
    return Path(artifacts[digest]).resolve()


def _checkpoint_path(
    manifest: dict,
    training_seed: int,
    artifacts: dict[str, Path],
) -> Path:
    records = manifest["checkpoints"]
    if manifest["role"] in ("historical_cold_k8", "matched_cold_k3") and len(
        records
    ) == 1:
        return _record_path(records[0], artifacts)
    matches = [
        value
        for value in records
        if int(value["training_seed"]) == int(training_seed)
    ]
    if len(matches) != 1:
        raise ValueError("training seed must resolve one checkpoint")
    return _record_path(matches[0], artifacts)


def _support_path(
    manifest: dict,
    artifacts: dict[str, Path],
    name: str,
) -> Path:
    matches = [
        value
        for value in manifest["support_artifacts"]
        if value["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain one {name}")
    return _record_path(matches[0], artifacts)


def _runtime_config(
    manifest: dict,
    artifacts: dict[str, Path],
    config_name: str,
) -> str:
    name = Path(config_name).name
    registered = _support_path(manifest, artifacts, name)
    repository = Path(__file__).resolve().parent.parent / "config" / name
    if not repository.is_file():
        raise FileNotFoundError(repository)
    if file_sha256(registered) != file_sha256(repository):
        raise ValueError("runtime config differs from registered artifact")
    return name


class _ArtifactGuard:
    def __init__(self, manifest: dict, artifacts: dict[str, Path]):
        self._artifact_paths = tuple(sorted(
            (Path(path).resolve() for path in artifacts.values()),
            key=str,
        ))
        self._artifact_hashes = tuple(
            file_sha256(path) for path in self._artifact_paths
        )
        expected = {
            value["sha256"]
            for value in (
                list(manifest["checkpoints"])
                + list(manifest["support_artifacts"])
            )
        }
        if set(self._artifact_hashes) != expected:
            raise ValueError("resolved artifact set differs from manifest")

    def _artifacts_unchanged(self) -> bool:
        return self._artifact_hashes == tuple(
            file_sha256(path) for path in self._artifact_paths
        )


class V18BenchmarkAdapter(_ArtifactGuard):

    temporal_config_name = "temporal_receiver_v18_segment_safe.json"
    threshold_name = "segment_safe_pd_l8_threshold.json"

    @staticmethod
    def _load_temporal_config(path):
        from cjlr_final_runtime import load_v18_segment_safe_config

        return load_v18_segment_safe_config(path)

    @staticmethod
    def _load_memory_threshold(path):
        from temporal_generalization_protocol import (
            load_segment_safe_threshold_artifact,
        )

        return load_segment_safe_threshold_artifact(path)["calibration"][
            "threshold"
        ]

    @staticmethod
    def _memory_type():
        from pilot_domain_memory_v19 import (
            OnlineStreamedSegmentSafePilotDomainMemory,
        )

        return OnlineStreamedSegmentSafePilotDomainMemory

    @staticmethod
    def _runtime_type():
        from cjlr_final_runtime import FrozenV18Runtime

        return FrozenV18Runtime

    @staticmethod
    def _base_checkpoint(manifest, training_seed, artifacts):
        return _checkpoint_path(manifest, training_seed, artifacts)

    @staticmethod
    def _runtime_kwargs(manifest, training_seed, artifacts):
        del manifest, training_seed, artifacts
        return {}

    def __init__(
        self,
        *,
        manifest,
        method,
        training_seed,
        artifacts,
        config_name,
        parameters,
        e2e_model,
        tf_module,
    ):
        del e2e_model
        _verify_method(manifest, method)
        if method not in ("candidate_persistent", "candidate_reset"):
            raise ValueError("v18 adapter requires a candidate method")
        super().__init__(manifest, artifacts)
        checkpoint = self._base_checkpoint(
            manifest, training_seed, artifacts
        )
        config_name = _runtime_config(
            manifest, artifacts, config_name
        )
        config_path = _support_path(
            manifest, artifacts, self.temporal_config_name
        )
        threshold_path = _support_path(
            manifest, artifacts, self.threshold_name
        )
        config = self._load_temporal_config(config_path)
        threshold = self._load_memory_threshold(threshold_path)
        self.manifest = manifest
        self.method = method
        self.tf = tf_module
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.receiver_config = config["receiver"]
        self.parameters = parameters
        self.threshold = threshold
        self.runtime_kwargs = self._runtime_kwargs(
            manifest, training_seed, artifacts
        )
        self.maximum_history_tbs = config["memory"][
            "maximum_history_tbs"
        ]
        self.runtime_type = self._runtime_type()
        self.temporal_config_digest = file_sha256(config_path)
        runtime_key = self._runtime_key(1)
        if runtime_key not in _TEMPORAL_RUNTIME_CACHE:
            _TEMPORAL_RUNTIME_CACHE[runtime_key] = self.runtime_type(
                tf_module,
                config_name,
                config["receiver"],
                checkpoint,
                batch_size=1,
                **self.runtime_kwargs,
            )
        self.runtime = _TEMPORAL_RUNTIME_CACHE[runtime_key]
        self._used_runtimes = {1: self.runtime}
        self.memory_type = self._memory_type()
        self.memories = []
        self.start_episode()

    def _runtime_key(self, batch_size: int):
        return (
            id(self.tf),
            self.runtime_type.__module__,
            self.runtime_type.__qualname__,
            self.temporal_config_digest,
            self.config_name,
            str(self.checkpoint),
            tuple(sorted(
                (name, str(value))
                for name, value in self.runtime_kwargs.items()
            )),
            int(batch_size),
        )

    def _runtime(self, batch_size: int):
        key = self._runtime_key(batch_size)
        if key not in _TEMPORAL_RUNTIME_CACHE:
            _TEMPORAL_RUNTIME_CACHE[key] = self.runtime_type(
                self.tf,
                self.config_name,
                self.receiver_config,
                self.checkpoint,
                batch_size=int(batch_size),
                **self.runtime_kwargs,
            )
        runtime = _TEMPORAL_RUNTIME_CACHE[key]
        self._used_runtimes[int(batch_size)] = runtime
        return runtime

    def _new_memory(self):
        return self.memory_type(
            self.runtime.receiver,
            self.parameters,
            self.threshold,
            self.tf,
            maximum_history_tbs=self.maximum_history_tbs,
        )

    def start_episode(self) -> None:
        self.start_episode_batch(1)

    def start_episode_batch(self, batch_size: int) -> None:
        if int(batch_size) < 1:
            raise ValueError("episode batch must be positive")
        self.memories = [self._new_memory() for _ in range(int(batch_size))]
        self.memory = self.memories[0] if len(self.memories) == 1 else None

    def reset_memory(self) -> None:
        for memory in self.memories:
            memory.reset_memory()

    def state_is_empty(self) -> bool:
        return all(memory.state_is_empty() for memory in self.memories)

    def decode_tb(self, inputs):
        output = self.decode_tb_batch([inputs])
        return {name: value[0] for name, value in output.items()}

    def decode_tb_batch(self, inputs):
        if len(inputs) != len(self.memories):
            raise ValueError("input and memory batch sizes differ")
        memories = [
            memory.update(point)
            for memory, point in zip(self.memories, inputs)
        ]
        tf = self.tf
        runtime = self._runtime(len(inputs))
        output = runtime.decode_candidate(
            tf.convert_to_tensor(np.stack([
                point["received_grid"] for point in inputs
            ])),
            tf.convert_to_tensor(np.stack([
                point["ls_channel"] for point in inputs
            ]), tf.float32),
            tf.convert_to_tensor(np.concatenate([
                memory.posterior_channel for memory in memories
            ]), tf.float32),
            tf.convert_to_tensor(np.concatenate([
                memory.posterior_variance for memory in memories
            ]), tf.float32),
            tf.convert_to_tensor(np.concatenate([
                memory.observation_variance for memory in memories
            ]), tf.float32),
            tf.convert_to_tensor(np.concatenate([
                memory.history_valid for memory in memories
            ]), tf.bool),
            tf.convert_to_tensor(np.concatenate([
                memory.reset_mask for memory in memories
            ]), tf.bool),
            tf.convert_to_tensor(np.stack([
                point["active"] for point in inputs
            ]), tf.bool),
        )
        logits, hard, crc = (np.asarray(value.numpy()) for value in output)
        return {
            "hard_bits": hard.astype(bool),
            "crc_status": crc.astype(bool),
            "logits": logits.astype(np.float32),
        }

    def synchronize(self) -> None:
        return None

    def checksums_unchanged(self) -> bool:
        return bool(
            self._artifacts_unchanged()
            and all(
                runtime.checksums_unchanged()
                for runtime in self._used_runtimes.values()
            )
        )


class ColdNeuralRxBenchmarkAdapter(_ArtifactGuard):

    def __init__(
        self,
        *,
        manifest,
        method,
        training_seed,
        artifacts,
        config_name,
        parameters,
        e2e_model,
        tf_module,
    ):
        del parameters, e2e_model
        _verify_method(manifest, method)
        if manifest["role"] not in (
            "matched_cold_k3",
            "matched_cold_k8",
            "historical_cold_k8",
        ):
            raise ValueError("cold adapter requires a cold role")
        super().__init__(manifest, artifacts)
        from cjlr_receiver import (
            OneLayerGridCodeMap,
            StatefulSoftTBInformationDecoder,
        )
        from temporal_receiver_core import (
            build_backbone,
            frozen_cold_state,
            model_checksum,
            prepare_cgnn_inputs,
            readout_llr_grid,
        )
        from utils import load_weights

        config_name = _runtime_config(
            manifest, artifacts, config_name
        )
        self.manifest = manifest
        self.tf = tf_module
        self.iterations = int(manifest["inference"]["iterations"])
        checkpoint = _checkpoint_path(
            manifest, training_seed, artifacts
        )
        _, model = build_backbone(
            config_name,
            num_it=self.iterations,
            training=False,
            num_tx_eval=int(manifest["radio"]["maximum_active_ues"]),
            n_size_bwp_override=int(manifest["radio"]["n_size_bwp"]),
        )
        self.model = model
        self.receiver = model._receiver
        self.backbone = self.receiver._neural_rx._cgnn
        if checkpoint.suffix == ".npz":
            raise ValueError("cold checkpoint must contain all weights")
        if checkpoint.name.endswith(".weights.h5"):
            self.backbone.load_weights(str(checkpoint))
        else:
            load_weights(model, str(checkpoint))
        self.grid_map = OneLayerGridCodeMap(
            self.receiver._neural_rx,
            int(manifest["radio"]["maximum_active_ues"]),
            int(manifest["radio"]["bits_per_symbol"]),
        )
        self.decoder = StatefulSoftTBInformationDecoder(
            self.receiver._tb_encoders[0],
            self.receiver._tb_decoders[0],
            int(self.receiver._tb_decoders[0]._decoder._num_iter),
        )
        self.prepare_inputs = prepare_cgnn_inputs
        self.frozen_state = frozen_cold_state
        self.readout = readout_llr_grid
        self.model_checksum = model_checksum
        self._model_hash = model_checksum(self.model)
        self._decode = self._build_graph()

    def _build_graph(self):
        tf = self.tf

        @tf.function(reduce_retracing=True, jit_compile=False)
        def decode(y, channel, active):
            inputs = self.prepare_inputs(
                self.receiver, y, channel, active
            )
            state = self.frozen_state(self.backbone, inputs)
            for index in range(self.iterations):
                state = self.backbone._iterations[index]([
                    state,
                    inputs[1],
                    inputs[3],
                ])
            grid = self.readout(self.backbone, state)
            output = self.decoder(self.grid_map.grid_to_coded(grid))
            return output.logits, output.hard_bits, output.crc_status

        return decode

    def start_episode(self) -> None:
        return None

    def start_episode_batch(self, batch_size: int) -> None:
        if int(batch_size) < 1:
            raise ValueError("episode batch must be positive")

    def reset_memory(self) -> None:
        return None

    def state_is_empty(self) -> bool:
        return True

    def decode_tb(self, inputs):
        output = self.decode_tb_batch([inputs])
        return {name: value[0] for name, value in output.items()}

    def decode_tb_batch(self, inputs):
        output = self._decode(
            self.tf.convert_to_tensor(np.stack([
                point["received_grid"] for point in inputs
            ])),
            self.tf.convert_to_tensor(np.stack([
                point["ls_channel"] for point in inputs
            ]), self.tf.float32),
            self.tf.convert_to_tensor(np.stack([
                point["active"] for point in inputs
            ]), self.tf.bool),
        )
        logits, hard, crc = (np.asarray(value.numpy()) for value in output)
        return {
            "hard_bits": hard.astype(bool),
            "crc_status": crc.astype(bool),
            "logits": logits.astype(np.float32),
        }

    def synchronize(self) -> None:
        return None

    def checksums_unchanged(self) -> bool:
        return bool(
            self._artifacts_unchanged()
            and self._model_hash == self.model_checksum(self.model)
        )


class PerfectCSIKBestBenchmarkAdapter(_ArtifactGuard):

    def __init__(
        self,
        *,
        manifest,
        method,
        training_seed,
        artifacts,
        config_name,
        parameters,
        e2e_model,
        tf_module,
    ):
        del training_seed, parameters, e2e_model
        _verify_method(manifest, method)
        super().__init__(manifest, artifacts)
        from utils import Parameters
        from utils.baseline_rx import BaselineReceiver

        config_name = _runtime_config(
            manifest, artifacts, config_name
        )
        self.manifest = manifest
        self.tf = tf_module
        oracle_parameters = Parameters(
            config_name,
            training=False,
            num_tx_eval=int(manifest["radio"]["maximum_active_ues"]),
            n_size_bwp_override=int(manifest["radio"]["n_size_bwp"]),
            system="baseline_perf_csi_kbest",
        )
        self.receiver = BaselineReceiver(
            oracle_parameters,
            return_tb_status=True,
            mcs_arr_eval_idx=0,
        )

    def start_episode(self) -> None:
        return None

    def start_episode_batch(self, batch_size: int) -> None:
        if int(batch_size) < 1:
            raise ValueError("episode batch must be positive")

    def reset_memory(self) -> None:
        return None

    def state_is_empty(self) -> bool:
        return True

    def decode_tb(self, inputs):
        output = self.decode_tb_batch([inputs])
        return {name: value[0] for name, value in output.items()}

    def decode_tb_batch(self, inputs):
        tf = self.tf
        output = self.receiver([
            tf.convert_to_tensor(np.stack([
                point["received_grid"] for point in inputs
            ])),
            tf.convert_to_tensor(np.stack([
                point["perfect_channel"] for point in inputs
            ])),
            tf.convert_to_tensor(np.asarray([
                point["noise_variance"] for point in inputs
            ]), tf.float32),
        ])
        hard, crc = (np.asarray(value.numpy()) for value in output)
        hard = hard.astype(bool)
        crc = crc.astype(bool)
        logits = np.where(hard, 1.0, -1.0).astype(np.float32)
        return {
            "hard_bits": hard,
            "crc_status": crc,
            "logits": logits,
        }

    def synchronize(self) -> None:
        return None

    def checksums_unchanged(self) -> bool:
        return self._artifacts_unchanged()


def make_v18_segment_safe_adapter(**kwargs):
    return V18BenchmarkAdapter(**kwargs)


def make_cold_nrx_adapter(**kwargs):
    return ColdNeuralRxBenchmarkAdapter(**kwargs)


def make_perfect_csi_kbest_adapter(**kwargs):
    return PerfectCSIKBestBenchmarkAdapter(**kwargs)
