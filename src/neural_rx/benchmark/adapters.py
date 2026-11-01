#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np

from temporal_benchmark_candidate import checked_decoder_tensors, decoder_output, stack_receiver_grids, validate_pretrained_checkpoint
from temporal_benchmark_protocol import METHOD_ROLES
from artifact_io import file_sha256, validate_file_record


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
    if len(records) != 1:
        raise ValueError("receiver manifest must contain exactly one checkpoint")
    record = records[0]
    if (not manifest["role"].startswith("frozen_")
            and int(record["training_seed"]) != int(training_seed)):
        raise ValueError("training seed differs from the checkpoint")
    return _record_path(record, artifacts)


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
        raise ValueError("Registered radio configuration differs from the runtime configuration")
    return name


class _ArtifactGuard:
    include_logits = False

    def __init__(self, manifest: dict, artifacts: dict[str, Path]):
        records = list(manifest["checkpoints"]) + list(manifest["support_artifacts"])
        expected = {record["sha256"] for record in records}
        if set(artifacts) != expected:
            raise ValueError("resolved artifact set differs from manifest")
        for record in records:
            validate_file_record(artifacts[record["sha256"]], record)
        verified = sorted((Path(path).resolve(), digest) for digest, path in artifacts.items())
        self._artifact_paths = tuple(path for path, _ in verified)
        self._artifact_hashes = tuple(digest for _, digest in verified)

    def _artifacts_unchanged(self) -> bool:
        return self._artifact_hashes == tuple(
            file_sha256(path) for path in self._artifact_paths
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
        tf_module,
    ):
        _verify_method(manifest, method)
        if manifest["role"] not in (
            "frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8",
        ):
            raise ValueError("cold adapter requires a cold role")
        validate_pretrained_checkpoint(manifest, config_name)
        super().__init__(manifest, artifacts)
        from receiver_decoding import (
            OneLayerGridCodeMap,
            SoftTBInformationDecoder,
        )
        from temporal_receiver_core import (
            build_backbone,
            model_checksum,
            prepare_cgnn_inputs,
            readout_llr_grid,
        )

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
            checkpoint=checkpoint,
        )
        self.model = model
        self.receiver = model._receiver
        self.backbone = self.receiver._neural_rx._cgnn
        self.backbone.trainable = False
        self.grid_map = OneLayerGridCodeMap(
            self.receiver._neural_rx,
            int(manifest["radio"]["maximum_active_ues"]),
            int(manifest["radio"]["bits_per_symbol"]),
        )
        self.decoder = SoftTBInformationDecoder(
            self.receiver._tb_encoders[0],
            self.receiver._tb_decoders[0],
            int(self.receiver._tb_decoders[0]._decoder._num_iter),
        )
        self.prepare_inputs = prepare_cgnn_inputs
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
            state = self.backbone.initialize_state(inputs)
            for index in range(self.iterations):
                state = self.backbone._iterations[index]([
                    state,
                    inputs[1],
                    inputs[3],
                ])
            grid = self.readout(self.backbone, state)
            output = self.decoder(self.grid_map.grid_to_coded(grid))
            return checked_decoder_tensors(output)

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
            *stack_receiver_grids(inputs, self.tf),
            self.tf.convert_to_tensor(np.stack([
                point["active"] for point in inputs
            ]), self.tf.bool),
        )
        return decoder_output(*output, include_logits=self.include_logits)

    def checksums_unchanged(self) -> bool:
        return bool(
            self._artifacts_unchanged()
            and self._model_hash == self.model_checksum(self.model)
        )
