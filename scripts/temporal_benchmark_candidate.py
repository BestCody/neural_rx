from __future__ import annotations

import ast
import configparser
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from artifact_io import file_sha256
from temporal_generalization_protocol import verify_payload


def pretrained_checkpoint(config_name="nrx_large.cfg"):
    root = Path(__file__).resolve().parent.parent
    config = configparser.ConfigParser()
    with (root / "config" / config_name).open() as stream:
        config.read_file(stream)
    label = ast.literal_eval(config["global"]["label"])
    return root / "models/pretrained" / f"{label}_weights"


def validate_pretrained_checkpoint(manifest, config_name="nrx_large.cfg"):
    expected = artifact_record(pretrained_checkpoint(config_name))
    records = manifest["checkpoints"]
    if len(records) != 1 or any(records[0].get(k) != v for k, v in expected.items()):
        raise ValueError("Receiver must use the shipped pretrained neural_rx checkpoint")
    if manifest["training"] != {"additional_updates": 0, "provenance": "shipped_neural_rx"}:
        raise ValueError("Pretrained reference cannot contain additional training")


def validate_manifest(manifest, protocol):
    verify_payload(manifest, field="manifest_sha256")
    verify_payload(protocol, field="protocol_sha256")
    if protocol.get("protocol") != "pretrained_nrx_experiment_v1":
        raise ValueError("Frozen-reference manifest requires the versioned experiment")
    if manifest["benchmark_protocol_sha256"] != protocol["protocol_sha256"] or manifest["radio"] != protocol["radio"]:
        raise ValueError("Frozen experiment/radio binding differs")
    role = manifest["role"]
    counts = {"candidate": 2, "frozen_cold_k2": 2, "frozen_cold_k3": 3,
              "frozen_cold_k8": 8}
    if role not in counts or manifest["inference"]["iterations"] != counts[role]:
        raise ValueError("Frozen experiment receiver rounds differ")
    if bool(manifest["inference"]["persistent_memory"]) != (role == "candidate"):
        raise ValueError("Frozen experiment memory mode differs")
    if len(manifest["checkpoints"]) != 1:
        raise ValueError("Exactly one frozen checkpoint is required")
    for record in manifest["checkpoints"] + manifest["support_artifacts"]:
        if len(record["sha256"]) != 64 or record["bytes"] <= 0:
            raise ValueError("Invalid artifact record")
    if role.startswith("frozen_"):
        validate_pretrained_checkpoint(manifest)


def artifact_record(path: Path | str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "name": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def manifest_artifacts(manifest: dict) -> list[dict]:
    return list(manifest.get("checkpoints", [])) + list(
        manifest.get("support_artifacts", [])
    )


def resolve_manifest_artifacts(
    manifest: dict, artifact_map: dict[str, str]
) -> dict[str, Path]:
    resolved = {}
    for record in manifest_artifacts(manifest):
        digest = record["sha256"]
        if digest not in artifact_map:
            raise FileNotFoundError(f"artifact map lacks {digest}")
        path = Path(artifact_map[digest]).expanduser().resolve()
        resolved[digest] = path
    return resolved


def stack_receiver_grids(inputs, tf):
    return (
        tf.convert_to_tensor(np.stack([point["received_grid"] for point in inputs])),
        tf.convert_to_tensor(np.stack([point["ls_channel"] for point in inputs]), tf.float32),
    )


def checked_decoder_tensors(output):
    import tensorflow as tf
    logits = tf.debugging.check_numerics(output.logits, "Decoder logits are not finite")
    with tf.control_dependencies([logits]):
        return logits, tf.cast(output.hard_bits, tf.bool), tf.identity(output.crc_status)


def decoder_output(logits, hard, crc, *, include_logits=False):
    result = {
        "hard_bits": np.asarray(hard.numpy(), dtype=bool),
        "crc_status": np.asarray(crc.numpy(), dtype=bool),
    }
    if include_logits:
        result["logits"] = np.asarray(logits.numpy(), dtype=np.float32)
    return result


@runtime_checkable
class BenchmarkDecoderAdapter(Protocol):

    manifest: dict

    def start_episode(self) -> None:
        ...

    def start_episode_batch(self, batch_size: int) -> None:
        ...

    def reset_memory(self) -> None:
        ...

    def state_is_empty(self) -> bool:
        ...

    def decode_tb(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...

    def decode_tb_batch(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...

    def checksums_unchanged(self) -> bool:
        ...


def validate_adapter_output(output: dict, bits_shape: tuple[int, ...]) -> None:
    required = {"hard_bits", "crc_status"}
    if set(output) not in (required, required | {"logits"}):
        raise ValueError("adapter output fields differ")
    expected_crc = bits_shape[:-1]
    hard = np.asarray(output["hard_bits"])
    crc = np.asarray(output["crc_status"])
    if hard.shape != bits_shape:
        raise ValueError("adapter hard-bit shape differs")
    if crc.shape != expected_crc:
        raise ValueError("adapter CRC shape differs")
    if "logits" in output:
        logits = np.asarray(output["logits"])
        if logits.shape != bits_shape:
            raise ValueError("adapter logits shape differs")
        if not np.all(np.isfinite(logits)):
            raise ValueError("adapter logits are not finite")
