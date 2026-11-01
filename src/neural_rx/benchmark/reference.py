from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from artifact_io import atomic_json, canonical_json, file_sha256
from artifact_provenance import source_records
from temporal_benchmark_adapters import ColdNeuralRxBenchmarkAdapter
from temporal_benchmark_candidate import resolve_manifest_artifacts, validate_manifest
from temporal_benchmark_episodes import CachedScenarioBank
from temporal_benchmark_runtime import evaluate_method_set
from temporal_generalization_protocol import seal_payload, verify_payload

ROOT = Path(__file__).resolve().parent.parent


def reference_runtime(tf):
    devices = [tf.config.experimental.get_device_details(device)
               for device in tf.config.get_visible_devices("GPU")]
    return {
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "tensorflow", "sionna")},
        "gpu_devices": [{key: device[key] for key in ("device_name", "compute_capability") if key in device}
                        for device in devices],
        "floatx": tf.keras.backend.floatx(),
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
        "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
    }


def reference_binding(protocol, plan, identity, manifests, episode_hashes, runtime, root=ROOT):
    methods = list(manifests)
    fields = ("role", "radio", "inference", "training", "checkpoints", "support_artifacts")
    return json.loads(canonical_json({
        "schema": "frozen_reference_v3",
        "methods": methods,
        "receivers": {method: {key: manifest[key] for key in fields}
                      for method, manifest in manifests.items()},
        "episodes": {"identity": identity, "sha256": episode_hashes},
        "jobs": plan["jobs"],
        "stage": protocol["stages"][plan["stage"]],
        "episodes_per_condition": plan["episodes_per_condition"],
        "snr_point_names": plan["snr_point_names"],
        "execution": {
            "accuracy_episode_batch_size": protocol["execution"]["accuracy_episode_batch_size"],
            "observable_receiver_inputs": protocol["execution"]["observable_receiver_inputs"],
            "persistent_methods": [m for m in methods if m in protocol["execution"].get("persistent_methods", [])],
        },
        "runtime": runtime,
        "sources": source_records(("scripts/temporal_benchmark_reference.py",), root),
    }))


def frozen_result_cache(directory, binding, compute):
    key = hashlib.sha256(canonical_json(binding).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/key
    with (directory/(key+".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            payload = json.loads((destination/"result.json").read_text())
            verify_payload(payload, field="cache_sha256")
            if (payload["cache"] != "frozen_reference_v3" or payload["binding"] != binding
                    or not all(payload["preflight"].values())):
                raise ValueError("Frozen reference result binding differs")
            raw = destination/"errors.npz"
            if file_sha256(raw) != payload["raw_sha256"]:
                raise ValueError("Frozen reference result checksum differs")
            with np.load(raw, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            preflight, seconds = payload["preflight"], payload["seconds"]
        else:
            arrays, preflight, seconds = compute()
            if not all(preflight.values()):
                raise RuntimeError("Frozen reference preflight failed")
            temp = Path(tempfile.mkdtemp(prefix=".reference-", dir=directory))
            try:
                raw = temp/"errors.npz"
                np.savez_compressed(raw, **arrays)
                atomic_json(temp/"result.json", seal_payload(dict(cache="frozen_reference_v3",
                    binding=binding, preflight=preflight, seconds=seconds,
                    raw_sha256=file_sha256(raw)), field="cache_sha256"))
                os.replace(temp, destination)
            finally:
                if temp.exists():
                    shutil.rmtree(temp)
        return arrays, preflight, seconds



def evaluate_frozen_references(directory, protocol, plan, provider: CachedScenarioBank, registrations, tf):
    manifests = {method: record[0] for method, record in registrations.items()}
    for manifest in manifests.values():
        validate_manifest(manifest, protocol)
        if not manifest["role"].startswith("frozen_cold_"):
            raise ValueError("Only cold receivers can use the frozen-reference cache")
    methods = tuple(manifests)
    binding = reference_binding(protocol, plan, provider.identity, manifests,
                                provider.content_hashes(plan["jobs"]), reference_runtime(tf))

    def compute():
        adapters = {
            method: ColdNeuralRxBenchmarkAdapter(
                manifest=manifest, method=method, training_seed=seed,
                artifacts=resolve_manifest_artifacts(manifest, mapping),
                config_name="nrx_large.cfg", tf_module=tf)
            for method, (manifest, mapping, seed) in registrations.items()
        }
        return evaluate_method_set(protocol, plan, provider, adapters, methods)

    return frozen_result_cache(directory, binding, compute)
