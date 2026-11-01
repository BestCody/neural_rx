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

from neural_rx.benchmark.runtime import BenchmarkEpisode, validate_episode
from neural_rx.provenance.artifacts import canonical_json, file_sha256
from neural_rx.benchmark.generalization_protocol import verify_payload
from neural_rx.paths import PACKAGE, ROOT


def scenario_identity(config_path, protocol):
    root = ROOT
    paths = [PACKAGE / name for name in (
        "benchmark/scenarios.py", "data/temporal.py", "data/geometry.py",
        "backbone/core.py", "benchmark/generalization_protocol.py",
    )]
    paths += sorted((PACKAGE / "backbone").glob("*.py"))
    paths += [Path(config_path)]
    identity = {
        "schema": "temporal_benchmark_episode_v2",
        "radio": protocol["radio"],
        "sources": {str(p.relative_to(root)): file_sha256(p)
                    for p in paths},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "tensorflow", "sionna")},
    }
    return identity


def ray_traced_identity(banks):
    from neural_rx.rt.bank import rt_scenario_sha256
    return {
        "sources": {str((PACKAGE/name).relative_to(ROOT)): file_sha256(PACKAGE/name)
                    for name in ("rt/bank.py", "rt/data.py")},
        "banks": {bank.sha256: {"seed": bank.config["seed"], "split": bank.config["split"],
                                "scenario_sha256": rt_scenario_sha256(bank.manifest)}
                  for bank in banks},
    }


class RoutedScenarioBank:
    """Keep ray-traced episodes in their own cache beside the statistical bank."""

    def __init__(self, statistical, ray_traced):
        self.statistical, self.ray_traced = statistical, ray_traced
        self.identity = dict(statistical.identity, ray_traced=ray_traced.identity["ray_traced"])

    def _bank(self, job):
        return self.ray_traced if job["provider"] == "ray_traced" else self.statistical

    def ensure_capacity(self, jobs):
        for bank in (self.statistical, self.ray_traced):
            bank.ensure_capacity([job for job in jobs if self._bank(job) is bank])

    def generate(self, job, episode_index):
        return self._bank(job).generate(job, episode_index)

    def content_hashes(self, jobs):
        return [value for job in jobs for value in self._bank(job).content_hashes([job])]


class CachedScenarioBank:
    def __init__(self, provider, directory, protocol, identity, *, readonly=False):
        self.provider = provider
        self.directory = Path(directory).expanduser().resolve()
        self.protocol = protocol
        self.identity = json.loads(canonical_json(identity))
        self.readonly = bool(readonly)
        dataset = self.directory / "dataset.json"
        if dataset.is_file():
            saved = json.loads(dataset.read_text())
            verify_payload(saved)
            if any(saved["identity"].get(key) != self.identity.get(key) for key in
                   ("schema", "radio", "ray_traced")):
                raise ValueError("Saved dataset format or radio configuration differs")
            self.identity = saved["identity"]
            self.readonly = True
        if not self.readonly:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _binding(self, job, episode_index):
        if not 0 <= int(episode_index) < int(job["episode_count"]):
            raise ValueError("episode index is outside the job")
        return {"identity": self.identity, "job": job, "episode_index": int(episode_index)}

    def path_for(self, job, episode_index):
        key = hashlib.sha256(canonical_json(self._binding(job, episode_index)).encode()).hexdigest()
        return self.directory / (key + ".npz")

    def ensure_capacity(self, jobs):
        required = 0
        largest = 0
        for job in jobs:
            missing = [index for index in range(int(job["episode_count"]))
                       if not self.path_for(job, index).is_file()]
            if not missing:
                continue
            if self.readonly:
                raise FileNotFoundError(f"Required shared episode is absent: {self.path_for(job, missing[0])}. "
                                        "Use another --episode-cache directory to generate a different dataset.")
            self.generate(job, missing[0])
            sample_bytes = self.path_for(job, missing[0]).stat().st_size
            required += sample_bytes * (len(missing) - 1)
            largest = max(largest, sample_bytes)
        estimate = int(required * 1.25) + largest * 4 + 1024**3
        if required and shutil.disk_usage(self.directory).free < estimate:
            raise OSError(f"Shared episode bank needs approximately {estimate / 1024**3:.1f} GiB more free space. "
                          "Use --episode-cache on a larger volume before starting evaluation.")

    def content_hashes(self, jobs):
        hashes = []
        for job in jobs:
            for index in range(int(job["episode_count"])):
                path = self.path_for(job, index)
                if not path.is_file():
                    self.generate(job, index)
                hashes.append(file_sha256(path))
        return hashes

    def _read(self, path, binding, job):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].tobytes())
            if metadata["binding"] != binding:
                raise ValueError("episode cache binding differs")
            arrays = {name: archive[name] for name in archive.files if name != "metadata"}
        common = {name: arrays[name] for name in
                  ("bits", "active", "ue_ids", "channel_change_event")}
        observable = self.protocol["execution"]["observable_receiver_inputs"]
        expected = set(common) | {"receiver/" + n for n in observable}
        if set(arrays) != expected:
            raise ValueError("episode cache array fields differ")
        receiver = []
        for snr in range(len(job["snr_db"])):
            points = []
            for tb in range(int(job["num_tbs"])):
                point = {n: arrays["receiver/" + n][snr, tb] for n in observable}
                points.append(point)
            receiver.append(tuple(points))
        episode = BenchmarkEpisode(**common, receiver_inputs=tuple(receiver))
        episode = validate_episode(episode, self.protocol, job)
        if episode.fingerprint.tobytes().hex() != metadata["fingerprint"]:
            raise ValueError("episode cache fingerprint differs")
        return episode

    def generate(self, job, episode_index):
        binding = self._binding(job, episode_index)
        path = self.path_for(job, episode_index)
        if path.is_file():
            return self._read(path, binding, job)
        if self.readonly:
            raise FileNotFoundError(f"Required shared episode is absent: {path}")
        with path.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_file():
                return self._read(path, binding, job)
            episode = self.provider.generate(job, episode_index)
            episode = validate_episode(episode, self.protocol, job)
            arrays = {name: getattr(episode, name) for name in
                      ("bits", "active", "ue_ids", "channel_change_event")}
            for name in self.protocol["execution"]["observable_receiver_inputs"]:
                arrays["receiver/" + name] = np.stack([
                    np.stack([point[name] for point in snr]) for snr in episode.receiver_inputs
                ])
            size = sum(value.nbytes for value in arrays.values())
            if shutil.disk_usage(self.directory).free < size * 2 + 1024**3:
                raise OSError("Insufficient space to publish a complete shared episode cache")
            metadata = {"binding": binding, "fingerprint": episode.fingerprint.tobytes().hex()}
            arrays["metadata"] = np.frombuffer(canonical_json(metadata).encode(), np.uint8)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return episode
