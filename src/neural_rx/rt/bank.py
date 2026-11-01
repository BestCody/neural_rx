from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from artifact_io import canonical_json, file_sha256
from temporal_generalization_protocol import verify_payload


def rt_scenario_sha256(manifest):
    settings = {key: value for key, value in manifest["config"].items()
                if key not in ("seed", "episodes", "split")}
    return hashlib.sha256(canonical_json({
        "config": settings, "scene_files": manifest["scene_files"],
        "packages": manifest["packages"],
    }).encode()).hexdigest()


class RTChannelBank:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        verify_payload(self.manifest, field="bank_sha256")
        if self.manifest["schema"] != "sionna_rt_paths_v1":
            raise ValueError("Unsupported ray-traced channel bank")
        self.config = self.manifest["config"]
        self.sha256 = self.manifest["bank_sha256"]
        self.records = self.manifest["episodes"]
        if len(self.records) != self.config["episodes"] or not self.records:
            raise ValueError("Incomplete ray-traced channel bank")
        for record in self.records:
            path = self.directory / record["file"]
            if path.resolve().parent != self.directory or file_sha256(path) != record["sha256"]:
                raise ValueError("Ray-traced path file changed")

    def load(self, index):
        with np.load(self.directory / self.records[index]["file"], allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

