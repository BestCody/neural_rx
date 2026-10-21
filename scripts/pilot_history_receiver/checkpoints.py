from __future__ import annotations

import fcntl
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
from artifact_io import atomic_json, validate_checkpoint_files


def recover_latest(output, *, lock_held=False):
    output = Path(output)
    if not output.exists():
        return None
    with (nullcontext() if lock_held else (output / "run.lock").open("a+")) as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _recover_latest(output)


def recover_history(path, checkpoint_step):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b") as stream:
        offset, last_update = 0, 0
        while line := stream.readline():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError("Training history contains a corrupt complete record") from None
                if last_update < checkpoint_step:
                    raise ValueError("Incomplete history precedes the restored checkpoint") from None
                stream.seek(offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            if not isinstance(record, dict) or type(record.get("update")) is not int or record["update"] < 1:
                raise ValueError("Training history contains an invalid update")
            if record["update"] > checkpoint_step:
                stream.seek(offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            if record["update"] <= last_update:
                raise ValueError("Committed training history has repeated or unordered updates")
            last_update = record["update"]
            offset = stream.tell()
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                return


def _recover_latest(output):
    output = Path(output)
    candidates = []
    for path in output.glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match[1]), path))
    latest = output / "latest.json"
    if not candidates:
        if latest.exists():
            raise FileNotFoundError("The latest pointer exists but no published checkpoint remains")
        return None
    step, path = max(candidates, key=lambda item: item[0])
    manifest = json.loads((path / "manifest.json").read_text())
    if type(manifest["update"]) is not int or manifest["update"] != step:
        raise ValueError("Published checkpoint step differs from its directory")
    files = manifest["files"]
    required = {"receiver.weights.h5", "cnn_autoencoder_finalist.json", "state.index"}
    if not required.issubset(files) or not any(name.startswith("state.data-") for name in files):
        raise ValueError("Published checkpoint is missing model or optimizer state")
    validate_checkpoint_files(path, files)
    recovered = {"checkpoint": path.name, "update": step}
    try:
        current = json.loads(latest.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        current = None
    if current != recovered:
        atomic_json(latest, recovered)
    return recovered
