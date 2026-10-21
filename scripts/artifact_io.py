from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_checkpoint_files(directory, files):
    directory = Path(directory).resolve()
    for name, expected in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Checkpoint files must be local filenames")
        path = directory / name
        if path.resolve().parent != directory:
            raise ValueError("Checkpoint files must remain inside their directory")
        if file_sha256(path) != expected:
            raise ValueError(f"Checkpoint checksum differs: {name}")


def validate_file_record(path, record):
    path = Path(path)
    if file_sha256(path) != record["sha256"] or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Artifact changed: {path.name}")
