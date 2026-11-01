from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np

from artifact_io import file_sha256


def load_complete_receiver(model, checkpoint, expected_sha256):
    checkpoint = Path(checkpoint)
    if file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Source checkpoint checksum differs")
    mapping = []
    with tempfile.TemporaryDirectory() as directory:
        migrated = Path(directory)/"receiver.weights.h5"
        model.save_weights(str(migrated))
        with h5py.File(checkpoint, "r") as source, h5py.File(migrated, "r+") as target:
            old, new = {}, {}
            source.visititems(lambda name, value: old.update({name: value}) if isinstance(value, h5py.Dataset) else None)
            target.visititems(lambda name, value: new.update({name: value}) if isinstance(value, h5py.Dataset) else None)
            hidden = int(model.blend_kernel.shape[1])
            added = {f"vars/{i+2}": shape for i, shape in enumerate(((8, hidden), (hidden,), (hidden,), ()))}
            if (set(new)-set(old) != set(added) or set(old)-set(new)
                    or any(new[name].shape != shape for name, shape in added.items())):
                raise ValueError("Migration requires the complete calibrated-feedback receiver and exactly four new blend tensors")
            for name, tensor in new.items():
                if name in added:
                    value = tensor[()]
                else:
                    value = old[name][()]
                    if value.shape != tensor.shape or value.dtype != tensor.dtype:
                        raise ValueError(f"Inherited tensor shape or dtype differs: {name}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite source tensor: {name}")
                tensor[...] = value
                mapping.append(dict(target=name, source=None if name in added else name,
                                    shape=list(tensor.shape), dtype=str(tensor.dtype),
                                    sha256=hashlib.sha256(np.asarray(value).tobytes()).hexdigest()))
        model.load_weights(str(migrated))
    if file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Source checkpoint changed during migration")
    return mapping
