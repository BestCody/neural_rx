import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from artifact_io import atomic_json, canonical_json, validate_checkpoint_files


class ArtifactIOTest(unittest.TestCase):
    def test_canonical_json_preserves_identifier_bytes_and_rejects_nonfinite_values(self):
        value = {"z": [True, None, 1.5], "a": {"ue": "é", "b": 0}}
        expected = '{"a":{"b":0,"ue":"\\u00e9"},"z":[true,null,1.5]}'
        self.assertEqual(canonical_json(value), expected)
        self.assertEqual(canonical_json(dict(reversed(list(value.items())))), expected)
        self.assertEqual(json.loads(expected), value)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                canonical_json({"nested": [invalid]})

    def test_interrupted_json_replacement_preserves_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            atomic_json(path, {"step": 1})
            original = path.read_bytes()
            with mock.patch("artifact_io.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_json(path, {"step": 2})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                atomic_json(path, {"step": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [path])
            atomic_json(path, {"step": 2})
            self.assertEqual(json.loads(path.read_text()), {"step": 2})

    def test_checkpoint_validation_rejects_corruption_missing_files_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "checkpoint"
            directory.mkdir()
            path = directory / "weights"
            path.write_bytes(b"weights")
            expected = hashlib.sha256(b"weights").hexdigest()
            validate_checkpoint_files(directory, {"weights": expected})
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                validate_checkpoint_files(directory, {"weights": expected})
            with self.assertRaises(FileNotFoundError):
                validate_checkpoint_files(directory, {"missing": expected})
            outside = root / "outside"
            outside.write_bytes(b"weights")
            for name in ("../outside", str(outside), "sub/weights", ""):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "local filenames"):
                    validate_checkpoint_files(directory, {name: expected})
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "inside their directory"):
                validate_checkpoint_files(directory, {"weights": expected})

    def test_artifact_and_export_helpers_do_not_initialize_tensorflow(self):
        code = "import artifact_io, temporal_benchmark_frozen, sys; assert 'tensorflow' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=15,
                       env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent)))


if __name__ == "__main__":
    unittest.main()
