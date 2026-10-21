import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from artifact_provenance import (
    TRAINING_SCHEMA, evaluation_source_records, source_records, training_source_records,
    validate_evaluation_sources, validate_training_binding,
)


class ArtifactProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()

    def test_resume_preserves_training_semantics_but_allows_repacking(self):
        recipe = {"protocol": "cnn_memory_training", "schema_version": TRAINING_SCHEMA, "learning_rate": .01}
        checksum = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        saved = {"sources": {"old.py": "old hash"}, "recipe": recipe,
                 "recipe_sha256": checksum(recipe), "seed": 42,
                 "architecture": {"filter_hidden": 16},
                 "execution": {"version": "pilot_filter_learned_blend_joint_v1", "tf32_enabled": False,
                               "detector_batch_tbs": 8, "descriptor_chunk_tbs": 16}}
        current = copy.deepcopy(saved)
        current["sources"] = {"new.py": "new hash"}
        current["execution"] = {"version": "pilot_filter_learned_blend_joint_v1",
                                "tf32_enabled": False, "detector_batch_tbs": 16,
                                "descriptor_chunk_tbs": 32}
        validate_training_binding(saved, current)
        for key in ("recipe", "seed", "architecture", "execution"):
            changed = copy.deepcopy(current)
            if key == "recipe":
                changed[key]["learning_rate"] *= 2
                changed["recipe_sha256"] = checksum(changed[key])
            elif key == "execution":
                changed[key]["tf32_enabled"] = True
            elif key == "architecture":
                changed[key]["filter_hidden"] *= 2
            else:
                changed[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_binding(saved, changed)
        with self.assertRaisesRegex(ValueError, "recipe checksum"):
            validate_training_binding(saved, dict(current, recipe_sha256="corrupt"))
        legacy = copy.deepcopy(saved)
        legacy["recipe"]["schema_version"] = 2
        legacy["recipe_sha256"] = checksum(legacy["recipe"])
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_training_binding(legacy, current)

    def test_import_closure_tracks_relative_lazy_and_package_imports(self):
        package = self.root / "scripts/package"
        package.mkdir()
        (package / "__init__.py").write_text("VALUE = 1\n")
        (package / "entry.py").write_text("def run():\n    from . import worker\n")
        (package / "worker.py").write_text("import package.entry\n")
        (self.root / "scripts/unrelated.py").write_text("unused = True\n")
        expected = {"scripts/package/entry.py", "scripts/package/__init__.py", "scripts/package/worker.py"}
        first = source_records(["scripts/package/entry.py"], self.root)
        self.assertEqual(set(first), expected)
        (package / "worker.py").write_text("import package.entry\nVALUE = 2\n")
        self.assertNotEqual(first, source_records(["scripts/package/entry.py"], self.root))

    def test_current_bindings_cover_receiver_dependencies_and_radio_configuration(self):
        sources = evaluation_source_records()
        required = {"scripts/receiver_decoding.py", "scripts/pilot_history_receiver/benchmark_adapter.py",
                    "scripts/pilot_history_receiver/model.py", "scripts/pilot_history_receiver/native_ops/pair_correction.cc",
                    "scripts/pilot_history_receiver/native_ops/pair_correction.cu.cc", "config/nrx_large.cfg"}
        self.assertTrue(required.issubset(sources))
        for name in required:
            incomplete = {k: v for k, v in sources.items() if k != name}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "source changed"):
                validate_evaluation_sources(incomplete)
        training = training_source_records("nrx_large.cfg")
        self.assertTrue({name for name in required if '/native_ops/' in name}.issubset(training))
        self.assertTrue({"config/nrx_large.cfg", "utils/channel_models.py", "utils/__init__.py"}.issubset(training))


if __name__ == "__main__":
    unittest.main()
