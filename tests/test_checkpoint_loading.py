import json
import fcntl
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from neural_rx.receiver import training
from neural_rx.receiver.checkpoints import recover_latest
import neural_rx.benchmark.adapters as adapters
from neural_rx.benchmark.candidate import artifact_record, resolve_manifest_artifacts
from neural_rx.benchmark.frozen import load_receiver_checkpoint_export
import neural_rx.backbone.core as core
ROOT = Path(__file__).resolve().parent.parent


class TinyE2E(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.backbone = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=False)])
        self._receiver = SimpleNamespace(_neural_rx=SimpleNamespace(_cgnn=self.backbone))

    def call(self, batch_size, ebno):
        return self.backbone(tf.ones([batch_size, 1]))


class CheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_artifact_guard_hashes_once_and_detects_later_changes(self):
        path = self.directory / "weights"
        path.write_bytes(b"weights")
        record = artifact_record(path)
        manifest = {"checkpoints": [record], "support_artifacts": []}
        artifact_map = {record["sha256"]: str(path)}
        checksum = mock.Mock(wraps=core.file_sha256)
        with mock.patch("neural_rx.provenance.artifacts.file_sha256", checksum), \
                mock.patch.object(adapters, "file_sha256", checksum):
            artifacts = resolve_manifest_artifacts(manifest, artifact_map)
            guard = adapters._ArtifactGuard(manifest, artifacts)
            self.assertEqual(checksum.call_count, 1)
            self.assertTrue(guard._artifacts_unchanged())
            path.write_bytes(b"changed")
            self.assertFalse(guard._artifacts_unchanged())
            with self.assertRaisesRegex(ValueError, "Artifact changed"):
                adapters._ArtifactGuard(manifest, artifacts)

    def test_recovery_restores_published_weights_optimizer_and_step_after_pointer_failure(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
        model(tf.ones([1,1]))
        optimizer = tf.keras.optimizers.Adam()
        optimizer.build(model.trainable_variables)
        trainer = training.MemoryTrainer.__new__(training.MemoryTrainer)
        trainer.model = model
        trainer.variant = "pilot_memory"
        trainer.settings = json.loads((ROOT / "config/memory_receiver.json").read_text())
        trainer.manager = SimpleNamespace(capacity=4, expiry_slots=8)
        trainer.step = tf.Variable(0,dtype=tf.int64)
        trainer.binding = lambda: {}
        trainer.schedule_state = lambda: {}
        trainer.checkpoint = tf.train.Checkpoint(model=model,optimizer=optimizer,step=trainer.step)
        trainer.save(self.directory)
        optimizer.apply_gradients((tf.ones_like(v),v) for v in model.trainable_variables)
        trainer.step.assign(1)
        variables = list(model.weights) + list(optimizer.variables()) + [trainer.step]
        expected = [v.numpy().copy() for v in variables]
        original = training.atomic_json
        def interrupt(path, value):
            if Path(path).name == "latest.json":
                raise OSError("interrupted pointer update")
            return original(path,value)
        with mock.patch.object(training,"atomic_json",side_effect=interrupt):
            with self.assertRaisesRegex(OSError,"interrupted pointer"):
                trainer.save(self.directory)
        self.assertEqual(json.loads((self.directory/"latest.json").read_text())["update"],0)
        latest = recover_latest(self.directory)
        self.assertEqual(latest["update"],1)
        for v in variables:
            v.assign(tf.zeros_like(v))
        trainer.checkpoint.read(str(self.directory/latest["checkpoint"]/"state")).assert_consumed()
        for v, value in zip(variables,expected, strict=True):
            np.testing.assert_array_equal(v.numpy(),value)
        trainer.step.assign_add(1)
        trainer.save(self.directory)
        self.assertEqual(recover_latest(self.directory)["update"],2)
        with (self.directory/"run.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                recover_latest(self.directory)
        (self.directory/".checkpoint-unpublished").mkdir()
        self.assertEqual(recover_latest(self.directory)["update"],2)
        (self.directory/"checkpoint_000002"/"receiver.weights.h5").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"checksum differs"):
            recover_latest(self.directory)

    def export(self, features):
        weights = self.directory / "receiver.weights.h5"
        weights.write_bytes(b"checkpoint fixture")
        settings = json.loads((ROOT / "config/memory_receiver.json").read_text())
        settings["architecture"].update(fit_tbs=features)
        (self.directory / "cnn_autoencoder_finalist.json").write_text(json.dumps(settings))
        self.seal_export(features)
        return weights

    def seal_export(self, features):
        files = {name: core.file_sha256(self.directory / name)
                 for name in ("receiver.weights.h5", "cnn_autoencoder_finalist.json")}
        architecture = json.loads((ROOT / "config/memory_receiver.json").read_text())["architecture"]
        architecture["fit_tbs"] = features
        (self.directory / "manifest.json").write_text(json.dumps(
            {"binding": {"sources": {}, "architecture": architecture, "seed": 0}, "update": 7, "files": files}))

    def test_export_rejects_configuration_missing_changed_or_bound_to_other_features(self):
        weights = self.export(8)
        settings_path = self.directory / "cnn_autoencoder_finalist.json"
        settings = json.loads(settings_path.read_text())
        settings["architecture"].update(fit_tbs=1)
        settings_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            load_receiver_checkpoint_export(weights)
        self.seal_export(8)
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            load_receiver_checkpoint_export(weights)
        settings_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_receiver_checkpoint_export(weights)

    def test_initializer_loads_configured_weights_in_both_supported_formats(self):
        expected = np.array([[7.]], np.float32)
        path = self.directory / "initializer.pickle"
        with path.open("wb") as stream:
            pickle.dump([expected], stream)
        with mock.patch.object(core, "Parameters", return_value=SimpleNamespace(label="unused")), \
                mock.patch.object(core, "E2E_Model", TinyE2E):
            _, model = core.build_backbone("fixture.cfg", 2, True, n_size_bwp_override=4,
                checkpoint=path, checkpoint_sha256=core.file_sha256(path))
            backbone = model._receiver._neural_rx._cgnn
            np.testing.assert_array_equal(backbone(tf.ones([1, 1])).numpy(), expected)
            self.assertTrue(backbone.trainable)
            self.assertEqual(backbone.num_it, 2)

    def test_initializer_rejects_checkpoint_changes_before_and_during_loading(self):
        path = self.directory / "initializer"
        with path.open("wb") as stream:
            pickle.dump([np.array([[7.]], np.float32)], stream)
        recipe = {"radio_config": "fixture.cfg", "initializer_checkpoint": str(path),
                  "initializer_sha256": core.file_sha256(path)}
        with mock.patch.object(core, "Parameters", return_value=SimpleNamespace(label="unused")), \
                mock.patch.object(core, "E2E_Model", TinyE2E):
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                core.build_backbone(recipe["radio_config"], 2, True, n_size_bwp_override=4,
                    checkpoint=path, checkpoint_sha256="0" * 64)
            original = core.load_weights

            def replace_after_load(model, filename):
                original(model, filename)
                path.write_bytes(b"replacement")

            with mock.patch.object(core, "load_weights", side_effect=replace_after_load):
                with self.assertRaisesRegex(ValueError, "changed while loading"):
                    core.build_backbone(recipe["radio_config"], 2, True, n_size_bwp_override=4,
                        checkpoint=path, checkpoint_sha256=recipe["initializer_sha256"])

if __name__ == "__main__":
    unittest.main()
