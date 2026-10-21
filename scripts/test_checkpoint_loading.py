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

from pilot_history_receiver import training
from pilot_history_receiver.checkpoints import recover_latest
import smoke_temporal_benchmark_adapters as smoke
import temporal_benchmark_adapters as adapters
from temporal_benchmark_candidate import artifact_record, resolve_manifest_artifacts
from temporal_benchmark_frozen import load_receiver_checkpoint_export, make_manifest, protocol_and_plan
from temporal_generalization_protocol import seal_payload
from run_memory_training import verified_result
import temporal_receiver_core as core


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
        with mock.patch("artifact_io.file_sha256", checksum), mock.patch.object(adapters, "file_sha256", checksum):
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
        for v, value in zip(variables,expected):
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
        settings["architecture"].update(blend_hidden=features)
        (self.directory / "cnn_autoencoder_finalist.json").write_text(json.dumps(settings))
        self.seal_export(features)
        return weights

    def seal_export(self, features):
        files = {name: core.file_sha256(self.directory / name)
                 for name in ("receiver.weights.h5", "cnn_autoencoder_finalist.json")}
        architecture = json.loads((ROOT / "config/memory_receiver.json").read_text())["architecture"]
        architecture["blend_hidden"] = features
        (self.directory / "manifest.json").write_text(json.dumps(
            {"binding": {"sources": {}, "architecture": architecture, "seed": 0}, "update": 7, "files": files}))

    def test_smoke_uses_the_exported_memory_architecture_for_both_path_forms(self):
        weights = self.export(8)

        class InspectedManifest(Exception):
            pass

        def inspect(protocol, role, checkpoint):
            manifest, artifacts, seed = make_manifest(protocol, role, checkpoint)
            self.assertEqual(Path(artifacts[manifest["checkpoints"][0]["sha256"]]), weights)
            self.assertEqual(seed, manifest["checkpoints"][0]["training_seed"])
            settings = self.directory / "cnn_autoencoder_finalist.json"
            for path in (settings, self.directory / "manifest.json"):
                self.assertIn(artifact_record(path), manifest["support_artifacts"])
            self.assertEqual(json.loads(settings.read_text())["architecture"]["blend_hidden"], 8)
            raise InspectedManifest

        for path in (self.directory, weights):
            with self.subTest(path=path), mock.patch("temporal_benchmark_frozen.make_manifest", side_effect=inspect), \
                    mock.patch.object(tf.config, "list_physical_devices", return_value=[]), \
                    mock.patch("sys.argv", ["smoke",
                                           "--candidate-checkpoint", str(path), "--prbs", "4"]), \
                    mock.patch.dict("os.environ", {}, clear=False):
                with self.assertRaises(InspectedManifest):
                    smoke.main()

    def test_export_rejects_configuration_missing_changed_or_bound_to_other_features(self):
        weights = self.export(8)
        settings_path = self.directory / "cnn_autoencoder_finalist.json"
        settings = json.loads(settings_path.read_text())
        settings["architecture"].update(blend_hidden=1)
        settings_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            load_receiver_checkpoint_export(weights)
        self.seal_export(8)
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            load_receiver_checkpoint_export(weights)
        settings_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_receiver_checkpoint_export(weights)

    def test_result_reuse_checks_evaluated_configuration_even_after_export_is_resealed(self):
        weights = self.export(8)
        protocol, _ = protocol_and_plan("smoke", 4)
        report = self.directory / "report"
        report.mkdir()
        np.savez(report / "errors.npz", errors=np.zeros(2, bool))
        (report / "protocol.json").write_text(json.dumps(protocol))
        manifests = {role: make_manifest(protocol, role, weights)[0]
                     for role in ("candidate", "frozen_cold_k8")}
        result = dict(phase="smoke", protocol_sha256=protocol["protocol_sha256"], manifests=manifests,
                      raw_sha256=core.file_sha256(report / "errors.npz"))

        def publish(checkpoint):
            (report / "results.json").write_text(json.dumps(seal_payload(
                dict(result, checkpoint=str(checkpoint)), field="result_sha256")))

        for checkpoint in (self.directory, weights):
            publish(checkpoint)
            self.assertEqual(verified_result(report)["manifests"], manifests)
        original_settings = (self.directory / "cnn_autoencoder_finalist.json").read_bytes()
        original_manifest = (self.directory / "manifest.json").read_bytes()
        original_weights = weights.read_bytes()
        for changed in ("expiry", "capacity", "weights", "manifest", "missing_config_record"):
            with self.subTest(changed=changed):
                (self.directory / "cnn_autoencoder_finalist.json").write_bytes(original_settings)
                (self.directory / "manifest.json").write_bytes(original_manifest)
                weights.write_bytes(original_weights)
                publish(self.directory)
                if changed in ("expiry", "capacity"):
                    settings = json.loads(original_settings)
                    settings["benchmark_memory"]["expiry_tbs" if changed == "expiry" else "capacity"] += 1
                    (self.directory / "cnn_autoencoder_finalist.json").write_text(json.dumps(settings))
                    self.seal_export(8)
                elif changed == "weights":
                    weights.write_bytes(b"replacement weights")
                    self.seal_export(8)
                elif changed == "manifest":
                    saved = json.loads(original_manifest)
                    saved["update"] += 1
                    (self.directory / "manifest.json").write_text(json.dumps(saved))
                else:
                    saved_result = json.loads((report / "results.json").read_text())
                    candidate = saved_result["manifests"]["candidate"]
                    candidate["support_artifacts"] = [r for r in candidate["support_artifacts"]
                                                      if r["name"] != "cnn_autoencoder_finalist.json"]
                    saved_result["manifests"]["candidate"] = seal_payload(candidate, field="manifest_sha256")
                    (report / "results.json").write_text(json.dumps(seal_payload(saved_result, field="result_sha256")))
                with self.assertRaisesRegex(ValueError, "artifacts or settings changed"):
                    verified_result(report)

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
