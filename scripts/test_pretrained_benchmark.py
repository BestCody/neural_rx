import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from temporal_benchmark_candidate import artifact_record, pretrained_checkpoint, validate_manifest
from temporal_benchmark_frozen import make_manifest, protocol_and_plan
from temporal_benchmark_runtime import evaluate_method_set, _measure_latency
from temporal_benchmark_scoring import crc_diagnostics
from temporal_benchmark_frozen import rates
from temporal_generalization_protocol import seal_payload
from test_benchmark_fixtures import SyntheticScenarioBank, ZeroAdapter, synthetic_candidate_manifest
from temporal_benchmark_protocol import load_benchmark_protocol, validate_benchmark_protocol


class PretrainedBenchmarkTest(unittest.TestCase):
    def test_checkpoint_selection_requires_one_artifact_and_matching_candidate_seed(self):
        from temporal_benchmark_adapters import _checkpoint_path
        record = {"sha256": "weights", "training_seed": 5}
        artifacts = {"weights": Path("models/pretrained/test")}
        manifest = {"role": "candidate", "checkpoints": [record]}
        self.assertEqual(_checkpoint_path(manifest, 5, artifacts), artifacts["weights"].resolve())
        with self.assertRaisesRegex(ValueError, "seed differs"):
            _checkpoint_path(manifest, 6, artifacts)
        for records in ([], [record, record]):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _checkpoint_path(dict(manifest, checkpoints=records), 5, artifacts)
        manifest["role"] = "frozen_cold_k8"
        self.assertEqual(_checkpoint_path(manifest, 6, artifacts), artifacts["weights"].resolve())

    def test_accuracy_batch_size_controls_development_but_not_latency(self):
        protocol = load_benchmark_protocol(Path(__file__).resolve().parents[1] / "config/temporal_benchmark_v1.json")
        protocol["execution"]["accuracy_episode_batch_size"] = 7
        protocol = seal_payload(protocol, field="protocol_sha256")
        validate_benchmark_protocol(protocol)
        anchors = {name: {"anchor_snr_db": 3.} for name in protocol["conditions"]}
        with mock.patch("temporal_benchmark_frozen.load_benchmark_protocol", side_effect=lambda _: copy.deepcopy(protocol)):
            for phase, expected in (("development", 7), ("confirmation", 7), ("calibration", 7), ("smoke", 1), ("latency", 1)):
                actual, _ = protocol_and_plan(phase, 132, episodes=1, anchors=anchors)
                self.assertEqual(actual["execution"]["accuracy_episode_batch_size"], expected)

    def test_invalid_or_unsupported_settings_are_rejected(self):
        original = load_benchmark_protocol(Path(__file__).resolve().parents[1] / "config/temporal_benchmark_v1.json")
        for section, key, value in (("statistics", "confidence_level", .8),
                                     ("execution", "paired_reference_cache_required", False),
                                     ("execution", "accuracy_episode_batch_size", 0),
                                     ("execution", "accuracy_episode_batch_size", 2.5)):
            protocol = copy.deepcopy(original)
            protocol[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_benchmark_protocol(seal_payload(protocol, field="protocol_sha256"))
        for key, value in (("seed_offset", 999), ("pathloss", False), ("shadow_fading", True), ("role", "stress")):
            protocol = copy.deepcopy(original)
            protocol["conditions"]["natural_umi"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unsupported condition"):
                validate_benchmark_protocol(seal_payload(protocol, field="protocol_sha256"))

    def test_shared_adapter_conversion_preserves_batch_order_values_and_types(self):
        import tensorflow as tf
        from temporal_benchmark_candidate import decoder_output, stack_receiver_grids
        inputs = [{"received_grid": np.full((1, 2, 3), index + 1j, np.complex64),
                   "ls_channel": np.full((2, 3), index, np.float64)} for index in (7, 2)]
        y, channel = stack_receiver_grids(inputs, tf)
        np.testing.assert_array_equal(y.numpy(), np.stack([p["received_grid"] for p in inputs]))
        np.testing.assert_array_equal(channel.numpy(), np.stack([p["ls_channel"] for p in inputs]))
        self.assertEqual(channel.dtype, tf.float32)
        expected = np.array([[.25, -1.], [3., 2.]], np.float64)
        from types import SimpleNamespace
        from temporal_benchmark_candidate import checked_decoder_tensors
        hard, crc = tf.constant(expected > 0), tf.constant([True, False])
        unread = mock.Mock()
        unread.numpy.side_effect = AssertionError("Normal scoring must not copy logits")
        scored = decoder_output(unread, hard, crc)
        self.assertEqual(set(scored), {"hard_bits", "crc_status"})
        result = decoder_output(tf.constant(expected), hard, crc, include_logits=True)
        for name, value in scored.items():
            np.testing.assert_array_equal(value, result[name])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            tf.function(lambda: checked_decoder_tensors(SimpleNamespace(
                logits=tf.constant([[float("nan")]]), hard_bits=hard, crc_status=crc)))()
        np.testing.assert_array_equal(result["logits"], expected.astype(np.float32))
        np.testing.assert_array_equal(result["hard_bits"], expected > 0)
        np.testing.assert_array_equal(result["crc_status"], [True, False])
        self.assertEqual(result["logits"].dtype, np.float32)
        self.assertEqual(result["hard_bits"].dtype, np.bool_)
        self.assertEqual(result["crc_status"].dtype, np.bool_)

    def setUp(self):
        self.protocol, self.plan = protocol_and_plan("smoke", 132, episodes=2)

    def test_valid_crc_disagreements_are_counted_without_rejecting_evaluation(self):
        import tensorflow as tf
        from sionna.fec.crc import CRCEncoder, CRCDecoder
        encoder = CRCEncoder('CRC24A')
        payload = tf.constant([[1.,1.,1.,1.],[0.,0.,0.,0.]])
        coded = encoder(payload).numpy()
        coded[1,-1] = 1.-coded[1,-1]
        bits, crc = CRCDecoder(encoder)(coded)
        hard, status = bits.numpy().astype(bool),crc.numpy().reshape(2).astype(bool)
        np.testing.assert_array_equal(status,[True,False])
        from smoke_temporal_benchmark_adapters import _crc_diagnostics
        self.assertEqual(_crc_diagnostics(dict(hard_bits=hard,crc_status=status),
            np.zeros_like(hard),np.ones(2,bool)),
            dict(undetected_payload_errors=1,crc_failures_with_correct_payload=1))
        class Adapter(ZeroAdapter):
            def decode_tb(self,inputs):
                return dict(hard_bits=hard.copy(),crc_status=status.copy(),logits=np.zeros_like(hard,np.float32))
            def decode_tb_batch(self,inputs):
                output = self.decode_tb(inputs[0])
                return {key:np.stack([value]*len(inputs)) for key,value in output.items()}
        manifest = synthetic_candidate_manifest(self.protocol)
        methods = ('candidate_persistent','candidate_reset')
        adapters = {method:Adapter(manifest) for method in methods}
        provider = SyntheticScenarioBank()
        arrays,checks,_ = evaluate_method_set(self.protocol,self.plan,provider,adapters,methods)
        self.assertTrue(all(checks.values()))
        for method in methods:
            errors,disagreement = arrays[f'errors_{method}'],arrays[f'crc_disagreement_{method}']
            self.assertTrue(errors[...,0].all())
            self.assertFalse(errors[...,1].any())
            self.assertTrue(disagreement.all())
            counts = crc_diagnostics(errors,disagreement)
            self.assertEqual(counts['undetected_payload_errors'],errors[...,0].size)
            self.assertEqual(counts['crc_failures_with_correct_payload'],errors[...,1].size)
            self.assertEqual(rates(arrays,[method],1)[method],[[.5]])
        protocol = copy.deepcopy(self.protocol)
        protocol['execution']['latency'].update(warmup_samples=1,sample_count=3)
        episode = provider.generate(self.plan['jobs'][0],0)
        with mock.patch('temporal_benchmark_runtime._latency_inputs',return_value=([episode],0)):
            timings = _measure_latency(protocol,self.plan,provider,adapters,methods)
        for method in methods:
            self.assertEqual(timings[f'latency_ms_{method}'].shape,(3,))
            self.assertEqual(crc_diagnostics(timings[f'latency_errors_{method}'],
                timings[f'latency_crc_disagreement_{method}']),
                dict(undetected_payload_errors=3,crc_failures_with_correct_payload=3))

    def test_all_references_bind_the_same_shipped_weights(self):
        expected = artifact_record(pretrained_checkpoint())
        for role in ("frozen_cold_k2", "frozen_cold_k3", "frozen_cold_k8"):
            manifest, _, _ = make_manifest(self.protocol, role)
            for name, value in expected.items():
                self.assertEqual(manifest["checkpoints"][0][name], value)
            self.assertEqual(manifest["training"]["additional_updates"], 0)

    def test_alternative_weights_are_rejected_even_with_a_valid_manifest_hash(self):
        manifest, _, _ = make_manifest(self.protocol, "frozen_cold_k2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.weights.h5"
            path.write_bytes(b"different trained weights")
            manifest["checkpoints"][0].update(artifact_record(path))
            manifest = seal_payload(manifest, field="manifest_sha256")
            with self.assertRaisesRegex(ValueError, "shipped pretrained"):
                validate_manifest(manifest, self.protocol)


    def test_persistent_and_reset_share_one_checkpoint(self):
        manifest = synthetic_candidate_manifest(self.protocol)
        methods = ("candidate_persistent", "candidate_reset")
        adapters = {method: ZeroAdapter(manifest) for method in methods}
        arrays, checks, _ = evaluate_method_set(self.protocol, self.plan, SyntheticScenarioBank(), adapters, methods)
        self.assertTrue(all(checks.values()))
        np.testing.assert_array_equal(arrays["errors_candidate_persistent"], arrays["errors_candidate_reset"])
        self.assertEqual(arrays["errors_candidate_persistent"].shape, (1, 1, 2, 8, 2))

    def test_reference_group_accepts_only_pretrained_manifests(self):
        methods = tuple(self.protocol["execution"]["accuracy_reference_methods"])
        adapters = {method: ZeroAdapter(make_manifest(self.protocol, method)[0]) for method in methods}
        _, checks, _ = evaluate_method_set(self.protocol, self.plan, SyntheticScenarioBank(), adapters, methods)
        self.assertTrue(all(checks.values()))
        self.assertIn("frozen_reference_artifacts_unchanged", checks)
        changed = copy.deepcopy(adapters[methods[0]].manifest)
        changed["training"]["additional_updates"] = 500
        adapters[methods[0]].manifest = seal_payload(changed, field="manifest_sha256")
        with self.assertRaises(ValueError):
            evaluate_method_set(self.protocol, self.plan, SyntheticScenarioBank(), adapters, methods)

    def test_development_and_confirmation_use_separate_predeclared_banks(self):
        anchors = {name: {"anchor_snr_db": 3.} for name in self.protocol["conditions"]}
        development, dev_plan = protocol_and_plan("development", 132, episodes=1, anchors=anchors)
        confirmation, final_plan = protocol_and_plan("confirmation", 132, episodes=1, anchors=anchors)
        self.assertEqual({j["condition"] for j in dev_plan["jobs"]}, set(self.protocol["development_conditions"]))
        self.assertEqual({j["condition"] for j in final_plan["jobs"]}, set(self.protocol["conditions"]))
        self.assertEqual(development["core_conditions"], confirmation["core_conditions"])
        self.assertIn("natural_uma", development["core_conditions"])
        self.assertTrue({j["seed_streams"]["channel"] for j in dev_plan["jobs"]}.isdisjoint(
                        {j["seed_streams"]["channel"] for j in final_plan["jobs"]}))


if __name__ == "__main__":
    unittest.main()
