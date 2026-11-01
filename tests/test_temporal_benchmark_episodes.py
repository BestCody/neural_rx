import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from temporal_benchmark_episodes import CachedScenarioBank
from temporal_benchmark_protocol import load_benchmark_protocol
from temporal_benchmark_runtime import episode_fingerprint
from test_benchmark_fixtures import SyntheticScenarioBank


class EpisodeCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.protocol = load_benchmark_protocol(Path(__file__).resolve().parents[1] / "config/temporal_benchmark_v1.json")
        self.job = {"condition": "test", "provider": "independent_tdl",
                    "provider_settings": {}, "episode_count": 2, "num_tbs": 2,
                    "snr_db": [1.0, 2.0], "seed_streams": {"channel": 100}}
        self.provider = Mock(wraps=SyntheticScenarioBank())
        self.bank = CachedScenarioBank(self.provider, self.temp.name, self.protocol, {"source": "one"})

    def test_shared_inputs_replay_without_generator_and_are_readonly(self):
        expected = SyntheticScenarioBank().generate(self.job, 0)
        first = self.bank.generate(self.job, 0)
        replay = CachedScenarioBank(None, self.temp.name, self.protocol, {"source": "one"}, readonly=True).generate(self.job, 0)
        self.assertEqual(self.provider.generate.call_count, 1)
        np.testing.assert_array_equal(first.fingerprint, episode_fingerprint(expected))
        np.testing.assert_array_equal(replay.fingerprint, first.fingerprint)
        np.testing.assert_array_equal(episode_fingerprint(expected), episode_fingerprint(first))
        np.testing.assert_array_equal(episode_fingerprint(first), episode_fingerprint(replay))
        with self.assertRaises(ValueError):
            replay.receiver_inputs[0][0]["received_grid"][0, 0] = -1
        self.assertNotIn("perfect_channel", replay.receiver_inputs[0][0])
        with np.load(self.bank.path_for(self.job, 0), allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"metadata", "bits", "active", "ue_ids", "channel_change_event"}
                             | {"receiver/" + n for n in self.protocol["execution"]["observable_receiver_inputs"]})

    def test_binding_covers_noise_schedule_sources_and_episode(self):
        original = self.bank.path_for(self.job, 0)
        changed = copy.deepcopy(self.job)
        changed["snr_db"][0] += 1
        self.assertNotEqual(original, self.bank.path_for(changed, 0))
        changed = copy.deepcopy(self.job)
        changed["provider_settings"]["scheduling"] = True
        self.assertNotEqual(original, self.bank.path_for(changed, 0))
        self.assertNotEqual(original, self.bank.path_for(self.job, 1))
        other = CachedScenarioBank(None, self.temp.name, self.protocol, {"source": "two"}, readonly=True)
        self.assertNotEqual(original, other.path_for(self.job, 0))
        with self.assertRaises(FileNotFoundError):
            other.generate(self.job, 0)

    def test_modified_received_signal_is_rejected(self):
        self.bank.generate(self.job, 0)
        hashes = self.bank.content_hashes([self.job])
        path = self.bank.path_for(self.job, 0)
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in source.files}
        arrays["receiver/received_grid"][0, 0, 0, 0] += 1
        np.savez(path, **arrays)
        changed = self.bank.content_hashes([self.job])
        self.assertNotEqual(hashes[0], changed[0])
        self.assertEqual(hashes[1], changed[1])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.bank.generate(self.job, 0)
        self.assertEqual(self.provider.generate.call_count, 2)

    def test_evaluator_loads_each_episode_once_and_uses_verified_fingerprints(self):
        from temporal_benchmark_frozen import protocol_and_plan
        from temporal_benchmark_runtime import evaluate_method_set
        from test_benchmark_fixtures import ZeroAdapter, synthetic_candidate_manifest
        protocol, plan = protocol_and_plan("smoke", 132, episodes=2)
        bank = CachedScenarioBank(SyntheticScenarioBank(), self.temp.name, protocol, {"source": "test"})
        methods = ("candidate_persistent", "candidate_reset")
        manifest = synthetic_candidate_manifest(protocol)
        adapters = {method: ZeroAdapter(manifest) for method in methods}
        with patch("temporal_benchmark_runtime.validate_episode", side_effect=AssertionError("Repeated validation")), \
                patch.object(bank, "generate", wraps=bank.generate) as generate:
            evaluate_method_set(protocol, plan, bank, adapters, methods)
        self.assertEqual(generate.call_count, sum(job["episode_count"] for job in plan["jobs"]))

    def test_fixed_dataset_replays_original_inputs_without_regeneration_or_source_rules(self):
        from temporal_generalization_protocol import seal_payload
        old_identity = {"schema": "test", "radio": {"prbs": 132},
                        "sources": {"generator.py": "before"}}
        new_identity = dict(old_identity, sources={"different.py": "after"})
        old = CachedScenarioBank(self.provider, self.temp.name, self.protocol, old_identity)
        expected = old.generate(self.job, 0)
        path = Path(self.temp.name) / "dataset.json"
        path.write_text(json.dumps(seal_payload({"identity": old_identity})))
        current = CachedScenarioBank(self.provider, self.temp.name, self.protocol, new_identity)
        self.assertEqual(current.identity, old_identity)
        self.assertEqual(current.path_for(self.job, 0), old.path_for(self.job, 0))
        actual = current.generate(self.job, 0)
        np.testing.assert_array_equal(episode_fingerprint(expected), episode_fingerprint(actual))
        with self.assertRaises(FileNotFoundError):
            current.generate(self.job, 1)
        self.assertEqual(self.provider.generate.call_count, 1)
        with self.assertRaisesRegex(ValueError, "radio configuration"):
            CachedScenarioBank(None, self.temp.name, self.protocol, dict(new_identity, radio={"prbs": 12}))
        value = json.loads(path.read_text())
        value["identity"]["sources"] = {}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "does not match"):
            CachedScenarioBank(None, self.temp.name, self.protocol, new_identity)

    def test_failed_generation_does_not_publish_partial_cache(self):
        self.provider.generate.side_effect = RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            self.bank.generate(self.job, 0)
        self.assertFalse(self.bank.path_for(self.job, 0).exists())
        self.assertFalse(list(Path(self.temp.name).glob("*.tmp")))

    def test_concurrent_readers_publish_one_identical_episode(self):
        other = CachedScenarioBank(self.provider, self.temp.name, self.protocol, {"source": "one"})
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(bank.generate, self.job, 0) for bank in (self.bank, other)]
            episodes = [future.result() for future in futures]
        self.assertEqual(self.provider.generate.call_count, 1)
        np.testing.assert_array_equal(episode_fingerprint(episodes[0]), episode_fingerprint(episodes[1]))

    def test_space_check_stops_before_generating_the_whole_bank(self):
        with patch("temporal_benchmark_episodes.shutil.disk_usage", side_effect=[
            Mock(free=100 * 1024**3), Mock(free=0),
        ]):
            with self.assertRaisesRegex(OSError, "larger volume"):
                self.bank.ensure_capacity([self.job])
        self.assertEqual(self.provider.generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
