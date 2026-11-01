import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from artifact_provenance import evaluation_source_records
from temporal_benchmark_episodes import CachedScenarioBank
from temporal_benchmark_frozen import make_manifest, protocol_and_plan
from temporal_benchmark_reference import evaluate_frozen_references, frozen_result_cache, reference_binding
from temporal_generalization_protocol import seal_payload
from test_benchmark_fixtures import SyntheticScenarioBank, ZeroAdapter

ROOT = Path(__file__).resolve().parent.parent


class BenchmarkEntrypointTest(unittest.TestCase):
    def test_entrypoints_reach_argument_validation_without_pythonpath_from_any_directory(self):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            cases = (
                ("run_temporal_benchmark.py", ["--experiment-phase", "latency", "--experiment-prbs", "4",
                 "--output-dir", str(Path(temporary) / "unused")], "Full experiments require"),
                ("smoke_temporal_benchmark_adapters.py", ["--prbs", "0", "--candidate-checkpoint",
                 str(ROOT / "models/learned_blend_joint_20261019/checkpoint_005000")], "Invalid smoke-test PRB count"),
            )
            for cwd in (ROOT, Path(temporary)):
                for name, arguments, message in cases:
                    with self.subTest(cwd=cwd, command=name):
                        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / name),
                                                 "--gpu", "-1", *arguments], cwd=cwd, env=environment,
                                                capture_output=True, text=True, timeout=45)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(message, result.stderr)
                        self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((Path(temporary) / "unused").exists())


class ReferenceReuseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol, self.plan = protocol_and_plan("smoke", 4, episodes=1)
        self.methods = tuple(self.protocol["execution"]["accuracy_reference_methods"])
        self.manifests = {m: make_manifest(self.protocol, m)[0] for m in self.methods}
        self.runtime = {"packages": {"tensorflow": "test"}, "tf32": True}
        self.identity = {"radio": self.protocol["radio"], "source": "test"}

    def binding(self, root=ROOT):
        return reference_binding(self.protocol, self.plan, self.identity, self.manifests,
                                 ["episode content hash"], self.runtime, root=root)

    def test_cnn_source_edits_reuse_cold_results_but_shared_decoder_edits_recompute(self):
        clone = self.directory / "source"
        for name in evaluation_source_records():
            destination = clone / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        baseline = self.binding(clone)
        for name in ("scripts/receiver_decoding.py", "scripts/temporal_benchmark_runtime.py",
                     "scripts/temporal_benchmark_scoring.py", "utils/neural_rx.py"):
            self.assertIn(name, baseline["sources"])
        self.assertFalse(any("pilot_history_receiver" in name or name.endswith("temporal_benchmark_frozen.py")
                             for name in baseline["sources"]))
        compute = mock.Mock(return_value=({"errors": np.zeros(2, bool)}, {"checked": True}, 0.))
        cache = self.directory / "cache"
        frozen_result_cache(cache, baseline, compute)
        old_protocol = self.protocol["protocol_sha256"]
        for name in ("scripts/pilot_history_receiver/model.py", "scripts/pilot_history_receiver/pilot_history.py",
                     "scripts/pilot_history_receiver/benchmark_adapter.py"):
            path = clone / name
            path.write_text(path.read_text() + "\nSOURCE_CHANGE = True\n")
        self.protocol["runtime_sources"] = evaluation_source_records(clone)
        self.protocol = seal_payload(self.protocol, field="protocol_sha256")
        self.manifests = {m: make_manifest(self.protocol, m)[0] for m in self.methods}
        self.assertNotEqual(self.protocol["protocol_sha256"], old_protocol)
        changed = self.binding(clone)
        self.assertEqual(changed, baseline)
        frozen_result_cache(cache, changed, compute)
        self.assertEqual(compute.call_count, 1)
        path = clone / "scripts/receiver_decoding.py"
        path.write_text(path.read_text() + "\nSOURCE_CHANGE = True\n")
        changed = self.binding(clone)
        self.assertNotEqual(changed, baseline)
        frozen_result_cache(cache, changed, compute)
        self.assertEqual(compute.call_count, 2)

    def test_reference_identity_covers_inputs_weights_rounds_radio_and_execution(self):
        baseline = self.binding()
        protocol = copy.deepcopy(self.protocol)
        protocol["execution"]["accuracy_episode_batch_size"] = 7
        changes = ((self.protocol, ["different input"], self.runtime),
                   (self.protocol, ["episode content hash"], dict(self.runtime, tf32=False)),
                   (protocol, ["episode content hash"], self.runtime))
        compute = mock.Mock(return_value=({"errors": np.zeros(2, bool)}, {"checked": True}, 0.))
        cache = self.directory / "cache"
        frozen_result_cache(cache, baseline, compute)
        for protocol, episodes, runtime in changes:
            changed = reference_binding(protocol, self.plan, self.identity, self.manifests, episodes, runtime)
            frozen_result_cache(cache, changed, compute)
        for field, key, value in (("inference", "iterations", 3), ("radio", "n_size_bwp", 12)):
            manifests = copy.deepcopy(self.manifests)
            manifests[self.methods[0]][field][key] = value
            changed = reference_binding(self.protocol, self.plan, self.identity, manifests,
                                        ["episode content hash"], self.runtime)
            self.assertNotEqual(changed, baseline)
            frozen_result_cache(cache, changed, compute)
        manifests = copy.deepcopy(self.manifests)
        manifests[self.methods[0]]["checkpoints"][0]["sha256"] = "changed weights"
        changed = reference_binding(self.protocol, self.plan, self.identity, manifests,
                                    ["episode content hash"], self.runtime)
        frozen_result_cache(cache, changed, compute)
        self.assertEqual(compute.call_count, 7)

    def test_integrated_reference_evaluation_skips_receiver_construction_on_cache_hit(self):
        bank = CachedScenarioBank(SyntheticScenarioBank(), self.directory / "episodes",
                                  self.protocol, self.identity)
        def build(**kwargs):
            return ZeroAdapter(kwargs["manifest"])
        with mock.patch("temporal_benchmark_reference.ColdNeuralRxBenchmarkAdapter", side_effect=build) as constructor, \
                mock.patch("temporal_benchmark_reference.reference_runtime", return_value=self.runtime):
            registrations = {m: make_manifest(self.protocol, m) for m in self.methods}
            first = evaluate_frozen_references(self.directory / "cache", self.protocol, self.plan,
                                              bank, registrations, None)[0]
            self.protocol["runtime_sources"]["scripts/pilot_history_receiver/model.py"] = "changed memory receiver"
            self.protocol = seal_payload(self.protocol, field="protocol_sha256")
            registrations = {m: make_manifest(self.protocol, m) for m in self.methods}
            second = evaluate_frozen_references(self.directory / "cache", self.protocol, self.plan,
                                               bank, registrations, None)[0]
            self.assertEqual(constructor.call_count, len(self.methods))
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
