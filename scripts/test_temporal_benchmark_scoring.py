import copy
from pathlib import Path
import unittest

import numpy as np

from run_memory_training import selection_key
from temporal_benchmark_frozen import development_report, protocol_and_plan, rates
from temporal_benchmark_plan import calibrate_condition
from temporal_benchmark_scoring import (
    TBLER_DEFINITION, adaptation_summary, block_error_rate, late_drift_summary,
    method_summary, paired_summary, payload_error_flags,
)


class EnumeratedBootstrap:
    def randint(self, low, high, size):
        if size == 4:
            return np.zeros(4, np.int64)
        if size == (4, 2):
            return np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        raise AssertionError((low, high, size))


class PooledScoringTest(unittest.TestCase):
    def test_payload_and_crc_flags_exclude_inactive_users_and_preserve_batches(self):
        from smoke_temporal_benchmark_adapters import _crc_diagnostics
        truth = np.zeros((4, 2, 3), np.int8)
        hard = truth.copy()
        hard[[1, 3], :, 1] = 1
        crc = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], bool)
        active = np.array([[1, 0]] * 4, bool)
        expected_errors = np.array([[0, 0], [1, 0], [0, 0], [1, 0]], bool)
        expected_disagreement = np.array([[0, 0], [1, 0], [1, 0], [0, 0]], bool)
        for index in (slice(None), 0, 1, 2, 3):
            errors, disagreement = payload_error_flags(hard[index], crc[index], truth[index], active[index])
            np.testing.assert_array_equal(errors, expected_errors[index])
            np.testing.assert_array_equal(disagreement, expected_disagreement[index])
        self.assertEqual(_crc_diagnostics(dict(hard_bits=hard, crc_status=crc), truth, active),
                         dict(undetected_payload_errors=1, crc_failures_with_correct_payload=1))

    def test_payload_flags_reject_shapes_that_would_broadcast_across_users(self):
        hard = np.zeros((2, 3), bool)
        for truth, crc, active in ((hard[:1], [1, 1], [1, 1]),
                                   (hard, [[1], [1]], [1, 1]),
                                   (hard, [1, 1], [1])):
            with self.subTest(truth=truth.shape, crc=crc, active=active), self.assertRaises(ValueError):
                payload_error_flags(hard, crc, truth, active)

    def test_empty_sample_handling_is_shared_without_fabricating_zero_errors(self):
        result = block_error_rate([1, 0, -1], [2, 0, 2], allow_empty=True)
        np.testing.assert_allclose(result, [.5, np.nan, -.5], equal_nan=True)
        with self.assertRaises(ValueError):
            block_error_rate([1, 0], [2, 0])
        with self.assertRaises(ValueError):
            block_error_rate([0], [-1], allow_empty=True)

    def setUp(self):
        active = np.ones((1, 2, 4, 2), bool)
        active[0, 1, 1:, 1] = False
        left = np.zeros((1, 1, 2, 4, 2), bool)
        right = left.copy()
        left[0, 0, 0, :3, 0] = True
        right[0, 0, 1, :2, 0] = True
        self.arrays = dict(active=active, errors_a=left, errors_b=right)
        self.protocol = {
            "statistics": {"primary_snr_point": "anchor"},
            "stages": {"development": {"num_tbs": 4, "scoring_window": [1, 4],
                "condition_names": ["schedule"], "snr_points": ["anchor"]}},
            "core_conditions": ["schedule"],
        }

    def test_report_selection_and_comparison_agree_with_unequal_episode_sizes(self):
        reported = rates(self.arrays, ["a", "b"], 1)
        summaries = {m: method_summary([self.arrays], m, self.protocol, "development")
                     for m in ("a", "b")}
        for method, expected in (("a", 3/13), ("b", 2/13)):
            self.assertAlmostEqual(reported[method][0][0], expected)
            self.assertAlmostEqual(summaries[method]["core_tbler"], expected)
        result = lambda m: dict(summary={"candidate_persistent": summaries[m]},
                               manifests={"candidate": {"training": {"additional_updates": 1}}})
        self.assertLess(selection_key(result("b")), selection_key(result("a")))
        paired = paired_summary([self.arrays], "a", "b", self.protocol,
                                "development", 4, EnumeratedBootstrap())
        self.assertAlmostEqual(paired["difference"], 1/13)
        expected = np.quantile([3/8, 1/13, 1/13, -2/5], [.025, .975])
        np.testing.assert_allclose(paired["ci95"], expected)
        self.assertEqual(paired["bootstrap_valid_samples"], 4)

    def test_inactive_blocks_and_empty_episodes_do_not_dilute_error_rate(self):
        arrays = copy.deepcopy(self.arrays)
        arrays["active"][0, 1] = False
        arrays["errors_a"][0, 0, 1] = True
        self.assertAlmostEqual(rates(arrays, ["a"], 1)["a"][0][0], 3/8)
        paired = paired_summary([arrays], "a", "b", self.protocol,
                                "development", 4, EnumeratedBootstrap())
        self.assertAlmostEqual(paired["difference"], 3/8)
        np.testing.assert_allclose(paired["ci95"], [3/8, 3/8])
        self.assertEqual(paired["bootstrap_valid_samples"], 3)
        arrays["active"][:] = False
        with self.assertRaisesRegex(ValueError, "no active"):
            method_summary([arrays], "a", self.protocol, "development")

    def test_counts_pool_across_runs_before_division(self):
        other = copy.deepcopy(self.arrays)
        other["active"][:] = False
        other["active"][..., 0, 0] = True
        other["errors_a"][:] = other["active"][:, None]
        other["errors_b"][:] = False
        summary = method_summary([self.arrays, other], "a", self.protocol, "development")
        self.assertAlmostEqual(summary["core_tbler"], 5/15)
        paired = paired_summary([self.arrays, other], "a", "b", self.protocol,
                                "development", 100, np.random.RandomState(4))
        self.assertAlmostEqual(paired["difference"], 3/15)

    def test_conditions_pool_transmitted_blocks_without_equal_condition_weight(self):
        arrays = {key: np.repeat(value, 2, axis=0) for key, value in self.arrays.items()}
        arrays["active"][1] = False
        arrays["active"][1, 0, 0, 0] = True
        protocol = copy.deepcopy(self.protocol)
        protocol["core_conditions"].append("other")
        protocol["stages"]["development"]["condition_names"].append("other")
        summary = method_summary([arrays], "a", protocol, "development")
        self.assertAlmostEqual(summary["per_condition_tbler"]["other"], 1.)
        self.assertAlmostEqual(summary["core_tbler"], 4/14)

    def test_late_drift_uses_each_windows_actual_transmission_count(self):
        arrays = {key: np.concatenate([value, value], axis=-2)
                  for key, value in self.arrays.items()}
        arrays["active"][0, 0, 4:, 1] = False
        arrays["active"][0, 1, 4:] = True
        arrays["errors_candidate_persistent"] = arrays.pop("errors_a")
        arrays["errors_candidate_reset"] = arrays.pop("errors_b")
        arrays["errors_candidate_persistent"][0, 0, 1, 4, 0] = True
        protocol = copy.deepcopy(self.protocol)
        protocol["stages"]["development"]["num_tbs"] = 8
        protocol["windows"] = dict(mature=[1, 4], late=[5, 8])
        protocol["stationary_conditions"] = ["schedule"]
        for reference in (None, "candidate_reset"):
            with self.subTest(reference=reference):
                observed = late_drift_summary([arrays], protocol, "development", 4,
                    EnumeratedBootstrap(), reference=reference)
                def estimate(indices):
                    value = 0.
                    for window, sign in ((slice(4, 8), 1), (slice(0, 4), -1)):
                        active = arrays["active"][0, indices, window]
                        errors = arrays["errors_candidate_persistent"][0, 0, indices, window]
                        numerator = int((errors & active).sum())
                        if reference:
                            numerator -= int((arrays[f"errors_{reference}"][0, 0, indices, window] & active).sum())
                        value += sign * numerator / active.sum()
                    return value
                self.assertAlmostEqual(observed["difference"], estimate([0, 1]))
                expected = np.quantile([estimate(indices) for indices in ([0, 0], [0, 1], [1, 0], [1, 1])], [.025, .975])
                np.testing.assert_allclose(observed["ci95"], expected)

    def test_development_windows_match_direct_counts(self):
        from temporal_benchmark_protocol import load_benchmark_protocol
        names = load_benchmark_protocol(Path(__file__).resolve().parents[1]/"config/temporal_benchmark_v1.json")["development_conditions"]
        protocol, _ = protocol_and_plan("development", 132, episodes=2,
            anchors={name: {"anchor_snr_db": 3.} for name in names})
        protocol["statistics"]["bootstrap_samples"] = 40
        arrays = {"active": np.ones((len(names), 2, 64, 2), bool),
                  "ue_ids": np.broadcast_to(np.arange(2), (len(names), 2, 64, 2))}
        arrays["active"][:, 1, ::2, 1] = False
        methods = ("candidate_persistent", "candidate_reset", "frozen_cold_k2", "frozen_cold_k8")
        random = np.random.RandomState(9).rand(len(names), 1, 2, 64, 2)
        for method, threshold in zip(methods, [.1, .2, .3, .4]):
            arrays[f"errors_{method}"] = random < threshold
        report = development_report(arrays, methods, protocol)
        self.assertEqual(report["tbler_definition"], TBLER_DEFINITION)
        for name, (first, last) in dict(protocol["windows"], tb1=[1, 1], whole_episode=[1, 64]).items():
            active = arrays["active"][:, :, first-1:last]
            for method in methods:
                error = arrays[f"errors_{method}"][:, :, :, first-1:last]
                expected = (error & active[:, None]).sum(axis=(2, 3, 4)) / active.sum(axis=(1, 2, 3))[:, None]
                np.testing.assert_allclose(report["windows"][name][method], expected)
        for method in methods:
            indices = [names.index(name) for name in protocol["core_conditions"]]
            mask = arrays["active"][indices, :, 16:]
            errors = arrays[f"errors_{method}"][indices, :, :, 16:]
            expected = (errors & mask[:, None]).sum()/mask.sum()
            self.assertAlmostEqual(report["summary"][method]["core_tbler"], expected)

    def test_adaptation_window_pools_counts_instead_of_averaging_tb_rates(self):
        arrays = {"active": np.ones((2, 2, 64, 2), bool),
                  "ue_ids": np.broadcast_to(np.arange(2), (2, 2, 64, 2)),
                  "errors_candidate_persistent": np.zeros((2, 1, 2, 64, 2), bool),
                  "errors_frozen_cold_k8": np.zeros((2, 1, 2, 64, 2), bool)}
        arrays["active"][0, :, 33] = False
        arrays["active"][0, 0, 33, 0] = True
        arrays["errors_candidate_persistent"][0, 0, 0, 32:34] = True
        protocol = copy.deepcopy(self.protocol)
        protocol["stages"]["development"]["condition_names"] = ["controlled_topology_change", "schedule_gap_reentry"]
        protocol["windows"] = dict(immediate_post_change=[33, 34], late_post_change=[57, 64])
        observed = adaptation_summary([arrays], protocol, "development")
        self.assertAlmostEqual(observed["immediate_mean_gap"], 3/5)
        self.assertEqual(observed["per_tb_gap"][32:34], [.5, 1.])

    def test_calibration_raw_rates_are_exact_counts_including_zero_errors(self):
        points = [dict(snr_db=i, errors=errors, blocks=100) for i, errors in enumerate([20, 0, 0])]
        result = calibrate_condition(points, .1)
        self.assertEqual(result["raw_tbler"], [.2, 0., 0.])
        self.assertTrue(result["target_bracketed"])
        self.assertAlmostEqual(result["snr_at_target_db"], .5)


if __name__ == "__main__":
    unittest.main()
