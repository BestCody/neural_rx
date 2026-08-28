#!/usr/bin/env python3
"""CPU-only tests for the 64-TB evaluator's methodology and metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_temporal_64tb as evaluator


class Temporal64TBEvaluatorTest(unittest.TestCase):
    def test_research_defaults_use_64_tbs_and_paired_baselines(self):
        parser = evaluator.build_parser()
        self.assertEqual(parser.get_default("num_tbs"), 64)
        self.assertEqual(parser.get_default("trajectory_snr"), 2.75)
        self.assertEqual(parser.get_default("execution_mode"), "graph")
        self.assertTrue(parser.get_default("temporal_only"))

    def test_snr_grid_is_inclusive_and_stable(self):
        self.assertEqual(
            evaluator.make_snr_grid(1.5, 2.0, 0.25),
            [1.5, 1.75, 2.0],
        )
        with self.assertRaises(ValueError):
            evaluator.make_snr_grid(2.0, 1.0, 0.25)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = evaluator.wilson_interval(10, 100)
        self.assertLess(low, 0.10)
        self.assertGreater(high, 0.10)
        self.assertEqual(evaluator.wilson_interval(0, 0), [None, None])

    def test_log_crossing_uses_tb2plus_counts(self):
        points = [
            {"snr_db": 2.0, "errors_tb2plus": 20, "blocks_tb2plus": 100},
            {"snr_db": 3.0, "errors_tb2plus": 5, "blocks_tb2plus": 100},
        ]
        crossing = evaluator.log_bler_crossing(points, target=0.10)
        self.assertIsNotNone(crossing)
        self.assertGreater(crossing, 2.0)
        self.assertLess(crossing, 3.0)

    def test_counter_separates_cold_start_and_tb2plus(self):
        counter = evaluator.new_counter(64)
        evaluator.add_counts(counter, 0, errors=2, blocks=4)
        evaluator.add_counts(counter, 1, errors=1, blocks=4)
        evaluator.add_counts(counter, 63, errors=3, blocks=4)
        result = evaluator.finalize_counter(counter, snr_db=2.75, batches=1)
        self.assertEqual(result["errors_all"], 6)
        self.assertEqual(result["errors_tb2plus"], 4)
        self.assertEqual(result["blocks_tb2plus"], 8)
        self.assertEqual(result["segments"]["cold_start"]["errors"], 2)
        self.assertEqual(result["segments"]["late"]["errors"], 3)
        self.assertEqual(len(result["per_tb"]), 64)

    def test_checkpoint_provenance_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "model.weights.h5"
            checkpoint.touch()
            summary = {
                "architecture": evaluator.ARCHITECTURE,
                "training_method": dict(evaluator.REQUIRED_TRAINING_INVARIANTS),
                "config": "nrx_large.cfg",
                "pooling": "mean",
                "compression": "writer",
                "d_mem": 32,
                "num_it": 1,
                "train_steps": 6000,
                "stream_len": 64,
                "tbptt_window": 4,
                "memory_reset_prob": 0.05,
                "seed": 1,
                "ue_pool_size": 4,
                "dynamic_scheduling": False,
                "memory_expiry_slots": 8,
                "schedule_switch_prob": 0.65,
                "schedule_reorder_prob": 0.50,
            }
            (directory / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            spec = evaluator.load_checkpoint_spec(
                str(checkpoint), expected_k=1, expected_config="nrx_large.cfg"
            )
            self.assertEqual(spec["num_it"], 1)
            self.assertEqual(spec["stream_len"], 64)

            summary["training_method"]["state_carried_across_tbptt_windows"] = False
            (directory / "training_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "state_carried"):
                evaluator.load_checkpoint_spec(
                    str(checkpoint), expected_k=1,
                    expected_config="nrx_large.cfg"
                )

    def test_trailing_mean_is_causal(self):
        np.testing.assert_allclose(
            evaluator._trailing_mean([1.0, 3.0, 5.0, 7.0], window=2),
            [1.0, 2.0, 4.0, 6.0],
        )


if __name__ == "__main__":
    unittest.main()
