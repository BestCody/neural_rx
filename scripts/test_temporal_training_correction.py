#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from register_temporal_benchmark_stage import validate_training_pair
from train_temporal_benchmark import (
    candidate_user_weights,
    effective_user_difficulty_db,
    episode_difficulty_scores,
    learning_rate_at_step,
    power_user_weights,
)


ROOT = Path(__file__).resolve().parents[1]


class TemporalTrainingCorrectionTest(unittest.TestCase):
    def test_v2_contract_has_balanced_general_ue_training(self):
        data = json.loads((
            ROOT / "config" / "temporal_benchmark_training_data_v2.json"
        ).read_text(encoding="utf-8"))
        recipe = json.loads((
            ROOT / "config" / "temporal_benchmark_training_recipe_v2.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(data["effective_ue_pool_size_range"], [3, 8])
        self.assertEqual(data["updates_per_sequence"], 5)
        self.assertEqual(len(data["target_strata"]), 5)
        self.assertEqual(
            recipe["candidate_teacher"]["source"],
            "matched_stage_checkpoint",
        )
        self.assertTrue(
            recipe["candidate_teacher"]["exact_checkpoint_required"]
        )

    def test_v3_contract_targets_weak_users(self):
        data = json.loads((
            ROOT / "config" / "temporal_benchmark_training_data_v3.json"
        ).read_text(encoding="utf-8"))
        recipe = json.loads((
            ROOT / "config" / "temporal_benchmark_training_recipe_v3.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            data["target_selection"],
            "event_relative_unique_near_far_v3",
        )
        power = data["received_power_augmentation"]
        self.assertEqual(power["episode_span_db"], [9.0, 15.0])
        self.assertEqual(
            recipe["candidate_initialization"]["source"],
            "matched_teacher_executed_k2_path",
        )

    def test_power_priority_favors_teachable_weak_user(self):
        attenuation = tf.constant([[0.0, 15.0]])
        power = power_user_weights(attenuation, 15.0, 3.0, tf).numpy()
        np.testing.assert_allclose(power, [[1.0, 3.0]])
        combined = candidate_user_weights(
            attenuation,
            tf.constant([[0.20, 0.30]]),
            tf.constant([[0.20, 0.20]]),
            15.0,
            {
                "power_max_multiplier": 3.0,
                "teacher_advantage_max_multiplier": 2.0,
                "teacher_advantage_scale": 0.05,
                "combined_max_multiplier": 4.0,
            },
            tf,
        ).numpy()
        np.testing.assert_allclose(combined, [[1.0, 4.0]])

    def test_difficulty_uses_actual_near_far_gap(self):
        channel = tf.reshape(
            tf.constant([1.0 + 0.0j, 10.0 ** -0.5 + 0.0j]),
            [1, 1, 1, 2, 1, 1, 1],
        )
        difficulty, power = effective_user_difficulty_db(
            channel,
            tf.constant([[1.0, 1.0]]),
            tf.zeros([1, 2]),
            tf,
        )
        np.testing.assert_allclose(power.numpy(), [[0.0, -10.0]], atol=1e-5)
        np.testing.assert_allclose(
            difficulty.numpy(), [[0.0, 10.0]], atol=1e-5
        )

    def test_target_score_prefers_near_far_over_two_weak_users(self):
        amplitudes = tf.constant([
            [10.0 ** -0.5, 10.0 ** -0.55],
            [10.0 ** 0.25, 10.0 ** -0.25],
        ], tf.complex64)
        channel = tf.reshape(amplitudes, [1, 2, 1, 1, 2, 1, 1, 1])
        scores = episode_difficulty_scores({
            "h": channel,
            "active": tf.ones([1, 2, 2]),
        }, tf)
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_v3_learning_rate_warms_and_decays(self):
        recipe = json.loads((
            ROOT / "config" / "temporal_benchmark_training_recipe_v3.json"
        ).read_text(encoding="utf-8"))
        values = [
            learning_rate_at_step(recipe, step, 500)
            for step in (0, 24, 499)
        ]
        np.testing.assert_allclose(values, [2e-5, 1e-4, 1e-5])

    def test_registration_requires_exact_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "training_summary.json"
            summary.write_text("sealed teacher\n", encoding="utf-8")
            digest = hashlib.sha256(summary.read_bytes()).hexdigest()
            baseline = {
                "data_fingerprint": "data",
                "full_checkpoint_sha256": "checkpoint",
                "final_value_checksum": "model",
                "final_weight_signature": [{"shape": [1]}],
            }
            candidate = {
                "data_fingerprint": "data",
                "teacher_checkpoint_sha256": "checkpoint",
                "teacher_value_checksum": "model",
                "teacher_weight_signature": [{"shape": [1]}],
                "teacher_summary_sha256": digest,
            }
            validate_training_pair(candidate, baseline, summary)
            candidate["teacher_checkpoint_sha256"] = "other"
            with self.assertRaisesRegex(ValueError, "not matched K=8"):
                validate_training_pair(candidate, baseline, summary)


if __name__ == "__main__":
    unittest.main()
