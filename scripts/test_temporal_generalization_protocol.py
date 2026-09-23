#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from temporal_generalization_protocol import (
    assert_adapter_features,
    build_segment_safe_threshold_artifact,
    build_split_manifest,
    conformal_false_reset_threshold,
    episode_seed,
    innovation_jump_score,
    load_split_manifest,
    load_segment_safe_threshold_artifact,
    normalized_dmrs_innovation,
    role_profile_specs,
    seal_payload,
    seed_bundle,
    validate_split_manifest,
    validate_split_slice,
)


class TemporalGeneralizationProtocolTest(unittest.TestCase):
    def test_repository_manifest_is_sealed_and_disjoint(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "temporal_generalization_protocol.json"
        )
        manifest = load_split_manifest(path)
        self.assertEqual(
            [profile.model for profile in role_profile_specs(
                manifest, "training"
            )],
            ["A", "B", "C"],
        )
        self.assertEqual(
            [profile.model for profile in role_profile_specs(
                manifest, "ood"
            )],
            ["D", "E", "UMi"],
        )
        seeds = {
            episode_seed(manifest, role, domain, 0)
            for role in manifest["roles"]
            for domain in ("channel", "payload", "noise", "schedule")
        }
        self.assertEqual(len(seeds), 16)
        self.assertEqual(
            seed_bundle(manifest, "calibration", 7),
            {
                "channel": 2_000_007,
                "payload": 2_100_007,
                "noise": 2_200_007,
                "schedule": 2_300_007,
            },
        )

    def test_split_slice_cannot_cross_role_boundary(self):
        manifest = build_split_manifest()
        validate_split_slice(manifest, "calibration", 0, 128)
        with self.assertRaisesRegex(ValueError, "requests"):
            validate_split_slice(manifest, "calibration", 1, 128)

    def test_manifest_tampering_is_rejected(self):
        manifest = build_split_manifest()
        manifest["roles"]["ood"]["seed_base"] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_split_manifest(manifest)

    def test_benchmark_metadata_is_not_an_adapter_feature(self):
        assert_adapter_features([
            "posterior_delta",
            "posterior_log_precision",
            "posterior_reliability",
        ])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            assert_adapter_features([
                "posterior_delta", "channel_profile"
            ])
        with self.assertRaisesRegex(ValueError, "unapproved"):
            assert_adapter_features(["posterior_delta", "mystery"])

    def test_score_is_prefix_causal(self):
        rng = np.random.RandomState(8)
        innovation = rng.lognormal(size=[4, 20, 2])
        full, full_valid = innovation_jump_score(innovation, 8)
        prefix, prefix_valid = innovation_jump_score(
            innovation[:, :15], 8
        )
        np.testing.assert_array_equal(full[:, :15], prefix)
        np.testing.assert_array_equal(full_valid[:, :15], prefix_valid)
        self.assertFalse(np.any(full_valid[:, :9]))
        self.assertTrue(np.all(full_valid[:, 9:]))

    def test_dmrs_innovation_is_prefix_causal_and_noise_normalized(self):
        pilots = np.zeros([2, 5, 1, 1, 2, 1], np.complex64)
        pilots[:, 2:] = 2.0 + 0.0j
        errors = np.ones(pilots.shape, np.float64)
        full = normalized_dmrs_innovation(pilots, errors)
        prefix = normalized_dmrs_innovation(
            pilots[:, :4], errors[:, :4]
        )
        np.testing.assert_array_equal(full[:, :4], prefix)
        np.testing.assert_allclose(full[:, 2], 2.0)

    def test_threshold_uses_no_change_episode_maxima(self):
        score = np.zeros([128, 16, 2], np.float64)
        eligible = np.zeros_like(score, dtype=bool)
        eligible[:, 8:] = True
        for episode in range(128):
            score[episode, 8:, :] = 1.0 + episode / 128.0
        result = conformal_false_reset_threshold(
            score, eligible, alpha=0.01
        )
        self.assertEqual(result["order_statistic_rank"], 128)
        self.assertEqual(
            result["threshold"], result["maximum_episode_max"]
        )
        self.assertFalse(np.any(score[eligible] > result["threshold"]))

    def test_alpha_requires_enough_calibration_episodes(self):
        score = np.ones([64, 16, 1], np.float64)
        eligible = np.zeros_like(score, dtype=bool)
        eligible[:, 8:] = True
        with self.assertRaisesRegex(ValueError, "more calibration"):
            conformal_false_reset_threshold(
                score, eligible, alpha=0.01
            )

    def test_segment_safe_artifact_binds_state_machine(self):
        manifest = build_split_manifest()
        calibration = {
            "method": "episode_max_split_conformal",
            "alpha": 0.01,
            "calibration_episodes": 128,
            "order_statistic_rank": 128,
            "threshold": 1.9,
            "minimum_episode_max": 1.0,
            "maximum_episode_max": 1.9,
            "trigger_rule": "score_strictly_greater_than_threshold",
        }
        artifact = build_segment_safe_threshold_artifact(
            calibration, manifest["manifest_sha256"], "c" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment-safe.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            loaded = load_segment_safe_threshold_artifact(path, manifest)
            self.assertTrue(loaded["state_machine"]["probation_rebase"])
            artifact["state_machine"]["refractory_tbs"] = 9
            artifact = seal_payload(artifact, field="artifact_sha256")
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state machine"):
                load_segment_safe_threshold_artifact(path, manifest)


if __name__ == "__main__":
    unittest.main()
