#!/usr/bin/env python3
"""Test training-only received-power variation."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from temporal_power_training_data import (
    PowerAwareTemporalTrainingDataGenerator,
)


def make_power_sampler(seed: int):
    generator = PowerAwareTemporalTrainingDataGenerator.__new__(
        PowerAwareTemporalTrainingDataGenerator
    )
    generator.ue_pool_size = 4
    generator.power_rng = np.random.RandomState(seed)
    generator.received_power_augmentation = {
        "mode": "anonymous_episode_span_v1",
        "probability": 1.0,
        "episode_span_db": (9.0, 15.0),
        "scaling": "centered_raw_channel",
    }
    generator._power_schedule = np.asarray([
        [[0, 1], [2, 3]]
    ])
    return generator


class TemporalPowerTrainingDataTest(unittest.TestCase):
    def test_offsets_are_stable_and_broad(self):
        first, augmented = make_power_sampler(
            31
        )._sample_power_attenuation(1)
        second, _ = make_power_sampler(
            31
        )._sample_power_attenuation(1)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(augmented[0])
        self.assertEqual(float(np.min(first[0])), 0.0)
        self.assertGreaterEqual(float(np.max(first[0])), 9.0)
        self.assertLessEqual(float(np.max(first[0])), 15.0)

    def test_offsets_scale_channel_amplitude(self):
        channel = tf.ones(
            [1, 2, 1, 1, 2, 1, 1, 1], tf.complex64
        )
        scaled = (
            PowerAwareTemporalTrainingDataGenerator
            ._apply_power_attenuation(
                channel, np.asarray([[0.0, 10.0]], np.float32)
            )
            .numpy()
        )
        strong = np.mean(np.abs(scaled[:, :, :, :, 0]) ** 2)
        weak = np.mean(np.abs(scaled[:, :, :, :, 1]) ** 2)
        self.assertAlmostEqual(float(strong / weak), 10.0, places=5)
        self.assertAlmostEqual(float(strong * weak), 1.0, places=5)

    def test_power_stream_is_required(self):
        streams = {
            "channel": 1,
            "payload": 2,
            "noise": 3,
            "schedule": 4,
        }
        with self.assertRaisesRegex(ValueError, "five seed streams"):
            PowerAwareTemporalTrainingDataGenerator._validate_power_seed_streams(
                streams
            )


if __name__ == "__main__":
    unittest.main()
