#!/usr/bin/env python3
"""Test the corrected temporal training scheduler."""

from __future__ import annotations

import unittest

import numpy as np

from temporal_training_data import TemporalTrainingDataGenerator


def make_scheduler(seed: int):
    generator = TemporalTrainingDataGenerator.__new__(
        TemporalTrainingDataGenerator
    )
    generator.num_scheduled_tx = 2
    generator.ue_pool_size = 8
    generator.effective_ue_pool_size_range = (3, 8)
    generator.dynamic_scheduling = True
    generator.schedule_switch_prob = 0.65
    generator.schedule_reorder_prob = 0.50
    generator.schedule_switch_prob_range = (0.15, 0.75)
    generator.schedule_reorder_prob_range = (0.0, 0.75)
    generator.schedule_rng = np.random.RandomState(seed)
    return generator


class TemporalTrainingDataV2Test(unittest.TestCase):
    def test_pool_is_larger_than_simultaneous_positions(self):
        generator = make_scheduler(17)
        schedule = generator.sample_schedule(32, 16).numpy()
        self.assertEqual(schedule.shape, (32, 16, 2))
        self.assertTrue(np.all(schedule >= 0))
        self.assertTrue(np.all(schedule < 8))
        self.assertTrue(all(
            len(np.unique(row)) == 2 for row in schedule.reshape(-1, 2)
        ))
        sizes = [
            row["effective_ue_pool_size"]
            for row in generator.last_schedule_metadata
        ]
        self.assertTrue(all(3 <= value <= 8 for value in sizes))
        self.assertGreater(len(set(sizes)), 1)

    def test_schedule_is_deterministic(self):
        first = make_scheduler(23).sample_schedule(8, 16).numpy()
        second = make_scheduler(23).sample_schedule(8, 16).numpy()
        np.testing.assert_array_equal(first, second)

if __name__ == "__main__":
    unittest.main()
