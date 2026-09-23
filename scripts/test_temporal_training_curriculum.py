#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from temporal_training_curriculum import (
    STRATA,
    event_relative_candidates,
    select_event_relative_targets,
    select_unique_event_relative_targets,
)


class TemporalTrainingCurriculumTest(unittest.TestCase):
    def setUp(self):
        self.ids = np.asarray([
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 1],
            [0, 1],
            [0, 1],
        ])
        self.active = np.ones_like(self.ids, bool)
        self.reset = np.zeros_like(self.active)
        self.reset[[1, 4, 6, 7]] = True
        self.valid = np.ones_like(self.active)
        self.valid[0] = False
        self.valid[self.reset] = False

    def test_all_strata_are_detected(self):
        candidates = event_relative_candidates(
            self.ids, self.active, self.reset, self.valid
        )
        self.assertEqual(tuple(candidates), STRATA)
        self.assertIn(0, candidates["cold"])
        self.assertIn(6, candidates["return"])
        self.assertIn(3, candidates["partner_change"])
        self.assertIn(1, candidates["rebuild"])
        self.assertIn(2, candidates["mature"])

    def test_selection_is_balanced_and_deterministic(self):
        first = select_event_relative_targets(
            self.ids,
            self.active,
            self.reset,
            self.valid,
            np.random.RandomState(9),
        )
        second = select_event_relative_targets(
            self.ids,
            self.active,
            self.reset,
            self.valid,
            np.random.RandomState(9),
        )
        self.assertEqual(first, second)
        self.assertEqual(tuple(target.stratum for target in first), STRATA)

    def test_numeric_identity_relabeling_is_invariant(self):
        mapping = {0: 17, 1: 3, 2: 91}
        relabeled = np.vectorize(mapping.get)(self.ids)
        original = event_relative_candidates(
            self.ids, self.active, self.reset, self.valid
        )
        changed = event_relative_candidates(
            relabeled, self.active, self.reset, self.valid
        )
        self.assertEqual(original, changed)

    def test_v3_targets_use_distinct_transport_blocks(self):
        targets = select_unique_event_relative_targets(
            self.ids,
            self.active,
            self.reset,
            self.valid,
            np.random.RandomState(9),
        )
        self.assertIsNotNone(targets)
        self.assertEqual(tuple(target.stratum for target in targets), STRATA)
        self.assertEqual(len({target.tb for target in targets}), len(STRATA))

    def test_missing_return_rejects_episode(self):
        ids = np.tile(np.asarray([[0, 1]]), (9, 1))
        result = select_event_relative_targets(
            ids,
            self.active,
            self.reset,
            self.valid,
            np.random.RandomState(1),
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
