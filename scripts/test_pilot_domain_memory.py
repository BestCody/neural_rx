#!/usr/bin/env python3
"""Test causal PD-L8 routing and reset behavior."""

from __future__ import annotations

import unittest

import numpy as np

from pilot_domain_memory import (
    OnlineSegmentSafePilotDomainMemory,
    SegmentSafeDMRSGuard,
    SegmentSafePilotBank,
    active_dmrs_innovation,
    segment_safe_reset_detector,
    segmented_wiener_estimate,
)
from pilot_domain_channel import pilot_domain_wiener_estimate


class PilotDomainMemoryTest(unittest.TestCase):
    def test_innovation_ignores_inactive_ues(self):
        pilots = np.ones((1, 3, 2, 1, 2, 1), np.complex64)
        pilots[:, 1:, 1] = 10.0
        errors = np.ones(pilots.shape, np.float32)
        active = np.asarray([[[1, 1], [1, 0], [1, 1]]], bool)
        innovation = active_dmrs_innovation(pilots, errors, active)
        self.assertEqual(float(innovation[0, 1, 1]), 0.0)
        self.assertEqual(float(innovation[0, 2, 1]), 0.0)

    def test_segment_safe_guard_rejects_boundary_innovation(self):
        guard = SegmentSafeDMRSGuard(threshold=2.0)
        cold = guard.update(None, active=True, consecutive=False)
        probation = guard.update(1.0, active=True, consecutive=True)
        accepted = guard.update(1.0, active=True, consecutive=True)
        changed = guard.update(3.0, active=True, consecutive=True)
        self.assertEqual(cold.reason, "cold")
        self.assertEqual(probation.reason, "probation")
        self.assertEqual(accepted.reason, "accepted")
        self.assertEqual(changed.reason, "change")
        self.assertTrue(changed.reset)
        self.assertEqual(len(guard.innovations), 0)

    def test_segment_safe_tb9_boundary_is_guarded(self):
        innovation = np.ones((1, 12, 1), np.float64)
        innovation[:, 0] = 0.0
        innovation[:, 8] = 3.0
        active = np.ones_like(innovation, bool)
        hard_reset = np.zeros_like(active)
        reset, score, eligible, valid = segment_safe_reset_detector(
            innovation, active, hard_reset, threshold=2.0
        )
        self.assertTrue(reset[0, 1, 0])
        self.assertFalse(eligible[0, 1, 0])
        self.assertTrue(eligible[0, 8, 0])
        self.assertEqual(float(score[0, 8, 0]), 3.0)
        self.assertTrue(reset[0, 8, 0])
        self.assertTrue(reset[0, 9, 0])
        self.assertFalse(valid[0, 8, 0])
        self.assertFalse(valid[0, 9, 0])
        self.assertTrue(valid[0, 10, 0])

    def test_segment_safe_online_and_offline_match(self):
        requested = np.asarray(
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0]
        )
        pilots = np.zeros((1, len(requested), 1, 1, 8, 1), np.complex64)
        for tb in range(1, len(requested)):
            pilots[:, tb] = pilots[:, tb - 1] + np.sqrt(requested[tb])
        errors = np.full(pilots.shape, 0.5, np.float32)
        active = np.ones((1, len(requested), 1), bool)
        hard_reset = np.zeros_like(active)
        innovation = active_dmrs_innovation(pilots, errors, active)
        reset, _, _, valid = segment_safe_reset_detector(
            innovation, active, hard_reset, threshold=2.0
        )
        offline, offline_error = segmented_wiener_estimate(
            pilots, errors, reset, max_history=8
        )
        memory = OnlineSegmentSafePilotDomainMemory.__new__(
            OnlineSegmentSafePilotDomainMemory
        )
        memory.maximum_history_tbs = 8
        memory.threshold = 2.0
        memory.start_episode()
        online = []
        online_error = []
        online_reset = []
        online_valid = []
        for tb in range(len(requested)):
            memory._tb_index = tb
            history, variance, did_reset, is_valid = memory._update_track(
                0, pilots[0, tb, 0], errors[0, tb, 0], True
            )
            estimate, posterior, _ = pilot_domain_wiener_estimate(
                np.asarray(history)[None, :, None],
                np.asarray(variance)[None, :, None],
                max_history=min(8, len(history)),
            )
            online.append(estimate[0, -1, 0])
            online_error.append(posterior[0, -1, 0])
            online_reset.append(did_reset)
            online_valid.append(is_valid)
        np.testing.assert_array_equal(online_reset, reset[0, :, 0])
        np.testing.assert_array_equal(online_valid, valid[0, :, 0])
        np.testing.assert_allclose(online, offline[0, :, 0], atol=1e-6)
        np.testing.assert_allclose(
            online_error, offline_error[0, :, 0], atol=1e-6
        )

    def test_sparse_bank_keeps_opaque_identities_separate(self):
        bank = SegmentSafePilotBank(threshold=2.0, maximum_history_tbs=8)
        error = np.ones((1, 2, 1), np.float32)
        schedule = ((7, 2), (2, 7), (7, 9), (7, 2))
        points = []
        for tb, identities in enumerate(schedule):
            row = []
            for identity in identities:
                pilot = np.full(
                    error.shape, identity + 0.1 * tb, np.complex64
                )
                row.append(bank.update(
                    tb, identity, pilot, error, active=True
                ))
            points.append(row)
        self.assertEqual(points[1][0].decision.reason, "probation")
        self.assertEqual(points[1][1].decision.reason, "probation")
        self.assertEqual(len(points[1][0].pilots), 1)
        self.assertEqual(len(points[1][1].pilots), 1)
        self.assertTrue(points[2][0].decision.history_valid)
        self.assertEqual(points[2][1].decision.reason, "cold")
        self.assertEqual(points[3][1].decision.reason, "external")
        self.assertEqual(len(points[3][1].pilots), 1)
        self.assertTrue(all(
            np.allclose(np.real(value), 2.3)
            for value in points[3][1].pilots
        ))


if __name__ == "__main__":
    unittest.main()
