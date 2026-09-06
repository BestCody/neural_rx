#!/usr/bin/env python3
"""Tests for pilot-domain temporal channel estimation."""

import unittest

import numpy as np

from pilot_domain_channel import (
    cdm_average_raw_pilots,
    estimate_cdm_noise_variance,
    pilot_domain_wiener_estimate,
    project_toeplitz_psd,
)
from pilot_domain_channel_v19 import (
    pilot_domain_wiener_latest,
    project_toeplitz_psd_batch,
)


def complex_normal(rng, shape, variance=1.0):
    scale = np.sqrt(float(variance) / 2.0)
    return scale * (
        rng.normal(size=shape) + 1j * rng.normal(size=shape)
    )


def correlated_channels(
    seed=1,
    episodes=6,
    num_tbs=20,
    num_users=2,
    num_phases=2,
    num_pilots=48,
    num_rx=2,
    rho=0.99,
    noise_variance=0.15,
):
    rng = np.random.default_rng(seed)
    shape = (
        episodes,
        num_tbs,
        num_users,
        num_phases,
        num_pilots,
        num_rx,
    )
    truth = np.empty(shape, np.complex128)
    truth[:, 0] = complex_normal(rng, shape[:1] + shape[2:])
    innovation = 1.0 - rho**2
    for tb in range(1, num_tbs):
        truth[:, tb] = (
            rho * truth[:, tb - 1]
            + complex_normal(rng, shape[:1] + shape[2:], innovation)
        )
    noise = complex_normal(rng, shape, noise_variance)
    observation = truth + noise
    variance = np.full(shape, noise_variance, np.float64)
    folds = np.broadcast_to(
        (np.arange(num_pilots) // 4) % 2,
        (num_users, num_phases, num_pilots),
    )
    return truth, observation, variance, folds


def mse(value, truth, start_tb=7):
    return float(np.mean(np.abs(value[:, start_tb:] - truth[:, start_tb:]) ** 2))


class PilotDomainChannelTests(unittest.TestCase):
    def test_cdm_average_repeats_each_pair_mean(self):
        raw = np.asarray([[[[[1.0], [3.0], [5.0], [9.0]]]]])
        actual = cdm_average_raw_pilots(raw)
        expected = np.asarray([[[[[2.0], [2.0], [7.0], [7.0]]]]])
        np.testing.assert_array_equal(actual, expected)

    def test_cdm_pair_residual_recovers_post_average_noise(self):
        rng = np.random.default_rng(9)
        raw_variance = 0.4
        shape = (64, 2, 2, 200, 3)
        channel = complex_normal(rng, shape[:3] + (100, 1, shape[-1]))
        channel = np.repeat(channel, 2, axis=4).reshape(shape)
        raw = channel + complex_normal(rng, shape, raw_variance)
        estimate, diagnostics = estimate_cdm_noise_variance(raw)
        expected = raw_variance / 2.0
        self.assertAlmostEqual(float(np.mean(estimate)), expected, delta=0.01)
        self.assertEqual(diagnostics["pair_count_per_phase"], 100)

    def test_cdm_noise_estimator_rejects_odd_pilot_count(self):
        with self.assertRaisesRegex(ValueError, "even valid-pilot"):
            estimate_cdm_noise_variance(
                np.ones([1, 1, 1, 3, 1], np.complex64)
            )

    def test_projection_is_toeplitz_and_psd(self):
        rng = np.random.default_rng(11)
        matrix = complex_normal(rng, [8, 8])
        matrix = 0.5 * (matrix + matrix.conj().T)
        projected, diagnostics = project_toeplitz_psd(matrix)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(projected)), -1e-10)
        self.assertLess(diagnostics["toeplitz_residual"], 1e-7)

    def test_batched_projection_matches_scalar_projection(self):
        rng = np.random.default_rng(23)
        matrices = complex_normal(rng, [5, 8, 8])
        matrices = 0.5 * (
            matrices + np.swapaxes(matrices.conj(), -1, -2)
        )
        batched, diagnostics = project_toeplitz_psd_batch(matrices)
        scalar = np.stack([
            project_toeplitz_psd(matrix)[0] for matrix in matrices
        ])
        np.testing.assert_allclose(batched, scalar, rtol=1e-10, atol=1e-10)
        self.assertTrue(np.all(diagnostics["iterations"] > 0))
        self.assertLess(
            float(np.max(diagnostics["toeplitz_residual"])), 1e-7
        )

    def test_newest_only_matches_every_final_prefix(self):
        _, observation, variance, _ = correlated_channels(
            seed=29,
            episodes=2,
            num_tbs=8,
            num_pilots=24,
        )
        for history in range(1, 9):
            old, old_variance, _ = pilot_domain_wiener_estimate(
                observation[:, :history],
                variance[:, :history],
                max_history=history,
            )
            new, new_variance, diagnostics = pilot_domain_wiener_latest(
                observation[:, :history],
                variance[:, :history],
                max_history=history,
            )
            np.testing.assert_allclose(
                new, old[:, -1], rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                new_variance,
                old_variance[:, -1],
                rtol=1e-6,
                atol=1e-7,
            )
            self.assertEqual(diagnostics["total_updates"], 8)

    def test_newest_only_applies_the_history_cap(self):
        _, observation, variance, _ = correlated_channels(
            seed=31,
            episodes=1,
            num_tbs=12,
            num_users=1,
            num_phases=1,
            num_pilots=24,
        )
        capped, capped_variance, _ = pilot_domain_wiener_latest(
            observation, variance, max_history=8
        )
        sliced, sliced_variance, _ = pilot_domain_wiener_latest(
            observation[:, -8:], variance[:, -8:], max_history=8
        )
        np.testing.assert_allclose(capped, sliced, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            capped_variance, sliced_variance, rtol=0.0, atol=0.0
        )

    def test_older_correlated_history_improves_estimate(self):
        truth, observation, variance, folds = correlated_channels()
        l2, _, _ = pilot_domain_wiener_estimate(
            observation, variance, max_history=2, fold_ids=folds
        )
        l8, _, _ = pilot_domain_wiener_estimate(
            observation, variance, max_history=8, fold_ids=folds
        )
        self.assertLess(mse(l8, truth), mse(l2, truth))

    def test_shuffled_user_history_removes_benefit(self):
        truth, observation, variance, folds = correlated_channels(seed=3)
        correct, _, _ = pilot_domain_wiener_estimate(
            observation, variance, max_history=8, fold_ids=folds
        )
        shuffled, _, _ = pilot_domain_wiener_estimate(
            observation,
            variance,
            max_history=8,
            fold_ids=folds,
            history_source=np.roll(observation, -1, axis=2),
            history_error_variance=np.roll(variance, -1, axis=2),
        )
        self.assertLess(mse(correct, truth), mse(shuffled, truth))

    def test_warmup_branches_are_identical(self):
        _, observation, variance, folds = correlated_channels(seed=4)
        l2, _, _ = pilot_domain_wiener_estimate(
            observation, variance, max_history=2, fold_ids=folds
        )
        l8, _, _ = pilot_domain_wiener_estimate(
            observation, variance, max_history=8, fold_ids=folds
        )
        shuffled, _, _ = pilot_domain_wiener_estimate(
            observation,
            variance,
            max_history=8,
            fold_ids=folds,
            history_source=np.roll(observation, -1, axis=2),
            history_error_variance=np.roll(variance, -1, axis=2),
        )
        np.testing.assert_allclose(l2[:, :2], l8[:, :2], atol=1e-10)
        np.testing.assert_allclose(l2[:, :1], shuffled[:, :1], atol=1e-10)

    def test_independent_history_receives_small_weights(self):
        rng = np.random.default_rng(5)
        shape = (4, 16, 1, 1, 64, 2)
        truth = complex_normal(rng, shape)
        variance = np.full(shape, 0.1)
        observation = truth + complex_normal(rng, shape, 0.1)
        _, _, diagnostics = pilot_domain_wiener_estimate(
            observation, variance, max_history=8
        )
        weights = diagnostics["mean_mature_weights"]
        history_weight = max(
            abs(complex(value["real"], value["imag"]))
            for value in weights[1:]
        )
        self.assertLess(history_weight, 0.12)

    def test_long_sequence_stays_finite(self):
        truth, observation, variance, _ = correlated_channels(
            seed=6,
            episodes=1,
            num_tbs=1024,
            num_users=1,
            num_phases=1,
            num_pilots=16,
            num_rx=1,
        )
        estimate, posterior, diagnostics = pilot_domain_wiener_estimate(
            observation, variance, max_history=8
        )
        self.assertTrue(np.all(np.isfinite(estimate)))
        self.assertTrue(np.all(np.isfinite(posterior)))
        self.assertTrue(np.all(posterior >= 0.0))
        self.assertGreater(
            diagnostics["min_observation_covariance_eigenvalue"], 0.0
        )
        self.assertLess(mse(estimate, truth), mse(observation, truth))

    def test_weight_history_is_opt_in_and_bounded(self):
        _, observation, variance, _ = correlated_channels(
            seed=17,
            episodes=2,
            num_tbs=12,
            num_users=1,
            num_phases=1,
            num_pilots=24,
            num_rx=1,
        )
        _, _, diagnostics = pilot_domain_wiener_estimate(
            observation,
            variance,
            max_history=8,
            capture_weight_history=True,
        )
        weights = diagnostics["weight_history"]
        self.assertEqual(weights.shape, (2, 12, 1, 1, 8))
        self.assertTrue(np.all(np.isfinite(weights)))
        np.testing.assert_array_equal(weights[:, 0, ..., 1:], 0.0)

    def test_invalid_fold_geometry_is_rejected(self):
        _, observation, variance, _ = correlated_channels(seed=8)
        invalid = np.zeros(observation.shape[2:5], dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "folds zero and one"):
            pilot_domain_wiener_estimate(
                observation, variance, max_history=8, fold_ids=invalid
            )


if __name__ == "__main__":
    unittest.main()
