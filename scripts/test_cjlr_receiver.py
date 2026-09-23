#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cjlr_receiver import (
    candidate_log_likelihood,
    exact_joint_messages,
    exact_joint_support,
    two_stream_mutual_information,
)


class CJLRReceiverTest(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(81)
        self.points = tf.constant(
            [-1.0 + 0.0j, 1.0 + 0.0j], tf.complex64
        )
        self.labels = tf.constant([[0.0], [1.0]], tf.float32)
        self.candidates, self.bits, self.indices = exact_joint_support(
            self.points, self.labels, 2, 4
        )

    def test_support_is_complete_and_bit_complete(self):
        self.assertEqual(tuple(self.candidates.shape), (4, 2))
        for user in range(2):
            self.assertEqual(
                set(self.bits[:, user, 0].numpy().tolist()), {0.0, 1.0}
            )

    def test_vector_score_matches_scalar_reference(self):
        received = tf.constant([[0.25 + 0.4j, -0.2 + 0.1j]])
        channel = tf.constant([[[1.0 + 0.2j, 0.4 - 0.1j],
                                [0.3 + 0.1j, 0.8 + 0.2j]]])
        variance = tf.constant([[[0.03, 0.04], [0.02, 0.05]]])
        noise = tf.constant([[0.1, 0.12]])
        actual = candidate_log_likelihood(
            received, channel, variance, noise, self.candidates
        ).numpy()[0]
        expected = []
        for candidate in self.candidates.numpy():
            prediction = np.sum(channel.numpy()[0] * candidate[:, None], axis=0)
            total_variance = noise.numpy()[0] + np.sum(
                variance.numpy()[0] * np.abs(candidate[:, None]) ** 2,
                axis=0,
            )
            expected.append(-np.sum(
                np.abs(received.numpy()[0] - prediction) ** 2
                / total_variance + np.log(total_variance)
            ))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_zero_extrinsic_has_exact_zero_delta(self):
        score = tf.random.normal([7, 4])
        result = exact_joint_messages(
            score, tf.zeros([7, 2, 1]), self.bits
        )
        np.testing.assert_array_equal(
            result.delta.numpy(), np.zeros([7, 2, 1], np.float32)
        )

    def test_target_prior_is_excluded(self):
        score = tf.random.normal([3, 4])
        first = tf.random.normal([3, 2, 1])
        second = tf.tensor_scatter_nd_add(
            first,
            tf.constant([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            tf.constant([5.0, -4.0, 2.0]),
        )
        first_output = exact_joint_messages(score, first, self.bits)
        second_output = exact_joint_messages(score, second, self.bits)
        np.testing.assert_allclose(
            first_output.code_message[:, 0].numpy(),
            second_output.code_message[:, 0].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_stream_permutation_is_equivariant(self):
        received = tf.complex(
            tf.random.normal([5, 2]), tf.random.normal([5, 2])
        )
        channel = tf.complex(
            tf.random.normal([5, 2, 2]), tf.random.normal([5, 2, 2])
        )
        variance = tf.random.uniform([5, 2, 2], 0.01, 0.1)
        noise = tf.random.uniform([5, 2], 0.05, 0.2)
        extrinsic = tf.random.normal([5, 2, 1])
        score = candidate_log_likelihood(
            received, channel, variance, noise, self.candidates
        )
        direct = exact_joint_messages(score, extrinsic, self.bits)
        permuted_candidates = tf.reverse(self.candidates, axis=[1])
        permuted_bits = tf.reverse(self.bits, axis=[1])
        permuted_score = candidate_log_likelihood(
            received,
            tf.reverse(channel, axis=[1]),
            tf.reverse(variance, axis=[1]),
            noise,
            permuted_candidates,
        )
        permuted = exact_joint_messages(
            permuted_score, tf.reverse(extrinsic, axis=[1]), permuted_bits
        )
        np.testing.assert_allclose(
            direct.delta.numpy(),
            tf.reverse(permuted.delta, axis=[1]).numpy(),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_mutual_information_detects_dependence(self):
        independent = tf.constant([[0.25, 0.25, 0.25, 0.25]])
        dependent = tf.constant([[0.5, 0.0, 0.0, 0.5]])
        self.assertAlmostEqual(float(two_stream_mutual_information(
            independent, self.indices, 2
        )[0]), 0.0, places=6)
        self.assertAlmostEqual(float(two_stream_mutual_information(
            dependent, self.indices, 2
        )[0]), np.log(2.0), places=6)


if __name__ == "__main__":
    unittest.main()
