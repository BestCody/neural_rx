#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cjlr_optimized_receiver import ExactCartesianKernel
from cjlr_receiver import candidate_log_likelihood, exact_joint_messages


class ExactCartesianKernelTest(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(612)
        self.points = np.asarray([-1.0 + 0.0j, 1.0 + 0.0j], np.complex64)
        self.labels = np.asarray([[0.0], [1.0]], np.float32)

    def inputs(self, streams, batch=7, antennas=3):
        received = tf.complex(
            tf.random.normal([batch, antennas]),
            tf.random.normal([batch, antennas]),
        )
        channel = tf.complex(
            tf.random.normal([batch, streams, antennas]),
            tf.random.normal([batch, streams, antennas]),
        )
        variance = tf.random.uniform(
            [batch, streams, antennas], 0.01, 0.09
        )
        noise = tf.random.uniform([batch, antennas], 0.04, 0.2)
        extrinsic = tf.random.normal([batch, streams, 1])
        return received, channel, variance, noise, extrinsic

    def compare_reference(self, streams, layout):
        kernel = ExactCartesianKernel(
            self.points, self.labels, streams, layout
        )
        values = self.inputs(streams)
        kernel.validate_inputs(*values)
        result = kernel(*values)
        score = candidate_log_likelihood(
            *values[:4], kernel.candidates
        )
        reference = exact_joint_messages(
            score, values[4], kernel.candidate_bits
        )
        np.testing.assert_allclose(
            result.score.numpy(), score.numpy(), rtol=3e-6, atol=3e-6
        )
        np.testing.assert_allclose(
            result.delta.numpy(),
            reference.delta.numpy(),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_flat_matches_reference_for_one_to_four_streams(self):
        for streams in range(1, 5):
            with self.subTest(streams=streams):
                self.compare_reference(streams, "flat")

    def test_cartesian_matches_reference_for_one_to_four_streams(self):
        for streams in range(1, 5):
            with self.subTest(streams=streams):
                self.compare_reference(streams, "cartesian")

    def test_zero_extrinsic_has_zero_delta(self):
        for layout in ("flat", "cartesian"):
            kernel = ExactCartesianKernel(
                self.points, self.labels, 3, layout
            )
            values = list(self.inputs(3))
            values[-1] = tf.zeros_like(values[-1])
            delta = kernel(*values).delta.numpy()
            np.testing.assert_array_equal(delta, np.zeros_like(delta))

    def test_target_prior_is_excluded(self):
        kernel = ExactCartesianKernel(
            self.points, self.labels, 2, "cartesian"
        )
        values = list(self.inputs(2))
        first = kernel(*values).delta
        offset = tf.concat([
            tf.ones_like(values[-1][:, :1]) * 7.0,
            tf.zeros_like(values[-1][:, 1:]),
        ], axis=1)
        values[-1] += offset
        second = kernel(*values).delta
        np.testing.assert_allclose(
            first[:, 0].numpy(), second[:, 0].numpy(), rtol=2e-5, atol=2e-5
        )

    def test_stream_permutation_is_equivariant(self):
        kernel = ExactCartesianKernel(
            self.points, self.labels, 3, "cartesian"
        )
        values = self.inputs(3)
        direct = kernel(*values).delta
        order = tf.constant([2, 0, 1], tf.int32)
        permuted = kernel(
            values[0],
            tf.gather(values[1], order, axis=1),
            tf.gather(values[2], order, axis=1),
            values[3],
            tf.gather(values[4], order, axis=1),
        ).delta
        inverse = tf.argsort(order)
        np.testing.assert_allclose(
            direct.numpy(),
            tf.gather(permuted, inverse, axis=1).numpy(),
            rtol=3e-5,
            atol=3e-5,
        )


if __name__ == "__main__":
    unittest.main()
