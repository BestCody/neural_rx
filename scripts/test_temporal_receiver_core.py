#!/usr/bin/env python3
"""Test shared temporal receiver runtime helpers."""

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from temporal_receiver_core import (
    exact_contributions,
    expand_noise,
    model_checksum,
    model_value_checksum,
    model_weight_signature,
    received_to_grid,
    select_channel_input,
    select_channel_variance,
    stacked_to_complex_channel,
)


class TemporalReceiverCoreTest(unittest.TestCase):
    def test_portable_weight_checksum_ignores_runtime_names(self):
        class Model:
            def __init__(self, prefix):
                self.weights = [
                    tf.Variable([1.0, 2.0], name=f"{prefix}_first"),
                    tf.Variable([[3]], name=f"{prefix}_second"),
                ]

        first = Model("student")
        second = Model("teacher_17")
        self.assertNotEqual(model_checksum(first), model_checksum(second))
        self.assertEqual(
            model_weight_signature(first), model_weight_signature(second)
        )
        self.assertEqual(
            model_value_checksum(first), model_value_checksum(second)
        )
        second.weights[0].assign([1.0, 4.0])
        self.assertNotEqual(
            model_value_checksum(first), model_value_checksum(second)
        )

    def test_history_selection_and_variance_match(self):
        current = tf.zeros((1, 2, 3, 4, 4), tf.float32)
        posterior = tf.ones_like(current)
        available = tf.constant([[True, False]])
        selected, valid = select_channel_input(
            current, posterior, available
        )
        np.testing.assert_array_equal(valid.numpy(), available.numpy())
        self.assertTrue(np.all(selected.numpy()[:, 0] == 1.0))
        self.assertTrue(np.all(selected.numpy()[:, 1] == 0.0))
        observation = tf.ones((1, 2, 3, 4, 2), tf.float32)
        history = 0.25 * observation
        variance = select_channel_variance(
            history, observation, valid
        ).numpy()
        self.assertTrue(np.all(variance[:, 0] == 0.25))
        self.assertTrue(np.all(variance[:, 1] == 1.0))

    def test_tensor_layout_helpers(self):
        stacked = tf.constant([[[[[1.0, 2.0, 3.0, 4.0]]]]])
        complex_value = stacked_to_complex_channel(stacked).numpy()
        np.testing.assert_array_equal(
            complex_value, np.asarray([[[[[1 + 3j, 2 + 4j]]]]])
        )
        received = tf.reshape(tf.range(24, dtype=tf.float32), (1, 1, 2, 3, 4))
        self.assertEqual(tuple(received_to_grid(received).shape), (1, 4, 3, 2))
        noise = tf.ones((2, 3), tf.float32)
        target = tf.zeros((2, 3, 4, 5), tf.float32)
        self.assertEqual(tuple(expand_noise(noise, target).shape), (2, 3, 1, 1))

    def test_exact_contributions_sum_to_received_shape(self):
        channel = tf.ones((1, 1, 2, 2, 3, 4, 5), tf.complex64)
        transmitted = tf.ones((1, 2, 3, 4, 5), tf.complex64)
        contribution = exact_contributions(channel, transmitted)
        self.assertEqual(tuple(contribution.shape), (1, 2, 5, 4, 2))
        np.testing.assert_array_equal(contribution.numpy(), 3.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
