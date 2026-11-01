#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from temporal_receiver_core import expand_noise
from verification.radio import exact_contributions, received_to_grid


class TemporalReceiverCoreTest(unittest.TestCase):

    def test_tensor_layout_helpers(self):
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
