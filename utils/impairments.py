# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from tensorflow.keras.layers import Layer
import tensorflow as tf
from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.constants import PI

class FrequencyOffset(Layer):
    """Frequency Offset."""

    def __init__(self, max_rel_offset, input_domain, resource_grid=None,
                 constant_offset=False, **kwargs):
        super().__init__(**kwargs)
        self._max_rel_offset = tf.cast(max_rel_offset, tf.float32)

        if constant_offset:
            self._min_rel_offset = self._max_rel_offset
        else:
            self._min_rel_offset = -self._max_rel_offset

        self._input_domain = input_domain
        self._resource_grid = resource_grid

        if self._input_domain == "freq":
            assert self._resource_grid is not None, \
                "resource_grid must be provided when input_domain is 'freq'."
            self._modulator = OFDMModulator(resource_grid.cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(
                                    resource_grid.fft_size, 0,
                                    resource_grid.cyclic_prefix_length)

    def call(self, inputs):
        if self._input_domain == "freq":
            inputs = self._modulator(inputs)

        num_time_samples = tf.shape(inputs)[-1]

        s = tf.concat((tf.shape(inputs)[0:2], tf.ones((2,), tf.int32)), axis=0)

        fo = tf.random.uniform(s,
                               minval=self._min_rel_offset,
                               maxval=self._max_rel_offset,
                               dtype=tf.float32)

        phase_increment = fo * 2 * PI
        time_steps = tf.reshape(
                            tf.range(0, num_time_samples, dtype=tf.float32),
                            [1, 1, 1, -1])
        phase_shifts = time_steps * phase_increment

        exp = tf.cast(tf.exp(tf.complex(0., phase_shifts)), inputs.dtype)
        outputs = exp * inputs

        if self._input_domain == "freq":
            outputs = self._demodulator(outputs)

        return outputs
