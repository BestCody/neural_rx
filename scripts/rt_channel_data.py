from __future__ import annotations

import numpy as np
import tensorflow as tf

from temporal_training_data import TemporalTrainingDataGenerator


class RayTracedGenerator(TemporalTrainingDataGenerator):
    def __init__(self, parameters, model, bank, **kwargs):
        self.bank = bank
        self.episode_index = None
        super().__init__(parameters, model, ue_pool_size=len(bank.config["users"]), **kwargs)
        if not np.isclose(self.p.carrier_frequency, bank.config["carrier_frequency_hz"], rtol=1e-9):
            raise ValueError("Ray-traced carrier frequency differs from the radio configuration")

    def _make_channel(self, profile):
        return None

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        if batch_size != 1 or self.episode_index is None or channel_regime is not None:
            raise ValueError("Select one exported RT episode; synthetic channel replacements are unsupported")
        data = self.bank.load(self.episode_index)
        a = data["a"]
        if a.shape[:4] != (1, self.p.num_rx_antennas, self.ue_pool_size, self.p.num_antenna_ports):
            raise ValueError("Ray-traced antenna or UE dimensions differ from the receiver configuration")
        symbols = int(self.rg.num_ofdm_symbols)
        count = int(seq_len) * symbols
        duration = float(self.rg.ofdm_symbol_duration)
        if count * duration > self.bank.config["max_duration_s"]:
            raise ValueError("Episode exceeds the exported local-motion validity interval")
        interval_count = int(np.floor(self.bank.config["max_duration_s"] / duration))
        time = tf.range(interval_count, dtype=tf.float32) * duration
        phase = 2 * np.pi * tf.convert_to_tensor(data["doppler_hz"])[..., None] * time
        evolution = tf.exp(tf.complex(tf.zeros_like(phase), phase))
        coefficients = tf.convert_to_tensor(a)[..., None] * evolution
        delays = tf.convert_to_tensor(data["tau"])
        delays = tf.where(delays < 0, 0., delays)
        frequency_phase = -2 * np.pi * delays[..., None] * self.freqs
        response = tf.exp(tf.complex(tf.zeros_like(frequency_phase), frequency_phase))
        h = tf.matmul(coefficients, response, transpose_a=True)
        if self.bank.config["power_normalization"] == "per_ue_episode":
            power = tf.reduce_mean(tf.abs(h)**2, axis=(0, 1, 3, 4, 5), keepdims=True)
            tf.debugging.assert_positive(power, "A ray-traced UE has zero received power")
            h = h / tf.cast(tf.sqrt(power), h.dtype)
        tf.debugging.assert_all_finite(tf.math.real(h), "Invalid ray-traced channel")
        tf.debugging.assert_all_finite(tf.math.imag(h), "Invalid ray-traced channel")
        return tf.stack(tf.split(h[None, ..., :count, :], int(seq_len), axis=5), axis=1)
