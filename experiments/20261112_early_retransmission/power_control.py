"""Closed-loop uplink power control for the TBLER sweep's episodes.

After each slot in which a UE is scheduled, the base station measures the UE's received power, averaged over the
band and the BS antennas after the UE's fixed precoder; we assume this wideband measurement is exact. It then sends
an accumulated transmit power control (TPC) command of -1, 0, +1, or +3 dB, as in 3GPP TS 38.213, toward a target
received power of 0 dB. The command applies from the UE's next scheduled slot, and the UE's power stays within a
fixed range around its nominal power. Each UE starts at its nominal power.

The power gains scale each pool UE's channel slot by slot. They use no random numbers, so channels, payloads, noise,
and schedules equal those of experiments/20261104_tbler_sweep; with a zero power range the episodes are identical.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from neural_rx.benchmark.tbler_sweep import ChannelCapturingBank
from neural_rx.data.geometry import GeometryAwareMUMIMOGenerator

SETTINGS = {"target_db": 0., "minimum_db": -10., "maximum_db": 15., "initial_db": 0.}


def tpc_step(error_db):
    """Accumulated TPC command for the received power's distance below the target."""
    return 3. if error_db >= 3. else 1. if error_db >= 1. else -1. if error_db <= -1. else 0.


def closed_loop(channel_power, scheduled, target_db, minimum_db, maximum_db, initial_db):
    """Power gain in dB per [batch, slot, pool UE] from the effective channel power and the schedule."""
    gain = np.zeros(channel_power.shape)
    for b in range(channel_power.shape[0]):
        for ue in range(channel_power.shape[2]):
            state = initial_db
            for t in range(channel_power.shape[1]):
                gain[b, t, ue] = state
                if scheduled[b, t, ue]:
                    measured = 10*np.log10(channel_power[b, t, ue])+state
                    state = min(max(state+tpc_step(target_db-measured), minimum_db), maximum_db)
    return gain


class ClosedLoopPowerControlGenerator(GeometryAwareMUMIMOGenerator):
    """The benchmark's geometry generator with closed-loop power control applied to the pool channels."""

    def configure_power_control(self, precoder, **settings):
        self.power_settings = dict(SETTINGS, **settings)
        self.precoder = tf.reshape(tf.cast(precoder, tf.complex64), [1, 1, 1, 1, 1, -1, 1, 1])

    def sample_schedule(self, batch_size, seq_len):
        schedule = super().sample_schedule(batch_size, seq_len)
        self._power_schedule = np.asarray(schedule)
        return schedule

    def sample_scenario(self, *args, **kwargs):
        scenario = super().sample_scenario(*args, **kwargs)
        self._power_active = np.asarray(scenario["active"]) > 0
        return scenario

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        h = super().sample_pool_channel(batch_size, seq_len, channel_regime)
        effective = tf.reduce_sum(h*self.precoder, axis=5)
        power = tf.reduce_mean(tf.abs(effective)**2, axis=(2, 3, 5, 6)).numpy()
        pool = power.shape[2]
        scheduled = np.zeros(power.shape, bool)
        for ue in range(pool):
            scheduled[..., ue] = np.any((self._power_schedule == ue) & self._power_active, -1)
        gain = closed_loop(power, scheduled, **self.power_settings)
        self.last_power_gain_db = gain
        scale = tf.cast(tf.sqrt(10.**(gain/10.)), h.dtype)
        return h*tf.reshape(scale, [int(batch_size), int(seq_len), 1, 1, pool, 1, 1, 1])


class PowerControlBank(ChannelCapturingBank):
    """ChannelCapturingBank whose geometry generators apply closed-loop power control."""

    def __init__(self, parameters, e2e_model, tf_module, settings):
        super().__init__(parameters, e2e_model, tf_module)
        self.power_settings = dict(settings)
        self.precoder = parameters.transmitters[0]._precoder._w[0, :, 0]

    def _make_generator(self, job, episode_index):
        generator = super()._make_generator(job, episode_index)
        if type(generator) is not GeometryAwareMUMIMOGenerator:
            raise ValueError("power control is defined for the geometry-based conditions")
        generator.__class__ = ClosedLoopPowerControlGenerator  # adds methods only; the state is unchanged
        generator.configure_power_control(self.precoder, **self.power_settings)
        return generator

    def _generator(self, job, episode_index):
        self.last_generator = super()._generator(job, episode_index)
        return self.last_generator

    def generate(self, job, episode_index):
        episode = super().generate(job, episode_index)
        generator = self.last_generator
        ids = generator._power_schedule[0]
        self.last_power_gain_db = np.take_along_axis(generator.last_power_gain_db[0], ids, axis=1)
        return episode
