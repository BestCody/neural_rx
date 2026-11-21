"""Closed-loop power control in which every power step also breaks the UE's phase continuity.

Worst case for a receiver that carries channel estimates across slots: whenever a UE's transmit power changes from
one slot to the next (a nonzero accumulated TPC command took effect), its channel is rotated by a new random phase,
uniform on [0, 2 pi), which then persists until the next power step, as when a UE's power amplifier changes its
operating point and loses its phase reference. Slots without a power step keep the phase of the slot before.

The phases come from their own random generator, seeded per sequence (PHASE_SEED + sequence index), so channels,
payloads, noise, schedules, and power gains equal those of power_control.py.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from power_control import ClosedLoopPowerControlGenerator, PowerControlBank

PHASE_SEED = 77000000


class PhaseJumpGenerator(ClosedLoopPowerControlGenerator):
    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        h = super().sample_pool_channel(batch_size, seq_len, channel_regime)
        gain = self.last_power_gain_db                                      # [batch, slot, pool UE]
        steps = np.zeros(gain.shape, bool)
        steps[:, 1:] = gain[:, 1:] != gain[:, :-1]
        rng = np.random.default_rng(self.phase_seed)
        jumps = np.where(steps, rng.uniform(0., 2*np.pi, gain.shape), 0.)
        phase = np.cumsum(jumps, axis=1)
        self.last_phase_steps = steps
        rotation = tf.constant(np.exp(1j*phase).astype(np.complex64))
        return h*tf.reshape(rotation, [int(batch_size), int(seq_len), 1, 1, gain.shape[2], 1, 1, 1])


class PhaseJumpBank(PowerControlBank):
    def _make_generator(self, job, episode_index):
        generator = super()._make_generator(job, episode_index)
        generator.__class__ = PhaseJumpGenerator                           # adds methods only
        return generator

    def _generator(self, job, episode_index):
        # The bank reuses one generator per condition, so the phase seed is set for every sequence.
        generator = super()._generator(job, episode_index)
        generator.phase_seed = PHASE_SEED+int(episode_index)
        return generator

    def generate(self, job, episode_index):
        episode = super().generate(job, episode_index)
        ids = self.last_generator._power_schedule[0]
        self.last_phase_steps = np.take_along_axis(self.last_generator.last_phase_steps[0], ids, axis=1)
        return episode
