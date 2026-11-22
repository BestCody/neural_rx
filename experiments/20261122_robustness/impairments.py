"""Impairments outside the NRX's training, for the fallback study (option B): does LS + LMMSE, a receiver a real-time
BS runs, beat the proposed NRX anywhere? All with closed-loop power control on the seeds of the power-control
sequences, as in banks.py.

  doppler      every UE moves with a Doppler frequency in [0.9 f, f]; training covered 0 to 400 Hz
  cfo          a carrier frequency offset of +-f ppm per UE (banks.FrequencyOffsetBank; 3GPP allows 0.1 ppm)
  phase_noise  each UE's oscillator adds a Wiener phase noise whose increment per OFDM symbol has standard deviation
               f rad, continuous across slots; the signs, the phase noise, and the speeds come from their own
               generators, seeded per sequence
"""
from __future__ import annotations

import numpy as np
import sionna as sn
import tensorflow as tf

from banks import FrequencyOffsetBank, FrequencyOffsetGenerator, ClosedLoopPowerControlGenerator, PowerControlBank
from neural_rx.data import geometry

PHASE_NOISE_SEED = 79000000
KINDS = ("none", "doppler", "cfo", "phase_noise")


class DopplerGenerator(ClosedLoopPowerControlGenerator):
    def _topology_candidates(self, count, profile):
        speed_scale = sn.SPEED_OF_LIGHT/self.p.carrier_frequency
        return geometry.gen_single_sector_topology(int(count), self.ue_pool_size, self.geometry_scenario,
                                                   min_ut_velocity=.9*self.doppler_hz*speed_scale,
                                                   max_ut_velocity=self.doppler_hz*speed_scale, indoor_probability=0.0)


class PhaseNoiseGenerator(ClosedLoopPowerControlGenerator):
    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        h = super().sample_pool_channel(batch_size, seq_len, channel_regime)
        batch, slots, pool, symbols = int(batch_size), int(seq_len), int(h.shape[4]), int(h.shape[6])
        steps = np.random.default_rng(self.phase_noise_seed).normal(0., self.phase_noise_rad, (batch, pool, slots*symbols))
        phase = np.cumsum(steps, -1).reshape(batch, pool, slots, symbols).transpose(0, 2, 1, 3)   # [batch, slot, UE, symbol]
        return tf.einsum("bsxauzwf,bsuw->bsxauzwf", h, tf.constant(np.exp(1j*phase).astype(np.complex64)))


class ImpairedBank(PowerControlBank):
    def __init__(self, parameters, e2e_model, tf_module, settings, kind, strength):
        super().__init__(parameters, e2e_model, tf_module, settings)
        if kind not in KINDS:
            raise ValueError(f"unknown impairment {kind}")
        self.kind, self.strength = kind, float(strength)

    def _make_generator(self, job, episode_index):
        generator = super()._make_generator(job, episode_index)
        if self.kind == "doppler":
            generator.__class__ = DopplerGenerator
            generator.doppler_hz = self.strength
        elif self.kind == "cfo":
            generator.__class__ = FrequencyOffsetGenerator
            generator.cfo_ppm = self.strength
        elif self.kind == "phase_noise":
            generator.__class__ = PhaseNoiseGenerator
            generator.phase_noise_rad = self.strength
        return generator

    def _generator(self, job, episode_index):
        generator = super()._generator(job, episode_index)
        if self.kind == "cfo":
            generator.cfo_seed = FrequencyOffsetBank.seed(episode_index)
        elif self.kind == "phase_noise":
            generator.phase_noise_seed = PHASE_NOISE_SEED+int(episode_index)
        return generator
