"""Test sequences that relax two assumptions of the 16-QAM evaluation, both with closed-loop power control
(experiments/20261112_early_retransmission/power_control.py), on the seeds of the power-control test sequences.

TDD uplink slots (TDDBank). The evaluation's sequences use back-to-back uplink slots. In TDD, a UE's uplink slots
are spread over the frame: in DDDSU (period 5 slots, 2.5 ms at 30 kHz) one slot in five is an uplink slot, in
DDDDDDDSUU (period 10 slots, 5 ms) two adjacent slots in ten. The channel of every pool UE is sampled over the whole
TDD timeline, as in the evaluation one OFDM symbol after the other, and only the OFDM symbols of the uplink slots are
kept, so the channel ages by the true time between two uplink slots. The slot numbers that select the DMRS are the true
slot numbers. TB t of a sequence is the t-th uplink slot.

Carrier frequency offset (FrequencyOffsetBank). Every pool UE transmits with a constant carrier frequency offset of
+-0.1 ppm of the carrier, the largest frequency error that 3GPP TS 38.101-1 (6.4.1) allows a UE relative to the carrier
it receives from the BS, with a random sign per UE. At 2.14 GHz this is +-214 Hz, a phase rotation of 0.67 rad per
slot. The rotation advances with every OFDM symbol, within and across slots; the inter-carrier interference of an
offset of 0.7% of the subcarrier spacing is neglected. The signs come from their own generator, seeded per sequence.

Channels within a slot, payloads, noise, schedules and power control equal those of the power-control sequences except
for these changes, so all receivers decode the same received signals.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"experiments/20261112_early_retransmission"))
from power_control import SETTINGS as SETTINGS_PC, ClosedLoopPowerControlGenerator, PowerControlBank  # noqa: E402,F401
from sionna.channel import cir_to_ofdm_channel  # noqa: E402

PATTERNS = {"dddsu": (5, (4,)), "dddddddsuu": (10, (8, 9))}     # period in slots, uplink slots within the period
CFO_PPM = 0.1
CFO_SEED = 78000000


def uplink_slots(pattern, count):
    """True slot index of each of the first `count` uplink slots, counted from the first uplink slot."""
    period, uplink = PATTERNS[pattern]
    slots = np.asarray([(k//len(uplink))*period+uplink[k % len(uplink)] for k in range(int(count))], np.int64)
    return slots-slots[0]


class TDDGenerator(ClosedLoopPowerControlGenerator):
    def _slot_numbers(self, seq_len):
        numbers = super()._slot_numbers(seq_len)              # same random draws as the power-control sequences
        if not self.advance_slot_pilots:
            return numbers
        num_slots = int(self.p.pilots.shape[0])
        return ((int(numbers[0])+uplink_slots(self.tdd_pattern, seq_len)) % num_slots).astype(np.int32)

    def _sample_joint_slots(self, channel, batch_size, seq_len):
        symbols = int(self.rg.num_ofdm_symbols)
        sampling_frequency = 1.0/self.rg.ofdm_symbol_duration
        slots = uplink_slots(self.tdd_pattern, seq_len)
        a, tau = channel(int(batch_size), int(slots[-1]+1)*symbols, sampling_frequency)
        keep = (slots[:, None]*symbols+np.arange(symbols)[None, :]).reshape(-1)
        a = tf.gather(a, keep, axis=-1)                       # [batch, rx, rx antenna, UE, UE antenna, path, time]
        h = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
        if int(h.shape[3]) != self.ue_pool_size:
            raise RuntimeError("joint channel UE dimension differs")
        return tf.stack(tf.split(h, int(seq_len), axis=5), axis=1)


class TDDBank(PowerControlBank):
    def __init__(self, parameters, e2e_model, tf_module, settings, pattern):
        super().__init__(parameters, e2e_model, tf_module, settings)
        if pattern not in PATTERNS:
            raise ValueError(f"unknown TDD pattern {pattern}")
        self.tdd_pattern = pattern

    def _make_generator(self, job, episode_index):
        generator = super()._make_generator(job, episode_index)
        generator.__class__ = TDDGenerator                       # adds methods only
        generator.tdd_pattern = self.tdd_pattern
        return generator


class FrequencyOffsetGenerator(ClosedLoopPowerControlGenerator):
    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        h = super().sample_pool_channel(batch_size, seq_len, channel_regime)
        batch, slots, pool, symbols = int(batch_size), int(seq_len), int(h.shape[4]), int(h.shape[6])
        sign = np.random.default_rng(self.cfo_seed).choice([-1., 1.], size=(batch, pool))
        offset = sign*CFO_PPM*1e-6*float(self.p.carrier_frequency)                        # Hz, [batch, pool UE]
        time = (np.arange(slots)[:, None]*symbols+np.arange(symbols)[None, :])*float(self.rg.ofdm_symbol_duration)
        phase = 2*np.pi*offset[:, None, :, None]*time[None, :, None, :]                    # [batch, slot, UE, symbol]
        self.last_cfo_hz = offset
        rotation = tf.constant(np.exp(1j*phase).astype(np.complex64))
        # h is [batch, slot, rx, rx antenna, UE, UE antenna, symbol, subcarrier]; Mul and BroadcastTo cannot broadcast
        # this many alternating dimensions on the CPU, einsum can
        return tf.einsum("bsxauzwf,bsuw->bsxauzwf", h, rotation)


class FrequencyOffsetBank(PowerControlBank):
    def _make_generator(self, job, episode_index):
        generator = super()._make_generator(job, episode_index)
        generator.__class__ = FrequencyOffsetGenerator           # adds methods only
        return generator

    def _generator(self, job, episode_index):
        # The bank reuses one generator per condition, so the seed of the signs is set for every sequence.
        generator = super()._generator(job, episode_index)
        generator.cfo_seed = CFO_SEED+int(episode_index)
        return generator
