#!/usr/bin/env python3
"""Generate persistent-UE temporal training data."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="nrx_large.cfg")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--ebno-db", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ue-pool-size", type=int, default=4)
    p.add_argument("--dynamic-scheduling", action="store_true")
    p.add_argument("--scenario-training", action="store_true")
    p.add_argument(
        "--channel-role",
        choices=("training", "calibration", "known_comparison", "ood"),
        default="known_comparison",
    )
    p.add_argument(
        "--split-manifest",
        default="config/temporal_generalization_protocol.json",
    )
    p.add_argument("--schedule-switch-prob", type=float, default=0.65)
    p.add_argument("--schedule-reorder-prob", type=float, default=0.50)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(ARGS.gpu))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import sionna as sn
from sionna.channel import (
    ApplyOFDMChannel,
    cir_to_ofdm_channel,
    gen_single_sector_topology,
    subcarrier_frequencies,
)
from sionna.channel.tr38901 import PanelArray, TDL, UMi
from sionna.utils import ebnodb2no, expand_to_rank

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from utils import Parameters, E2E_Model
from utils.channel_models import ue_correlation_matrix, gnb_correlation_matrix
from temporal_generalization_protocol import (
    ChannelProfileSpec,
    load_split_manifest,
    role_profile_specs,
    seed_bundle,
    validate_split_slice,
)


NUM_TX = 2
NUM_RX_ANT = 4


def sample_burst_mask(
    batch_size,
    seq_len,
    probability,
    min_stable_tbs=4,
    max_duration=16,
    rng=None,
):
    """Sample one random impairment burst per sequence."""
    rng = np.random if rng is None else rng
    mask = np.zeros([int(batch_size), int(seq_len)], dtype=bool)
    if seq_len < 2 or probability <= 0.0:
        return mask

    first = min(max(int(min_stable_tbs), 1), int(seq_len) - 1)
    for batch_index in range(int(batch_size)):
        if rng.random() >= float(probability):
            continue
        start = int(rng.randint(first, int(seq_len)))
        available = int(seq_len) - start
        duration = int(rng.randint(1, min(int(max_duration), available) + 1))
        mask[batch_index, start:start + duration] = True
    return mask


def sample_activity_mask(
    batch_size,
    seq_len,
    num_scheduled_tx,
    gap_start_probability,
    max_gap_slots,
    rng=None,
):
    """Sample UE scheduling gaps while retaining one active UE."""
    rng = np.random if rng is None else rng
    shape = [int(batch_size), int(seq_len), int(num_scheduled_tx)]
    active = np.ones(shape, dtype=np.float32)
    if gap_start_probability <= 0.0:
        return active

    for batch_index in range(shape[0]):
        for ue_pos in range(shape[2]):
            slot = 1
            while slot < shape[1]:
                if rng.random() < float(gap_start_probability):
                    duration = int(rng.randint(1, int(max_gap_slots) + 1))
                    active[batch_index, slot:min(slot + duration, shape[1]), ue_pos] = 0
                    slot += duration
                else:
                    slot += 1
        for slot in range(shape[1]):
            if not np.any(active[batch_index, slot]):
                keep = int(rng.randint(0, shape[2]))
                active[batch_index, slot, keep] = 1
    return active


def controlled_change_mask(
    batch_size,
    seq_len,
    num_users,
    change_tb,
    duration=None,
):
    """Build a deterministic one-indexed channel-change mask."""
    change_tb = int(change_tb)
    seq_len = int(seq_len)
    if not 2 <= change_tb <= seq_len:
        raise ValueError("change_tb must be between 2 and seq_len")
    start = change_tb - 1
    stop = seq_len if duration is None else min(start + int(duration), seq_len)
    if stop <= start:
        raise ValueError("duration must be positive")
    mask = np.zeros(
        [int(batch_size), seq_len, int(num_users)], dtype=bool
    )
    mask[:, start:stop, :] = True
    return mask


class TemporalTrainingDataGenerator:
    """Generate TB sequences for stable UE IDs."""

    def __init__(
        self,
        sys_parameters,
        e2e_model,
        ue_pool_size=NUM_TX,
        effective_ue_pool_size_range=None,
        dynamic_scheduling=False,
        schedule_switch_prob=0.65,
        schedule_reorder_prob=0.50,
        schedule_switch_prob_range=None,
        schedule_reorder_prob_range=None,
        scenario_training=False,
        channel_change_prob=0.75,
        ebno_drop_prob=0.50,
        min_ebno_drop_db=0.5,
        max_ebno_drop_db=2.0,
        gap_start_prob=0.05,
        max_gap_slots=4,
        hard_reset_prob=0.02,
        min_stable_tbs=4,
        max_event_duration=16,
        random_slot_offset=True,
        advance_slot_pilots=True,
        event_channel_model="C",
        event_delay_spread_s=300e-9,
        event_min_doppler_hz=250,
        event_max_doppler_hz=400,
        channel_profiles=None,
        event_channel_profiles=None,
        randomize_profile_assignment=False,
        profile_seed=None,
        channel_role="known_comparison",
        seed_streams=None,
        seed_stream_limit=None,
    ):
        self.p = sys_parameters
        self.tx = e2e_model._transmitters[0]
        self.source = e2e_model._source
        self.rx = e2e_model._receiver
        self.rg = self.tx.resource_grid
        self.freqs = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel()
        self.apply_channel_noiseless = ApplyOFDMChannel(add_awgn=False)

        self.num_scheduled_tx = int(sys_parameters.max_num_tx)
        self.ue_pool_size = int(ue_pool_size)
        self.effective_ue_pool_size_range = self._validate_integer_range(
            effective_ue_pool_size_range,
            "effective_ue_pool_size_range",
        )
        self.dynamic_scheduling = bool(dynamic_scheduling)
        self.schedule_switch_prob = float(schedule_switch_prob)
        self.schedule_reorder_prob = float(schedule_reorder_prob)
        self.schedule_switch_prob_range = self._validate_probability_range(
            schedule_switch_prob_range, "schedule_switch_prob_range"
        )
        self.schedule_reorder_prob_range = self._validate_probability_range(
            schedule_reorder_prob_range, "schedule_reorder_prob_range"
        )
        self.scenario_training = bool(scenario_training)
        self.channel_change_prob = float(channel_change_prob)
        self.ebno_drop_prob = float(ebno_drop_prob)
        self.min_ebno_drop_db = float(min_ebno_drop_db)
        self.max_ebno_drop_db = float(max_ebno_drop_db)
        self.gap_start_prob = float(gap_start_prob)
        self.max_gap_slots = int(max_gap_slots)
        self.hard_reset_prob = float(hard_reset_prob)
        self.min_stable_tbs = int(min_stable_tbs)
        self.max_event_duration = int(max_event_duration)
        self.random_slot_offset = bool(random_slot_offset)
        self.advance_slot_pilots = bool(advance_slot_pilots)
        self.event_channel_model = str(event_channel_model)
        self.event_delay_spread_s = float(event_delay_spread_s)
        self.event_min_doppler_hz = float(event_min_doppler_hz)
        self.event_max_doppler_hz = float(event_max_doppler_hz)
        self.randomize_profile_assignment = bool(
            randomize_profile_assignment
        )
        self.seed_streams = self._validate_seed_streams(seed_streams)
        self.seed_stream_limit = (
            None if seed_stream_limit is None else int(seed_stream_limit)
        )
        if self.seed_stream_limit is not None and self.seed_stream_limit < 1:
            raise ValueError("seed_stream_limit must be positive")
        self.seed_stream_consumed = 0
        if self.seed_streams is None:
            self.channel_rng = np.random.RandomState(
                profile_seed if profile_seed is not None else 0
            )
            self.noise_rng = np.random
            self.schedule_rng = np.random
        else:
            self.channel_rng = np.random.RandomState(
                self.seed_streams["channel"]
            )
            self.noise_rng = np.random.RandomState(
                self.seed_streams["noise"]
            )
            self.schedule_rng = np.random.RandomState(
                self.seed_streams["schedule"]
            )
        self.profile_rng = self.channel_rng
        self.channel_role = str(channel_role)
        if self.channel_role not in (
            "training",
            "calibration",
            "known_comparison",
            "ood",
        ):
            raise ValueError(f"unknown channel role: {self.channel_role}")

        if self.ue_pool_size < self.num_scheduled_tx:
            raise ValueError(
                "ue_pool_size must be >= the receiver's max_num_tx "
                f"({self.num_scheduled_tx})"
            )
        if self.effective_ue_pool_size_range is not None:
            low, high = self.effective_ue_pool_size_range
            if low < self.num_scheduled_tx or high > self.ue_pool_size:
                raise ValueError("effective UE pool range is outside bounds")
        for name, value in [
            ("schedule_switch_prob", self.schedule_switch_prob),
            ("schedule_reorder_prob", self.schedule_reorder_prob),
            ("channel_change_prob", self.channel_change_prob),
            ("ebno_drop_prob", self.ebno_drop_prob),
            ("gap_start_prob", self.gap_start_prob),
            ("hard_reset_prob", self.hard_reset_prob),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_gap_slots < 1:
            raise ValueError("max_gap_slots must be positive")
        if self.min_stable_tbs < 1:
            raise ValueError("min_stable_tbs must be positive")
        if self.max_event_duration < 1:
            raise ValueError("max_event_duration must be positive")
        if not 0.0 <= self.min_ebno_drop_db <= self.max_ebno_drop_db:
            raise ValueError("invalid Eb/N0-drop range")
        if self.event_delay_spread_s <= 0.0:
            raise ValueError("event_delay_spread_s must be positive")
        if not 0.0 <= self.event_min_doppler_hz <= self.event_max_doppler_hz:
            raise ValueError("invalid event Doppler range")

        if channel_profiles is None:
            channel_profiles = (
                ChannelProfileSpec(
                    "TDL-B100",
                    "tdl",
                    "B100",
                    100e-9,
                    100e-9,
                    0.0,
                    400.0,
                ),
                ChannelProfileSpec(
                    "TDL-C300",
                    "tdl",
                    "C300",
                    300e-9,
                    300e-9,
                    0.0,
                    100.0,
                ),
            )
        if event_channel_profiles is None:
            event_channel_profiles = (
                ChannelProfileSpec(
                    f"TDL-{self.event_channel_model}",
                    "tdl",
                    self.event_channel_model,
                    self.event_delay_spread_s,
                    self.event_delay_spread_s,
                    self.event_min_doppler_hz,
                    self.event_max_doppler_hz,
                ),
            )
        self.channel_profiles = self._validate_profiles(channel_profiles)
        self.event_channel_profiles = self._validate_profiles(
            event_channel_profiles
        )
        self.steady_channels = tuple(
            self._make_channel(profile) for profile in self.channel_profiles
        )
        self.event_channels = tuple(
            self._make_channel(profile)
            for profile in self.event_channel_profiles
        )
        self.ue_profiles = self._initial_profile_metadata()

    @staticmethod
    def _validate_integer_range(value, name):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = (int(item) for item in value)
        if low < 1 or high < low:
            raise ValueError(f"{name} is invalid")
        return low, high

    @staticmethod
    def _validate_probability_range(value, name):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"{name} must contain two values")
        low, high = (float(item) for item in value)
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"{name} is invalid")
        return low, high

    @staticmethod
    def _validate_seed_streams(seed_streams):
        if seed_streams is None:
            return None
        required = {"channel", "payload", "noise", "schedule"}
        if set(seed_streams) != required:
            raise ValueError("seed_streams must define four random domains")
        values = {name: int(value) for name, value in seed_streams.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("seed streams cannot be negative")
        if len(set(values.values())) != len(values):
            raise ValueError("seed streams must be disjoint")
        return values

    def _reserve_seed_batch(self, batch_size):
        if self.seed_streams is None:
            return None
        batch_size = int(batch_size)
        stop = self.seed_stream_consumed + batch_size
        if self.seed_stream_limit is not None and stop > self.seed_stream_limit:
            raise ValueError("data request exceeds the frozen split slice")
        seeds = {
            name: value + self.seed_stream_consumed
            for name, value in self.seed_streams.items()
        }
        self.seed_stream_consumed = stop
        return seeds

    def _activate_seed_batch(self, seeds):
        if seeds is None:
            return
        self.channel_rng = np.random.RandomState(seeds["channel"])
        self.profile_rng = self.channel_rng
        self.noise_rng = np.random.RandomState(seeds["noise"])
        self.schedule_rng = np.random.RandomState(seeds["schedule"])
        tf.random.set_seed(seeds["channel"])

    def seed_stream_metadata(self):
        return {
            "enabled": self.seed_streams is not None,
            "base_seeds": self.seed_streams,
            "episode_capacity": self.seed_stream_limit,
            "episodes_consumed": self.seed_stream_consumed,
        }

    @staticmethod
    def _validate_profiles(profiles):
        values = []
        for value in profiles:
            profile = (
                value
                if isinstance(value, ChannelProfileSpec)
                else ChannelProfileSpec.from_dict(value)
            )
            profile.validate()
            values.append(profile)
        if not values:
            raise ValueError("channel profile pool cannot be empty")
        return tuple(values)

    def _make_tdl(self, profile):
        carrier_frequency = self.p.carrier_frequency
        scale = sn.SPEED_OF_LIGHT / carrier_frequency
        tx_corr = ue_correlation_matrix(self.p.num_antenna_ports, 0)
        rx_corr = gnb_correlation_matrix(NUM_RX_ANT, 0)
        delay_spread = self._nominal_delay(profile)
        return TDL(
            profile.model,
            delay_spread,
            carrier_frequency,
            min_speed=profile.min_doppler_hz * scale,
            max_speed=profile.max_doppler_hz * scale,
            num_tx_ant=self.p.num_antenna_ports,
            num_rx_ant=NUM_RX_ANT,
            rx_corr_mat=rx_corr,
            tx_corr_mat=tx_corr,
        )

    def _make_umi(self, profile):
        carrier_frequency = self.p.carrier_frequency
        if NUM_RX_ANT % 2:
            raise ValueError("UMi requires an even receive-antenna count")
        bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=NUM_RX_ANT // 2,
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.p.num_antenna_ports,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
        return UMi(
            carrier_frequency=carrier_frequency,
            o2i_model="low",
            bs_array=bs_array,
            ut_array=ut_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
        )

    def _make_channel(self, profile):
        if profile.family == "tdl":
            return self._make_tdl(profile)
        if profile.family == "umi":
            return self._make_umi(profile)
        raise ValueError(f"unsupported channel family: {profile.family}")

    @staticmethod
    def _nominal_delay(profile):
        if profile.min_delay_spread_s is None:
            return None
        low = float(profile.min_delay_spread_s)
        high = float(profile.max_delay_spread_s)
        return math.sqrt(low * high)

    def _sample_delay(self, profile):
        if profile.min_delay_spread_s is None:
            return None
        low = float(profile.min_delay_spread_s)
        high = float(profile.max_delay_spread_s)
        if low == high:
            return low
        return float(np.exp(self.profile_rng.uniform(
            np.log(low), np.log(high)
        )))

    def _initial_profile_metadata(self):
        profiles = []
        for ue_id in range(self.ue_pool_size):
            steady = self.channel_profiles[
                ue_id % len(self.channel_profiles)
            ]
            event = self.event_channel_profiles[
                ue_id % len(self.event_channel_profiles)
            ]
            profiles.append(self._profile_metadata(
                ue_id,
                steady,
                self._nominal_delay(steady),
                event,
                self._nominal_delay(event),
            ))
        return profiles

    def _profile_metadata(
        self, ue_id, steady, steady_delay, event, event_delay
    ):
        return {
            "ue_id": int(ue_id),
            "split_role": self.channel_role,
            "steady_family": steady.family,
            "steady_model": steady.model,
            "steady_delay_spread_s": steady_delay,
            "steady_delay_spread_range_s": [
                steady.min_delay_spread_s,
                steady.max_delay_spread_s,
            ],
            "steady_doppler_hz": [
                steady.min_doppler_hz,
                steady.max_doppler_hz,
            ],
            "harsh_family": event.family,
            "harsh_model": event.model,
            "harsh_delay_spread_s": event_delay,
            "harsh_delay_spread_range_s": [
                event.min_delay_spread_s,
                event.max_delay_spread_s,
            ],
            "harsh_doppler_hz": [
                event.min_doppler_hz,
                event.max_doppler_hz,
            ],
        }

    def _assignment_indices(self, size):
        if self.randomize_profile_assignment:
            return self.profile_rng.randint(
                0, int(size), size=self.ue_pool_size
            )
        return np.arange(self.ue_pool_size) % int(size)

    def _slot_numbers(self, seq_len):
        num_slots = int(self.p.pilots.shape[0])
        start = int(self.p.slot_number)
        if self.random_slot_offset:
            start = int(self.schedule_rng.randint(0, num_slots))
        if not self.advance_slot_pilots:
            return np.full([int(seq_len)], start, dtype=np.int32)
        return (start + np.arange(int(seq_len), dtype=np.int32)) % num_slots

    def _set_slot_pilots(self, slot_number):
        pilots = tf.gather(self.p.pilots, int(slot_number), axis=0)
        self.tx.pilot_pattern.pilots = pilots
        self.rx._ls_est._pilot_pattern.pilots = pilots

    def sample_scenario(
        self,
        batch_size,
        seq_len,
        ebno_db,
        enabled=None,
        controlled_change_tb=None,
        controlled_change_duration=None,
        controlled_ebno_drop_db=0.0,
    ):
        """Sample activity, channel events, and slot metadata."""
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        enabled = self.scenario_training if enabled is None else bool(enabled)
        channel_mask = np.zeros(
            [batch_size, seq_len, self.ue_pool_size], dtype=bool
        )
        ebno_mask = np.zeros([batch_size, seq_len], dtype=bool)
        if enabled:
            for ue_id in range(self.ue_pool_size):
                channel_mask[:, :, ue_id] = sample_burst_mask(
                    batch_size,
                    seq_len,
                    self.channel_change_prob,
                    self.min_stable_tbs,
                    self.max_event_duration,
                    rng=self.channel_rng,
                )
            ebno_mask = sample_burst_mask(
                batch_size,
                seq_len,
                self.ebno_drop_prob,
                self.min_stable_tbs,
                self.max_event_duration,
                rng=self.noise_rng,
            )
        if controlled_change_tb is not None:
            if enabled:
                raise ValueError(
                    "controlled changes cannot use random scenario training"
                )
            channel_mask = controlled_change_mask(
                batch_size,
                seq_len,
                self.ue_pool_size,
                controlled_change_tb,
                controlled_change_duration,
            )
            if float(controlled_ebno_drop_db) > 0.0:
                ebno_mask = channel_mask[:, :, 0]

        base_ebno = np.asarray(ebno_db, dtype=np.float32)
        if base_ebno.ndim == 0:
            base_ebno = np.full([batch_size], float(base_ebno), np.float32)
        if list(base_ebno.shape) != [batch_size]:
            raise ValueError("ebno_db must be scalar or have shape [batch_size]")
        if controlled_change_tb is None:
            drops = self.noise_rng.uniform(
                self.min_ebno_drop_db,
                self.max_ebno_drop_db,
                size=[batch_size],
            ).astype(np.float32)
        else:
            drops = np.full(
                [batch_size], float(controlled_ebno_drop_db), np.float32
            )
        ebno_schedule = base_ebno[:, None] - ebno_mask * drops[:, None]

        active = sample_activity_mask(
            batch_size,
            seq_len,
            self.num_scheduled_tx,
            self.gap_start_prob if enabled else 0.0,
            self.max_gap_slots,
            rng=self.schedule_rng,
        )
        hard_reset = np.zeros_like(active, dtype=bool)
        if enabled and self.hard_reset_prob > 0.0:
            hard_reset = (
                self.schedule_rng.random_sample(active.shape)
                < self.hard_reset_prob
            ) & active.astype(bool)
            hard_reset[:, 0] = False

        channel_event = channel_mask.copy()
        channel_event[:, 1:] = (
            channel_mask[:, 1:] != channel_mask[:, :-1]
        )
        channel_event[:, 0] = False
        ebno_event = ebno_mask.copy()
        ebno_event[:, 1:] = ebno_mask[:, 1:] != ebno_mask[:, :-1]
        ebno_event[:, 0] = False
        slot_numbers = self._slot_numbers(seq_len)
        return {
            "active": tf.convert_to_tensor(active, tf.float32),
            "hard_reset": tf.convert_to_tensor(hard_reset, tf.bool),
            "memory_compatible": tf.ones_like(
                tf.convert_to_tensor(active, tf.float32), dtype=tf.bool
            ),
            "channel_regime": tf.convert_to_tensor(channel_mask, tf.bool),
            "channel_change_event": tf.convert_to_tensor(channel_event, tf.bool),
            "ebno_impairment": tf.convert_to_tensor(ebno_mask, tf.bool),
            "ebno_change_event": tf.convert_to_tensor(ebno_event, tf.bool),
            "ebno_db": tf.convert_to_tensor(ebno_schedule, tf.float32),
            "slot_number": tf.convert_to_tensor(slot_numbers, tf.int32),
        }

    def sample_schedule(self, batch_size, seq_len):
        """Sample scheduled UE IDs."""
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        n = self.num_scheduled_tx
        pool = self.ue_pool_size

        if not self.dynamic_scheduling or pool == n:
            fixed = np.arange(n, dtype=np.int32)
            schedule = np.tile(fixed[None, None, :], [batch_size, seq_len, 1])
            if self.dynamic_scheduling and n > 1:
                for b in range(batch_size):
                    for t in range(1, seq_len):
                        if self.schedule_rng.random() < self.schedule_reorder_prob:
                            schedule[b, t] = self.schedule_rng.permutation(
                                schedule[b, t]
                            )
            return tf.convert_to_tensor(schedule, tf.int32)

        schedule = np.empty((batch_size, seq_len, n), dtype=np.int32)
        full_universe = np.arange(pool, dtype=np.int32)
        metadata = []

        for b in range(batch_size):
            if self.effective_ue_pool_size_range is None:
                effective_pool = pool
            else:
                low, high = self.effective_ue_pool_size_range
                effective_pool = int(self.schedule_rng.randint(low, high + 1))
            universe = self.schedule_rng.choice(
                full_universe, size=effective_pool, replace=False
            )
            switch_prob = self.schedule_switch_prob
            if self.schedule_switch_prob_range is not None:
                switch_prob = float(self.schedule_rng.uniform(
                    *self.schedule_switch_prob_range
                ))
            reorder_prob = self.schedule_reorder_prob
            if self.schedule_reorder_prob_range is not None:
                reorder_prob = float(self.schedule_rng.uniform(
                    *self.schedule_reorder_prob_range
                ))
            current = self.schedule_rng.choice(
                universe, size=n, replace=False
            )
            schedule[b, 0] = current
            metadata.append({
                "effective_ue_pool_size": effective_pool,
                "schedule_switch_prob": switch_prob,
                "schedule_reorder_prob": reorder_prob,
            })

            for t in range(1, seq_len):
                current = current.copy()

                if self.schedule_rng.random() < switch_prob:
                    replace_pos = int(self.schedule_rng.randint(0, n))
                    absent = np.setdiff1d(universe, current, assume_unique=False)
                    if len(absent):
                        current[replace_pos] = self.schedule_rng.choice(absent)

                if (
                    n > 1
                    and self.schedule_rng.random() < reorder_prob
                ):
                    current = self.schedule_rng.permutation(current)

                if len(np.unique(current)) != n:
                    raise RuntimeError("Scheduler generated duplicate physical UE IDs")
                schedule[b, t] = current

        self.last_schedule_metadata = metadata
        return tf.convert_to_tensor(schedule, tf.int32)

    def _sample_channel_slots(
        self, profile, channel, batch_size, seq_len
    ):
        num_sym = self.rg.num_ofdm_symbols
        sampling_frequency = 1.0 / self.rg.ofdm_symbol_duration
        delay_spread = self._sample_delay(profile)
        if profile.family == "tdl":
            if profile.model not in ("A30", "B100", "C300"):
                channel.delay_spread = tf.cast(
                    delay_spread, tf.float32
                )
        elif profile.family == "umi":
            scale = sn.SPEED_OF_LIGHT / self.p.carrier_frequency
            topology = gen_single_sector_topology(
                int(batch_size),
                1,
                "umi",
                min_ut_velocity=profile.min_doppler_hz * scale,
                max_ut_velocity=profile.max_doppler_hz * scale,
                indoor_probability=0.0,
            )
            channel.set_topology(*topology)
        else:
            raise ValueError(f"unsupported family: {profile.family}")
        a, tau = channel(
            batch_size, seq_len * num_sym, sampling_frequency
        )
        h = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
        slots = tf.stack(tf.split(h, seq_len, axis=5), axis=1)
        return slots, delay_spread

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        """Sample continuous per-UE channel trajectories."""
        steady_indices = self._assignment_indices(
            len(self.channel_profiles)
        )
        event_indices = self._assignment_indices(
            len(self.event_channel_profiles)
        )
        h_users = []
        metadata = []
        for ue_id in range(self.ue_pool_size):
            steady_index = int(steady_indices[ue_id])
            event_index = int(event_indices[ue_id])
            steady_profile = self.channel_profiles[steady_index]
            event_profile = self.event_channel_profiles[event_index]
            if (
                self.randomize_profile_assignment
                and len(self.event_channel_profiles) > 1
                and event_profile.name == steady_profile.name
            ):
                candidates = [
                    index
                    for index, profile in enumerate(
                        self.event_channel_profiles
                    )
                    if profile.name != steady_profile.name
                ]
                event_index = int(self.profile_rng.choice(candidates))
                event_profile = self.event_channel_profiles[event_index]
            steady, steady_delay = self._sample_channel_slots(
                steady_profile,
                self.steady_channels[steady_index],
                batch_size,
                seq_len,
            )
            event_delay = self._nominal_delay(event_profile)
            if channel_regime is not None:
                harsh, event_delay = self._sample_channel_slots(
                    event_profile,
                    self.event_channels[event_index],
                    batch_size,
                    seq_len,
                )
                mask = channel_regime[:, :, ue_id]
                mask = tf.reshape(
                    mask,
                    [batch_size, seq_len, 1, 1, 1, 1, 1, 1],
                )
                steady = tf.where(mask, harsh, steady)
            h_users.append(steady)
            metadata.append(self._profile_metadata(
                ue_id,
                steady_profile,
                steady_delay,
                event_profile,
                event_delay,
            ))
        self.ue_profiles = metadata
        return tf.concat(h_users, axis=4)

    def gather_scheduled_channel(self, h_pool, ue_ids):
        """Map stable UEs into receiver positions."""
        seq_len = ue_ids.shape[1]
        if seq_len is None:
            raise ValueError("seq_len must be statically known for sequence training")

        gathered = []
        for t in range(seq_len):
            h_t = tf.gather(
                h_pool[:, t],
                ue_ids[:, t],
                axis=3,
                batch_dims=1,
            )
            gathered.append(h_t)
        return tf.stack(gathered, axis=1)

    def ebno_to_no(self, ebno_db):
        return ebnodb2no(
            ebno_db,
            self.tx._num_bits_per_symbol,
            self.tx._target_coderate,
            self.tx.resource_grid,
        )

    def _sample_bits(self, shape, seed, tb):
        if seed is None:
            return self.source(shape)
        values = tf.random.stateless_uniform(
            shape,
            seed=tf.constant([int(seed), int(tb)], tf.int32),
            minval=0,
            maxval=2,
            dtype=tf.int32,
        )
        return tf.cast(values, self.source.dtype)

    @staticmethod
    def _add_seeded_noise(clean, no, seed, tb):
        real_dtype = clean.dtype.real_dtype
        shape = tf.shape(clean)
        real = tf.random.stateless_normal(
            shape,
            seed=tf.constant([int(seed), 2 * int(tb)], tf.int32),
            dtype=real_dtype,
        )
        imag = tf.random.stateless_normal(
            shape,
            seed=tf.constant([int(seed), 2 * int(tb) + 1], tf.int32),
            dtype=real_dtype,
        )
        unit_noise = tf.complex(real, imag) / tf.cast(
            tf.sqrt(tf.constant(2.0, real_dtype)), clean.dtype
        )
        variance = expand_to_rank(no, tf.rank(clean), axis=-1)
        scale = tf.cast(tf.sqrt(tf.cast(variance, real_dtype)), clean.dtype)
        return clean + unit_noise * scale

    def sample_batch(
        self,
        batch_size,
        seq_len,
        ebno_db,
        ue_ids=None,
        include_pool_channel=False,
        scenario_training=None,
        controlled_change_tb=None,
        controlled_change_duration=None,
        controlled_ebno_drop_db=0.0,
        include_clean_received=False,
        include_transmitted=False,
        include_ls=True,
    ):
        """Generate a full temporal training batch."""
        batch_seeds = self._reserve_seed_batch(batch_size)
        self._activate_seed_batch(batch_seeds)
        if ue_ids is None:
            ue_ids = self.sample_schedule(batch_size, seq_len)
        else:
            ue_ids = tf.cast(ue_ids, tf.int32)
            expected = [int(batch_size), int(seq_len), self.num_scheduled_tx]
            if list(ue_ids.shape) != expected:
                raise ValueError(
                    f"ue_ids must have shape {expected}, got {list(ue_ids.shape)}"
                )
            tf.debugging.assert_greater_equal(ue_ids, 0)
            tf.debugging.assert_less(ue_ids, self.ue_pool_size)

        scenario_enabled = (
            self.scenario_training
            if scenario_training is None
            else bool(scenario_training)
        )
        scenario = self.sample_scenario(
            batch_size,
            seq_len,
            ebno_db,
            enabled=scenario_enabled,
            controlled_change_tb=controlled_change_tb,
            controlled_change_duration=controlled_change_duration,
            controlled_ebno_drop_db=controlled_ebno_drop_db,
        )
        h_pool = self.sample_pool_channel(
            batch_size,
            seq_len,
            (
                scenario["channel_regime"]
                if scenario_enabled or controlled_change_tb is not None
                else None
            ),
        )
        h = self.gather_scheduled_channel(h_pool, ue_ids)

        bits = []
        received = []
        clean_received = []
        transmitted = []
        ls_estimates = []
        noise = []

        for t in range(seq_len):
            h_t = h[:, t]
            active_t = scenario["active"][:, t]
            self._set_slot_pilots(scenario["slot_number"][t])
            b_t = self._sample_bits([
                batch_size,
                self.p.max_num_tx,
                self.tx._tb_size,
            ], None if batch_seeds is None else batch_seeds["payload"], t)
            x_t = self.tx(b_t)
            x_t *= tf.cast(active_t[:, :, None, None, None], x_t.dtype)
            if include_transmitted:
                transmitted.append(x_t)
            no_t = self.ebno_to_no(scenario["ebno_db"][:, t])
            if include_clean_received or batch_seeds is not None:
                clean_t = self.apply_channel_noiseless([x_t, h_t])
                if include_clean_received:
                    clean_received.append(clean_t)
            if batch_seeds is None:
                y_t = self.apply_channel([x_t, h_t, no_t])
            else:
                y_t = self._add_seeded_noise(
                    clean_t, no_t, batch_seeds["noise"], t
                )
            ls_t = None
            if include_ls:
                ls_t = self.rx.estimate_channel(y_t, self.p.max_num_tx)

            bits.append(b_t)
            received.append(y_t)
            if include_ls:
                ls_estimates.append(ls_t)
            noise.append(no_t)

        result = {
            "bits": tf.stack(bits, axis=1),
            "y": tf.stack(received, axis=1),
            "h": h,
            "active": scenario["active"],
            "ue_ids": ue_ids,
            "no": tf.stack(noise, axis=1),
            "ebno_db": scenario["ebno_db"],
            "slot_number": scenario["slot_number"],
            "hard_reset": scenario["hard_reset"],
            "memory_compatible": scenario["memory_compatible"],
            "channel_regime": scenario["channel_regime"],
            "channel_change_event": scenario["channel_change_event"],
            "ebno_impairment": scenario["ebno_impairment"],
            "ebno_change_event": scenario["ebno_change_event"],
        }
        if include_ls:
            result["ls"] = tf.stack(ls_estimates, axis=1)
        if include_pool_channel:
            result["h_pool"] = h_pool
        if include_clean_received:
            result["y_clean"] = tf.stack(clean_received, axis=1)
        if include_transmitted:
            result["x"] = tf.stack(transmitted, axis=1)
        return result


def _complex_corr(a, b):
    a = np.asarray(a).reshape(a.shape[0], -1)
    b = np.asarray(b).reshape(b.shape[0], -1)
    num = np.abs(np.sum(np.conj(a) * b, axis=1))
    den = np.sqrt(
        np.sum(np.abs(a) ** 2, axis=1) *
        np.sum(np.abs(b) ** 2, axis=1)
    )
    return float(np.mean(num / np.maximum(den, 1e-12)))


def validate_batch(batch, dynamic_scheduling=False):
    channel_key = "h_pool" if "h_pool" in batch else "h"
    h = batch[channel_key].numpy()
    bits = batch["bits"].numpy()
    ue_ids = batch["ue_ids"].numpy()
    active = batch["active"].numpy()
    seq_len = h.shape[1]

    adjacent = [_complex_corr(h[:, t], h[:, t + 1])
                for t in range(seq_len - 1)]
    lag2 = [_complex_corr(h[:, t], h[:, t + 2])
            for t in range(seq_len - 2)] if seq_len > 2 else []

    bit_disagreement = [
        float(np.mean(bits[:, t] != bits[:, t + 1]))
        for t in range(seq_len - 1)
    ]

    schedule_changes = [
        float(np.mean(np.any(ue_ids[:, t] != ue_ids[:, t + 1], axis=-1)))
        for t in range(seq_len - 1)
    ]
    unique_per_tb = all(
        np.all([
            len(np.unique(ue_ids[b, t])) == ue_ids.shape[-1]
            for b in range(ue_ids.shape[0])
        ])
        for t in range(ue_ids.shape[1])
    )

    checks = {
        "nearby_channels_related": bool(np.mean(adjacent) > 0.5),
        "payloads_independent": bool(0.45 < np.mean(bit_disagreement) < 0.55),
        "unique_physical_ues_per_tb": bool(unique_per_tb),
        "at_least_one_active_ue": bool(np.all(np.sum(active, axis=-1) >= 1)),
    }
    if "slot_number" in batch and seq_len > 1:
        slots = batch["slot_number"].numpy()
        checks["slot_number_advances"] = bool(np.all(slots[1:] != slots[:-1]))
    if dynamic_scheduling and seq_len > 1:
        checks["schedule_changes_present"] = bool(np.mean(schedule_changes) > 0.0)

    summary = {
        "bits_shape": list(batch["bits"].shape),
        "y_shape": list(batch["y"].shape),
        "ls_shape": list(batch["ls"].shape),
        "h_shape": list(batch["h"].shape),
        "ue_ids_shape": list(batch["ue_ids"].shape),
        "ue_ids_example": ue_ids[0].tolist(),
        "active_shape": list(batch["active"].shape),
        "active_fraction": float(np.mean(active)),
        "adjacent_channel_corr_mean": float(np.mean(adjacent)),
        "adjacent_channel_corr": adjacent,
        "lag2_channel_corr_mean": float(np.mean(lag2)) if lag2 else None,
        "first_last_channel_corr": _complex_corr(h[:, 0], h[:, -1]),
        "adjacent_payload_bit_disagreement_mean": float(np.mean(bit_disagreement)),
        "adjacent_payload_bit_disagreement": bit_disagreement,
        "schedule_change_fraction_mean": (
            float(np.mean(schedule_changes)) if schedule_changes else 0.0
        ),
        "schedule_change_fraction": schedule_changes,
        "slot_numbers": (
            batch["slot_number"].numpy().tolist()
            if "slot_number" in batch else None
        ),
        "channel_change_fraction": (
            float(np.mean(batch["channel_change_event"].numpy()))
            if "channel_change_event" in batch else 0.0
        ),
        "ebno_change_fraction": (
            float(np.mean(batch["ebno_change_event"].numpy()))
            if "ebno_change_event" in batch else 0.0
        ),
        "checks": checks,
    }
    return summary


def main():
    np.random.seed(ARGS.seed)
    tf.random.set_seed(ARGS.seed)
    try:
        sn.config.seed = ARGS.seed
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    p = Parameters(
        ARGS.config,
        training=False,
        num_tx_eval=NUM_TX,
        system="nrx",
    )
    e2e = E2E_Model(p, training=False, mcs_arr_eval_idx=0)
    manifest_path = Path(ARGS.split_manifest)
    if not manifest_path.is_absolute():
        manifest_path = HERE.parent / manifest_path
    split_manifest = load_split_manifest(manifest_path)
    validate_split_slice(
        split_manifest, ARGS.channel_role, 0, ARGS.batch_size
    )
    channel_profiles = role_profile_specs(
        split_manifest, ARGS.channel_role
    )
    generator = TemporalTrainingDataGenerator(
        p,
        e2e,
        ue_pool_size=ARGS.ue_pool_size,
        dynamic_scheduling=ARGS.dynamic_scheduling,
        schedule_switch_prob=ARGS.schedule_switch_prob,
        schedule_reorder_prob=ARGS.schedule_reorder_prob,
        scenario_training=ARGS.scenario_training,
        channel_profiles=channel_profiles,
        event_channel_profiles=channel_profiles,
        randomize_profile_assignment=(
            ARGS.channel_role in ("training", "calibration")
        ),
        channel_role=ARGS.channel_role,
        seed_streams=seed_bundle(
            split_manifest, ARGS.channel_role, 0
        ),
        seed_stream_limit=ARGS.batch_size,
    )
    batch = generator.sample_batch(
        batch_size=ARGS.batch_size,
        seq_len=ARGS.seq_len,
        ebno_db=ARGS.ebno_db,
        include_pool_channel=True,
    )

    summary = validate_batch(
        batch, dynamic_scheduling=ARGS.dynamic_scheduling)
    summary["ue_pool_size"] = ARGS.ue_pool_size
    summary["ue_profiles"] = generator.ue_profiles
    summary["channel_role"] = ARGS.channel_role
    summary["split_manifest_sha256"] = split_manifest[
        "manifest_sha256"
    ]
    summary["seed_streams"] = generator.seed_stream_metadata()
    print("TEMPORAL_DATA_SUMMARY=" + json.dumps(summary, indent=2))

    if ARGS.output_json:
        path = Path(ARGS.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")

    if ARGS.validate:
        if not all(summary["checks"].values()):
            raise SystemExit("Temporal data validation failed")
        print("VALIDATION_PASSED")


if __name__ == "__main__":
    main()
