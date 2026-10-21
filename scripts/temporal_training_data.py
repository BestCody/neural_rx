import math

import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.channel import (
    ApplyOFDMChannel,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)
from sionna.channel.tr38901 import TDL
from sionna.utils import BinarySource, ebnodb2no, expand_to_rank

from utils.channel_models import ue_correlation_matrix, gnb_correlation_matrix
from temporal_generalization_protocol import ChannelProfileSpec


NUM_TX = 2
NUM_RX_ANT = 4


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
):
    """Build a deterministic one-indexed channel-change mask."""
    change_tb = int(change_tb)
    seq_len = int(seq_len)
    if not 2 <= change_tb <= seq_len:
        raise ValueError("change_tb must be between 2 and seq_len")
    mask = np.zeros(
        [int(batch_size), seq_len, int(num_users)], dtype=bool
    )
    mask[:, change_tb - 1:, :] = True
    return mask


class TemporalTrainingDataGenerator:
    """Generate TB sequences for stable UE IDs."""

    def __init__(
        self,
        sys_parameters,
        e2e_model=None,
        ue_pool_size=NUM_TX,
        dynamic_scheduling=False,
        schedule_switch_prob=0.65,
        schedule_reorder_prob=0.50,
        scenario_training=False,
        gap_start_prob=0.05,
        max_gap_slots=4,
        random_slot_offset=True,
        advance_slot_pilots=True,
        channel_profiles=None,
        event_channel_profiles=None,
        randomize_profile_assignment=False,
        seed_streams=None,
        seed_stream_limit=None,
    ):
        self.p = sys_parameters
        self.tx = sys_parameters.transmitters[0]
        self.source = BinarySource() if e2e_model is None else e2e_model._source
        self.rx = None if e2e_model is None else e2e_model._receiver
        self.rg = self.tx.resource_grid
        self.freqs = subcarrier_frequencies(
            self.rg.fft_size, self.rg.subcarrier_spacing)
        self.apply_channel = ApplyOFDMChannel()
        self.apply_channel_noiseless = ApplyOFDMChannel(add_awgn=False)

        self.num_scheduled_tx = int(sys_parameters.max_num_tx)
        self.ue_pool_size = int(ue_pool_size)
        self.dynamic_scheduling = bool(dynamic_scheduling)
        self.schedule_switch_prob = float(schedule_switch_prob)
        self.schedule_reorder_prob = float(schedule_reorder_prob)
        self.scenario_training = bool(scenario_training)
        self.gap_start_prob = float(gap_start_prob)
        self.max_gap_slots = int(max_gap_slots)
        self.random_slot_offset = bool(random_slot_offset)
        self.advance_slot_pilots = bool(advance_slot_pilots)
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
            self.channel_rng = np.random.RandomState(0)
            self.schedule_rng = np.random
        else:
            self.channel_rng = np.random.RandomState(
                self.seed_streams["channel"]
            )
            self.schedule_rng = np.random.RandomState(
                self.seed_streams["schedule"]
            )
        self.profile_rng = self.channel_rng
        if self.ue_pool_size < self.num_scheduled_tx:
            raise ValueError(
                "ue_pool_size must be >= the receiver's max_num_tx "
                f"({self.num_scheduled_tx})"
            )
        for name, value in [
            ("schedule_switch_prob", self.schedule_switch_prob),
            ("schedule_reorder_prob", self.schedule_reorder_prob),
            ("gap_start_prob", self.gap_start_prob),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_gap_slots < 1:
            raise ValueError("max_gap_slots must be positive")

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
                    "TDL-C",
                    "tdl",
                    "C",
                    300e-9,
                    300e-9,
                    250.0,
                    400.0,
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
        self.schedule_rng = np.random.RandomState(seeds["schedule"])
        tf.random.set_seed(seeds["channel"])


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

    def _make_channel(self, profile):
        if profile.family == "tdl":
            return self._make_tdl(profile)
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
    ):
        """Sample activity, channel events, and slot metadata."""
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        enabled = self.scenario_training if enabled is None else bool(enabled)
        channel_mask = np.zeros(
            [batch_size, seq_len, self.ue_pool_size], dtype=bool
        )
        if controlled_change_tb is not None:
            if enabled:
                raise ValueError(
                    "controlled changes cannot use random scheduling gaps"
                )
            channel_mask = controlled_change_mask(
                batch_size,
                seq_len,
                self.ue_pool_size,
                controlled_change_tb,
            )

        base_ebno = np.asarray(ebno_db, dtype=np.float32)
        if base_ebno.ndim == 0:
            base_ebno = np.full([batch_size], float(base_ebno), np.float32)
        if list(base_ebno.shape) != [batch_size]:
            raise ValueError("ebno_db must be scalar or have shape [batch_size]")
        active = sample_activity_mask(
            batch_size,
            seq_len,
            self.num_scheduled_tx,
            self.gap_start_prob if enabled else 0.0,
            self.max_gap_slots,
            rng=self.schedule_rng,
        )
        channel_event = channel_mask.copy()
        channel_event[:, 1:] = (
            channel_mask[:, 1:] != channel_mask[:, :-1]
        )
        channel_event[:, 0] = False
        slot_numbers = self._slot_numbers(seq_len)
        return {
            "active": tf.convert_to_tensor(active, tf.float32),
            "channel_regime": tf.convert_to_tensor(channel_mask, tf.bool),
            "channel_change_event": tf.convert_to_tensor(channel_event, tf.bool),
            "ebno_db": tf.convert_to_tensor(np.broadcast_to(base_ebno[:, None], (batch_size, seq_len)), tf.float32),
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
        for b in range(batch_size):
            universe = self.schedule_rng.choice(
                full_universe, size=pool, replace=False
            )
            switch_prob = self.schedule_switch_prob
            reorder_prob = self.schedule_reorder_prob
            current = self.schedule_rng.choice(
                universe, size=n, replace=False
            )
            schedule[b, 0] = current

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
        else:
            raise ValueError(f"unsupported family: {profile.family}")
        a, tau = channel(
            batch_size, seq_len * num_sym, sampling_frequency
        )
        h = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
        slots = tf.stack(tf.split(h, seq_len, axis=5), axis=1)
        return slots

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        """Sample continuous per-UE channel trajectories."""
        steady_indices = self._assignment_indices(
            len(self.channel_profiles)
        )
        event_indices = self._assignment_indices(
            len(self.event_channel_profiles)
        )
        h_users = []
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
            steady = self._sample_channel_slots(
                steady_profile,
                self.steady_channels[steady_index],
                batch_size,
                seq_len,
            )
            if channel_regime is not None:
                harsh = self._sample_channel_slots(
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
            seed=tf.stack([tf.cast(seed, tf.int32), tf.cast(tb, tf.int32)]),
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
            seed=tf.stack([tf.cast(seed, tf.int32), 2 * tf.cast(tb, tf.int32)]),
            dtype=real_dtype,
        )
        imag = tf.random.stateless_normal(
            shape,
            seed=tf.stack([tf.cast(seed, tf.int32), 2 * tf.cast(tb, tf.int32) + 1]),
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
        scenario_training=None,
        controlled_change_tb=None,
        include_clean_received=False,
        include_ls=True,
    ):
        """Generate a full temporal training batch."""
        batch_seeds = self._reserve_seed_batch(batch_size)
        self._activate_seed_batch(batch_seeds)
        ue_ids = self.sample_schedule(batch_size, seq_len)

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
        )
        h_pool = self.sample_pool_channel(
            batch_size,
            seq_len,
            (
                scenario["channel_regime"]
                if controlled_change_tb is not None
                else None
            ),
        )
        h = self.gather_scheduled_channel(h_pool, ue_ids)

        bits = []
        received = []
        clean_received = []
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
            "channel_regime": scenario["channel_regime"],
            "channel_change_event": scenario["channel_change_event"],
        }
        if include_ls:
            result["ls"] = tf.stack(ls_estimates, axis=1)
        if include_clean_received:
            result["y_clean"] = tf.stack(clean_received, axis=1)
        return result
