#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.channel import cir_to_ofdm_channel, gen_single_sector_topology
from sionna.channel.tr38901 import PanelArray, UMa, UMi

from temporal_generalization_protocol import ChannelProfileSpec
from temporal_training_data import TemporalTrainingDataGenerator


def azimuth_geometry(ut_loc, bs_loc):
    ut_loc = np.asarray(ut_loc, np.float64)
    bs_loc = np.asarray(bs_loc, np.float64)
    if ut_loc.ndim != 3 or ut_loc.shape[-1] != 3:
        raise ValueError("UT locations must have shape [batch,UE,3]")
    if bs_loc.shape != (ut_loc.shape[0], 1, 3):
        raise ValueError("BS locations must have shape [batch,1,3]")
    delta = ut_loc[..., :2] - bs_loc[:, :, :2]
    angles = np.mod(np.degrees(np.arctan2(
        delta[..., 1], delta[..., 0]
    )), 360.0)
    users = ut_loc.shape[1]
    if users < 2:
        return angles, np.full([ut_loc.shape[0]], 180.0)
    difference = np.abs(angles[:, :, None] - angles[:, None, :])
    difference = np.minimum(difference, 360.0 - difference)
    diagonal = np.arange(users)
    difference[:, diagonal, diagonal] = np.inf
    return angles, np.min(difference, axis=(1, 2))


class GeometryAwareMUMIMOGenerator(TemporalTrainingDataGenerator):

    def __init__(
        self,
        sys_parameters,
        e2e_model,
        ue_pool_size=2,
        geometry_scenario="umi",
        force_los=None,
        min_azimuth_separation_deg=0.0,
        max_azimuth_separation_deg=180.0,
        min_doppler_hz=0.0,
        max_doppler_hz=400.0,
        topology_draw_limit=64,
        **kwargs,
    ):
        scenario = str(geometry_scenario).lower()
        if scenario not in ("umi", "uma"):
            raise ValueError("geometry scenario must be umi or uma")
        if force_los is not None and not isinstance(force_los, bool):
            raise ValueError("force_los must be true, false, or none")
        low = float(min_azimuth_separation_deg)
        high = float(max_azimuth_separation_deg)
        if not 0.0 <= low <= high <= 180.0:
            raise ValueError("invalid azimuth-separation interval")
        if int(topology_draw_limit) < 1:
            raise ValueError("topology draw limit must be positive")
        forbidden = {
            "channel_profiles",
            "event_channel_profiles",
            "randomize_profile_assignment",
        }
        overlap = forbidden.intersection(kwargs)
        if overlap:
            raise ValueError("geometry profiles cannot be overridden")
        self.geometry_scenario = scenario
        self.force_los = force_los
        self.min_azimuth_separation_deg = low
        self.max_azimuth_separation_deg = high
        self.topology_draw_limit = int(topology_draw_limit)
        self.last_geometry = None
        profile = ChannelProfileSpec(
            name=f"Geometry-{scenario.upper()}",
            family="umi",
            model="UMi",
            min_delay_spread_s=None,
            max_delay_spread_s=None,
            min_doppler_hz=float(min_doppler_hz),
            max_doppler_hz=float(max_doppler_hz),
        )
        super().__init__(
            sys_parameters,
            e2e_model,
            ue_pool_size=ue_pool_size,
            channel_profiles=(profile,),
            event_channel_profiles=(profile,),
            randomize_profile_assignment=False,
            **kwargs,
        )

    def _make_umi(self, profile):
        carrier = self.p.carrier_frequency
        antennas = int(self.p.num_rx_antennas)
        if antennas == 1:
            columns = 1
            polarization = "single"
            polarization_type = "V"
        elif antennas % 2 == 0:
            columns = antennas // 2
            polarization = "dual"
            polarization_type = "cross"
        else:
            raise ValueError("receive antennas must be one or even")
        bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=columns,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern="38.901",
            carrier_frequency=carrier,
        )
        ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.p.num_antenna_ports,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier,
        )
        model = UMi if self.geometry_scenario == "umi" else UMa
        return model(
            carrier_frequency=carrier,
            o2i_model="low",
            bs_array=bs_array,
            ut_array=ut_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
        )

    def _topology_candidates(self, count, profile):
        speed_scale = sn.SPEED_OF_LIGHT / self.p.carrier_frequency
        return gen_single_sector_topology(
            int(count),
            self.ue_pool_size,
            self.geometry_scenario,
            min_ut_velocity=profile.min_doppler_hz * speed_scale,
            max_ut_velocity=profile.max_doppler_hz * speed_scale,
            indoor_probability=0.0,
        )

    def _draw_topology(self, batch_size, profile):
        batch_size = int(batch_size)
        natural = (
            self.min_azimuth_separation_deg == 0.0
            and self.max_azimuth_separation_deg == 180.0
        )
        if natural:
            return self._topology_candidates(batch_size, profile)
        selected = [[] for _ in range(6)]
        count = 0
        dtypes = None
        for _ in range(self.topology_draw_limit):
            candidate_count = max(8, 4 * (batch_size - count))
            candidates = self._topology_candidates(
                candidate_count, profile
            )
            arrays = [value.numpy() for value in candidates]
            dtypes = [value.dtype for value in candidates]
            _, separation = azimuth_geometry(arrays[0], arrays[1])
            valid = np.flatnonzero(
                (separation >= self.min_azimuth_separation_deg)
                & (separation <= self.max_azimuth_separation_deg)
            )
            take = valid[:batch_size - count]
            for output, value in zip(selected, arrays):
                if len(take):
                    output.append(value[take])
            count += len(take)
            if count == batch_size:
                break
        if count != batch_size or dtypes is None:
            raise RuntimeError("requested geometry interval was not sampled")
        return tuple(
            tf.convert_to_tensor(np.concatenate(parts, axis=0), dtype=dtype)
            for parts, dtype in zip(selected, dtypes)
        )

    def _set_joint_topology(self, channel, topology):
        channel.set_topology(*topology, los=self.force_los)
        ut_loc, bs_loc, ut_orientation, bs_orientation, velocity, indoor = (
            topology
        )
        azimuth, separation = azimuth_geometry(
            ut_loc.numpy(), bs_loc.numpy()
        )
        scenario = channel._scenario
        return {
            "ut_location_m": tf.identity(ut_loc),
            "bs_location_m": tf.identity(bs_loc),
            "ut_orientation_rad": tf.identity(ut_orientation),
            "bs_orientation_rad": tf.identity(bs_orientation),
            "ut_velocity_mps": tf.identity(velocity),
            "indoor": tf.identity(indoor),
            "azimuth_deg": tf.convert_to_tensor(azimuth, tf.float32),
            "minimum_azimuth_separation_deg": tf.convert_to_tensor(
                separation, tf.float32
            ),
            "los": tf.identity(scenario.los[:, 0]),
            "los_aoa_deg": tf.identity(scenario.los_aoa[:, 0]),
        }

    def _sample_joint_slots(self, channel, batch_size, seq_len):
        symbols = int(self.rg.num_ofdm_symbols)
        sampling_frequency = 1.0 / self.rg.ofdm_symbol_duration
        a, tau = channel(
            int(batch_size), int(seq_len) * symbols, sampling_frequency
        )
        h = cir_to_ofdm_channel(self.freqs, a, tau, normalize=False)
        if int(h.shape[3]) != self.ue_pool_size:
            raise RuntimeError("joint channel UE dimension differs")
        return tf.stack(tf.split(h, int(seq_len), axis=5), axis=1)

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        profile = self.channel_profiles[0]
        steady_topology = self._draw_topology(batch_size, profile)
        steady_metadata = self._set_joint_topology(
            self.steady_channels[0], steady_topology
        )
        steady = self._sample_joint_slots(
            self.steady_channels[0], batch_size, seq_len
        )
        event_metadata = None
        if channel_regime is not None:
            event_topology = self._draw_topology(batch_size, profile)
            event_metadata = self._set_joint_topology(
                self.event_channels[0], event_topology
            )
            event = self._sample_joint_slots(
                self.event_channels[0], batch_size, seq_len
            )
            mask = tf.reshape(
                channel_regime,
                [int(batch_size), int(seq_len), 1, 1,
                 self.ue_pool_size, 1, 1, 1],
            )
            steady = tf.where(mask, event, steady)
        self.last_geometry = {
            "steady": steady_metadata,
            "event": event_metadata,
        }
        self._update_profile_metadata(steady_metadata, event_metadata)
        return steady

    def _update_profile_metadata(self, steady, event):
        locations = steady["ut_location_m"].numpy()[0]
        azimuth = steady["azimuth_deg"].numpy()[0]
        los = steady["los"].numpy()[0]
        event_locations = (
            None if event is None else event["ut_location_m"].numpy()[0]
        )
        self.ue_profiles = []
        for ue_id in range(self.ue_pool_size):
            self.ue_profiles.append({
                "ue_id": ue_id,
                "split_role": self.channel_role,
                "steady_family": self.geometry_scenario,
                "steady_model": self.geometry_scenario.upper(),
                "steady_position_m": locations[ue_id].tolist(),
                "steady_azimuth_deg": float(azimuth[ue_id]),
                "steady_los": bool(los[ue_id]),
                "harsh_family": self.geometry_scenario,
                "harsh_model": self.geometry_scenario.upper(),
                "harsh_position_m": (
                    None
                    if event_locations is None
                    else event_locations[ue_id].tolist()
                ),
            })

    def sample_batch(self, *args, **kwargs):
        result = super().sample_batch(*args, **kwargs)
        if self.last_geometry is None:
            raise RuntimeError("joint geometry metadata is unavailable")
        for phase, metadata in self.last_geometry.items():
            if metadata is None:
                continue
            for name, value in metadata.items():
                result[f"geometry_{phase}_{name}"] = value
        result["geometry_scenario"] = tf.convert_to_tensor(
            self.geometry_scenario
        )
        result["geometry_force_los"] = tf.convert_to_tensor(
            -1 if self.force_los is None else int(self.force_los),
            tf.int8,
        )
        return result
