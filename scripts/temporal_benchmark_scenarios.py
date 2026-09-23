#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from geometry_mumimo_data import GeometryAwareMUMIMOGenerator
from temporal_benchmark_runtime import BenchmarkEpisode
from temporal_generalization_protocol import ChannelProfileSpec
from temporal_receiver_core import expand_noise, make_snr_batch
from temporal_training_data import TemporalTrainingDataGenerator


class PowerControlledGeometryGenerator(GeometryAwareMUMIMOGenerator):

    def __init__(self, *args, received_power_gap_db: float, **kwargs):
        self.received_power_gap_db = float(received_power_gap_db)
        if self.received_power_gap_db <= 0.0:
            raise ValueError("received-power gap must be positive")
        super().__init__(*args, **kwargs)

    def sample_pool_channel(self, batch_size, seq_len, channel_regime=None):
        h = super().sample_pool_channel(batch_size, seq_len, channel_regime)
        power = self.tf.reduce_mean(
            self.tf.abs(h) ** 2, axis=(2, 3, 5, 6, 7), keepdims=True
        )
        normalized = self.tf.math.divide_no_nan(
            h, self.tf.cast(self.tf.sqrt(power), h.dtype)
        )
        scales = np.ones(self.ue_pool_size, np.float32)
        scales[1] = 10.0 ** (-self.received_power_gap_db / 20.0)
        shape = [1, 1, 1, 1, self.ue_pool_size, 1, 1, 1]
        return normalized * self.tf.cast(
            self.tf.reshape(scales, shape), normalized.dtype
        )


class TemporalBenchmarkScenarioBank:

    def __init__(self, parameters, e2e_model, tf_module):
        self.parameters = parameters
        self.e2e_model = e2e_model
        self.receiver = e2e_model._receiver
        self.tf = tf_module
        self.generators = {}

    def _seed_streams(self, job: dict, episode_index: int) -> dict:
        return {
            name: int(value) + int(episode_index)
            for name, value in job["seed_streams"].items()
        }

    def _common_kwargs(self, job: dict, episode_index: int) -> dict:
        return {
            "seed_streams": self._seed_streams(job, episode_index),
            "seed_stream_limit": 1,
            "channel_role": "calibration",
            "random_slot_offset": True,
            "advance_slot_pilots": True,
            "channel_change_prob": 0.0,
            "ebno_drop_prob": 0.0,
            "hard_reset_prob": 0.0,
        }

    @staticmethod
    def _tdl_profile(settings: dict) -> ChannelProfileSpec:
        model = settings["profile"]
        delay = {"B100": 100e-9, "C300": 300e-9}[model]
        low, high = settings["doppler_hz"]
        return ChannelProfileSpec(
            name=f"TDL-{model}",
            family="tdl",
            model=model,
            min_delay_spread_s=delay,
            max_delay_spread_s=delay,
            min_doppler_hz=float(low),
            max_doppler_hz=float(high),
        )

    def _geometry_range(
        self, settings: dict, episode_index: int
    ) -> tuple[float, float]:
        strata = settings.get("doppler_strata_hz", [[0.0, 400.0]])
        low, high = strata[int(episode_index) % len(strata)]
        return float(low), float(high)

    def _make_generator(self, job: dict, episode_index: int):
        provider = job["provider"]
        settings = job["provider_settings"]
        common = self._common_kwargs(job, episode_index)
        if provider == "independent_tdl":
            profile = self._tdl_profile(settings)
            return TemporalTrainingDataGenerator(
                self.parameters,
                self.e2e_model,
                ue_pool_size=2,
                dynamic_scheduling=False,
                scenario_training=False,
                channel_profiles=(profile,),
                event_channel_profiles=(profile,),
                **common,
            )
        low, high = self._geometry_range(settings, episode_index)
        geometry = {
            "geometry_scenario": settings["scenario"],
            "force_los": settings.get("force_los"),
            "min_doppler_hz": low,
            "max_doppler_hz": high,
        }
        separation = settings.get("angular_separation_deg", [0.0, 180.0])
        geometry["min_azimuth_separation_deg"] = float(separation[0])
        geometry["max_azimuth_separation_deg"] = float(separation[1])
        scheduling = provider == "shared_geometry_schedule"
        base = {
            **common,
            **geometry,
            "ue_pool_size": int(settings.get("ue_pool_size", 2)),
            "dynamic_scheduling": scheduling,
            "scenario_training": scheduling,
            "schedule_switch_prob": float(
                settings.get("schedule_switch_probability", 0.0)
            ),
            "schedule_reorder_prob": float(
                settings.get("schedule_reorder_probability", 0.0)
            ),
            "gap_start_prob": float(
                settings.get("gap_start_probability", 0.0)
            ),
            "max_gap_slots": int(settings.get("maximum_gap_tbs", 4)),
        }
        if provider in (
            "shared_geometry",
            "shared_geometry_change",
            "shared_geometry_schedule",
        ):
            return GeometryAwareMUMIMOGenerator(
                self.parameters, self.e2e_model, **base
            )
        if provider == "shared_geometry_power":
            gaps = settings["received_power_gap_db"]
            gap = float(gaps[int(episode_index) % len(gaps)])
            generator = PowerControlledGeometryGenerator(
                self.parameters,
                self.e2e_model,
                received_power_gap_db=gap,
                **base,
            )
            generator.tf = self.tf
            return generator
        raise ValueError(f"unknown scenario provider {provider}")

    def _generator(self, job: dict, episode_index: int):
        settings = job["provider_settings"]
        strata = settings.get("doppler_strata_hz", [[0.0, 400.0]])
        stratum = int(episode_index) % len(strata)
        gaps = settings.get("received_power_gap_db", [0.0])
        gap = int(episode_index) % len(gaps)
        key = (job["condition"], stratum, gap)
        if key not in self.generators:
            self.generators[key] = self._make_generator(job, episode_index)
        generator = self.generators[key]
        generator.seed_streams = self._seed_streams(job, episode_index)
        generator.seed_stream_consumed = 0
        generator.seed_stream_limit = 1
        return generator

    def generate(self, job: dict, episode_index: int) -> BenchmarkEpisode:
        if not 0 <= int(episode_index) < int(job["episode_count"]):
            raise ValueError("episode index is outside the job")
        generator = self._generator(job, episode_index)
        settings = job["provider_settings"]
        controlled_change = (
            settings.get("change_tb")
            if job["provider"] == "shared_geometry_change"
            else None
        )
        scenario_training = job["provider"] == "shared_geometry_schedule"
        snrs = [float(value) for value in job["snr_db"]]
        base = generator.sample_batch(
            1,
            int(job["num_tbs"]),
            snrs[0],
            scenario_training=scenario_training,
            controlled_change_tb=controlled_change,
            include_clean_received=True,
            include_transmitted=True,
            include_ls=False,
        )
        scale = self.tf.cast(
            self.tf.sqrt(expand_noise(base["no"], base["y_clean"])),
            self.tf.complex64,
        )
        unit_noise = self.tf.math.divide_no_nan(
            base["y"] - base["y_clean"], scale
        )
        batches = [
            make_snr_batch(
                base, unit_noise, snr, generator, self.receiver
            )
            for snr in snrs
        ]
        active = base["active"].numpy()[0].astype(bool)
        bits = base["bits"].numpy()[0]
        ue_ids = base["ue_ids"].numpy()[0].astype(np.int32)
        pool_change = base["channel_change_event"].numpy()[0].astype(bool)
        scheduled_change = np.take_along_axis(pool_change, ue_ids, axis=1)
        receiver_inputs = []
        oracle_inputs = []
        for batch in batches:
            point_receiver = []
            point_oracle = []
            for tb in range(int(job["num_tbs"])):
                inputs = {
                    "active": active[tb],
                    "ls_channel": batch["ls"].numpy()[0, tb],
                    "noise_variance": batch["no"].numpy()[0, tb],
                    "received_grid": batch["y"].numpy()[0, tb],
                    "slot_number": np.asarray(
                        batch["slot_number"].numpy()[tb], np.int32
                    ),
                    "ue_ids": batch["ue_ids"].numpy()[0, tb],
                }
                point_receiver.append(inputs)
                point_oracle.append({
                    **inputs,
                    "perfect_channel": batch["h"].numpy()[0, tb],
                })
            receiver_inputs.append(tuple(point_receiver))
            oracle_inputs.append(tuple(point_oracle))
        return BenchmarkEpisode(
            bits=bits,
            active=active,
            ue_ids=ue_ids,
            channel_change_event=scheduled_change,
            receiver_inputs=tuple(receiver_inputs),
            oracle_inputs=tuple(oracle_inputs),
        )
