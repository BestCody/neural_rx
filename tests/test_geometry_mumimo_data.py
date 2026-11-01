#!/usr/bin/env python3

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from geometry_mumimo_data import (
    GeometryAwareMUMIMOGenerator,
    azimuth_geometry,
)
from verification.radio import exact_contributions, received_to_grid
from utils import E2E_Model, Parameters


def seeds(base):
    return {
        "channel": base,
        "payload": base + 100_000,
        "noise": base + 200_000,
        "schedule": base + 300_000,
    }


class GeometryMUMIMODataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = Parameters(
            "nrx_large.cfg",
            training=True,
            num_tx_eval=2,
            n_size_bwp_override=4,
        )
        cls.e2e = E2E_Model(cls.parameters, training=True)

    def make_generator(self, base, **kwargs):
        return GeometryAwareMUMIMOGenerator(
            self.parameters,
            self.e2e,
            ue_pool_size=2,
            dynamic_scheduling=False,
            scenario_training=False,
            gap_start_prob=0.0,
            random_slot_offset=True,
            advance_slot_pilots=True,
            seed_streams=seeds(base),
            seed_stream_limit=1,
            **kwargs,
        )

    def test_azimuth_wrap_and_minimum(self):
        ut = np.asarray([[[1.0, 0.0, 1.5], [0.0, -1.0, 1.5]]])
        bs = np.asarray([[[0.0, 0.0, 10.0]]])
        angle, separation = azimuth_geometry(ut, bs)
        np.testing.assert_allclose(angle, [[0.0, 270.0]])
        np.testing.assert_allclose(separation, [90.0])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "azimuth"):
            self.make_generator(
                31_000_000,
                min_azimuth_separation_deg=30.0,
                max_azimuth_separation_deg=10.0,
            )
        with self.assertRaisesRegex(ValueError, "umi or uma"):
            self.make_generator(31_000_000, geometry_scenario="tdl")

    def test_joint_los_sequence_and_reconstruction(self):
        generator = self.make_generator(
            31_000_000,
            force_los=True,
            min_azimuth_separation_deg=20.0,
            max_azimuth_separation_deg=100.0,
            min_doppler_hz=0.0,
            max_doppler_hz=0.0,
        )
        batch = generator.sample_batch(
            1,
            3,
            4.0,
            include_clean_received=True,
            include_ls=False,
        )
        self.assertEqual(batch["h"].shape, (1, 3, 1, 4, 2, 2, 14, 48))
        self.assertEqual(batch["y"].shape, (1, 3, 1, 4, 14, 48))
        self.assertTrue(np.all(batch["geometry_steady_los"].numpy()))
        separation = float(
            batch["geometry_steady_minimum_azimuth_separation_deg"].numpy()[0]
        )
        self.assertGreaterEqual(separation, 20.0)
        self.assertLessEqual(separation, 100.0)
        locations = batch["geometry_steady_ut_location_m"].numpy()[0]
        self.assertFalse(np.array_equal(locations[0], locations[1]))
        np.testing.assert_allclose(
            batch["h"][:, 0].numpy(),
            batch["h"][:, 2].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        for tb in range(3):
            generator._set_slot_pilots(batch["slot_number"][tb])
            transmitted = generator.tx(batch["bits"][:, tb])
            transmitted *= tf.cast(batch["active"][:, tb, :, None, None, None], transmitted.dtype)
            self.assertEqual(transmitted.shape, (1, 2, 2, 14, 48))
            contribution = exact_contributions(
                batch["h"][:, tb], transmitted
            )
            reconstructed = tf.reduce_sum(contribution, axis=1)
            clean = received_to_grid(batch["y_clean"][:, tb])
            np.testing.assert_allclose(
                reconstructed.numpy(),
                clean.numpy(),
                rtol=2e-5,
                atol=2e-5,
            )

    def test_moving_channel_evolves_continuously(self):
        generator = self.make_generator(
            34_000_100,
            force_los=True,
            min_azimuth_separation_deg=20.0,
            max_azimuth_separation_deg=100.0,
            min_doppler_hz=100.0,
            max_doppler_hz=100.0,
        )
        batch = generator.sample_batch(1, 3, 4.0, include_ls=False)
        first = batch["h"][:, 0].numpy().reshape(-1)
        second = batch["h"][:, 1].numpy().reshape(-1)
        last = batch["h"][:, 2].numpy().reshape(-1)
        correlation = abs(np.vdot(first, second)) / np.sqrt(
            np.vdot(first, first).real * np.vdot(second, second).real
        )
        self.assertGreater(correlation, 0.8)
        self.assertGreater(np.max(np.abs(first - last)), 1e-5)

    def test_natural_mode_records_realized_state(self):
        batch = self.make_generator(
            35_000_000,
            force_los=None,
            min_doppler_hz=0.0,
            max_doppler_hz=0.0,
        ).sample_batch(1, 1, 4.0, include_ls=False)
        self.assertEqual(int(batch["geometry_force_los"].numpy()), -1)
        self.assertEqual(batch["geometry_steady_los"].shape, (1, 2))
        self.assertEqual(
            batch["geometry_steady_azimuth_deg"].shape, (1, 2)
        )

    def test_independent_generators_reproduce_episode(self):
        settings = {
            "force_los": True,
            "min_azimuth_separation_deg": 20.0,
            "max_azimuth_separation_deg": 100.0,
            "min_doppler_hz": 0.0,
            "max_doppler_hz": 100.0,
        }
        first = self.make_generator(32_000_000, **settings).sample_batch(
            1, 2, 4.0, include_clean_received=True, include_ls=False
        )
        second = self.make_generator(32_000_000, **settings).sample_batch(
            1, 2, 4.0, include_clean_received=True, include_ls=False
        )
        for name in (
            "bits",
            "h",
            "y",
            "y_clean",
            "slot_number",
            "geometry_steady_ut_location_m",
            "geometry_steady_ut_velocity_mps",
            "geometry_steady_los",
        ):
            np.testing.assert_array_equal(
                first[name].numpy(), second[name].numpy()
            )

    def test_scheduling_gaps_do_not_generate_a_channel_change(self):
        generator = self.make_generator(32_500_000)
        generator.gap_start_prob = 1.0
        with patch.object(generator, "_sample_joint_slots", wraps=generator._sample_joint_slots) as sample:
            batch = generator.sample_batch(1, 4, 4.0, scenario_training=True, include_ls=False)
        self.assertEqual(sample.call_count, 1)
        self.assertFalse(np.all(batch["active"].numpy()))
        self.assertFalse(np.any(batch["channel_change_event"].numpy()))
        self.assertNotIn("geometry_event_ut_location_m", batch)
        generator.seed_stream_consumed = 0
        with patch.object(generator, "_sample_joint_slots", wraps=generator._sample_joint_slots) as sample:
            batch = generator.sample_batch(1, 4, 4.0, controlled_change_tb=3, include_ls=False)
        self.assertEqual(sample.call_count, 2)
        self.assertTrue(np.any(batch["channel_change_event"].numpy()))

    def test_forced_nlos_uma_is_joint(self):
        generator = self.make_generator(
            33_000_000,
            geometry_scenario="uma",
            force_los=False,
            min_doppler_hz=0.0,
            max_doppler_hz=0.0,
        )
        batch = generator.sample_batch(
            1, 1, 4.0, include_clean_received=True, include_ls=False
        )
        self.assertFalse(np.any(batch["geometry_steady_los"].numpy()))
        self.assertEqual(batch["h"].shape[4], 2)
        self.assertEqual(batch["geometry_scenario"].numpy(), b"uma")

    def test_compiled_renderer_matches_eager_generator(self):
        streams = seeds(35_700_000)
        generator = self.make_generator(35_700_000, min_doppler_hz=0.0, max_doppler_hz=100.0)
        batch = generator.sample_batch(1,3,4.0,include_ls=True)
        from pilot_history_receiver.training_render import TrainingRenderer
        rendered = TrainingRenderer(generator)(batch["h"],batch["active"],batch["ue_ids"],
            batch["slot_number"],batch["ebno_db"],streams["payload"],streams["noise"])
        np.testing.assert_array_equal(batch["bits"],rendered["bits"])
        np.testing.assert_array_equal(batch["no"],rendered["no"])
        for name in ("y","ls"):
            np.testing.assert_allclose(batch[name],rendered[name],rtol=2e-5,atol=2e-6)



if __name__ == "__main__":
    unittest.main()
