from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern, ResourceGrid
from sionna.nr import PUSCHLSChannelEstimator

from neural_rx.receiver.model import PilotHistoryReceiver, architecture_settings
from neural_rx.receiver.pilot_layout import PilotLayout
from neural_rx.receiver.pilot_history import PilotHistory
from neural_rx.backbone.cgnn import CGNN


def pilot_layout(prbs=1, dmrs_length=1):
    mask = np.zeros([2, 1, 4, prbs*12], bool)
    mask[:, :, (0, 3) if dmrs_length == 1 else (0, 1)] = True
    pilots = np.zeros([2, 1, 24*prbs], np.complex64)
    pilots[0, :, ::2] = 1.
    pilots[1, :, 1::2] = 1.
    grid = ResourceGrid(num_ofdm_symbols=4, fft_size=prbs*12, subcarrier_spacing=30000,
                        num_tx=2, num_streams_per_tx=1, pilot_pattern=PilotPattern(mask, pilots))
    estimator = PUSCHLSChannelEstimator(grid, dmrs_length, 1 if dmrs_length == 1 else 0, 2)
    return PilotLayout(estimator, grid, 1), estimator


def inputs(prbs=1):
    layout = pilot_layout(prbs)[0]
    channel = layout.expand(tf.random.normal([1, 2, layout.count, 8]))
    return [tf.random.normal([1, 12*prbs, 4, 8]), tf.random.uniform([2, 12*prbs, 4, 2]),
            channel, tf.ones([1, 2]), tf.ones([1, 2, 1])]


def singleton_group_layout():
    layout = PilotLayout.__new__(PilotLayout)
    layout.users, layout.frequency, layout.symbols = layout.key = (2, 2, 2)
    layout.count = 3
    layout.grid_indices = np.array([[0, 1, 1, 2], [2, 2, 1, 0]], np.int32)
    layout.group_indices = np.array([[[0, 1], [1, 2], [3, 3]], [[3, 0], [2, 3], [0, 1]]], np.int32)
    layout.group_valid = np.array([[[1, 0], [1, 1], [1, 0]], [[1, 0], [1, 0], [1, 1]]], bool)
    layout.group_sizes = layout.group_valid.sum(-1).astype(np.int32)
    return layout


def build_model(prbs=1):
    base = CGNN(num_bits_per_symbol=[4], num_rx_ant=4, num_it=2, d_s=56,
        num_units_init=[128, 128], num_units_agg=[[64], [64]],
        num_units_state=[[128, 128], [128, 128]], num_units_readout=[128],
        layer_type_dense="dense", layer_type_conv="sepconv", layer_type_readout="dense", training=False)
    base(inputs(prbs))
    return PilotHistoryReceiver(base, pilot_layout(prbs)[0])


def valid_history(model, sample, noise):
    obs = model.observe(sample[2], noise)
    history = model.empty_history(obs)
    return PilotHistory(tf.repeat(obs[:, :, None], model.history_tbs, 2),
        tf.broadcast_to(tf.range(1, model.history_tbs+1)[None, None], history.age.shape),
        tf.broadcast_to(tf.range(2)[None, :, None], history.ports.shape), tf.ones_like(history.valid))


def records(channel, noise_variance, rng):
    """Packed estimates [1, users, cells, 2*antennas+1] of a channel [users, cells, antennas] with LS noise."""
    noise = (rng.normal(size=channel.shape)+1j*rng.normal(size=channel.shape))*np.sqrt(noise_variance/2)
    h = channel+noise
    packed = np.concatenate([h.real, h.imag, np.full([*h.shape[:2], 1], noise_variance)], -1)
    return tf.constant(packed[None], tf.float32)


def history_of(model, entries):
    """A memory holding `entries` (newest first), all received on the UE's current port."""
    observations = tf.stack(entries, 2)
    shape = observations.shape[:3]
    return PilotHistory(observations, tf.broadcast_to(tf.range(1, shape[2]+1)[None, None], shape),
                        tf.broadcast_to(tf.range(shape[1])[None, :, None], shape), tf.ones(shape, tf.bool))


class ChannelFilterTest(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)
        self.tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, self.tf32)

    def test_architecture_requires_the_pilot_filter(self):
        settings = json.loads((Path(__file__).resolve().parents[1]/"config/memory_receiver.json").read_text())
        architecture_settings(settings)
        settings["architecture"]["memory"] = "pilot_history_attention"
        with self.assertRaisesRegex(ValueError, "current pilot-history receiver"):
            architecture_settings(settings)

    def test_packing_preserves_sionna_ls_and_noise_for_both_cdm_lengths(self):
        for length in (1, 2):
            layout, estimator = pilot_layout(3, length)
            y = tf.complex(tf.random.normal([2, 1, 4, 4, 36]), tf.random.normal([2, 1, 4, 4, 36]))
            noise = tf.constant([.1, .8])
            h, variance = estimator([y, noise])
            h = tf.transpose(h[:, 0, :, :, 0], [0, 2, 4, 3, 1])
            channel = tf.concat([tf.math.real(h), tf.math.imag(h)], -1)
            observations = layout.pack(channel, noise)
            np.testing.assert_array_equal(layout.expand(observations[..., :-1]), channel)
            expanded = layout.expand(observations[..., -1:])
            expected = tf.transpose(variance[:, 0, 0, :, 0], [0, 1, 3, 2])[..., None]
            np.testing.assert_allclose(expanded, expected, atol=1e-7)
            self.assertEqual(layout.count, 18 if length == 1 else 9)
            rotated = tf.concat([-channel[..., 4:], channel[..., :4]], -1)
            actual = layout.pack(rotated, noise)
            np.testing.assert_array_equal(actual[..., :4], -observations[..., 4:8])
            np.testing.assert_array_equal(actual[..., 4:8], observations[..., :4])

    def test_neighbors_use_physical_coordinates_when_pilot_ports_change(self):
        layout, _ = pilot_layout(2)
        records = tf.reshape(tf.range(2*2*layout.count, dtype=tf.float32), [1, 2, 2, layout.count, 1])
        ports = tf.constant([[[1, 0], [0, 1]]])
        selected, valid, delta = layout.neighbors(records, ports)
        for user in range(2):
            for old in range(2):
                source = int(ports[0, user, old])
                for i in range(layout.count):
                    for j in np.flatnonzero(valid[0, user, old, i]):
                        index = layout.neighbor_indices[user, source, i, j]
                        self.assertEqual(selected[0, user, old, i, j, 0], records[0, user, old, index, 0])
                        self.assertAlmostEqual(float(delta[0,user,old,i,j,1]),
                            float((layout.coordinates[source,index,1]-layout.coordinates[user,i,1])*.3), places=5)   # 30 kHz, in 100 kHz

    def test_expansion_derivative_is_a_masked_sum_and_adjoint(self):
        rng = np.random.RandomState(207)
        for batch, features, layout in ((1, 1, pilot_layout(2, 1)[0]), (3, 8, pilot_layout(2, 2)[0]),
                                        (2, 112, pilot_layout(2, 1)[0]), (1, 5, singleton_group_layout())):
            records = tf.constant(rng.normal(size=[batch, 2, layout.count, features]), tf.float32)
            incoming = tf.constant(rng.normal(size=[batch, *layout.key, features]), tf.float32)
            @tf.function(jit_compile=True)
            def derivative(records):
                with tf.GradientTape() as tape:
                    tape.watch(records)
                    loss = tf.reduce_sum(layout.expand(records)*incoming)
                return loss, tape.gradient(loss, records)
            loss, gradient = derivative(records)
            flat = np.asarray(incoming).reshape(batch, 2, -1, features)
            expected = np.stack([np.stack([flat[:, u, layout.grid_indices[u] == g].sum(1)
                for g in range(layout.count)], 1) for u in range(2)], 1)
            np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(loss, tf.reduce_sum(records*gradient), rtol=2e-6, atol=2e-5)

    def test_solver_matches_the_library_solver(self):
        from neural_rx.receiver.channel_filter import solve_positive_definite
        rng = np.random.RandomState(3)
        a = rng.normal(size=[2, 3, 40, 30])+1j*rng.normal(size=[2, 3, 40, 30])
        matrix = np.einsum('...ki,...kj->...ij', a.conj(), a)
        matrix += 1e-3*np.trace(matrix, axis1=-2, axis2=-1)[..., None, None]/30*np.eye(30)
        rhs = rng.normal(size=[2, 3, 30])+1j*rng.normal(size=[2, 3, 30])
        actual = solve_positive_definite(tf.constant(matrix, tf.complex64), tf.constant(rhs, tf.complex64))
        np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs[..., None])[..., 0], rtol=2e-3, atol=2e-5)

    def test_filter_has_no_weights_and_passes_no_gradient_to_its_inputs(self):
        model, sample, noise = build_model(2), inputs(2), tf.constant([.1])
        self.assertFalse(hasattr(model.channel_filter, "trainable_variables"))
        layout = model.layout_for(sample[2])
        observation = model.observe(sample[2], noise)
        history = valid_history(model, inputs(2), noise)
        with tf.GradientTape() as tape:
            tape.watch([observation, history.observations])
            coefficients, channel = model.prepare(sample, observation, history, layout)
            loss = tf.reduce_mean(tf.square(channel))+tf.reduce_mean(tf.square(tf.abs(coefficients)))
        self.assertTrue(np.isfinite(channel).all())
        self.assertEqual(tape.gradient(loss, [observation, history.observations]), [None, None])

    def test_empty_memory_returns_the_ls_estimate_and_invalid_entries_are_ignored(self):
        model, sample, noise = build_model(), inputs(), tf.constant([.1])
        observation, layout = model.observe(sample[2], noise), model.layout_for(sample[2])
        empty = model.empty_history(observation)
        poisoned = empty._replace(observations=tf.fill(empty.observations.shape, float('nan')))
        for history in (empty, poisoned):
            coefficients = model.channel_filter(observation, history, sample[3], layout)
            np.testing.assert_array_equal(layout.expand_channel(coefficients),
                                          tf.complex(sample[2][..., :4], sample[2][..., 4:]))
        inactive = model.channel_filter(observation, poisoned, tf.zeros_like(sample[3]), layout)
        np.testing.assert_array_equal(inactive, 0.)

    def test_phase_and_scale_equivariance(self):
        model, sample, noise = build_model(), inputs(), tf.constant([.1])
        observation = model.observe(sample[2], noise)
        history = valid_history(model, inputs(), noise)
        layout = model.layout_for(sample[2])
        original = model.channel_filter(observation, history, sample[3], layout)
        factor = tf.constant(1.2+.7j, tf.complex64)
        def transform(record):
            from neural_rx.receiver.channel_filter import complex_channel
            h = complex_channel(record[..., :-1])*factor
            return tf.concat([tf.math.real(h), tf.math.imag(h), record[..., -1:]*tf.abs(factor)**2], -1)
        changed = model.channel_filter(transform(observation),
                                       history._replace(observations=transform(history.observations)),
                                       sample[3], layout)
        np.testing.assert_allclose(changed, original*factor, rtol=2e-3, atol=2e-4)

    def test_memory_of_the_same_channel_reduces_the_error_and_of_another_channel_does_not(self):
        rng = np.random.RandomState(11)
        model = build_model(8)
        layout = next(iter(model.layouts.values()))
        variance, shape = .2, (2, 1, 4)
        draw = lambda: (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)*np.ones([1, layout.count, 1])
        channel = draw()                                          # constant over the grid
        observation, active = records(channel, variance, rng), tf.ones([1, 2])
        truth = np.broadcast_to(channel[:, :1, None], (2, layout.frequency, layout.symbols, 4))
        def error(history):
            estimate = layout.expand_channel(model.channel_filter(observation, history, active, layout))[0]
            return float(np.mean(np.abs(estimate-truth)**2))
        ls = error(model.empty_history(observation))
        self.assertAlmostEqual(ls, variance, delta=.2*variance)
        same = error(history_of(model, [records(channel, variance, rng) for _ in range(model.history_tbs)]))
        self.assertLess(same, .25*ls)
        # entries of a UE whose channel has changed since do not predict the current LS estimates, so the filter
        # leaves them out and keeps the weights of the current TB's own neighbours
        other = draw()
        changed = error(history_of(model, [records(other, variance, rng) for _ in range(model.history_tbs)]))
        self.assertLess(changed, .7*ls)
        # only the newest entry is of the current channel: the filter uses it and leaves the older ones out
        mixed = [records(channel, variance, rng)]+[records(other, variance, rng) for _ in range(model.history_tbs-1)]
        self.assertLess(error(history_of(model, mixed)), .6*ls)
        # a single entry is enough to fit the weights of the current TB's own neighbours
        one = model.empty_history(observation)
        entry = records(channel, variance, rng)
        one = PilotHistory(tf.concat([entry[:, :, None], one.observations[:, :, 1:]], 2),
                           tf.concat([tf.ones_like(one.age[..., :1]), one.age[..., 1:]], 2), 
                           tf.broadcast_to(tf.range(2)[None, :, None], one.ports.shape),
                           tf.concat([tf.ones_like(one.valid[..., :1]), one.valid[..., 1:]], 2))
        self.assertLess(error(one), .7*ls)


    def test_weights_are_fitted_for_the_ages_of_the_entries_after_a_pause(self):
        rng = np.random.RandomState(5)
        model = build_model(8)
        layout = next(iter(model.layouts.values()))
        variance, shape, turn = .2, (2, 1, 4), .9                # the channel turns by 0.9 rad per slot
        base = (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)*np.ones([1, layout.count, 1])
        at = lambda age: records(base*np.exp(-1j*turn*age), variance, rng)
        observation, active = at(0), tf.ones([1, 2])
        truth = np.broadcast_to(base[:, :1, None], (2, layout.frequency, layout.symbols, 4))
        def error(ages):
            entries = tf.stack([at(age) for age in ages], 2)
            history = PilotHistory(entries, tf.broadcast_to(tf.constant(ages)[None, None], entries.shape[:3]),
                                   tf.broadcast_to(tf.range(2)[None, :, None], entries.shape[:3]),
                                   tf.ones(entries.shape[:3], tf.bool))
            estimate = layout.expand_channel(model.channel_filter(observation, history, active, layout))[0]
            return float(np.mean(np.abs(estimate-truth)**2))
        self.assertLess(error([1, 2, 3, 4, 5, 6, 7]), .35*variance)      # scheduled in every slot
        self.assertLess(error([3, 4, 5, 6, 7, 8, 9]), .35*variance)      # back after two slots away
        self.assertLess(error([1, 4, 5, 7, 8, 9, 10]), .5*variance)      # irregular


if __name__ == '__main__':
    unittest.main()
