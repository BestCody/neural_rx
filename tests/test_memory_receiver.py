from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf
from sionna.ofdm import PilotPattern, ResourceGrid
from sionna.mapping import Mapper
from sionna.nr import PUSCHLSChannelEstimator

from pilot_history_receiver.model import PilotHistoryReceiver, architecture_settings
from pilot_history_receiver.pilot_layout import PilotLayout
from pilot_history_receiver.pilot_history import PilotHistory
from utils.neural_rx import CGNN


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
    return PilotHistoryReceiver(base, pilot_layout(prbs)[0], Mapper("qam", 4))


def valid_history(model, sample, noise):
    obs = model.observe(sample[2], noise)
    history = model.empty_history(obs)
    return PilotHistory(tf.repeat(obs[:, :, None], model.history_tbs, 2),
        tf.broadcast_to(tf.range(1, model.history_tbs+1)[None, None], history.age.shape),
        tf.broadcast_to(tf.range(2)[None, :, None], history.ports.shape), tf.ones_like(history.valid))


def activate_physical_path(model, sample):
    model(sample, tf.fill([tf.shape(sample[0])[0]], .1))
    for layer in (model.channel_memory.filter, model.channel_memory.error):
        for variable in layer.trainable_variables:
            variable.assign(tf.random.normal(variable.shape, stddev=.03))


class ChannelMemoryTest(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)
        self.tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        self.addCleanup(tf.config.experimental.enable_tensor_float_32_execution, self.tf32)

    def test_architecture_requires_decoder_assisted_filter(self):
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
                            float((layout.coordinates[source,index,1]-layout.coordinates[user,i,1])*.03), places=6)

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

    def test_compiled_preparation_preserves_weight_gradients_without_input_gradients(self):
        model, sample, noise = build_model(2), inputs(2), tf.constant([.1])
        activate_physical_path(model, sample)
        layout = model.layout_for(sample[2])
        observation = model.observe(sample[2], noise)
        history = valid_history(model, inputs(2), noise)
        variables = model.channel_memory.trainable_variables

        @tf.function(jit_compile=True)
        def reference(observation, history, active):
            estimate = model.channel_memory(observation, history, active, layout=layout)
            channel = layout.expand_channel(estimate.coefficients)
            return estimate, tf.concat([tf.math.real(channel), tf.math.imag(channel)], -1)

        outputs, gradients = [], []
        def revised(observation, history, active):
            context, channel = model.prepare(sample, observation, history, layout)
            return model.channel_memory.complete(context, layout), channel
        for call in (reference, revised):
            with tf.GradientTape() as tape:
                tape.watch([observation, history.observations])
                estimate, channel = call(observation, history, sample[3])
                loss = (tf.reduce_mean(tf.square(channel))+
                        tf.reduce_mean(tf.square(tf.abs(estimate.coefficients)))+
                        tf.reduce_mean(tf.math.real(tf.linalg.trace(estimate.covariance))))
            outputs.append((*estimate, channel, loss))
            gradients.append(tape.gradient(loss, [observation, history.observations, *variables]))
        for actual, expected in zip(outputs[1], outputs[0]):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        for gradient in gradients[0][:2]:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)
        self.assertEqual(gradients[1][:2], [None, None])
        for actual, expected in zip(gradients[1][2:], gradients[0][2:]):
            self.assertIsNotNone(actual)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        gradient_map = {id(v): g for v, g in zip(variables, gradients[1][2:])}
        for layer in (model.channel_memory.filter, model.channel_memory.error):
            self.assertGreater(float(tf.linalg.global_norm(
                [gradient_map[id(v)] for v in layer.trainable_variables])), 0.)

    def test_deferred_uncertainty_preserves_precision_layouts_and_concurrent_calls(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        model, sample, noise = build_model(48), inputs(48), tf.constant([.1])
        activate_physical_path(model, sample)
        layout = model.layout_for(sample[2])
        observation = model.observe(sample[2], noise)
        history = valid_history(model, inputs(48), noise)
        original, _ = model.prepare(sample, observation, history, layout)
        for tf32 in (False, True):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            for users in (1, 3):
                indices = tf.range(users) % 2
                context = type(original)(*(tf.gather(v, indices, axis=1) for v in original))
                geometry = SimpleNamespace(r_layout=layout.r_layout[np.arange(users) % 2])
                reference = tf.function(lambda c:model.channel_memory.uncertainty(c, geometry), jit_compile=True)
                revised = tf.function(lambda c:model.channel_memory.complete(c, geometry).covariance)
                expected = reference(context)
                np.testing.assert_allclose(revised(context), expected, rtol=2e-5, atol=2e-6)
                with ThreadPoolExecutor(max_workers=2) as workers:
                    outputs = list(workers.map(lambda _:revised(context).numpy(), range(2)))
                np.testing.assert_array_equal(outputs[0], outputs[1])
                np.testing.assert_allclose(outputs[0], expected, rtol=2e-5, atol=2e-6)


    def test_invalid_history_is_ignored_and_current_neighbors_remain_available(self):
        model, sample, noise = build_model(), inputs(), tf.constant([.1])
        activate_physical_path(model, sample)
        observation, layout = model.observe(sample[2], noise), model.layout_for(sample[2])
        empty = model.empty_history(observation)
        poisoned = empty._replace(observations=tf.fill(empty.observations.shape, float('nan')))
        original = model.channel_memory(observation, empty, sample[3], layout=layout)
        actual = model.channel_memory(observation, poisoned, sample[3], layout=layout)
        for a, b in zip(original, actual):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(actual.covariance).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(actual.covariance) > 0))
        inactive = model.channel_memory(observation, poisoned, tf.zeros_like(sample[3]), layout=layout)
        for value in inactive:
            np.testing.assert_array_equal(value, 0.)

    def test_phase_and_scale_equivariance(self):
        model, sample, noise = build_model(), inputs(), tf.constant([.1])
        activate_physical_path(model, sample)
        observation = model.observe(sample[2], noise)
        history = valid_history(model, inputs(), noise)
        layout = model.layout_for(sample[2])
        original = model.channel_memory(observation, history, sample[3], layout=layout)
        factor = tf.constant(1.2+.7j, tf.complex64)
        def transform(record):
            from pilot_history_receiver.channel_filter import complex_channel
            h = complex_channel(record[..., :-1])*factor
            return tf.concat([tf.math.real(h), tf.math.imag(h), record[..., -1:]*tf.abs(factor)**2], -1)
        changed = model.channel_memory(transform(observation), history._replace(observations=transform(history.observations)), sample[3], layout=layout)
        np.testing.assert_allclose(changed.coefficients, original.coefficients*factor, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(changed.covariance, original.covariance*tf.cast(tf.abs(factor)**2, tf.complex64), rtol=3e-5, atol=2e-6)

    def test_compact_matrix_channel_loss_matches_direct_positions(self):
        from pilot_history_receiver.channel_filter import complex_channel
        from pilot_history_receiver.training_execution import channel_nll
        model, sample = build_model(), inputs()
        activate_physical_path(model, sample)
        layout = model.layout_for(sample[2])
        observation = model.observe(sample[2], tf.constant([.1]))
        estimate = model.channel_memory(observation, model.empty_history(observation), sample[3], layout=layout)
        truth = tf.random.normal(sample[2].shape)
        targets = layout.channel_targets(truth)
        self.assertEqual(targets.shape[-1], 40)
        got = channel_nll(estimate, targets, layout)
        error = complex_channel(truth)-layout.expand_channel(estimate.coefficients)
        covariance = layout.expand(tf.reshape(estimate.covariance, [1, 2, layout.count, 16]))
        covariance = tf.reshape(covariance, [1, 2, layout.frequency, layout.symbols, 4, 4])
        factor = tf.linalg.cholesky(covariance)
        white = tf.linalg.triangular_solve(factor, error[..., None])[..., 0]
        direct = (tf.reduce_sum(tf.abs(white)**2, -1)+2*tf.reduce_sum(tf.math.log(tf.math.real(tf.linalg.diag_part(factor))), -1))/4
        expected = tf.reduce_sum(direct*layout.data_mask[None], [2,3])/layout.data_mask.sum((1,2))[None]
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
        changed = tf.where(layout.data_mask[None, ..., None], truth, truth+7.)
        np.testing.assert_array_equal(layout.channel_targets(changed), targets)

    def test_partner_ties_and_reordering_use_physical_identity(self):
        from pilot_history_receiver.decoder_correction import select_partners
        channel = tf.ones([1, 2, 3, 4], tf.complex64)
        variance = tf.zeros([1, 2, 3, 4])
        ids = tf.constant([[9, 3, 8, 2]])
        expected = np.array([[[3, 3, 3, 1], [3, 3, 3, 1]]])
        np.testing.assert_array_equal(select_partners(channel, variance, tf.ones([1]), tf.ones([1,4]), ids), expected)
        order = [2,0,3,1]
        chosen = select_partners(tf.gather(channel, order, axis=-1), variance, tf.ones([1]), tf.ones([1,4]), tf.gather(ids, order, axis=1))
        np.testing.assert_array_equal(np.array(order)[chosen], expected[:,:,order])


class NativeCorrectionTests(unittest.TestCase):
    def test_blend_endpoints_shared_fit_ties_and_sign_symmetry(self):
        from pilot_history_receiver.decoder_correction import DecoderCorrection
        kernel = DecoderCorrection(Mapper('qam', 2), 2)
        rng = np.random.RandomState(417)
        y = tf.constant([[[1+.5j], [2-.5j]]], tf.complex64)
        factor = tf.ones([1,1,1,1,1], tf.complex64)
        white = tf.zeros([1,1,1,1,6], tf.complex64)
        logdet = tf.zeros([1,1,1])
        evidence = tf.constant(rng.normal(size=[1,2,2,2]), tf.float32)
        base = tf.constant(rng.normal(size=[1,2,2,2]), tf.float32)
        calibration = tf.constant(rng.normal(size=7)*.1, tf.float32)
        strength = tf.constant(.4)
        blend = [tf.constant(rng.normal(size=s)*.2, tf.float32) for s in ((8,8),(8,),(8,),())]
        phi = tf.ones([2,2,3]); cells = tf.zeros([2], tf.int64)
        enabled = tf.ones([1,1], tf.bool)
        indices = tf.reshape(tf.range(4,dtype=tf.int64),[1,2,2]); indices = tf.repeat(indices,2,0)
        def native(values, signs=tf.ones([2,2,2]), fit_override=None):
            scores, fit = kernel.score(values[0],values[1:4],phi,cells,enabled)
            return kernel.native.pilot_pair_correction(*values[:5],phi,cells,tf.constant([[0,1]],tf.int64),
                kernel.symbols,kernel.energy_indices,kernel.labels,enabled,indices,signs,tf.ones([1,2,2],tf.bool),
                *values[5:8],tf.ones([1,2]),scores,fit if fit_override is None else fit_override,*values[8:],llr_limit=20.)
        values = [y,factor,white,logdet,evidence,base,calibration,strength,*blend]
        upstream = tf.constant(rng.normal(size=base.shape),tf.float32)
        outputs, gradients = [], []
        for use_native in (False, True):
            with tf.GradientTape() as tape:
                tape.watch(values)
                result = native(values) if use_native else self.reference(*values[:5],phi,cells,
                    kernel.symbols,np.asarray(kernel.labels),np.asarray(kernel.energy_indices),
                    *values[5:7],20.,strength,blend)
                loss = tf.reduce_sum(result*tf.cast(upstream,result.dtype))
            outputs.append(result); gradients.append(tape.gradient(loss,values))
        np.testing.assert_allclose(*outputs,rtol=3e-5,atol=2e-6)
        for actual, expected in zip(gradients[1],gradients[0]):
            np.testing.assert_allclose(actual,tf.convert_to_tensor(expected),rtol=3e-3,atol=2e-5)
        self.assertGreater(float(tf.linalg.norm(gradients[1][0])),1e-6)
        signed = [*values[:4],-evidence,-base,*values[6:]]
        np.testing.assert_array_equal(native(signed,-tf.ones([2,2,2])),-native(values))
        low, high = native(values,fit_override=tf.zeros([1,2])),native(values,fit_override=tf.ones([1,2])*5)
        self.assertGreater(float(tf.linalg.norm(low-high)),1e-6)
        removed = tf.concat([blend[0][:7],tf.zeros_like(blend[0][7:])],0)
        ignored = [*values[:8],removed,*values[9:]]
        np.testing.assert_array_equal(native(ignored,fit_override=tf.zeros([1,2])),native(ignored,fit_override=tf.ones([1,2])*5))
        endpoints = [y,factor,tf.ones_like(white)*complex(.1,.15),*values[3:]]
        _, physical, delta, _ = self.reference(*endpoints[:5],phi,cells,kernel.symbols,np.asarray(kernel.labels),
            np.asarray(kernel.energy_indices),*values[5:7],20.,strength,blend,True)
        self.assertGreater(float(tf.linalg.norm(delta)),1e-6)
        for endpoint in (0.,1.):
            forced = [*endpoints[:10],tf.zeros([8]),tf.constant(-1000. if endpoint==0 else 1000.)]
            expected = float(strength)*delta+endpoint*(physical-tf.cast(base,tf.float64))
            np.testing.assert_allclose(native(forced),expected,rtol=2e-5,atol=2e-6)

    def test_preparation_matches_tensorflow_outputs_gradients_and_failure_recovery(self):
        from pilot_history_receiver.decoder_correction import DecoderCorrection
        from concurrent.futures import ThreadPoolExecutor
        rng = np.random.RandomState(902)
        for bits, batch, cells, antennas in ((2, 1, 1, 1), (2, 1, 1, 4), (4, 2, 7, 4), (4, 1, 3, 3), (6, 1, 5, 8)):
            kernel = DecoderCorrection(Mapper('qam', bits), bits)
            z = lambda shape: tf.constant(rng.normal(size=shape)+1j*rng.normal(size=shape), tf.complex64)
            coefficients = z([batch, cells, 2, antennas, 3])
            a = z([batch, cells, 2, antennas, antennas])*.2
            covariance = tf.linalg.matmul(a, a, adjoint_b=True)
            a = z([batch, cells, antennas, antennas])*.1
            background = tf.linalg.matmul(a, a, adjoint_b=True)
            cases = [(tf.ones([batch])*.2, covariance, background), (tf.zeros([batch]), covariance, background),
                     (tf.zeros([batch]), tf.zeros_like(covariance), tf.zeros_like(background))]
            for noise, current_covariance, current_background in cases:
                values = [coefficients, current_covariance, noise, current_background]
                gradients, outputs = [], []
                weights = None
                for native in (False, True):
                    with tf.GradientTape() as tape:
                        tape.watch(values)
                        if native:
                            output = kernel.prepare(*values)
                        else:
                            matrix, rhs = kernel._matrix_and_rhs(*values)
                            factor = tf.linalg.cholesky(matrix)
                            output = (factor, tf.linalg.triangular_solve(factor, rhs),
                                      2*tf.reduce_sum(tf.math.log(tf.math.real(tf.linalg.diag_part(factor))), -1))
                        if weights is None:
                            weights = [tf.cast(rng.normal(size=x.shape), x.dtype) for x in output]
                        loss = sum(tf.reduce_sum(tf.math.real(x*w)) for x, w in zip(output, weights))
                    outputs.append(output)
                    gradients.append(tape.gradient(loss, values))
                for actual, expected in zip(outputs[1], outputs[0]):
                    np.testing.assert_array_equal(actual, expected)
                for actual, expected in zip(gradients[1], gradients[0]):
                    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
        @tf.function
        def invoke(covariance):
            return kernel.prepare(coefficients, covariance, noise, background)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            invoke(tf.fill(tf.shape(covariance), tf.constant(complex('nan'), tf.complex64)))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            invoke(-100*tf.eye(antennas, batch_shape=[batch,cells,2], dtype=tf.complex64))
        varied = [covariance*(1+i*.1) for i in range(3)]
        expected = [[x.numpy() for x in invoke(value)] for value in varied]
        with ThreadPoolExecutor(max_workers=3) as workers:
            actual = list(workers.map(lambda value: [x.numpy() for x in invoke(value)], varied))
        for result, reference_result in zip(actual, expected):
            for value, reference in zip(result, reference_result):
                np.testing.assert_array_equal(value, reference)

    @staticmethod
    def reference(y, factor, white, logdet, evidence, phi, cells, symbols, labels, energy,
                  effective, calibration, limit, strength, blend, return_parts=False):
        y, factor, white, symbols = [tf.cast(x,tf.complex128) for x in (y,factor,white,symbols)]
        logdet, evidence, phi, effective, calibration = [tf.cast(x,tf.float64) for x in
                                                       (logdet,evidence,phi,effective,calibration)]
        factor = tf.linalg.set_diag(factor, tf.cast(tf.math.real(tf.linalg.diag_part(factor)), tf.complex128))
        factor, white, logdet = [tf.gather(x, cells, axis=1) for x in (factor, white, logdet)]
        yw = tf.linalg.triangular_solve(factor, y[:, :, None, :, None])[..., 0]
        ha = tf.reduce_sum(white[..., :3]*tf.cast(phi[None, :, None, 0, None, :], tf.complex128), -1)
        hb = tf.reduce_sum(white[..., 3:]*tf.cast(phi[None, :, None, 1, None, :], tf.complex128), -1)
        m, q = labels.shape
        group_count = int(round(np.sqrt(factor.shape[2])))
        pair_groups = (energy[:, None]*group_count+energy[None]).reshape(-1)
        left = tf.repeat(symbols, m); right = tf.tile(symbols, [m])
        residual = tf.gather(yw, pair_groups, axis=2)-tf.gather(ha, pair_groups, axis=2)*left[None,None,:,None]-tf.gather(hb, pair_groups, axis=2)*right[None,None,:,None]
        residual_energy = tf.reduce_sum(tf.abs(residual)**2, -1)
        rho = tf.reduce_min(residual_energy, -1)/tf.cast(tf.shape(y)[-1], tf.float64)
        scores = -residual_energy-tf.gather(logdet, pair_groups, axis=2)
        bits = np.concatenate([np.repeat(labels, m, axis=0), np.tile(labels, (m,1))], -1)
        e = tf.reshape(tf.transpose(evidence, [0,2,1,3]), [tf.shape(evidence)[0], tf.shape(evidence)[2], 2*q])
        l = tf.reshape(tf.transpose(effective, [0,2,1,3]), tf.shape(e))
        def margin(v, bit):
            return (tf.reduce_logsumexp(tf.gather(v,np.flatnonzero(bits[:,bit]),axis=-1),-1)
                    -tf.reduce_logsumexp(tf.gather(v,np.flatnonzero(1-bits[:,bit]),axis=-1),-1))
        physical = tf.stack([margin(scores, bit) for bit in range(2*q)], -1)
        re, rl, rp = [x/(limit+tf.abs(x)) for x in (e, l, physical)]
        features = tf.stack([tf.abs(re),tf.abs(rl),tf.abs(rp),re*rl,re*rp,rl*rp], -1)
        e *= 2*tf.sigmoid(tf.reduce_sum(features*calibration[:6], -1)+calibration[6])
        values = []
        for bit in range(2*q):
            other = bits.copy(); other[:, bit] = 0
            scored = scores+tf.einsum('bnq,sq->bns',e,tf.constant(other, tf.float64))
            values.append(margin(scored,bit)-physical[...,bit])
        delta = tf.stack(values, -1)
        u = tf.cast(strength, tf.float64)*delta
        b0, b1 = l+u, physical+u
        normalizer = limit+tf.abs(b0)+tf.abs(b1)+tf.abs(u)
        x0, x1, xu = b0/normalizer, b1/normalizer, u/normalizer
        features = tf.stack([tf.abs(x0), tf.abs(x1), x0*x1, tf.abs(x1-x0), tf.abs(xu), x0*xu, x1*xu,
                            tf.broadcast_to((rho/(1+rho))[..., None], tf.shape(l))], -1)
        kernel, bias, head, offset = [tf.cast(v, tf.float64) for v in blend]
        coefficient = tf.sigmoid(tf.reduce_sum(tf.nn.relu(features@kernel+bias)*head, -1)+offset)
        result = u+coefficient*(physical-l)
        to_user = lambda value: tf.transpose(tf.reshape(value,[tf.shape(y)[0],tf.shape(y)[1],2,q]),[0,2,1,3])
        return (to_user(result), to_user(physical), to_user(delta), rho) if return_parts else to_user(result)

    def test_multi_user_routing_and_single_active_user_match_direct_pairs(self):
        from types import SimpleNamespace
        from scipy.special import logsumexp
        from pilot_history_receiver.decoder_correction import DecoderCorrection, select_partners
        from pilot_history_receiver.channel_filter import ChannelEstimate
        rng = np.random.RandomState(146)
        kernel = DecoderCorrection(Mapper('qam', 2), 2)
        symbols, labels = np.asarray(kernel.symbols), np.asarray(kernel.labels)
        calibration = rng.normal(size=7).astype(np.float32)
        strength = tf.constant(.37)
        blend = [tf.constant(rng.normal(size=shape)*.3, tf.float32) for shape in ((8, 8), (8,), (8,), ())]
        for users in (1, 2, 3, 4):
            n, r = 3, 3
            theta = (rng.normal(size=[1,users,1,r,3])+1j*rng.normal(size=[1,users,1,r,3])).astype(np.complex64)*.3
            a = (rng.normal(size=[1,users,1,r,r])+1j*rng.normal(size=[1,users,1,r,r])).astype(np.complex64)*.1
            covariance = a@a.conj().swapaxes(-1,-2)+.1*np.eye(r,dtype=np.complex64)
            phi = rng.normal(size=[n,users,3]).astype(np.float32); phi[...,0] = 1
            layout = SimpleNamespace(users=users, code_positions=np.arange(n,dtype=np.int32),
                code_phi=phi, code_cells=np.zeros(n,np.int32), cell_positions=[np.arange(n,dtype=np.int32)],
                joint=lambda x: tf.transpose(x,[0,2,1,3,4]))
            y = (rng.normal(size=[1,n,r])+1j*rng.normal(size=[1,n,r])).astype(np.complex64)
            received = tf.constant(np.concatenate([y.real,y.imag],-1)[:,:,None])
            llr = rng.normal(size=[1,users,n*2]).astype(np.float32)
            evidence = rng.normal(size=llr.shape).astype(np.float32)
            ids = tf.constant([rng.permutation(users)],tf.int32)
            for active in (np.ones([1,users],np.float32),np.eye(1,users,dtype=np.float32)):
                actual = kernel(received, ChannelEstimate(tf.constant(theta),tf.constant(covariance)),tf.constant([.4]),
                    tf.constant(active),ids,tf.constant(llr),tf.reshape(evidence+llr,[1,users,n,2]),layout,
                    tf.tile(tf.reshape(tf.range(n*2),[1,n,2]),[users,1,1]),
                    tf.ones([users,n,2]), tf.reshape(llr,[1,users,n,2]),
                    tf.constant(calibration), 20., strength, blend).numpy().reshape(users,n,2)
                h = np.einsum('urq,nuq->nur',theta[0,:,0],phi)
                cell_h = theta[0,:,0,:,0].T[None,None]*active[:,None,None]
                cell_v = np.diagonal(covariance[0,:,0],axis1=-2,axis2=-1).real.T[None,None]*active[:,None,None]
                partners = np.asarray(select_partners(cell_h,cell_v,tf.constant([.4]),tf.constant(active),ids))[0,0]
                proposals = llr.reshape(users,n,2)@labels.T
                p = np.exp(proposals-logsumexp(proposals,axis=-1,keepdims=True))
                means = np.sum(p*symbols,-1)
                variance = np.sum(p*abs(symbols-means[...,None])**2,-1)
                energy = np.sum(p*abs(symbols)**2,-1)
                e = evidence.reshape(users,n,2)
                expected = llr.reshape(users,n,2).copy()
                for u in range(users):
                    if not active[0,u]:
                        continue
                    v = int(partners[u])
                    choices = [(i,j) for i in range(4) for j in (range(4) if v >= 0 else [0])]
                    pair_bits = np.array([np.r_[labels[i],labels[j]] if v>=0 else labels[i] for i,j in choices])
                    for position in range(n):
                        background = [w for w in range(users) if active[0,w] and w not in (u,v)]
                        residual_y = y[0,position]-sum((h[position,w]*means[w,position] for w in background),start=np.zeros(r,complex))
                        matrix_bg = .4*np.eye(r,dtype=complex)
                        for w in background:
                            matrix_bg += variance[w,position]*np.outer(h[position,w],h[position,w].conj())+energy[w,position]*covariance[0,w,0]
                        scores=[]; distances=[]
                        for i,j in choices:
                            matrix = matrix_bg+abs(symbols[i])**2*covariance[0,u,0]
                            residual = residual_y-h[position,u]*symbols[i]
                            if v >= 0:
                                matrix = matrix+abs(symbols[j])**2*covariance[0,v,0]
                                residual -= h[position,v]*symbols[j]
                            distance=np.vdot(residual,np.linalg.solve(matrix,residual)).real
                            distances.append(distance)
                            scores.append(-distance-np.linalg.slogdet(matrix)[1])
                        scores=np.array(scores)
                        prior = np.r_[e[u,position],e[v,position]] if v>=0 else e[u,position].copy()
                        base = np.r_[llr[0,u].reshape(n,2)[position],llr[0,v].reshape(n,2)[position]] if v>=0 else llr[0,u].reshape(n,2)[position]
                        physical = np.array([logsumexp(scores[pair_bits[:,bit] == 1])-logsumexp(scores[pair_bits[:,bit] == 0])
                                             for bit in range(pair_bits.shape[1])])
                        re,rl,rp = [x/(20+np.abs(x)) for x in (prior,base,physical)]
                        features = np.stack([abs(re),abs(rl),abs(rp),re*rl,re*rp,rl*rp],-1)
                        prior *= 2/(1+np.exp(-(features@calibration[:6]+calibration[6])))
                        for bit in range(2):
                            selected=pair_bits[:,bit].astype(bool)
                            other=pair_bits.copy();other[:,bit]=0
                            z=scores+other@prior
                            delta=(logsumexp(z[selected])-logsumexp(z[~selected]))-physical[bit]
                            correction=float(strength)*delta
                            b0,b1=base[bit]+correction,physical[bit]+correction
                            denominator=20+abs(b0)+abs(b1)+abs(correction)
                            x0,x1,xu=b0/denominator,b1/denominator,correction/denominator
                            rho=min(distances)/r
                            features=np.array([abs(x0),abs(x1),x0*x1,abs(x1-x0),abs(xu),x0*xu,x1*xu,rho/(1+rho)])
                            coefficient=1/(1+np.exp(-(np.maximum(features@blend[0].numpy()+blend[1].numpy(),0)@blend[2].numpy()+blend[3].numpy())))
                            expected[u,position,bit]=base[bit]+correction+coefficient*(physical[bit]-base[bit])
                np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=5e-6)

    def test_native_forward_full_gradients_and_zero_evidence(self):
        from pilot_history_receiver.native_ops import load
        from pilot_history_receiver.decoder_correction import DecoderCorrection
        tf.keras.utils.set_random_seed(901)
        tf.config.experimental.enable_op_determinism()
        kernel = DecoderCorrection(Mapper('qam',4),4)
        rng = np.random.RandomState(88)
        complex_value = lambda shape: tf.constant(rng.normal(size=shape)+1j*rng.normal(size=shape),tf.complex64)
        y = complex_value([2,5,3])
        factor = tf.linalg.band_part(complex_value([2,2,9,3,3])*.1,-1,0)
        factor = tf.linalg.set_diag(factor,tf.cast(tf.ones([2,2,9,3])*1.3,tf.complex64))
        white = complex_value([2,2,9,3,6])*.2
        logdet = tf.constant(rng.normal(size=[2,2,9]),tf.float32)
        evidence = tf.constant(rng.normal(size=[2,2,5,4]),tf.float32)
        effective = tf.constant(rng.uniform(-20,20,size=evidence.shape),tf.float32)
        phi = tf.constant(rng.normal(size=[5,2,3]),tf.float32)
        cells = tf.constant([0,1,0,1,1]); positions = tf.constant([[0,2,-1],[1,3,4]])
        mapping = [tf.tile(tf.reshape(tf.range(20),[1,5,4]),[2,1,1]),
                   tf.ones([2,5,4]), tf.ones([2,2,5],tf.bool)]
        incoming = tf.constant(rng.normal(size=evidence.shape),tf.float32)
        strength = tf.constant(.37)
        blend = [tf.constant(rng.normal(size=shape)*.3, tf.float32) for shape in ((8, 8), (8,), (8,), ())]
        def arguments(values, enabled):
            scores = kernel.score(values[0], values[1:4], phi, tf.cast(cells, tf.int64), enabled)
            return (*values[:5],phi,tf.cast(cells,tf.int64),tf.cast(positions,tf.int64),kernel.symbols,kernel.energy_indices,kernel.labels,
                    enabled,tf.cast(mapping[0],tf.int64),*mapping[1:],*values[5:8],tf.ones([2,2]),*scores,*values[8:])
        def call(values, enabled=tf.ones([2,2],tf.bool)):
            return load().pilot_pair_correction(*arguments(values,enabled),llr_limit=20.)
        for calibration in (tf.zeros([7]),tf.constant(rng.normal(size=7),tf.float32)):
            for source in (evidence,tf.zeros_like(evidence),evidence*30):
                values = [y,factor,white,logdet,source,effective,calibration,strength,*blend]
                outputs, gradients = [], []
                for native in (False, True):
                    with tf.GradientTape() as tape:
                        tape.watch(values)
                        output = call(values) if native else self.reference(*values[:5],phi,cells,
                            kernel.symbols,np.asarray(kernel.labels),np.asarray(kernel.energy_indices),
                            *values[5:7],20.,values[7],values[8:])
                        loss = tf.reduce_sum(output*tf.cast(incoming,output.dtype))
                    outputs.append(output); gradients.append(tape.gradient(loss,values))
                np.testing.assert_allclose(*outputs,rtol=3e-4,atol=2e-5)
                for actual, expected in zip(gradients[1],gradients[0]):
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual,tf.convert_to_tensor(expected),rtol=3e-3,atol=8e-5)
                if not np.any(source):
                    _, _, delta, rho = self.reference(*values[:5],phi,cells,kernel.symbols,
                        np.asarray(kernel.labels),np.asarray(kernel.energy_indices),*values[5:7],20.,values[7],values[8:],True)
                    np.testing.assert_allclose(delta, 0., atol=1e-13)
                    np.testing.assert_allclose(kernel.score(y,(factor,white,logdet),phi,tf.cast(cells,tf.int64),
                        tf.ones([2,2],tf.bool))[1], rho, rtol=3e-6, atol=3e-6)
                    self.assertGreater(float(tf.linalg.norm(gradients[1][0])),1e-6)
                    self.assertGreater(float(tf.linalg.norm(gradients[1][4])),1e-6)
                repeated = load().pilot_pair_correction_grad(*arguments(values,tf.ones([2,2],tf.bool)),
                                                            incoming,llr_limit=20.)
                gate_gradient=tf.reduce_sum(repeated[8],[0,1])
                unpacked=[tf.reshape(gate_gradient[:64],[8,8]),gate_gradient[64:72],gate_gradient[72:80],gate_gradient[80]]
                for a,b in zip(gradients[1],[*repeated[:6],tf.reduce_sum(repeated[6],[0,1]),tf.reduce_sum(repeated[7]),*unpacked]):
                    np.testing.assert_array_equal(a,b)
        modified = source.numpy(); modified[:,0,:,1] += 10.
        changed = call([y,factor,white,logdet,tf.constant(modified),effective,calibration,strength,*blend])
        np.testing.assert_array_equal(changed[:,0,:,1],outputs[1][:,0,:,1])
        with tf.GradientTape() as tape:
            tape.watch(source)
            target = tf.reduce_sum(call(values)[:,0,:,1])
        np.testing.assert_array_equal(tape.gradient(target,source)[:,0,:,1],0.)
        disabled = tf.zeros([2,2],tf.bool)
        np.testing.assert_array_equal(call(values,disabled),0.)
        gradients = load().pilot_pair_correction_grad(*arguments(values,disabled),incoming,llr_limit=20.)
        for value in gradients:
            np.testing.assert_array_equal(value,0.)
        from concurrent.futures import ThreadPoolExecutor
        def concurrent(offset):
            return call([y+tf.cast(offset,tf.complex64),factor,white,logdet,
                         evidence+offset,effective,calibration+offset,strength,*blend]).numpy()
        expected = [concurrent(i*.1) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(concurrent,[i*.1 for i in range(4)]))
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a,b)

    def test_scrambled_padded_correction_and_gradients_with_asymmetric_requests(self):
        from pilot_history_receiver.decoder_correction import DecoderCorrection
        rng = np.random.RandomState(179)
        for q in (2, 6):
            kernel = DecoderCorrection(Mapper('qam',q),q)
            groups = len(kernel.energies)**2
            n, total = 3, 3*q+5
            y = tf.constant(rng.normal(size=[2,n,1])+1j*rng.normal(size=[2,n,1]),tf.complex64)
            factor = tf.ones([2,1,groups,1,1],tf.complex64)
            white = tf.constant(.1*(rng.normal(size=[2,1,groups,1,6])+1j*rng.normal(size=[2,1,groups,1,6])),tf.complex64)
            logdet = tf.zeros([2,1,groups])
            evidence = tf.constant(rng.normal(size=[2,2,1,total]),tf.float32)
            effective = tf.constant(rng.uniform(-20,20,size=evidence.shape),tf.float32)
            calibration = tf.constant(rng.normal(size=7),tf.float32)
            indices = np.stack([rng.permutation(total)[:n*q].reshape(n,q) for _ in range(2)])
            signs = rng.choice([-1.,1.],size=[2,n,q]).astype(np.float32)
            permutation = tf.one_hot(indices,total)*signs[...,None]
            wanted = tf.constant([[[True,False,True],[False,True,False]],[[False,True,True],[True,False,True]]])
            phi = tf.constant(rng.normal(size=[n,2,3]),tf.float32)
            incoming = tf.constant(rng.normal(size=evidence.shape),tf.float32)
            strength = tf.constant(.37)
            width = 5 if q == 2 else 33
            blend = [tf.constant(rng.normal(size=shape)*.3, tf.float32) for shape in ((8,width),(width,),(width,),())]
            inputs = [y,factor,white,logdet,evidence,effective,calibration,strength,*blend]
            outputs, gradients = [], []
            for native in (False,True):
                with tf.GradientTape() as tape:
                    tape.watch(inputs)
                    if native:
                        scores = kernel.score(y, (factor,white,logdet), phi, tf.zeros([n],tf.int64), tf.ones([2,1],tf.bool))
                        output = kernel.native.pilot_pair_correction(*inputs[:5],phi,tf.zeros([n],tf.int64),
                            tf.range(n,dtype=tf.int64)[None],kernel.symbols,kernel.energy_indices,kernel.labels,tf.ones([2,1],tf.bool),
                            tf.constant(indices,tf.int64),tf.constant(signs),wanted,*inputs[5:8],
                            tf.ones([2,2]),*scores,*blend,llr_limit=20.)
                    else:
                        symbol_evidence = tf.einsum('buc,unqc->bunq',evidence[:,:,0],permutation)
                        symbol_base = tf.einsum('buc,unqc->bunq',effective[:,:,0],permutation)
                        symbol_delta = self.reference(y,factor,white,logdet,symbol_evidence,phi,tf.zeros([n],tf.int32),
                            kernel.symbols,np.asarray(kernel.labels),np.asarray(kernel.energy_indices),
                            symbol_base,calibration,20.,strength,blend)
                        output = tf.einsum('bunq,unqc->buc',symbol_delta*tf.cast(wanted[...,None],tf.float64),
                                           tf.cast(permutation,tf.float64))[:,:,None]
                    loss = tf.reduce_sum(output*tf.cast(incoming,output.dtype))
                outputs.append(output); gradients.append(tape.gradient(loss,inputs))
            np.testing.assert_allclose(*outputs,rtol=2e-4,atol=8e-6)
            for actual,expected in zip(gradients[1],gradients[0]):
                np.testing.assert_allclose(actual,tf.convert_to_tensor(expected),rtol=3e-3,atol=3e-5)
            for u in range(2):
                padding = np.setdiff1d(np.arange(total),indices[u].ravel())
                np.testing.assert_array_equal(outputs[1][:,u,0].numpy()[:,padding],0.)
                for index in (4,5):
                    np.testing.assert_array_equal(gradients[1][index][:,u,0].numpy()[:,padding],0.)
            active = tf.constant([[1., 0.], [1., 1.]])
            requests = wanted & tf.cast(active[..., None], tf.bool)
            values = inputs
            outputs, gradients = [], []
            for fused in (False, True):
                with tf.GradientTape() as tape:
                    tape.watch(values)
                    information = evidence if fused else (evidence-effective)*active[..., None, None]
                    scores = kernel.score(y, (factor,white,logdet), phi, tf.zeros([n],tf.int64), tf.ones([2,1],tf.bool))
                    output = kernel.native.pilot_pair_correction(y,factor,white,logdet,information,phi,
                        tf.zeros([n],tf.int64),tf.range(n,dtype=tf.int64)[None],kernel.symbols,kernel.energy_indices,
                        kernel.labels,tf.ones([2,1],tf.bool),tf.constant(indices,tf.int64),tf.constant(signs),
                        requests,effective,calibration,strength,active,*scores,*blend,llr_limit=20.,apply_update=fused)
                    if not fused:
                        output = effective+output*active[..., None, None]
                    loss = tf.reduce_sum(output*incoming)
                outputs.append(output)
                gradients.append(tape.gradient(loss,values))
            np.testing.assert_allclose(*outputs, rtol=2e-6, atol=2e-6)
            for actual, expected in zip(gradients[1], gradients[0]):
                np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
            self.assertGreater(abs(float(gradients[1][7])), 1e-6)


if __name__ == '__main__':
    unittest.main()
