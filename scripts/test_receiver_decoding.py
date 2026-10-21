import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from sionna.nr import TBDecoder, TBEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from receiver_decoding import OneLayerGridCodeMap, SoftTBInformationDecoder, DecoderAssistedTB


class ReceiverDecodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.config.experimental.enable_op_determinism()

    def test_mapping_preserves_each_users_data_positions(self):
        indices = tf.constant([[[0, 2]], [[1, 3]]], tf.int32)
        ofdm = SimpleNamespace(
            _layer_demappers=[SimpleNamespace(
                _mapper=SimpleNamespace(num_layers=1), _num_bits_per_symbol=2)],
            _rg_demapper=SimpleNamespace(_data_ind=indices),
        )
        mapping = OneLayerGridCodeMap(ofdm, 2, 2)
        grid = tf.Variable(tf.reshape(tf.range(16, dtype=tf.float32), [1, 2, 2, 2, 2]))
        with tf.GradientTape() as tape:
            coded = mapping.grid_to_coded(grid)
            loss = tf.reduce_sum(coded)
        np.testing.assert_array_equal(coded, [[[0, 1, 2, 3], [12, 13, 14, 15]]])
        np.testing.assert_array_equal(tape.gradient(loss, grid),
            [[[[[1, 1], [1, 1]], [[0, 0], [0, 0]]],
              [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]]])

    def test_split_continuation_matches_declared_iterations_and_clipped_evidence(self):
        for size, coded_size, total in ((1000, 2400, 18), (1000, 2400, 20), (9991, 24000, 18)):
            encoder = TBEncoder(size, coded_size, .5, 4, n_rnti=[1, 2], n_id=[11, 22])
            tb_decoder = TBDecoder(encoder, num_bp_iter=total)
            ordinary = SoftTBInformationDecoder(encoder, tb_decoder, total)
            split = DecoderAssistedTB(encoder, TBDecoder(encoder, num_bp_iter=total), total, 8)
            reference_first = LDPC5GDecoder(encoder.ldpc_encoder, hard_out=False, return_infobits=False,
                                            stateful=True, cn_type='boxplus-phi', num_iter=8)
            reference_last = LDPC5GDecoder(encoder.ldpc_encoder, hard_out=False, return_infobits=True,
                                           stateful=True, cn_type='boxplus-phi', num_iter=total-8)
            bits = tf.cast(tf.random.stateless_uniform([1, 2, size], [87, 91], maxval=2, dtype=tf.int32), tf.float32)
            coded = encoder(bits)
            inputs, e, state, _ = split.evidence(tf.zeros_like(coded))
            split.finish(inputs, state)
            ordinary(tf.zeros_like(coded))
            for scale in (1.2, 35.):
                llr = (2*coded-1)*scale+tf.random.stateless_normal(tf.shape(coded), [17, 21])
                @tf.function
                def execute(value):
                    inputs, evidence, state, after_first = split.evidence(value)
                    posterior, _ = split.first.decoder([split.first.to_code_blocks(value), None])
                    maximum = split.first.llr_limit
                    expected = split.first.from_code_blocks(posterior)-tf.clip_by_value(value, -maximum, maximum)
                    tf.debugging.assert_equal(after_first, posterior)
                    return split.first.from_code_blocks(evidence), split.finish(inputs, state), ordinary(value), expected
                evidence, continued, reference, expected = execute(llr)
                np.testing.assert_allclose(continued.logits, reference.logits, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(continued.hard_bits, reference.hard_bits)
                np.testing.assert_array_equal(continued.crc_status, reference.crc_status)
                code_blocks = split.first.to_code_blocks(llr)
                np.testing.assert_array_equal(split.first.from_code_blocks(code_blocks), llr)
                np.testing.assert_allclose(evidence, expected, atol=2e-6)
                effective, _, state, _ = split.evidence(llr)
                effective_tb = split.first.from_code_blocks(effective)
                np.testing.assert_array_equal(effective_tb, tf.clip_by_value(llr,-split.first.llr_limit,split.first.llr_limit))
                perturbation = -1.5*effective_tb
                adjusted = split.first.to_code_blocks(effective_tb+perturbation)
                recovered = tb_decoder._descrambler(llr)
                recovered = tf.concat([recovered,tf.zeros([1,2,split.first.num_fillers])],-1)
                recovered = tf.gather(recovered,encoder.output_perm_inv,axis=-1)
                recovered = tf.reshape(recovered,[1,2,encoder.num_cbs,encoder.ldpc_encoder.n])
                _, expected_state = reference_first([recovered,None])
                corrected = tb_decoder._descrambler(effective_tb+perturbation)
                corrected = tf.concat([corrected,tf.zeros([1,2,split.first.num_fillers])],-1)
                corrected = tf.reshape(tf.gather(corrected,encoder.output_perm_inv,axis=-1),tf.shape(recovered))
                info, _ = reference_last([corrected,expected_state])
                actual, expected_payload = split.finish(adjusted,state), split.last.payload(info)
                np.testing.assert_allclose(actual.logits,expected_payload.logits,rtol=2e-6,atol=2e-6)
                np.testing.assert_array_equal(actual.hard_bits,expected_payload.hard_bits)
                np.testing.assert_array_equal(actual.crc_status,expected_payload.crc_status)
                probe = tf.random.stateless_normal(tf.shape(llr), [55, 67])*.8
                with tf.GradientTape() as tape:
                    tape.watch(probe)
                    inputs, e, state, _ = split.evidence(probe)
                    logits = split.finish(inputs+.2*tf.tanh(e), state).logits
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=bits, logits=logits))
                gradient = tape.gradient(loss, probe)
                self.assertTrue(np.isfinite(gradient).all())
                if scale < 20:
                    self.assertGreater(float(tf.linalg.global_norm([gradient])), 1e-6)
                    branches = []
                    for keep_input in (False, True):
                        with tf.GradientTape() as tape:
                            tape.watch(probe)
                            inputs, e, state, _ = split.evidence(probe)
                            corrected = inputs+.2*tf.tanh(e)
                            if keep_input:
                                state = tf.nest.map_structure(tf.stop_gradient, state)
                            else:
                                corrected = tf.stop_gradient(corrected)
                            logits = split.finish(corrected,state).logits
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=bits,logits=logits))
                        branches.append(tape.gradient(loss,probe))
                        self.assertIsNotNone(branches[-1])
                        self.assertTrue(np.isfinite(branches[-1]).all())
                        self.assertTrue(np.any(branches[-1].numpy() != 0.))
                    np.testing.assert_allclose(gradient,branches[0]+branches[1],rtol=3e-4,atol=2e-6)

    def test_correction_addresses_preserve_scrambling_padding_and_gradients(self):
        for size, coded_size, scramble in ((1000, 2400, True), (9991, 24000, True), (9991, 24000, False)):
            encoder = TBEncoder(size, coded_size, .5, 4, n_rnti=[1, 2], n_id=[11, 22], use_scrambler=scramble)
            decoder = DecoderAssistedTB(encoder, TBDecoder(encoder, num_bp_iter=20), 20, 8)
            original = tf.random.stateless_normal([2, 2, coded_size], [121, 153])*35.
            delta = tf.random.stateless_normal(tf.shape(original), [171, 183])
            mapping = np.asarray(decoder.symbol_indices).reshape(2, -1)
            signs = tf.reshape(decoder.symbol_signs, [2, coded_size])
            total = decoder.first.num_cbs*decoder.first.cb_length
            addresses = [[b, u, int(mapping[u, i])] for b in range(2) for u in range(2) for i in range(coded_size)]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(delta)
                mapped = tf.reshape(tf.scatter_nd(addresses, tf.reshape(delta*signs, [-1]), [2, 2, total]),
                                    [2, 2, decoder.first.num_cbs, decoder.first.cb_length])
                actual = decoder.first.to_code_blocks(original)+.37*mapped
                expected = decoder.first.to_code_blocks(original+.37*delta)
                weights = tf.random.stateless_normal(tf.shape(actual), [181, 192])
                actual_loss = tf.reduce_sum(actual*weights)
                expected_loss = tf.reduce_sum(expected*weights)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(decoder.first.from_code_blocks(mapped), delta)
            np.testing.assert_array_equal(tape.gradient(actual_loss, delta), tape.gradient(expected_loss, delta))
            for u in range(2):
                padding = np.setdiff1d(np.arange(total), mapping[u])
                np.testing.assert_array_equal(tf.reshape(mapped, [2, 2, total])[:, u].numpy()[:, padding], 0.)

    def test_payload_loss_has_finite_numerically_correct_ldpc_gradients(self):
        encoder = TBEncoder(1000, 2400, .5, 4, n_rnti=[1, 2], n_id=[11, 22])
        decoder = SoftTBInformationDecoder(encoder, TBDecoder(encoder, num_bp_iter=20), 20)
        bits = tf.cast(tf.random.stateless_uniform([1,2,1000], [53,61],
            maxval=2, dtype=tf.int32), tf.float32)
        coded = encoder(bits)
        llr = (2*coded-1)*1.2 + tf.random.stateless_normal(tf.shape(coded), [71,83])
        @tf.function
        def loss(value):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=bits, logits=decoder(value).logits))
        with tf.GradientTape() as tape:
            tape.watch(llr)
            value = loss(llr)
        gradient = tape.gradient(value,llr)
        self.assertIsNotNone(gradient)
        self.assertTrue(np.isfinite(gradient).all())
        norm = tf.linalg.global_norm([gradient])
        self.assertGreater(float(norm),1e-6)
        direction = gradient/norm
        epsilon = .01
        numerical = (loss(llr+epsilon*direction)-loss(llr-epsilon*direction))/(2*epsilon)
        np.testing.assert_allclose(numerical,tf.reduce_sum(gradient*direction),rtol=.02,atol=2e-5)
        self.assertFalse(decoder.decoder.trainable_variables)

    def test_payload_and_crc_match_sionna_for_single_and_multiple_code_blocks(self):
        for size, coded_size in ((1000, 2400), (9991, 24000)):
            with self.subTest(payload_size=size):
                encoder = TBEncoder(size, coded_size, .5, 4, n_rnti=[1, 2], n_id=[11, 22])
                reference = TBDecoder(encoder, num_bp_iter=20)
                decoder = SoftTBInformationDecoder(encoder, reference, 20)
                bits = tf.cast(tf.random.stateless_uniform(
                    [2, 2, size], [17, 28], maxval=2, dtype=tf.int32), tf.float32)
                coded = encoder(bits)
                noise = tf.random.stateless_normal(tf.shape(coded), [31, 42])
                reference((2 * coded - 1) * 8.)
                decoder((2 * coded - 1) * 8.)
                decode = tf.function(decoder)
                for llr in ((2 * coded - 1) * 8., (2 * coded - 1) * 1.5 + noise * 2.):
                    actual = decode(llr)
                    hard, crc = reference(llr)
                    np.testing.assert_array_equal(actual.hard_bits, hard)
                    np.testing.assert_array_equal(actual.crc_status, crc)
                    np.testing.assert_array_equal(actual.hard_bits, actual.logits > 0.)
                    self.assertEqual(actual.logits.shape, bits.shape)
                    self.assertTrue(np.isfinite(actual.logits).all())
                self.assertFalse(decoder.decoder._stateful)


if __name__ == "__main__":
    unittest.main()
