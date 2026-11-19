import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from sionna.nr import TBDecoder, TBEncoder

from neural_rx.decoding import OneLayerGridCodeMap, SoftTBInformationDecoder


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
