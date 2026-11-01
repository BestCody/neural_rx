from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sionna.fec.ldpc.decoding import LDPCBPDecoder
from sionna.fec.crc import CRCEncoder, CRCDecoder


class SoftInformationOutput(NamedTuple):

    logits: tf.Tensor
    hard_bits: tf.Tensor
    crc_status: tf.Tensor


class OneLayerGridCodeMap:

    def __init__(self, ofdm, num_tx: int, bits_per_symbol: int):
        self.num_tx = int(num_tx)
        self.bits_per_symbol = int(bits_per_symbol)
        if self.num_tx < 1 or self.bits_per_symbol < 1:
            raise ValueError("invalid grid-code dimensions")
        layer_demappers = ofdm._layer_demappers
        if layer_demappers is None or len(layer_demappers) != 1:
            raise ValueError("Grid mapping requires one fixed MCS")
        layer = layer_demappers[0]
        if int(layer._mapper.num_layers) != 1:
            raise ValueError("Grid mapping requires one layer per UE")
        if int(layer._num_bits_per_symbol) != self.bits_per_symbol:
            raise ValueError("modulation order differs from the protocol")
        indices = np.asarray(ofdm._rg_demapper._data_ind.numpy(), np.int32)
        if indices.shape[0] < self.num_tx or indices.shape[1] != 1:
            raise ValueError("unsupported data-index geometry")
        indices = indices[:self.num_tx, 0]
        self.data_indices = tf.convert_to_tensor(indices, tf.int32)
        self.num_data_symbols = int(indices.shape[1])

    def gather_stream_grid(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.convert_to_tensor(value)
        tf.debugging.assert_rank(value, 5)
        tf.debugging.assert_equal(tf.shape(value)[1], self.num_tx)
        value = tf.transpose(value, [1, 0, 3, 2, 4])
        shape = tf.shape(value)
        value = tf.reshape(
            value,
            [self.num_tx, shape[1], shape[2] * shape[3], shape[4]],
        )
        value = tf.gather(
            value,
            self.data_indices,
            axis=2,
            batch_dims=1,
        )
        return tf.transpose(value, [1, 0, 2, 3])

    def grid_to_coded(self, llr_grid: tf.Tensor) -> tf.Tensor:
        values = self.gather_stream_grid(llr_grid)
        tf.debugging.assert_equal(
            tf.shape(values)[-1], self.bits_per_symbol
        )
        shape = tf.shape(values)
        return tf.reshape(
            values,
            [shape[0], self.num_tx, self.num_data_symbols * self.bits_per_symbol],
        )


class SoftTBInformationDecoder:

    def __init__(self, tb_encoder, tb_decoder, num_iter: int, *, stateful=False, return_infobits=True):
        self.tb_encoder = tb_encoder
        self.tb_decoder = tb_decoder
        self.num_iter = int(num_iter)
        if self.num_iter < 0:
            raise ValueError("BP iteration count cannot be negative")
        base = tb_decoder._decoder
        self.decoder = LDPCBPDecoder(
            base.pcm,
            hard_out=False,
            num_iter=self.num_iter,
            cn_type=str(base._cn_type),
            stateful=stateful,
            output_dtype=tf.float32,
        )
        self.num_tx = int(tb_encoder.num_tx)
        self.num_cbs = int(tb_encoder.num_cbs)
        self.tb_length = int(tb_encoder.n)
        encoder = tb_encoder.ldpc_encoder
        self.cb_length = int(base.num_vns)
        self.info_length = int(encoder.k)
        self.num_fillers = int(encoder.n*self.num_cbs - np.sum(np.asarray(tb_encoder.cw_lengths)))
        if self.num_fillers < 0:
            raise ValueError("invalid TB code-block filler count")
        self.cb_crc_length = (
            0
            if tb_decoder._cb_crc_decoder is None
            else int(tb_decoder._cb_crc_decoder._encoder.crc_length)
        )
        self.tb_crc_length = int(
            tb_decoder._tb_crc_decoder._encoder.crc_length
        )
        self.decoder.llr_max = base.llr_max
        self.llr_limit = float(base.llr_max)
        if encoder.n > encoder.n_ldpc-(encoder.k_ldpc-encoder.k)-2*encoder.z:
            raise ValueError("Repeated rate-matched bits require an explicit inverse accumulation")
        with tf.init_scope():
            positions = np.arange(encoder.n, dtype=np.int32)+2*encoder.z
            positions += (positions >= encoder.k)*(encoder.k_ldpc-encoder.k)
            if encoder.num_bits_per_symbol is not None:
                positions = positions[np.asarray(encoder.out_int)]
            if np.any(positions >= self.cb_length):
                raise ValueError("Rate-recovered positions exceed the decoder graph")
            graph_positions = (np.arange(self.num_cbs, dtype=np.int32)[:, None]*self.cb_length
                               + positions[None]).reshape(-1)
            inverse = np.asarray(tb_encoder.output_perm_inv)
            addresses = graph_positions[np.argsort(inverse)[:self.tb_length]]
            recovery = np.full(self.num_cbs*self.cb_length, self.tb_length, np.int32)
            recovery[graph_positions] = np.minimum(inverse, self.tb_length)
            recovery = recovery.reshape(self.num_cbs, self.cb_length)
            recovery[:, encoder.k:encoder.k_ldpc] = self.tb_length+1
            self.recovery_indices = tf.constant(recovery)
            self.transmitted_indices = tf.constant(addresses)
            signs = tf.ones([1, self.num_tx, self.tb_length], tf.float32)
            if tb_decoder._descrambler is not None:
                signs = tb_decoder._descrambler(signs)
            self.scrambling_signs = signs
        self.payload_crc = None
        if return_infobits:
            with tf.init_scope():
                self.payload_crc = CRCDecoder(CRCEncoder(tb_encoder.tb_crc_encoder.crc_degree))
                self.payload_crc(tf.zeros([1, self.num_tx, int(tb_decoder.tb_size)+self.tb_crc_length]))

    def to_code_blocks(self, llr: tf.Tensor) -> tf.Tensor:
        llr = tf.convert_to_tensor(llr, tf.float32)
        tf.debugging.assert_rank(llr, 3)
        tf.debugging.assert_equal(tf.shape(llr)[1], self.num_tx)
        tf.debugging.assert_equal(tf.shape(llr)[2], self.tb_length)
        zero = tf.zeros([tf.shape(llr)[0], self.num_tx, 1], llr.dtype)
        known = tf.fill(tf.shape(zero), -tf.cast(self.decoder.llr_max, llr.dtype))
        values = tf.concat([llr*self.scrambling_signs, zero, known], -1)
        return tf.gather(values, self.recovery_indices, axis=-1)

    def from_code_blocks(self, values):
        values = tf.reshape(values, [tf.shape(values)[0], self.num_tx, -1])
        return tf.gather(values, self.transmitted_indices, axis=-1)*self.scrambling_signs

    def __call__(self, llr: tf.Tensor) -> SoftInformationOutput:
        return self.payload(self.decoder(self.to_code_blocks(llr))[..., :self.info_length])

    def payload(self, info_logits):
        tf.debugging.assert_all_finite(
            info_logits, "soft information logits are not finite"
        )
        hard = tf.cast(info_logits > 0.0, tf.float32)
        if self.cb_crc_length:
            soft_no_cb_crc = info_logits[..., :-self.cb_crc_length]
            hard_no_cb_crc = hard[..., :-self.cb_crc_length]
        else:
            soft_no_cb_crc = info_logits
            hard_no_cb_crc = hard
        full_length = int(self.tb_decoder.tb_size) + self.tb_crc_length
        soft_tb = tf.reshape(
            soft_no_cb_crc,
            [tf.shape(info_logits)[0], self.num_tx, full_length],
        )
        hard_tb = tf.reshape(
            hard_no_cb_crc,
            [tf.shape(info_logits)[0], self.num_tx, full_length],
        )
        soft_payload = soft_tb[..., : int(self.tb_decoder.tb_size)]
        hard_payload, crc_status = self.payload_crc(hard_tb)
        if self.tb_encoder.k_padding > 0:
            padding = int(self.tb_encoder.k_padding)
            soft_payload = soft_payload[..., :-padding]
            hard_payload = hard_payload[..., :-padding]
        return SoftInformationOutput(
            tf.cast(soft_payload, tf.float32),
            tf.cast(hard_payload, self.tb_decoder.dtype),
            tf.squeeze(tf.cast(crc_status, tf.bool), axis=-1),
        )


class DecoderAssistedTB:
    def __init__(self, tb_encoder, tb_decoder, num_iter=20, first_iterations=8):
        if not 0 < first_iterations < num_iter:
            raise ValueError("The first decoding stage must use part of the total iteration budget")
        self.first = SoftTBInformationDecoder(tb_encoder, tb_decoder, first_iterations,
                                             stateful=True, return_infobits=False)
        self.last = SoftTBInformationDecoder(tb_encoder, tb_decoder, num_iter-first_iterations,
                                            stateful=True)
        bits = int(tb_encoder._num_bits_per_symbol)
        with tf.init_scope():
            shape = (self.first.num_tx, self.first.tb_length // bits, bits)
            addresses = np.broadcast_to(np.asarray(self.first.transmitted_indices),
                                        (self.first.num_tx, self.first.tb_length)).reshape(shape)
            self.symbol_indices = tf.constant(addresses, tf.int32)
            self.symbol_signs = tf.reshape(self.first.scrambling_signs, shape)

    def evidence(self, llr):
        inputs = self.first.to_code_blocks(llr)
        maximum = self.first.decoder.llr_max
        inputs = tf.clip_by_value(inputs, -maximum, maximum)
        posterior, state = self.first.decoder([inputs, None])
        return inputs, posterior-inputs, state, posterior

    def finish(self, llr, state):
        logits, _ = self.last.decoder([llr, state])
        return self.last.payload(logits[..., :self.last.info_length])
