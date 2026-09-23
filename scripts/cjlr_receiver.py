#!/usr/bin/env python3

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sionna.fec.ldpc.decoding import LDPC5GDecoder


class SoftBPOutput(NamedTuple):

    app: tf.Tensor
    extrinsic: tf.Tensor
    state: tf.RaggedTensor
    clipped_input: tf.Tensor


class ExactJointOutput(NamedTuple):

    zero_message: tf.Tensor
    code_message: tf.Tensor
    delta: tf.Tensor
    joint_probability: tf.Tensor


class SoftInformationOutput(NamedTuple):

    logits: tf.Tensor
    hard_bits: tf.Tensor
    crc_status: tf.Tensor
    state: tf.RaggedTensor


class OneLayerGridCodeMap:

    def __init__(self, ofdm, num_tx: int, bits_per_symbol: int):
        self.num_tx = int(num_tx)
        self.bits_per_symbol = int(bits_per_symbol)
        if self.num_tx < 1 or self.bits_per_symbol < 1:
            raise ValueError("invalid grid-code dimensions")
        layer_demappers = ofdm._layer_demappers
        if layer_demappers is None or len(layer_demappers) != 1:
            raise ValueError("CJLR Stage 0 requires one fixed MCS")
        layer = layer_demappers[0]
        if int(layer._mapper.num_layers) != 1:
            raise ValueError("CJLR Stage 0 requires one layer per UE")
        if int(layer._num_bits_per_symbol) != self.bits_per_symbol:
            raise ValueError("modulation order differs from the protocol")
        indices = np.asarray(ofdm._rg_demapper._data_ind.numpy(), np.int32)
        if indices.shape[0] < self.num_tx or indices.shape[1] != 1:
            raise ValueError("unsupported data-index geometry")
        indices = indices[:self.num_tx, 0]
        if not all(np.array_equal(indices[0], value) for value in indices):
            raise ValueError("CJLR Stage 0 requires aligned data resources")
        self.data_indices = tf.convert_to_tensor(indices, tf.int32)
        self.common_indices = tf.convert_to_tensor(indices[0], tf.int32)
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

    def gather_shared_grid(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.convert_to_tensor(value)
        tf.debugging.assert_rank(value, 4)
        value = tf.transpose(value, [0, 2, 1, 3])
        shape = tf.shape(value)
        value = tf.reshape(value, [shape[0], shape[1] * shape[2], shape[3]])
        return tf.gather(value, self.common_indices, axis=1)

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

    def coded_to_data(self, coded: tf.Tensor) -> tf.Tensor:
        coded = tf.convert_to_tensor(coded)
        tf.debugging.assert_rank(coded, 3)
        tf.debugging.assert_equal(tf.shape(coded)[1], self.num_tx)
        tf.debugging.assert_equal(
            tf.shape(coded)[2],
            self.num_data_symbols * self.bits_per_symbol,
        )
        return tf.reshape(
            coded,
            [
                tf.shape(coded)[0],
                self.num_tx,
                self.num_data_symbols,
                self.bits_per_symbol,
            ],
        )


class StatefulSoftTBCode:

    def __init__(self, tb_encoder, tb_decoder, num_iter: int):
        self.tb_encoder = tb_encoder
        self.tb_decoder = tb_decoder
        self.num_iter = int(num_iter)
        if self.num_iter < 0:
            raise ValueError("BP iteration count cannot be negative")
        base = tb_decoder._decoder
        self.decoder = LDPC5GDecoder(
            tb_encoder.ldpc_encoder,
            hard_out=False,
            return_infobits=False,
            prune_pcm=bool(base._prune_pcm),
            num_iter=self.num_iter,
            cn_type=str(base._cn_type),
            stateful=True,
            output_dtype=tf.float32,
        )
        self.num_tx = int(tb_encoder.num_tx)
        self.num_cbs = int(tb_encoder.num_cbs)
        self.tb_length = int(tb_encoder.n)
        self.cb_length = int(tb_encoder.ldpc_encoder.n)
        self.num_fillers = int(
            self.cb_length * self.num_cbs
            - np.sum(np.asarray(tb_encoder.cw_lengths))
        )
        if self.num_fillers < 0:
            raise ValueError("invalid TB code-block filler count")
        self.llr_max = tf.cast(base._llr_max, tf.float32)

    def to_code_blocks(self, llr: tf.Tensor) -> tf.Tensor:
        llr = tf.convert_to_tensor(llr, tf.float32)
        tf.debugging.assert_rank(llr, 3)
        tf.debugging.assert_equal(tf.shape(llr)[1], self.num_tx)
        tf.debugging.assert_equal(tf.shape(llr)[2], self.tb_length)
        if self.tb_decoder._descrambler is not None:
            llr = self.tb_decoder._descrambler(llr)
        if self.num_fillers:
            llr = tf.concat(
                [
                    llr,
                    tf.zeros(
                        [tf.shape(llr)[0], self.num_tx, self.num_fillers],
                        llr.dtype,
                    ),
                ],
                axis=-1,
            )
        llr = tf.gather(llr, self.tb_encoder.output_perm_inv, axis=-1)
        return tf.reshape(
            llr,
            [tf.shape(llr)[0], self.num_tx, self.num_cbs, self.cb_length],
        )

    def from_code_blocks(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.convert_to_tensor(value, tf.float32)
        tf.debugging.assert_rank(value, 4)
        tf.debugging.assert_equal(tf.shape(value)[1], self.num_tx)
        tf.debugging.assert_equal(tf.shape(value)[2], self.num_cbs)
        tf.debugging.assert_equal(tf.shape(value)[3], self.cb_length)
        value = tf.reshape(
            value,
            [tf.shape(value)[0], self.num_tx, self.num_cbs * self.cb_length],
        )
        value = tf.gather(value, self.tb_encoder._output_perm, axis=-1)
        value = value[..., :self.tb_length]
        if self.tb_decoder._descrambler is not None:
            value = self.tb_decoder._descrambler(value)
        return value

    def __call__(
        self,
        llr: tf.Tensor,
        state: tf.RaggedTensor | None = None,
    ) -> SoftBPOutput:
        code_blocks = self.to_code_blocks(llr)
        clipped = tf.clip_by_value(
            code_blocks,
            -self.llr_max,
            self.llr_max,
        )
        app, next_state = self.decoder([code_blocks, state])
        extrinsic = app - clipped
        result = SoftBPOutput(
            self.from_code_blocks(app),
            self.from_code_blocks(extrinsic),
            next_state,
            self.from_code_blocks(clipped),
        )
        for value in result[:2]:
            tf.debugging.assert_all_finite(value, "soft BP output is not finite")
        return result


class StatefulSoftTBInformationDecoder(StatefulSoftTBCode):

    def __init__(self, tb_encoder, tb_decoder, num_iter: int):
        super().__init__(tb_encoder, tb_decoder, num_iter)
        base = tb_decoder._decoder
        self.decoder = LDPC5GDecoder(
            tb_encoder.ldpc_encoder,
            hard_out=False,
            return_infobits=True,
            prune_pcm=bool(base._prune_pcm),
            num_iter=self.num_iter,
            cn_type=str(base._cn_type),
            stateful=True,
            output_dtype=tf.float32,
        )
        self.cb_crc_length = (
            0
            if tb_decoder._cb_crc_decoder is None
            else int(tb_decoder._cb_crc_decoder._encoder.crc_length)
        )
        self.tb_crc_length = int(
            tb_decoder._tb_crc_decoder._encoder.crc_length
        )

    def __call__(
        self,
        llr: tf.Tensor,
        state: tf.RaggedTensor | None = None,
    ) -> SoftInformationOutput:
        code_blocks = self.to_code_blocks(llr)
        info_logits, next_state = self.decoder([code_blocks, state])
        tf.debugging.assert_all_finite(
            info_logits, "soft information logits are not finite"
        )
        hard = tf.cast(info_logits > 0.0, tf.float32)
        if self.cb_crc_length:
            soft_no_cb_crc = info_logits[..., :-self.cb_crc_length]
            hard_no_cb_crc, _ = self.tb_decoder._cb_crc_decoder(hard)
        else:
            soft_no_cb_crc = info_logits
            hard_no_cb_crc = hard
        full_length = int(self.tb_decoder.tb_size) + self.tb_crc_length
        soft_tb = tf.reshape(
            soft_no_cb_crc,
            [tf.shape(llr)[0], self.num_tx, full_length],
        )
        hard_tb = tf.reshape(
            hard_no_cb_crc,
            [tf.shape(llr)[0], self.num_tx, full_length],
        )
        soft_payload = soft_tb[..., : int(self.tb_decoder.tb_size)]
        hard_payload, crc_status = self.tb_decoder._tb_crc_decoder(hard_tb)
        if self.tb_encoder.k_padding > 0:
            padding = int(self.tb_encoder.k_padding)
            soft_payload = soft_payload[..., :-padding]
            hard_payload = hard_payload[..., :-padding]
        return SoftInformationOutput(
            tf.cast(soft_payload, tf.float32),
            tf.cast(hard_payload, self.tb_decoder.dtype),
            tf.squeeze(tf.cast(crc_status, tf.bool), axis=-1),
            next_state,
        )


def constellation_table(mapper) -> tuple[tf.Tensor, tf.Tensor]:
    points = tf.convert_to_tensor(mapper.constellation.points)
    bases = np.asarray(mapper._binary_base.numpy(), np.int32)
    indices = np.arange(int(points.shape[0]), dtype=np.int32)[:, None]
    labels = (indices // bases[None]) % 2
    mapped = mapper(tf.convert_to_tensor(labels.reshape(1, -1), tf.float32))
    tf.debugging.assert_near(tf.reshape(mapped, [-1]), points)
    return points, tf.convert_to_tensor(labels, tf.float32)


def exact_joint_support(
    points: tf.Tensor,
    labels: tf.Tensor,
    num_streams: int,
    candidate_cap: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    points = tf.convert_to_tensor(points)
    labels = tf.convert_to_tensor(labels, tf.float32)
    num_streams = int(num_streams)
    order = int(points.shape[0])
    count = order ** num_streams
    if count > int(candidate_cap):
        raise ValueError("joint support exceeds the frozen candidate cap")
    indices = np.indices([order] * num_streams, dtype=np.int32)
    indices = np.reshape(indices, [num_streams, -1]).T
    indices_tf = tf.convert_to_tensor(indices, tf.int32)
    return (
        tf.gather(points, indices_tf),
        tf.gather(labels, indices_tf),
        indices_tf,
    )


def candidate_log_likelihood(
    received: tf.Tensor,
    channel: tf.Tensor,
    channel_variance: tf.Tensor,
    received_noise: tf.Tensor,
    candidates: tf.Tensor,
) -> tf.Tensor:
    received = tf.convert_to_tensor(received)
    channel = tf.cast(channel, received.dtype)
    candidates = tf.cast(candidates, received.dtype)
    channel_variance = tf.cast(
        tf.convert_to_tensor(channel_variance), received.dtype.real_dtype
    )
    received_noise = tf.cast(
        tf.convert_to_tensor(received_noise), received.dtype.real_dtype
    )
    tf.debugging.assert_rank(received, 2)
    tf.debugging.assert_rank(channel, 3)
    tf.debugging.assert_equal(tf.shape(channel), tf.shape(channel_variance))
    tf.debugging.assert_equal(tf.shape(received), tf.shape(received_noise))
    tf.debugging.assert_equal(tf.shape(channel)[1], tf.shape(candidates)[1])
    tf.debugging.assert_non_negative(channel_variance)
    tf.debugging.assert_positive(received_noise)
    prediction = tf.einsum("nur,cu->ncr", channel, candidates)
    energy = tf.square(tf.abs(candidates))
    variance = received_noise[:, None] + tf.einsum(
        "nur,cu->ncr", channel_variance, energy
    )
    tf.debugging.assert_positive(variance)
    residual = received[:, None] - prediction
    score = -tf.reduce_sum(
        tf.square(tf.abs(residual)) / variance + tf.math.log(variance),
        axis=-1,
    )
    tf.debugging.assert_all_finite(score, "candidate score is not finite")
    return score


def exact_joint_messages(
    score: tf.Tensor,
    extrinsic: tf.Tensor,
    candidate_bits: tf.Tensor,
) -> ExactJointOutput:
    score = tf.convert_to_tensor(score, tf.float32)
    extrinsic = tf.convert_to_tensor(extrinsic, tf.float32)
    candidate_bits = tf.convert_to_tensor(candidate_bits, tf.float32)
    tf.debugging.assert_rank(score, 2)
    tf.debugging.assert_rank(extrinsic, 3)
    tf.debugging.assert_rank(candidate_bits, 3)
    tf.debugging.assert_equal(tf.shape(score)[1], tf.shape(candidate_bits)[0])
    tf.debugging.assert_equal(
        tf.shape(extrinsic)[1:], tf.shape(candidate_bits)[1:]
    )
    full_prior = tf.einsum("num,cum->nc", extrinsic, candidate_bits)
    zero_messages = []
    code_messages = []
    users = int(candidate_bits.shape[1])
    bits_per_symbol = int(candidate_bits.shape[2])
    for user in range(users):
        zero_row = []
        code_row = []
        for bit in range(bits_per_symbol):
            labels = candidate_bits[:, user, bit]
            one = tf.equal(labels, 1.0)
            zero = tf.logical_not(one)
            tf.debugging.assert_positive(tf.reduce_sum(tf.cast(one, tf.int32)))
            tf.debugging.assert_positive(tf.reduce_sum(tf.cast(zero, tf.int32)))
            zero_message = tf.reduce_logsumexp(
                tf.boolean_mask(score, one, axis=1), axis=1
            ) - tf.reduce_logsumexp(
                tf.boolean_mask(score, zero, axis=1), axis=1
            )
            target_prior = extrinsic[:, user, bit, None] * labels[None]
            leave_target = score + full_prior - target_prior
            code_message = tf.reduce_logsumexp(
                tf.boolean_mask(leave_target, one, axis=1), axis=1
            ) - tf.reduce_logsumexp(
                tf.boolean_mask(leave_target, zero, axis=1), axis=1
            )
            zero_row.append(zero_message)
            code_row.append(code_message)
        zero_messages.append(tf.stack(zero_row, axis=-1))
        code_messages.append(tf.stack(code_row, axis=-1))
    zero_message = tf.stack(zero_messages, axis=1)
    code_message = tf.stack(code_messages, axis=1)
    delta = code_message - zero_message
    probability = tf.nn.softmax(score + full_prior, axis=-1)
    for value in (zero_message, code_message, delta, probability):
        tf.debugging.assert_all_finite(value, "joint output is not finite")
    return ExactJointOutput(
        zero_message,
        code_message,
        delta,
        probability,
    )


def two_stream_mutual_information(
    probability: tf.Tensor,
    candidate_indices: tf.Tensor,
    constellation_size: int,
) -> tf.Tensor:
    probability = tf.convert_to_tensor(probability, tf.float32)
    candidate_indices = tf.convert_to_tensor(candidate_indices, tf.int32)
    constellation_size = int(constellation_size)
    tf.debugging.assert_equal(tf.shape(candidate_indices)[1], 2)
    expected = constellation_size * constellation_size
    tf.debugging.assert_equal(tf.shape(candidate_indices)[0], expected)
    joint = tf.reshape(probability, [-1, constellation_size, constellation_size])
    left = tf.reduce_sum(joint, axis=2, keepdims=True)
    right = tf.reduce_sum(joint, axis=1, keepdims=True)
    product = left * right
    tiny = tf.cast(np.finfo(np.float32).tiny, joint.dtype)
    log_ratio = tf.math.log(tf.maximum(joint, tiny)) - tf.math.log(
        tf.maximum(product, tiny)
    )
    term = tf.where(joint > 0.0, joint * log_ratio, tf.zeros_like(joint))
    result = tf.maximum(tf.reduce_sum(term, axis=(1, 2)), 0.0)
    tf.debugging.assert_all_finite(result, "mutual information is not finite")
    return result
