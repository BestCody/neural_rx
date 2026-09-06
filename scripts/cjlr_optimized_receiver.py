#!/usr/bin/env python3
"""Optimized exact CJLR inference primitives."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from cjlr_receiver import SoftInformationOutput


class ExactKernelOutput(NamedTuple):
    """Hold an exact counterfactual delta."""

    delta: tf.Tensor
    score: tf.Tensor


class FusedDecoderTrace(NamedTuple):
    """Hold fused-decoder boundary values."""

    output: SoftInformationOutput
    early_extrinsic: tf.Tensor
    delta: tf.Tensor


class ExactCartesianKernel:
    """Evaluate an exact Cartesian support without bit scans."""

    def __init__(self, points, labels, num_streams: int, layout: str):
        points = np.asarray(points)
        labels = np.asarray(labels, np.float32)
        self.num_streams = int(num_streams)
        self.layout = str(layout)
        if points.ndim != 1 or not np.iscomplexobj(points):
            raise ValueError("constellation points must be complex rank one")
        if labels.ndim != 2 or labels.shape[0] != points.shape[0]:
            raise ValueError("constellation labels do not match points")
        if self.num_streams < 1:
            raise ValueError("at least one stream is required")
        if self.layout not in ("flat", "cartesian"):
            raise ValueError("unsupported exact-kernel layout")
        if not np.all(np.logical_or(labels == 0.0, labels == 1.0)):
            raise ValueError("constellation labels must be binary")
        for bit in range(labels.shape[1]):
            if set(labels[:, bit].tolist()) != {0.0, 1.0}:
                raise ValueError("each modulation bit needs both values")
        self.order = int(points.shape[0])
        self.bits_per_symbol = int(labels.shape[1])
        indices = np.indices(
            [self.order] * self.num_streams, dtype=np.int32
        ).reshape(self.num_streams, -1).T
        self.support_width = int(indices.shape[0])
        self.points = tf.convert_to_tensor(points, tf.complex64)
        self.labels = tf.convert_to_tensor(labels, tf.float32)
        self.indices = tf.convert_to_tensor(indices, tf.int32)
        self.candidates = tf.gather(self.points, self.indices)
        self.candidate_bits = tf.gather(self.labels, self.indices)
        self.one_indices = tuple(
            tf.convert_to_tensor(
                np.flatnonzero(labels[:, bit] == 1.0), tf.int32
            )
            for bit in range(self.bits_per_symbol)
        )
        self.zero_indices = tuple(
            tf.convert_to_tensor(
                np.flatnonzero(labels[:, bit] == 0.0), tf.int32
            )
            for bit in range(self.bits_per_symbol)
        )

    def validate_inputs(
        self,
        received,
        channel,
        channel_variance,
        received_noise,
        extrinsic,
    ) -> None:
        """Validate one operator input outside the timed graph."""
        received = tf.convert_to_tensor(received)
        channel = tf.convert_to_tensor(channel)
        channel_variance = tf.convert_to_tensor(channel_variance)
        received_noise = tf.convert_to_tensor(received_noise)
        extrinsic = tf.convert_to_tensor(extrinsic)
        tf.debugging.assert_rank(received, 2)
        tf.debugging.assert_rank(channel, 3)
        tf.debugging.assert_equal(tf.shape(channel), tf.shape(channel_variance))
        tf.debugging.assert_equal(tf.shape(received), tf.shape(received_noise))
        tf.debugging.assert_equal(tf.shape(channel)[1], self.num_streams)
        tf.debugging.assert_equal(tf.shape(extrinsic)[0], tf.shape(received)[0])
        tf.debugging.assert_equal(tf.shape(extrinsic)[1], self.num_streams)
        tf.debugging.assert_equal(
            tf.shape(extrinsic)[2], self.bits_per_symbol
        )
        tf.debugging.assert_non_negative(channel_variance)
        tf.debugging.assert_positive(received_noise)
        for value in (
            received,
            channel,
            channel_variance,
            received_noise,
            extrinsic,
        ):
            if value.dtype.is_complex:
                finite = tf.math.is_finite(tf.math.real(value))
                finite &= tf.math.is_finite(tf.math.imag(value))
            else:
                finite = tf.math.is_finite(value)
            tf.debugging.assert_equal(tf.reduce_all(finite), True)

    def _flat_score(
        self, received, channel, channel_variance, received_noise
    ):
        candidates = tf.cast(self.candidates, received.dtype)
        prediction = tf.einsum("nur,cu->ncr", channel, candidates)
        energy = tf.square(tf.abs(candidates))
        variance = received_noise[:, None] + tf.einsum(
            "nur,cu->ncr", channel_variance, energy
        )
        residual = received[:, None] - prediction
        return -tf.reduce_sum(
            tf.square(tf.abs(residual)) / variance + tf.math.log(variance),
            axis=-1,
        )

    def _cartesian_score(
        self, received, channel, channel_variance, received_noise
    ):
        batch = tf.shape(received)[0]
        antennas = int(received.shape[1])
        joint_shape = [batch] + [self.order] * self.num_streams + [antennas]
        prediction = tf.zeros(joint_shape, received.dtype)
        variance = tf.broadcast_to(
            tf.reshape(
                received_noise,
                [batch] + [1] * self.num_streams + [antennas],
            ),
            joint_shape,
        )
        point_energy = tf.square(tf.abs(self.points))
        for user in range(self.num_streams):
            shape = (
                [batch]
                + [1] * user
                + [self.order]
                + [1] * (self.num_streams - user - 1)
                + [antennas]
            )
            contribution = (
                channel[:, user, None]
                * tf.cast(self.points[None, :, None], received.dtype)
            )
            prediction += tf.reshape(contribution, shape)
            uncertainty = (
                channel_variance[:, user, None]
                * point_energy[None, :, None]
            )
            variance += tf.reshape(uncertainty, shape)
        observation = tf.reshape(
            received,
            [batch] + [1] * self.num_streams + [antennas],
        )
        residual = observation - prediction
        score = -tf.reduce_sum(
            tf.square(tf.abs(residual)) / variance + tf.math.log(variance),
            axis=-1,
        )
        return tf.reshape(score, [batch, self.support_width])

    def _factorized_delta(self, score, extrinsic):
        batch = tf.shape(score)[0]
        score_joint = tf.reshape(
            score, [batch] + [self.order] * self.num_streams
        )
        symbol_prior = tf.einsum("num,qm->nuq", extrinsic, self.labels)
        posterior = score_joint
        for user in range(self.num_streams):
            shape = (
                [batch]
                + [1] * user
                + [self.order]
                + [1] * (self.num_streams - user - 1)
            )
            posterior = posterior + tf.reshape(symbol_prior[:, user], shape)
        paired = tf.stack([score_joint, posterior], axis=-1)
        users = []
        for user in range(self.num_streams):
            axes = tuple(
                axis + 1
                for axis in range(self.num_streams)
                if axis != user
            )
            symbol_scores = (
                tf.reduce_logsumexp(paired, axis=axes)
                if axes
                else paired
            )
            bits = []
            for bit in range(self.bits_per_symbol):
                one = tf.reduce_logsumexp(
                    tf.gather(symbol_scores, self.one_indices[bit], axis=1),
                    axis=1,
                )
                zero = tf.reduce_logsumexp(
                    tf.gather(symbol_scores, self.zero_indices[bit], axis=1),
                    axis=1,
                )
                marginal = one - zero
                bits.append(
                    marginal[:, 1]
                    - extrinsic[:, user, bit]
                    - marginal[:, 0]
                )
            users.append(tf.stack(bits, axis=-1))
        return tf.stack(users, axis=1)

    def __call__(
        self,
        received,
        channel,
        channel_variance,
        received_noise,
        extrinsic,
    ) -> ExactKernelOutput:
        """Return the exact counterfactual correction."""
        received = tf.convert_to_tensor(received)
        channel = tf.cast(channel, received.dtype)
        channel_variance = tf.cast(
            channel_variance, received.dtype.real_dtype
        )
        received_noise = tf.cast(
            received_noise, received.dtype.real_dtype
        )
        extrinsic = tf.cast(extrinsic, tf.float32)
        score = (
            self._flat_score(
                received, channel, channel_variance, received_noise
            )
            if self.layout == "flat"
            else self._cartesian_score(
                received, channel, channel_variance, received_noise
            )
        )
        return ExactKernelOutput(
            self._factorized_delta(tf.cast(score, tf.float32), extrinsic),
            tf.cast(score, tf.float32),
        )


def make_chunked_exact_delta(
    kernel: ExactCartesianKernel,
    chunk_size: int,
    jit_compile: bool = True,
):
    """Build one bounded exact-list graph."""
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    @tf.function(reduce_retracing=True, jit_compile=bool(jit_compile))
    def run(received, channel, variance, noise, extrinsic):
        total = received.shape[0]
        if total is None or int(total) % chunk_size:
            raise ValueError("exact-list length must divide into equal chunks")
        chunks = []
        for start in range(0, int(total), chunk_size):
            stop = start + chunk_size
            chunks.append(kernel(
                received[start:stop],
                channel[start:stop],
                variance[start:stop],
                noise[start:stop],
                extrinsic[start:stop],
            ).delta)
        return tf.concat(chunks, axis=0)

    return run


class FusedCodedJointTBDecoder:
    """Run split BP around one coded joint callback."""

    def __init__(
        self,
        tb_encoder,
        tb_decoder,
        first_iter: int,
        last_iter: int,
        reference_decoder=None,
    ):
        self.tb_encoder = tb_encoder
        self.tb_decoder = tb_decoder
        self.first_iter = int(first_iter)
        self.last_iter = int(last_iter)
        if min(self.first_iter, self.last_iter) < 1:
            raise ValueError("both BP segments must be positive")
        base = tb_decoder._decoder
        self.decoder = reference_decoder
        if self.decoder is None:
            self.decoder = LDPC5GDecoder(
                tb_encoder.ldpc_encoder,
                hard_out=False,
                return_infobits=False,
                prune_pcm=bool(base._prune_pcm),
                num_iter=self.first_iter + self.last_iter,
                cn_type=str(base._cn_type),
                stateful=True,
                output_dtype=tf.float32,
            )
        if not bool(self.decoder._stateful):
            raise ValueError("fused BP requires a stateful decoder")
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
        self.vn_splits = tf.constant(
            self.decoder._vn_row_splits, tf.int32
        )
        self.cn_mask = tf.ragged.constant(
            self.decoder._gen_node_mask(self.decoder._cn_con),
            row_splits_dtype=tf.int32,
        )
        self.cb_crc_length = (
            0
            if tb_decoder._cb_crc_decoder is None
            else int(tb_decoder._cb_crc_decoder._encoder.crc_length)
        )
        self.tb_crc_length = int(
            tb_decoder._tb_crc_decoder._encoder.crc_length
        )

    def _to_code_blocks(self, llr):
        llr = tf.cast(llr, tf.float32)
        if self.tb_decoder._descrambler is not None:
            llr = self.tb_decoder._descrambler(llr)
        if self.num_fillers:
            llr = tf.concat([
                llr,
                tf.zeros(
                    [tf.shape(llr)[0], self.num_tx, self.num_fillers],
                    llr.dtype,
                ),
            ], axis=-1)
        llr = tf.gather(llr, self.tb_encoder.output_perm_inv, axis=-1)
        return tf.reshape(
            llr,
            [tf.shape(llr)[0], self.num_tx, self.num_cbs, self.cb_length],
        )

    def _from_code_blocks(self, value):
        value = tf.reshape(
            value,
            [tf.shape(value)[0], self.num_tx, self.num_cbs * self.cb_length],
        )
        value = tf.gather(value, self.tb_encoder._output_perm, axis=-1)
        value = value[..., :self.tb_length]
        if self.tb_decoder._descrambler is not None:
            value = self.tb_decoder._descrambler(value)
        return value

    def _rate_to_internal(self, value, filler):
        decoder = self.decoder
        flat = tf.reshape(value, [-1, self.cb_length])
        if decoder._encoder.num_bits_per_symbol is not None:
            flat = tf.gather(
                flat, decoder._encoder.out_int_inv, axis=-1
            )
        batch = tf.shape(flat)[0]
        flat = tf.concat([
            tf.zeros([batch, 2 * decoder.encoder.z], flat.dtype),
            flat,
        ], axis=1)
        k_filler = decoder.encoder.k_ldpc - decoder.encoder.k
        punctured = (
            decoder.encoder.n_ldpc
            - k_filler
            - decoder.encoder.n
            - 2 * decoder.encoder.z
            - decoder._nb_pruned_nodes
        )
        flat = tf.concat([
            flat,
            tf.zeros([batch, punctured], flat.dtype),
        ], axis=1)
        systematic = flat[:, :decoder.encoder.k]
        parity_length = (
            decoder.encoder.n_ldpc
            - k_filler
            - decoder.encoder.k
            - decoder._nb_pruned_nodes
        )
        parity = flat[
            :, decoder.encoder.k:decoder.encoder.k + parity_length
        ]
        fillers = tf.fill(
            [batch, k_filler], tf.cast(filler, flat.dtype)
        )
        return tf.concat([systematic, fillers, parity], axis=1)

    def _internal_to_rate(self, value, reference):
        decoder = self.decoder
        flat = tf.reshape(value, [-1, decoder._n_pruned])
        first = flat[:, :decoder.encoder.k]
        second = flat[:, decoder.encoder.k_ldpc:]
        no_filler = tf.concat([first, second], axis=1)
        rate = no_filler[
            :,
            2 * decoder.encoder.z:
            2 * decoder.encoder.z + decoder.encoder.n,
        ]
        if decoder._encoder.num_bits_per_symbol is not None:
            rate = tf.gather(rate, decoder._encoder.out_int, axis=-1)
        shape = tf.concat([
            tf.shape(reference)[:-1],
            tf.constant([self.cb_length], tf.int32),
        ], axis=0)
        return tf.reshape(rate, shape)

    def _bp(self, llr, messages, count: int):
        decoder = self.decoder

        def body(current_llr, current_messages, iteration):
            iteration += 1
            ragged = tf.RaggedTensor.from_row_splits(
                current_messages, self.vn_splits
            )
            ragged = decoder._vn_update(ragged, current_llr)
            if decoder._has_weights:
                ragged = tf.ragged.map_flat_values(
                    decoder._mult_weights, ragged
                )
            check = tf.gather(ragged.flat_values, self.cn_mask, axis=None)
            check = decoder._cn_update(check)
            current_messages = tf.gather(
                check.flat_values, decoder._ind_cn_inv, axis=None
            )
            return current_llr, current_messages, iteration

        def condition(current_llr, current_messages, iteration):
            del current_llr, current_messages
            return iteration < int(count)

        _, messages, _ = tf.while_loop(
            condition,
            body,
            (llr, messages, tf.constant(0)),
            parallel_iterations=1,
            maximum_iterations=int(count),
        )
        return messages

    def _app(self, llr, messages):
        ragged = tf.RaggedTensor.from_row_splits(messages, self.vn_splits)
        return -tf.transpose(llr + tf.reduce_sum(ragged, axis=1))

    def _finish(self, info_logits, messages):
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
            [tf.shape(info_logits)[0], self.num_tx, full_length],
        )
        hard_tb = tf.reshape(
            hard_no_cb_crc,
            [tf.shape(info_logits)[0], self.num_tx, full_length],
        )
        soft_payload = soft_tb[..., :int(self.tb_decoder.tb_size)]
        hard_payload, crc_status = self.tb_decoder._tb_crc_decoder(hard_tb)
        if self.tb_encoder.k_padding > 0:
            padding = int(self.tb_encoder.k_padding)
            soft_payload = soft_payload[..., :-padding]
            hard_payload = hard_payload[..., :-padding]
        state = tf.RaggedTensor.from_row_splits(messages, self.vn_splits)
        return SoftInformationOutput(
            tf.cast(soft_payload, tf.float32),
            tf.cast(hard_payload, self.tb_decoder.dtype),
            tf.squeeze(tf.cast(crc_status, tf.bool), axis=-1),
            state,
        )

    def trace(self, coded, refine):
        """Decode and expose the two fusion boundaries."""
        code_blocks = self._to_code_blocks(coded)
        raw_internal = self._rate_to_internal(code_blocks, -self.llr_max)
        early_internal = tf.clip_by_value(
            raw_internal, -self.llr_max, self.llr_max
        )
        early_llr = -tf.transpose(early_internal)
        messages = tf.zeros(
            [self.decoder._num_edges, tf.shape(early_llr)[1]], tf.float32
        )
        messages = self._bp(early_llr, messages, self.first_iter)
        early_app = self._app(early_llr, messages)
        early_rate = self._internal_to_rate(early_app, code_blocks)
        clipped_rate = tf.clip_by_value(
            code_blocks, -self.llr_max, self.llr_max
        )
        early_extrinsic = self._from_code_blocks(early_rate - clipped_rate)
        delta = tf.cast(refine(early_extrinsic), tf.float32)
        delta_blocks = self._to_code_blocks(delta)
        delta_internal = self._rate_to_internal(delta_blocks, 0.0)
        final_internal = tf.clip_by_value(
            raw_internal + delta_internal, -self.llr_max, self.llr_max
        )
        final_llr = -tf.transpose(final_internal)
        messages = self._bp(final_llr, messages, self.last_iter)
        final_app = self._app(final_llr, messages)
        info_length = int(self.decoder.encoder.k)
        info = final_app[:, :info_length]
        info = tf.reshape(info, [
            tf.shape(code_blocks)[0],
            self.num_tx,
            self.num_cbs,
            info_length,
        ])
        return FusedDecoderTrace(
            self._finish(info, messages), early_extrinsic, delta
        )

    def __call__(self, coded, refine):
        """Decode with an exact refinement between BP segments."""
        return self.trace(coded, refine).output


class UnrolledCodedJointTBDecoder(FusedCodedJointTBDecoder):
    """Insert one exact refinement in a statically unrolled BP graph."""

    def _step(self, llr, messages):
        ragged = tf.RaggedTensor.from_row_splits(messages, self.vn_splits)
        ragged = self.decoder._vn_update(ragged, llr)
        if self.decoder._has_weights:
            ragged = tf.ragged.map_flat_values(
                self.decoder._mult_weights, ragged
            )
        check = tf.gather(ragged.flat_values, self.cn_mask, axis=None)
        check = self.decoder._cn_update(check)
        return tf.gather(
            check.flat_values, self.decoder._ind_cn_inv, axis=None
        )

    def trace(self, coded, refine):
        """Decode with a fixed unrolled BP8-refine-BP12 schedule."""
        coded = tf.cast(coded, tf.float32)
        code_blocks = self._to_code_blocks(coded)
        raw_internal = self._rate_to_internal(code_blocks, -self.llr_max)
        early_internal = tf.clip_by_value(
            raw_internal, -self.llr_max, self.llr_max
        )
        llr = -tf.transpose(early_internal)
        messages = tf.zeros(
            [self.decoder._num_edges, tf.shape(llr)[1]], tf.float32
        )
        early_extrinsic = tf.zeros_like(coded)
        delta = tf.zeros_like(coded)
        for iteration in range(self.first_iter + self.last_iter):
            messages = self._step(llr, messages)
            if iteration + 1 == self.first_iter:
                early_app = self._app(llr, messages)
                early_rate = self._internal_to_rate(early_app, code_blocks)
                clipped_rate = tf.clip_by_value(
                    code_blocks, -self.llr_max, self.llr_max
                )
                early_extrinsic = self._from_code_blocks(
                    early_rate - clipped_rate
                )
                delta = tf.cast(refine(early_extrinsic), tf.float32)
                delta_internal = self._rate_to_internal(
                    self._to_code_blocks(delta), 0.0
                )
                final_internal = tf.clip_by_value(
                    raw_internal + delta_internal,
                    -self.llr_max,
                    self.llr_max,
                )
                llr = -tf.transpose(final_internal)
        final_app = self._app(llr, messages)
        info_length = int(self.decoder.encoder.k)
        info = tf.reshape(
            final_app[:, :info_length],
            [
                tf.shape(code_blocks)[0],
                self.num_tx,
                self.num_cbs,
                info_length,
            ],
        )
        return FusedDecoderTrace(
            self._finish(info, messages), early_extrinsic, delta
        )
