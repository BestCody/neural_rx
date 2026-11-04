# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Classical PUSCH baselines restored from utils/baseline_rx.py of NVlabs/neural_rx
# (commit d4c5420), extended with per-slot DMRS and per-TB UE activity.

from __future__ import annotations

from itertools import combinations

import numpy as np
import tensorflow as tf
from sionna.mimo import StreamManagement
from sionna.nr import PUSCHLSChannelEstimator, PUSCHReceiver, PUSCHTransmitter, TBDecoder
from sionna.ofdm import KBestDetector, LinearDetector, LMMSEInterpolator
from sionna.utils import flatten_dims, flatten_last_dims, split_dim

KBEST_LIST_SIZE = 64
SYSTEMS = ("baseline_lslin_lmmse", "baseline_lmmse_kbest", "baseline_perf_csi_kbest")


def lmmse_reduction(num_prbs):
    """Upstream's automatic split: the smallest PRB-count factor product above 20."""
    prime_factors, value, factor = [], num_prbs, 2
    while factor < num_prbs:
        if value % factor == 0:
            prime_factors.append(factor)
            value //= factor
        else:
            factor += 1
    products = [int(np.prod(subset)) for size in range(1, len(prime_factors)+1)
                for subset in combinations(prime_factors, size)]
    best = min(product for product in products if product > 20)
    if num_prbs % best:
        raise ValueError("The PRB count must be a multiple of the LMMSE block size")
    return num_prbs//best


class LowComplexityPUSCHLMMSEEstimator(PUSCHLSChannelEstimator):
    """LMMSE interpolation over consecutive PRB blocks, keeping large grids feasible."""

    def __init__(self, resource_grid, dmrs_length, dmrs_additional_position, num_cdm_groups_without_data,
                 interpolator, num_rx_antennas, num_tx, reduction):
        super().__init__(resource_grid=resource_grid, dmrs_length=dmrs_length,
                         dmrs_additional_position=dmrs_additional_position,
                         num_cdm_groups_without_data=num_cdm_groups_without_data,
                         interpolator=interpolator, dtype=tf.complex64)
        self._reduction = reduction
        self._num_rx_antennas = num_rx_antennas
        self._num_tx = num_tx
        self._num_pilots = int(resource_grid.num_pilot_symbols.numpy())

    def _split(self, value):
        value = split_dim(value, [2, -1], axis=tf.rank(value)-1)
        value = split_dim(value, [self._reduction, -1], axis=tf.rank(value)-1)
        value = tf.transpose(value, perm=[6, 0, 1, 2, 3, 4, 5, 7])
        return flatten_dims(flatten_last_dims(value, 2), 2, 0)

    def _merge(self, value):
        value = split_dim(value, [self._reduction, -1], 0)
        value = tf.transpose(value, perm=[1, 2, 3, 4, 5, 6, 0, 7])
        return flatten_last_dims(value, 2)

    def call(self, inputs):
        y, no = inputs
        y_flat = flatten_last_dims(self._removed_nulled_scs(y))
        y_pilots = tf.gather(y_flat, self._pilot_ind, axis=-1)
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))
        shape = [y.shape[0], 1, self._num_rx_antennas, self._num_tx, 1, self._num_pilots]
        h_hat, err_var = tf.ensure_shape(h_hat, shape), tf.ensure_shape(err_var, shape)
        h_hat, err_var = self._interpol(self._split(h_hat), self._split(err_var))
        return self._merge(h_hat), self._merge(tf.maximum(err_var, tf.cast(0, err_var.dtype)))


def _lmmse_estimator(parameters, covariance):
    tx = parameters.transmitters[0]
    reduction = lmmse_reduction(parameters.n_size_bwp)
    configs = []
    for config in parameters.pusch_configs[0]:
        small = config.clone()
        small.carrier.n_size_grid = parameters.n_size_bwp//reduction
        configs.append(small)
    small_grid = PUSCHTransmitter(configs).resource_grid
    frequency = np.asarray(covariance["frequency"])[:small_grid.fft_size, :small_grid.fft_size]
    interpolator = LMMSEInterpolator(small_grid.pilot_pattern,
                                     cov_mat_time=tf.constant(covariance["time"], tf.complex64),
                                     cov_mat_freq=tf.constant(frequency, tf.complex64),
                                     cov_mat_space=tf.constant(covariance["space"], tf.complex64),
                                     order="s-f-t")
    dmrs = parameters.pusch_configs[0][0].dmrs
    return LowComplexityPUSCHLMMSEEstimator(tx._resource_grid, dmrs.length, dmrs.additional_position,
                                            dmrs.num_cdm_groups_without_data, interpolator,
                                            parameters.num_rx_antennas, parameters.max_num_tx, reduction)


class ClassicalBaselineReceiver:
    """A stateless baseline receiver; the batch size is fixed by the LMMSE block estimator."""

    def __init__(self, parameters, system, covariance=None):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown baseline {system}")
        self.parameters, self.system = parameters, system
        self.perfect = system == "baseline_perf_csi_kbest"
        tx = parameters.transmitters[0]
        self.tx = tx
        grid, bits = tx._resource_grid, tx._num_bits_per_symbol
        dmrs = parameters.pusch_configs[0][0].dmrs
        if self.perfect:
            estimator = "perfect"
        elif system == "baseline_lslin_lmmse":
            estimator = PUSCHLSChannelEstimator(
                resource_grid=grid, dmrs_length=dmrs.length, dmrs_additional_position=dmrs.additional_position,
                num_cdm_groups_without_data=dmrs.num_cdm_groups_without_data, interpolation_type="lin")
        else:
            if covariance is None:
                raise ValueError("LMMSE estimation requires covariance matrices")
            estimator = _lmmse_estimator(parameters, covariance)
        if system == "baseline_lslin_lmmse":
            detector = LinearDetector("lmmse", "bit", "maxlog", grid, parameters.sm, constellation_type="qam",
                                      num_bits_per_symbol=bits)
        else:
            detector = KBestDetector("bit", parameters.max_num_tx, KBEST_LIST_SIZE, grid, parameters.sm,
                                     constellation_type="qam", num_bits_per_symbol=bits)
        decoder = TBDecoder(tx._tb_encoder, num_bp_iter=parameters.num_bp_iter, cn_type=parameters.cn_type)
        self.receiver = PUSCHReceiver(tx, channel_estimator=estimator, mimo_detector=detector, tb_decoder=decoder,
                                      stream_management=None, input_domain="freq", return_tb_crc_status=True)
        self.estimator = None if self.perfect else estimator
        # With one scheduled UE, a two-stream list would only repeat each symbol for the paused UE.
        self.one_stream = None
        if system != "baseline_lslin_lmmse":
            self.one_stream = [KBestDetector("bit", 1, 2**bits, PUSCHTransmitter([config.clone()])._resource_grid,
                                             StreamManagement(np.ones([1, 1], int), 1), constellation_type="qam",
                                             num_bits_per_symbol=bits)
                               for config in parameters.pusch_configs[0]]
        self._decode, self._decode_one = self._build_graph()

    def set_slot(self, slot_number):
        """Use the DMRS of this slot, as the transmitter did."""
        pilots = tf.gather(self.parameters.pilots, int(slot_number), axis=0)
        self.tx.pilot_pattern.pilots = pilots
        if self.estimator is not None:
            self.estimator._pilot_pattern.pilots = pilots

    def _effective_channel(self, h):
        h = tf.transpose(h, perm=[0, 1, 3, 5, 6, 2, 4])
        return tf.transpose(tf.matmul(h, self.receiver._w), perm=[0, 1, 5, 2, 6, 3, 4])

    def _build_graph(self):
        receiver = self.receiver

        # One trace per batch size: the LMMSE block estimator needs a static batch dimension.
        @tf.function(jit_compile=False)
        def decode(y, h, no, active):
            scheduled = tf.cast(active, tf.float32)[:, None, None, :, None, None, None]
            if self.perfect:
                h_hat = self._effective_channel(h)*tf.cast(scheduled, tf.complex64)
                err_var = tf.zeros([], tf.float32)
            else:
                h_hat, err_var = receiver._channel_estimator([y, no])
                h_hat, err_var = h_hat*tf.cast(scheduled, h_hat.dtype), err_var*scheduled
            llr = receiver._layer_demapper(receiver._mimo_detector([y, h_hat, err_var, no]))
            return receiver._tb_decoder(llr)

        @tf.function(jit_compile=False)
        def decode_one(y, h, no, position):
            if self.perfect:
                h_hat, err_var = self._effective_channel(h), tf.zeros([], tf.float32)
            else:
                h_hat, err_var = receiver._channel_estimator([y, no])
                err_var = err_var[:, :, :, position:position+1]
            llr = self.one_stream[position]([y, h_hat[:, :, :, position:position+1], err_var, no])
            paused = tf.zeros_like(llr)
            llr = tf.concat([llr, paused] if position == 0 else [paused, llr], axis=1)
            return receiver._tb_decoder(receiver._layer_demapper(llr))

        return decode, decode_one

    def decode(self, y, no, active, h=None, slot_number=None):
        if slot_number is not None:
            self.set_slot(slot_number)
        placeholder = tf.zeros([0], tf.complex64) if h is None else tf.convert_to_tensor(h, tf.complex64)
        y, no = tf.convert_to_tensor(y, tf.complex64), tf.convert_to_tensor(no, tf.float32)
        scheduled = np.asarray(active)[0]
        if self.one_stream is not None and not scheduled.all():
            hard, crc = self._decode_one(y, placeholder, no, int(np.argmax(scheduled)))
        else:
            hard, crc = self._decode(y, placeholder, no, tf.convert_to_tensor(active, tf.bool))
        return hard.numpy(), crc.numpy()


def covariance_from_channels(samples):
    """Accumulate E[hh^H] over [draw, UE, antenna, symbol, subcarrier] effective channels."""
    samples = tf.convert_to_tensor(samples, tf.complex64)
    draws = tf.reshape(samples, [-1, *samples.shape[-3:]])
    power = tf.reduce_mean(tf.abs(draws)**2)
    per_subcarrier = tf.transpose(draws, [0, 1, 3, 2])
    per_antenna = tf.transpose(draws, [0, 2, 1, 3])
    time = tf.reduce_mean(tf.matmul(draws, draws, adjoint_b=True), axis=(0, 1))/tf.cast(draws.shape[-1], draws.dtype)
    frequency = (tf.reduce_mean(tf.matmul(per_subcarrier, per_subcarrier, adjoint_b=True), axis=(0, 1))
                 /tf.cast(draws.shape[-2], draws.dtype))
    space = (tf.reduce_mean(tf.matmul(per_antenna, per_antenna, adjoint_b=True), axis=(0, 1))
             /tf.cast(draws.shape[-1], draws.dtype))
    return {"time": time.numpy(), "frequency": frequency.numpy(), "space": space.numpy(), "power": float(power)}


def normalized_covariance(total, count):
    """Unit-diagonal time and frequency factors; the space factor carries the mean channel power."""
    averaged = {name: total[name]/count for name in ("time", "frequency", "space")}
    power = float(total["power"]/count)
    result = {}
    for name, matrix in averaged.items():
        result[name] = matrix/np.mean(np.real(np.diag(matrix)))
    result["space"] = result["space"]*power
    result["power"] = power
    return result
