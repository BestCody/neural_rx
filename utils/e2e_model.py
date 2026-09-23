# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import tensorflow as tf
from tensorflow.keras import Model
from sionna.channel import gen_single_sector_topology
from sionna.utils import BinarySource, ebnodb2no, expand_to_rank, log10
from .baseline_rx import BaselineReceiver
from .neural_rx import NeuralPUSCHReceiver


class E2E_Model(Model):

    def __init__(self, sys_parameters, training=False, return_tb_status=False,
                 mcs_arr_eval_idx=0):

        super().__init__()

        assert isinstance(mcs_arr_eval_idx, int), "E2E Model can only evaluate one MCS at a time."

        self._sys_parameters = sys_parameters
        self._training = training
        self._return_tb_status = return_tb_status
        self._mcs_arr_eval_idx = mcs_arr_eval_idx


        self._source = BinarySource()
        self._transmitters = sys_parameters.transmitters

        self._channel = sys_parameters.channel


        if self._sys_parameters.system == 'baseline_perf_csi_kbest':
            self._sys_name = "Baseline - Perf. CSI & K-Best"
            self._receiver = BaselineReceiver(
                                self._sys_parameters,
                                return_tb_status=return_tb_status,
                                mcs_arr_eval_idx=mcs_arr_eval_idx)

        elif self._sys_parameters.system == "nrx":
            self._sys_name = "Neural Receiver"
            self._receiver = NeuralPUSCHReceiver(
                                    self._sys_parameters,
                                    training)
        else:
            raise NotImplementedError("Unknown system selected!")

    def _active_dmrs_mask(self, batch_size, num_tx, max_num_tx):

        max_num_tx = tf.cast(max_num_tx, tf.int32)
        num_tx = tf.cast(num_tx, tf.int32)
        r = tf.range(max_num_tx, dtype=tf.int32)
        r = tf.expand_dims(r, axis=0)
        r = tf.tile(r, (batch_size,1))
        x = tf.where(r<tf.cast(num_tx, tf.int32),
                     tf.ones_like(r),
                     tf.zeros_like(r))
        x = tf.expand_dims(x, axis=-1)
        x_p = tf.map_fn(tf.random.shuffle, x)
        x_p = tf.cast(x_p, tf.float32)
        return tf.squeeze(x_p, axis=-1)

    def _mask_active_dmrs(self, b, b_hat, num_tx, active_dmrs,
                          mcs_arr_eval_idx, tb_crc_status=None):
        batch_size = tf.shape(b)[0]

        a_mask = expand_to_rank(active_dmrs, tf.rank(b_hat), axis=-1)
        a_mask = tf.broadcast_to(a_mask, tf.shape(b_hat))

        b_hat = tf.boolean_mask(b_hat, a_mask)
        b_hat = tf.reshape(b_hat,
                           (batch_size, num_tx, self._transmitters
                            [mcs_arr_eval_idx]._tb_size))

        b = tf.boolean_mask(b, a_mask)
        b = tf.reshape(b, (batch_size, num_tx, self._transmitters
                           [mcs_arr_eval_idx]._tb_size))

        if tb_crc_status is not None:
            a_mask = expand_to_rank(active_dmrs, tf.rank(tb_crc_status),
                                    axis=-1)
            a_mask = tf.broadcast_to(a_mask, tf.shape(tb_crc_status))
            tb_crc_status = tf.boolean_mask(tb_crc_status, a_mask)
            tb_crc_status = tf.reshape(tb_crc_status, (batch_size, num_tx))
            return b, b_hat, tb_crc_status

        return b, b_hat

    def _set_transmitter_random_pilots(self):
        pilot_set = self._sys_parameters.pilots
        num_pilots = tf.shape(pilot_set)[0]
        random_pilot_ind = tf.random.uniform((), 0, num_pilots, dtype=tf.int32)
        pilots = tf.gather(pilot_set, random_pilot_ind, axis=0)
        for mcs_list_idx in range(len(self._sys_parameters.mcs_index)):
            self._transmitters[mcs_list_idx].pilot_pattern.pilots = pilots

    def call(self, batch_size, ebno_db, num_tx=None, output_nrx_h_hat=False,
             mcs_arr_eval_idx=None, mcs_ue_mask=None, active_dmrs=None):

        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx

        if mcs_arr_eval_idx is None:
            mcs_arr_eval_idx = self._mcs_arr_eval_idx

        if active_dmrs is None:
            active_dmrs = self._active_dmrs_mask(
                                batch_size,
                                num_tx,
                                self._sys_parameters.max_num_tx)

        if mcs_ue_mask is None:
            assert isinstance(mcs_arr_eval_idx, int), "Pre-defined MCS UE mask only works if mcs_arr_eval_idx is an integer"
            mcs_ue_mask = tf.one_hot(mcs_arr_eval_idx,
                                     depth=len(self._sys_parameters.mcs_index))
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)
            mcs_ue_mask = tf.tile(mcs_ue_mask,
                                  multiples=[batch_size,
                                             self._sys_parameters.max_num_tx,
                                             1])
            mcs_arr_eval = [mcs_arr_eval_idx]
        else:
            if isinstance(mcs_arr_eval_idx, (list, tuple)):
                assert len(mcs_arr_eval_idx) == len(self._sys_parameters.mcs_index), "mcs_arr_eval_idx list not compatible with length of mcs_index array"
                mcs_arr_eval = mcs_arr_eval_idx
            else:
                mcs_arr_eval = list(range(len(self._sys_parameters.mcs_index)))


        b = []
        for idx in range(len(mcs_arr_eval)):
            b.append(
                self._source([batch_size,
                            self._sys_parameters.max_num_tx,
                            self._transmitters[mcs_arr_eval[idx]]._tb_size]))

        if self._training:
            self._set_transmitter_random_pilots()

        _mcs_ue_mask = tf.cast(expand_to_rank(
                                    tf.gather(mcs_ue_mask,
                                              indices=mcs_arr_eval[0],
                                              axis=2), 5, axis=-1),
                                              dtype=tf.complex64)
        x = _mcs_ue_mask * self._transmitters[mcs_arr_eval[0]](b[0])
        for idx in range(1, len(mcs_arr_eval)):
            _mcs_ue_mask = tf.cast(expand_to_rank(
                                        tf.gather(mcs_ue_mask,
                                                  indices=mcs_arr_eval[idx],
                                                  axis=2), 5, axis=-1),
                                                  dtype=tf.complex64)
            x = x + _mcs_ue_mask * self._transmitters[mcs_arr_eval[idx]](b[idx])

        a_tx = expand_to_rank(active_dmrs, tf.rank(x), axis=-1)
        x = tf.multiply(x, tf.cast(a_tx, tf.complex64))



        if self._sys_parameters.ebno:

            if self._sys_parameters.mask_pilots:
                tx = self._sys_parameters.transmitters[0]
                num_pilots = tf.cast(tx._resource_grid.num_pilot_symbols,
                                     tf.float32)
                num_res = tf.cast(tx._resource_grid.num_resource_elements,
                                  tf.float32)
                ebno_db -= 10.*log10(1. - num_pilots/num_res)

            no = ebnodb2no(
                    ebno_db,
                    self._transmitters[mcs_arr_eval[0]]._num_bits_per_symbol,
                    self._transmitters[mcs_arr_eval[0]]._target_coderate,
                    self._transmitters[mcs_arr_eval[0]]._resource_grid)

        else:
            no = 10**(-ebno_db/10)

        if self._sys_parameters.channel_type in ("UMi", "UMa"):
            if self._sys_parameters.channel_type == "UMi":
                ch_type = 'umi'
            else:
                ch_type = 'uma'
            topology = gen_single_sector_topology(
                        batch_size,
                        self._sys_parameters.max_num_tx,
                        ch_type,
                        min_ut_velocity=self._sys_parameters.min_ut_velocity,
                        max_ut_velocity=self._sys_parameters.max_ut_velocity,
                        indoor_probability=0.)
            self._sys_parameters.channel_model.set_topology(*topology)

        y, h = self._channel([x, no])


        if self._sys_parameters.system == 'baseline_perf_csi_kbest':

            b_hat = self._receiver([y, h, no])

            if self._return_tb_status:
                b_hat, tb_crc_status = b_hat
            else:
                tb_crc_status = None
            return self._mask_active_dmrs(b[0], b_hat, num_tx,
                                          active_dmrs, mcs_arr_eval[0],
                                          tb_crc_status)

        elif self._sys_parameters.system == "nrx":

            if self._training:
                losses = self._receiver([y, active_dmrs, b, h, mcs_ue_mask],
                                        mcs_arr_eval)
                return losses
            else:
                b_hat, h_hat_refined, h_hat, tb_crc_status = \
                                self._receiver((y, active_dmrs),
                                                mcs_arr_eval,
                                                mcs_ue_mask_eval=mcs_ue_mask)

                b, b_hat, tb_crc_status = self._mask_active_dmrs(b[0], b_hat,
                                                                num_tx,
                                                                active_dmrs,
                                                                mcs_arr_eval[0],
                                                                tb_crc_status)

                h_hat_output_shape = tf.concat([[batch_size, num_tx],
                                                tf.shape(h_hat_refined)[2:]],
                                                axis=0)
                a_mask = expand_to_rank(active_dmrs,
                                        tf.rank(h_hat_refined), axis=-1)
                a_mask = tf.broadcast_to(a_mask, tf.shape(h_hat_refined))
                if h_hat is not None:
                    h_hat = tf.boolean_mask(h_hat, a_mask)
                    h_hat = tf.reshape(h_hat, h_hat_output_shape)
                h_hat_refined = tf.boolean_mask(h_hat_refined, a_mask)
                h_hat_refined = tf.reshape(h_hat_refined, h_hat_output_shape)
                h = self._receiver.preprocess_channel_ground_truth(h)
                h = tf.boolean_mask(h, a_mask)
                h = tf.reshape(h, h_hat_output_shape)

                if self._return_tb_status:
                    if output_nrx_h_hat:
                        return b, b_hat, tb_crc_status, h, h_hat_refined, h_hat
                    else:
                        return b, b_hat, tb_crc_status
                else:
                    if output_nrx_h_hat:
                        return b, b_hat, h, h_hat_refined, h_hat
                    else:
                        return b, b_hat
        else:
            raise ValueError("Unknown system selected!")
