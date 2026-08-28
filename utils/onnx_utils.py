# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from tensorflow.keras import Model
from sionna.channel import OFDMChannel, gen_single_sector_topology
from sionna.utils import BinarySource, ebnodb2no, insert_dims
from sionna.ofdm import LSChannelEstimator
from sionna.utils import flatten_dims, flatten_last_dims, compute_ber, hard_decisions, expand_to_rank
import numpy as np
import tensorflow as tf
from sionna.nr import TBDecoder


class DataGeneratorAerial(Model):
    """DataGeneratorAerial(sys_parameters, **kwargs)"""

    def __init__(self, sys_parameters, training=False):

        super().__init__()

        self._sys_parameters = sys_parameters
        self._training = training

        self._source = BinarySource()
        self._transmitter = sys_parameters.transmitters[0]

        self._channel = sys_parameters.channel


        max_num_tx = sys_parameters.max_num_tx
        rg = sys_parameters.transmitters[0]._resource_grid

        self._ls_est = LSChannelEstimator(resource_grid=rg,
                                          interpolation_type=None,
                                          interpolator=None)

        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._w = sys_parameters.transmitters[0]._precoder._w
        else:
            self._w = tf.ones([sys_parameters.max_num_tx,
                               sys_parameters.num_antenna_ports, 1],
                               tf.complex64)

        self._w = insert_dims(self._w, 2, 1)

        self._ls_est_nn = LSChannelEstimator(resource_grid=rg,
                                             interpolation_type="nn")

        self._pilots = rg.pilot_pattern.pilots
        self._pilot_ind = self._ls_est._pilot_ind
        self._ls_nn_ind = self._ls_est_nn._interpol._gather_ind

        dmrs_ofdm_pos_ = []
        dmrs_subcarrier_pos_ = []
        rg_ = sys_parameters.transmitters[0].resource_grid.build_type_grid()
        rg_np = rg_.numpy()

        for i in range(rg_np.shape[0]):
            idx = np.where(rg_np[i,...]==1)
            dmrs_ofdm_pos_.append(np.unique(idx[1]))

            p = self._pilots[i,0,:]

            idx = np.unique(idx[-1])
            p = p.numpy()[idx]
            idx_active_pilots = idx[np.where(np.abs(p)>0)[0]]

            idx_per_prb = idx_active_pilots[np.where(idx_active_pilots<12)]
            dmrs_subcarrier_pos_.append(idx_per_prb)

        self._dmrs_ofdm_pos = tf.cast(np.stack(dmrs_ofdm_pos_), tf.int32)
        self._dmrs_subcarrier_pos = tf.cast(np.stack(dmrs_subcarrier_pos_),
                                            tf.int32)


        rg_type = rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        self._pilots = rg.pilot_pattern.pilots

        pilots_only = tf.scatter_nd(
                            pilot_ind,
                            flatten_last_dims(self._pilots , 3),
                            rg_type.shape)
        self.pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)

        pilot_ind = np.array(self.pilot_ind)

        pilot_ind_sorted = [ [] for _ in range(max_num_tx) ]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        pilots_dist_time = np.zeros([   max_num_tx,
                                        rg.num_ofdm_symbols,
                                        rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        rg.num_ofdm_symbols,
                                        rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(rg.num_ofdm_symbols)
        f_ind = np.arange(rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):

                pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
                pilots_dist_time[tx_ind, :, :, i] = pt

                pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
        nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
        nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                            axis=1, keepdims=True)
        std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
        nearest_pilot_dist_time = np.where(std_ > 0.,
                                           nearest_pilot_dist_time / std_,
                                           nearest_pilot_dist_time)
        nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                            axis=2, keepdims=True)
        std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
        nearest_pilot_dist_freq = np.where(std_ > 0.,
                                           nearest_pilot_dist_freq / std_,
                                           nearest_pilot_dist_freq)

        nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                        nearest_pilot_dist_freq],
                                        axis=-1)
        nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
        self._nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                                [0, 2, 1, 3])

        self._pe = self._nearest_pilot_dist[:self._sys_parameters.max_num_tx]

    def _active_dmrs_mask(self, batch_size, num_tx, max_num_tx):
        """Sample mask of num_tx active users"""

        max_num_tx = tf.cast(max_num_tx, tf.int32)
        num_tx = tf.cast(num_tx, tf.int32)
        r = tf.range(max_num_tx, dtype=tf.int32)
        r = tf.expand_dims(r, axis=0)
        r = tf.tile(r, (batch_size,1))
        x = tf.where(r<tf.cast(num_tx, tf.int32),
                     tf.ones_like(r),
                     tf.zeros_like(r))
        x = tf.expand_dims(x, axis=-1)
        x_p = tf.map_fn(lambda v: tf.random.shuffle(v), x)
        x_p = tf.cast(x_p, tf.float32)
        return tf.squeeze(x_p, axis=-1)

    def _set_transmitter_random_pilots(self):
        """Set transmitter random pilots."""
        pilot_set = self._sys_parameters.pilots
        num_pilots = tf.shape(pilot_set)[0]
        random_pilot_ind = tf.random.uniform((), 0, num_pilots, dtype=tf.int32)
        pilots = tf.gather(pilot_set, random_pilot_ind, axis=0)
        self._transmitter.pilot_pattern.pilots = pilots

    def call(self, batch_size, ebno_db, num_tx=None):
        """Call."""

        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx

        dmrs_port_mask = self._active_dmrs_mask(batch_size, num_tx,
                                                self._sys_parameters.max_num_tx)


        b = self._source([batch_size,
                          self._sys_parameters.max_num_tx,
                          self._transmitter._tb_size])

        c = self._sys_parameters.transmitters[0]._tb_encoder(b)

        if self._training:
            self._set_transmitter_random_pilots()

        x = self._transmitter(b)

        a_tx = expand_to_rank(dmrs_port_mask, tf.rank(x), axis=-1)
        x = tf.multiply(x, tf.cast(a_tx, tf.complex64))


        if self._sys_parameters.frequency_offset is not None:
            x = self._sys_parameters.frequency_offset(x)

        if self._sys_parameters.ebno:
            no = ebnodb2no(
                    ebno_db,
                    self._transmitter._num_bits_per_symbol,
                    self._transmitter._target_coderate,
                    self._transmitter._resource_grid)
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


        h_hat, _ = self._ls_est((y, 0.1))
        h = tf.transpose(h, perm=[0,1,3,5,6,2,4])
        h = tf.matmul(h, self._w)
        h = tf.transpose(h, perm=[0,1,5,2,6,3,4])


        y = tf.squeeze(y, axis=1)
        y = tf.transpose(y, (0,3,2,1))

        h_hat = tf.squeeze(h_hat, axis=(1,4))

        s = h_hat.shape.as_list()
        s[-1] = s[-1]//self._sys_parameters.num_cdm_groups_without_data
        h_hat = tf.gather_nd(h_hat, tf.where(tf.abs(h_hat)>1e-7))
        h_hat = tf.reshape(h_hat, s)

        h_hat = tf.transpose(h_hat, (0, 3, 2, 1))

        nrx_inputs = [tf.math.real(y).numpy(),
                      tf.math.imag(y).numpy(),
                      tf.math.real(h_hat).numpy(),
                      tf.math.imag(h_hat).numpy(),
                      dmrs_port_mask.numpy(),
                      self._dmrs_ofdm_pos.numpy(),
                      self._dmrs_subcarrier_pos.numpy(),]
        return nrx_inputs, c, b, h


class DataEvaluator():
    """Data Evaluator."""

    def __init__(self, sys_parameters):

        super().__init__()

        self.num_streams = sys_parameters.sm.num_streams_per_tx
        self.num_tx = sys_parameters.sm.num_tx
        self.num_bits_per_symbol \
                    = sys_parameters.transmitters[0]._num_bits_per_symbol

        rg = sys_parameters.transmitters[0]._resource_grid

        mask = rg.pilot_pattern.mask
        num_data_symbols = rg.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")

        self.data_ind = data_ind[...,:num_data_symbols]
        self.eff_sub_ind = rg.effective_subcarrier_ind
        self.stream_ind = sys_parameters.sm.stream_ind

        self._tb_decoder = TBDecoder(sys_parameters.transmitters[0]._tb_encoder)

    def post_process_llrs(self, llr):
        llr = -1.*llr


        llr = tf.transpose(llr, [0, 2, 4, 3, 1])

        llr = tf.gather(llr, self.eff_sub_ind, axis=-2)

        llr = tf.transpose(llr, [1, 2, 3, 4, 0])

        llr = flatten_dims(llr, 2, 1)

        llr = tf.gather(llr, self.data_ind[:,0,:], batch_dims=1, axis=1)

        llr = tf.transpose(llr, [3, 0, 1, 2])
        llr = llr[:,:self.num_tx]

        llr = flatten_last_dims(llr, 2)

        return llr

    def __call__(self, llrs, bits):

        llr = self.post_process_llrs(llrs)

        b_hat = hard_decisions(llr)
        ber = compute_ber(b_hat, bits)

        u_hat,_ = self._tb_decoder(llr)

        return llr, ber, u_hat


def precalculate_nnrx_indices(sys_parameters):
    """Pre-calculate static pilots and pilot indices"""


    max_num_tx = sys_parameters.max_num_tx
    rg = sys_parameters.transmitters[0]._resource_grid

    ls_est = LSChannelEstimator(
                    sys_parameters.transmitters[0]._resource_grid,
                    interpolation_type="nn")

    pilots = rg.pilot_pattern.pilots
    pilot_ind = ls_est._pilot_ind
    ls_nn_ind = ls_est._interpol._gather_ind


    rg_type = rg.build_type_grid()[:,0]
    p_ind = tf.where(rg_type==1)
    pilots = rg.pilot_pattern.pilots

    pilots_only = tf.scatter_nd(
                        p_ind,
                        flatten_last_dims(pilots , 3),
                        rg_type.shape)
    p_ind = tf.where(tf.abs(pilots_only) > 1e-3)
    p_ind = np.array(p_ind)

    p_ind_sorted = [ [] for _ in range(max_num_tx) ]

    for p_ind in p_ind:
        tx_ind = p_ind[0]
        re_ind = p_ind[1:]
        p_ind_sorted[tx_ind].append(re_ind)
    p_ind_sorted = np.array(p_ind_sorted)

    pilots_dist_time = np.zeros([   max_num_tx,
                                    rg.num_ofdm_symbols,
                                    rg.fft_size,
                                    p_ind_sorted.shape[1]])
    pilots_dist_freq = np.zeros([   max_num_tx,
                                    rg.num_ofdm_symbols,
                                    rg.fft_size,
                                    p_ind_sorted.shape[1]])

    t_ind = np.arange(rg.num_ofdm_symbols)
    f_ind = np.arange(rg.fft_size)

    for tx_ind in range(max_num_tx):
        for i, p_ind in enumerate(p_ind_sorted[tx_ind]):

            pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
            pilots_dist_time[tx_ind, :, :, i] = pt

            pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
            pilots_dist_freq[tx_ind, :, :, i] = pf

    nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
    nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
    nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                        axis=1, keepdims=True)
    std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
    nearest_pilot_dist_time = np.where(std_ > 0.,
                                        nearest_pilot_dist_time / std_,
                                        nearest_pilot_dist_time)
    nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                        axis=2, keepdims=True)
    std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
    nearest_pilot_dist_freq = np.where(std_ > 0.,
                                        nearest_pilot_dist_freq / std_,
                                        nearest_pilot_dist_freq)

    nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                    nearest_pilot_dist_freq],
                                    axis=-1)
    nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
    nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                            [0, 2, 1, 3])

    pe = nearest_pilot_dist[:sys_parameters.max_num_tx]

    return pilots.numpy(), pe.numpy(), pilot_ind.numpy(), ls_nn_ind.numpy()
