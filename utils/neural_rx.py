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
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, SeparableConv2D, Layer
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, insert_dims, expand_to_rank
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator

class StateInit(Layer):

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        if layer_type=="sepconv":
            layer = SeparableConv2D
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_conv = []
        for n in num_units:
            conv = layer(n, (3,3), padding='same',
                         activation='relu', dtype=dtype)
            self._hidden_conv.append(conv)

        self._output_conv = layer(d_s, (3,3), activation=None,
                                  padding='same', dtype=dtype)

    def call(self, inputs):
        y, pe, h_hat = inputs


        batch_size = tf.shape(y)[0]
        num_tx = tf.shape(pe)[0]


        y = tf.tile(tf.expand_dims(y, axis=1), [1, num_tx, 1, 1, 1])
        y = flatten_dims(y, 2, 0)

        pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
        pe = flatten_dims(pe, 2, 0)

        h_hat = flatten_dims(h_hat, 2, 0)
        z = tf.concat([y, pe, h_hat], axis=-1)

        layers = self._hidden_conv
        for conv in layers:
            z = conv(z)
        z = self._output_conv(z)

        s0 = split_dim(z, [batch_size, num_tx], 0)

        return s0

class AggregateUserStates(Layer):

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))
        self._output_layer = layer(d_s, activation=None, dtype=dtype)

    def call(self, inputs):

        s, active_tx = inputs

        sp = s
        for layer in self._hidden_layers:
            sp = layer(sp)
        sp = self._output_layer(sp)

        active_tx = expand_to_rank(active_tx, tf.rank(sp), axis=-1)
        sp = tf.multiply(sp, active_tx)

        a = tf.reduce_sum(sp, axis=1, keepdims=True) - sp

        p = tf.reduce_sum(active_tx, axis=1, keepdims=True) - 1.
        p = tf.nn.relu(p)

        p = tf.where(p==0., 1., tf.math.divide_no_nan(1.,p))

        a = tf.multiply(a, p)

        return a

class UpdateState(Layer):

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        if layer_type=="sepconv":
            layer = SeparableConv2D
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_conv = []
        for n in num_units:
            conv = layer(n, (3,3), padding='same',
                         activation="relu", dtype=dtype)
            self._hidden_conv.append(conv)

        self._output_conv = layer(d_s, (3,3), padding='same',
                                  activation=None, dtype=dtype)

    def call(self, inputs):
        s, a, pe = inputs


        batch_size = tf.shape(s)[0]
        num_tx = tf.shape(s)[1]

        pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
        pe = flatten_dims(pe, 2, 0)
        s = flatten_dims(s, 2, 0)
        a = flatten_dims(a, 2, 0)
        z = tf.concat([a, s, pe], axis=-1)

        layers = self._hidden_conv
        for conv in layers:
            z = conv(z)
        z = self._output_conv(z)
        z = z + s
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new

class CGNNIt(Layer):

    def __init__(   self,
                    d_s,
                    num_units_agg,
                    num_units_state_update,
                    layer_type_dense="dense",
                    layer_type_conv="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        self._state_aggreg = AggregateUserStates(d_s,
                                                 num_units_agg,
                                                 layer_type_dense,
                                                 dtype=dtype)

        self._state_update = UpdateState(d_s,
                                         num_units_state_update,
                                         layer_type_conv,
                                         dtype=dtype)

    def call(self, inputs):
        s, pe, active_tx = inputs


        a = self._state_aggreg((s, active_tx))

        s_new = self._state_update((s, a, pe))

        return s_new

class ReadoutLLRs(Layer):

    def __init__(   self,
                    num_bits_per_symbol,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))

        self._output_layer = layer(num_bits_per_symbol,
                                   activation=None, dtype=dtype)

    def call(self, s):


        z = s
        for layer in self._hidden_layers:
            z = layer(z)
        llr = self._output_layer(z)

        return llr

class ReadoutChEst(Layer):

    def __init__(   self,
                    num_rx_ant,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))
        self._output_layer = layer(2*num_rx_ant, activation=None, dtype=dtype)

    def call(self, s):


        z = s
        for layer in self._hidden_layers:
            z = layer(z)
        h_hat = self._output_layer(z)

        return h_hat

class CGNN(Model):

    def __init__(   self,
                    num_bits_per_symbol,
                    num_rx_ant,
                    num_it,
                    d_s,
                    num_units_init,
                    num_units_agg,
                    num_units_state ,
                    num_units_readout,
                    layer_type_dense,
                    layer_type_conv,
                    layer_type_readout,
                    training=False,
                    apply_multiloss=False,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype,**kwargs)

        self._training = training

        self._apply_multiloss = apply_multiloss

        self._s_init = []
        for _ in num_bits_per_symbol:
            self._s_init.append(
                StateInit(  d_s,
                            num_units_init,
                            layer_type=layer_type_conv,
                            dtype=dtype))

        self._iterations = []
        for i in range(num_it):
            it = CGNNIt(    d_s,
                            num_units_agg[i],
                            num_units_state[i],
                            layer_type_dense=layer_type_dense,
                            layer_type_conv=layer_type_conv,
                            dtype=dtype)
            self._iterations.append(it)
        self._num_it = num_it

        self._readout_llrs = []
        for num_bits in num_bits_per_symbol:
            self._readout_llrs.append(
                ReadoutLLRs(num_bits,
                                        num_units_readout,
                                        layer_type=layer_type_readout,
                                        dtype=dtype))
        self._readout_chest = ReadoutChEst(num_rx_ant,
                                           num_units_readout,
                                           layer_type=layer_type_readout,
                                           dtype=dtype)

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

    @property
    def apply_multiloss(self):
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

    @property
    def num_it(self):
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._iterations)),\
            "Invalid number of iterations"
        self._num_it = val

    @staticmethod
    def input_scale(y):
        norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
        return tf.math.divide_no_nan(1.0, tf.sqrt(norm))

    def initialize_state(self, inputs):
        y, pe, channel, _, mcs_mask = inputs
        norm = self.input_scale(y)
        y = y * norm
        if channel is not None:
            channel = channel * tf.expand_dims(norm, axis=1)
        initial_inputs = (y, pe, channel)
        state = self._s_init[0](initial_inputs) * expand_to_rank(
            tf.gather(mcs_mask, indices=0, axis=2), 5, axis=-1
        )
        for index in range(1, self._num_mcss_supported):
            state += self._s_init[index](initial_inputs) * expand_to_rank(
                tf.gather(mcs_mask, indices=index, axis=2), 5, axis=-1
            )
        return state

    def call(self, inputs):
        _, pe, _, active_tx, _ = inputs
        s = self.initialize_state(inputs)

        llrs = []
        h_hats = []
        for i in range(self._num_it):
            it = self._iterations[i]
            s = it([s, pe, active_tx])

            if (self._training and self._apply_multiloss) or i==self._num_it-1:
                llrs_ = []
                for idx in range(self._num_mcss_supported):
                    llrs_.append(self._readout_llrs[idx](s))
                llrs.append(llrs_)
                h_hats.append(self._readout_chest(s))

        return llrs, h_hats

class CGNNOFDM(Model):

    def __init__(self,
                 sys_parameters,
                 max_num_tx,
                 training,
                 num_it=5,
                 d_s=32,
                 num_units_init=[64],
                 num_units_agg=[[64]],
                 num_units_state=[[64]],
                 num_units_readout=[64],
                 layer_demappers=None,
                 layer_type_dense="dense",
                 layer_type_conv="sepconv",
                 layer_type_readout="dense",
                 nrx_dtype=tf.float32,
                 **kwargs):
        super().__init__(**kwargs)

        self._training = training
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype

        self._num_mcss_supported = len(sys_parameters.mcs_index)

        self._rg = sys_parameters.transmitters[0]._resource_grid

        num_bits_per_symbol = []
        for mcs_list_idx in range(self._num_mcss_supported):
             num_bits_per_symbol.append(
                        sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol)

        num_rx_ant = sys_parameters.num_rx_antennas

        self._cgnn = CGNN(num_bits_per_symbol,
                          num_rx_ant,
                          num_it,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          training=training,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          dtype=nrx_dtype)

        self._rg_demapper = ResourceGridDemapper(self._rg,
                                                 sys_parameters.sm)

        if training:
            self._bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
            self._mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)


        rg_type = self._rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        pilots_only = tf.scatter_nd(pilot_ind, pilots,
                                    rg_type.shape)
        pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)
        pilot_ind = np.array(pilot_ind)

        pilot_ind_sorted = [ [] for _ in range(max_num_tx) ]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        pilots_dist_time = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(self._rg.num_ofdm_symbols)
        f_ind = np.arange(self._rg.fft_size)

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

    @property
    def num_it(self):
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        self._cgnn.num_it = val

    def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None):

        if self._training:
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask = inputs
        else:
            y, h_hat_init, active_tx = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = tf.one_hot(mcs_arr_eval[0],
                                         depth=self._num_mcss_supported)
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)

        num_tx = tf.shape(active_tx)[1]

        y = y[:,0]
        y = tf.transpose(y, [0, 3, 2, 1])
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
        pe = self._nearest_pilot_dist[:num_tx]


        y = tf.cast(y, self._nrx_dtype)
        pe = tf.cast(pe, self._nrx_dtype)

        h_hat_init = tf.cast(h_hat_init, self._nrx_dtype)
        active_tx = tf.cast(active_tx, self._nrx_dtype)

        llrs_, h_hats_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask])

        indices = mcs_arr_eval

        llrs = []

        h_hats = []
        for llrs_, h_hat_ in zip(llrs_, h_hats_):

            h_hat_ = tf.cast(h_hat_, tf.float32)

            _llrs_ = []
            for idx in indices:

                llrs_[idx] = tf.cast(llrs_[idx], tf.float32)


                llrs_[idx] = tf.transpose(llrs_[idx], [0, 1, 3, 2, 4])
                llrs_[idx] = tf.expand_dims(llrs_[idx], axis=1)
                llrs_[idx] = self._rg_demapper(llrs_[idx])
                llrs_[idx] = llrs_[idx][:,:num_tx]

                llrs_[idx] = flatten_last_dims(llrs_[idx], 2)

                llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                _llrs_.append(llrs_[idx])


            llrs.append(_llrs_)
            h_hats.append(h_hat_)

        if self._training:

            loss_data = tf.constant(0.0, dtype=tf.float32)
            for llrs_ in llrs:
                for idx in range(len(indices)):
                    loss_data_ = self._bce(bits[idx], llrs_[idx])

                    mcs_ue_mask_ = expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=indices[idx], axis=2),
                        tf.rank(loss_data_), axis=-1)

                    loss_data_ = tf.multiply(loss_data_, mcs_ue_mask_)


                    active_tx_data = expand_to_rank(active_tx,
                                                    tf.rank(loss_data_),
                                                    axis=-1)
                    loss_data_ = tf.multiply(loss_data_, active_tx_data)
                    loss_data += tf.reduce_mean(loss_data_)

            loss_chest = tf.constant(0.0, dtype=tf.float32)
            if h_hats is not None:
                for h_hat_ in h_hats:
                    if h is not None:
                        loss_chest += self._mse(h, h_hat_)

            active_tx_chest = expand_to_rank(active_tx,
                                             tf.rank(loss_chest), axis=-1)
            loss_chest = tf.multiply(loss_chest, active_tx_chest)
            loss_chest = tf.reduce_mean(loss_chest)
            return loss_data, loss_chest
        else:
            return llrs[-1][0], h_hats[-1]

class NeuralPUSCHReceiver(Layer):

    def __init__(self,
                sys_parameters,
                training=False,
                **kwargs):


        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters

        self._training = training

        self._tb_encoders = []
        self._tb_decoders= []

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        for mcs_list_idx in range(self._num_mcss_supported):
                self._tb_encoders.append(
                    self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder)

                self._tb_decoders.append(
                    TBDecoder(self._tb_encoders[mcs_list_idx],
                              num_bp_iter=sys_parameters.num_bp_iter,
                              cn_type=sys_parameters.cn_type))

        self._precoding_mat = sys_parameters.transmitters[0]._precoder._w

        rg = sys_parameters.transmitters[0]._resource_grid
        pc =  sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimator(
                resource_grid=rg,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn")

        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))

        self._neural_rx = CGNNOFDM(
                    sys_parameters,
                    max_num_tx=sys_parameters.max_num_tx,
                    training=training,
                    num_it=sys_parameters.num_nrx_iter,
                    d_s=sys_parameters.d_s,
                    num_units_init=sys_parameters.num_units_init,
                    num_units_agg=sys_parameters.num_units_agg,
                    num_units_state=sys_parameters.num_units_state,
                    num_units_readout=sys_parameters.num_units_readout,
                    layer_demappers=self._layer_demappers,
                    layer_type_dense=sys_parameters.layer_type_dense,
                    layer_type_conv=sys_parameters.layer_type_conv,
                    layer_type_readout=sys_parameters.layer_type_readout,
                    dtype=sys_parameters.nrx_dtype)

    def estimate_channel(self, y, num_tx):


        if self._sys_parameters.initial_chest == 'ls':
            h_hat, _ = self._ls_est([y, 1e-1])

            h_hat = h_hat[:,0,:,:num_tx,0]
            h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)],
                              axis=-1)

        return h_hat

    def preprocess_channel_ground_truth(self, h):

        h = tf.squeeze(h, axis=1)

        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        h = tf.squeeze(tf.matmul(h, w), axis=-1)

        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)

        return h

    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):

        if self._training:
            y, active_tx, b, h, mcs_ue_mask  = inputs
            bits = []
            for idx in range(len(mcs_arr_eval)):
                bits.append(
                    self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx]))

            num_tx = tf.shape(active_tx)[1]
            h_hat = self.estimate_channel(y, num_tx)

            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            losses = self._neural_rx((y, h_hat, active_tx,
                                      bits, h, mcs_ue_mask),
                                      mcs_arr_eval)
            return losses

        else:
            y, active_tx = inputs

            num_tx = tf.shape(active_tx)[1]
            h_hat = self.estimate_channel(y, num_tx)

            llr, h_hat_refined = self._neural_rx(
                                            (y, h_hat, active_tx),
                                            [mcs_arr_eval[0]],
                                            mcs_ue_mask_eval=mcs_ue_mask_eval)

            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            return b_hat, h_hat_refined, h_hat, tb_crc_status
