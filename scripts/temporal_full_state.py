#!/usr/bin/env python3
"""Raw full-state temporal-memory upper bound for Neural RX.

This path intentionally does *no* pooling or dimensional compression before
persistence.  For every scheduled UE it stores the complete final CGNN state
[B,U,F,T,d_s] from TB_t and restores that state for the same physical UE at
TB_{t+1}.  The only learned temporal operation is a d_s-wide read gate that
controls how strongly the previous full state is added to the current initial
state.

The persistent memory manager still stores vectors, so the full tensor is
flattened only for storage/routing and reshaped losslessly on read.  No Dense,
PCA, autoencoder, pooling, or other compressor is applied to the stored state.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sionna.utils import expand_to_rank, flatten_last_dims

from temporal_training_data import TemporalTrainingDataGenerator
from ue_memory_manager import DifferentiableUEMemoryManager
from utils import Parameters, E2E_Model, load_weights


class FullStateTemporalCGNN(tf.keras.Model):
    """K-step CGNN carrying the exact previous final state per physical UE."""

    def __init__(self, base_cgnn, d_s, **kwargs):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_s = int(d_s)
        self.mem_gate = Dense(
            self.d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(-3.0),
            name="full_state_read_gate",
        )

    @property
    def memory_variables(self):
        return list(self.mem_gate.trainable_variables)

    def _initial_state(self, y, pe, h_hat, mcs_ue_mask):
        base = self.base
        norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
        norm = tf.math.divide_no_nan(1.0, tf.sqrt(norm))
        y = y * norm
        if h_hat is not None:
            h_hat = h_hat * tf.expand_dims(norm, axis=1)

        if base._var_mcs_masking:
            s = base._s_init[0]((y, pe, h_hat))
        else:
            s = base._s_init[0]((y, pe, h_hat)) * expand_to_rank(
                tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1)
            for idx in range(1, base._num_mcss_supported):
                s += base._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                    tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)
        return s

    def _iterations(self, s, pe, active_tx):
        for i in range(self.base._num_it):
            s = self.base._iterations[i]([s, pe, active_tx])
        return s

    def call(
        self,
        inputs,
        prev_memory=None,
        memory_gap=None,
        memory_valid=None,
        training=None,
    ):
        del training
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        base = self.base
        s = self._initial_state(y, pe, h_hat, mcs_ue_mask)
        shape = tf.shape(s)
        batch_size = shape[0]
        num_tx = shape[1]

        if memory_valid is None:
            memory_valid = tf.zeros([batch_size, num_tx], tf.bool)
        else:
            memory_valid = tf.cast(memory_valid, tf.bool)
        if memory_gap is None:
            memory_gap = tf.zeros([batch_size, num_tx], tf.int32)
        else:
            memory_gap = tf.cast(memory_gap, tf.int32)

        if prev_memory is None:
            prev_state = tf.zeros_like(s)
        else:
            prev_memory = tf.cast(prev_memory, s.dtype)
            prev_state = tf.reshape(prev_memory, tf.shape(s))

        valid_f = tf.cast(memory_valid, s.dtype)
        safe_prev = tf.where(
            memory_valid[..., None, None, None],
            prev_state,
            tf.zeros_like(prev_state),
        )
        pooled_init = tf.reduce_mean(s, axis=[2, 3])
        pooled_prev = tf.reduce_mean(safe_prev, axis=[2, 3])
        age = tf.math.log1p(tf.cast(tf.maximum(memory_gap, 0), s.dtype)) * valid_f
        gate_in = tf.concat(
            [pooled_init, pooled_prev, age[..., None], valid_f[..., None]],
            axis=-1,
        )
        gate = self.mem_gate(gate_in) * valid_f[..., None]
        s = s + gate[:, :, None, None, :] * safe_prev
        s = self._iterations(s, pe, active_tx)

        if base._var_mcs_masking:
            llr_grid = base._readout_llrs[0](s)
            llr_grid = tf.gather(
                llr_grid,
                indices=tf.range(base._num_bits_per_symbol[0]),
                axis=-1,
            )
        else:
            llr_grid = base._readout_llrs[0](s)
        h_refined = base._readout_chest(s)

        active_bool = tf.cast(active_tx, tf.bool)
        next_state = tf.where(
            active_bool[..., None, None, None], s, safe_prev)
        next_memory = tf.reshape(next_state, [batch_size, num_tx, -1])
        return llr_grid, h_refined, next_memory


def prepare_cgnn_inputs(receiver, y, h_hat, active_tx):
    ofdm = receiver._neural_rx
    num_tx = tf.shape(active_tx)[1]
    y2 = y[:, 0]
    y2 = tf.transpose(y2, [0, 3, 2, 1])
    y2 = tf.concat([tf.math.real(y2), tf.math.imag(y2)], axis=-1)
    pe = ofdm._nearest_pilot_dist[:num_tx]
    y2 = tf.cast(y2, ofdm._nrx_dtype)
    pe = tf.cast(pe, ofdm._nrx_dtype)
    h_hat = tf.cast(h_hat, ofdm._nrx_dtype)
    active = tf.cast(active_tx, ofdm._nrx_dtype)
    mcs_mask = tf.ones([tf.shape(y2)[0], num_tx, 1], tf.float32)
    return [y2, pe, h_hat, active, mcs_mask]


def demap_llr(ofdm, llr_grid, num_tx, mcs_idx=0):
    llr = tf.cast(llr_grid, tf.float32)
    llr = tf.transpose(llr, [0, 1, 3, 2, 4])
    llr = tf.expand_dims(llr, axis=1)
    llr = ofdm._rg_demapper(llr)
    llr = llr[:, :num_tx]
    llr = flatten_last_dims(llr, 2)
    if ofdm._layer_demappers is None:
        llr = tf.squeeze(llr, axis=-2)
    else:
        llr = ofdm._layer_demappers[mcs_idx](llr)
    return llr


def temporal_forward(
    receiver,
    model,
    y,
    h_hat,
    active_tx,
    memory,
    memory_gap,
    memory_valid,
    training=False,
):
    inputs = prepare_cgnn_inputs(receiver, y, h_hat, active_tx)
    llr_grid, h_ref, next_memory = model(
        inputs,
        prev_memory=memory,
        memory_gap=memory_gap,
        memory_valid=memory_valid,
        training=training,
    )
    llr = demap_llr(receiver._neural_rx, llr_grid, tf.shape(active_tx)[1], 0)
    return llr, tf.cast(h_ref, tf.float32), next_memory


def build_system(
    config="nrx_large.cfg",
    num_it=2,
    training=True,
    ue_pool_size=4,
    dynamic_scheduling=False,
    schedule_switch_prob=0.65,
    schedule_reorder_prob=0.50,
):
    p = Parameters(
        config,
        training=training,
        **({} if training else {"num_tx_eval": 2}),
        system="nrx",
    )
    e2e = E2E_Model(
        p,
        training=training,
        **({} if training else {"mcs_arr_eval_idx": 0}),
    )
    e2e(1, 1.0)
    load_weights(e2e, f"../weights/{p.label}_weights")
    base = e2e._receiver._neural_rx._cgnn
    base.num_it = int(num_it)
    model = FullStateTemporalCGNN(base, d_s=p.d_s, name="temporal_full_state")
    generator = TemporalTrainingDataGenerator(
        p,
        e2e,
        ue_pool_size=ue_pool_size,
        dynamic_scheduling=dynamic_scheduling,
        schedule_switch_prob=schedule_switch_prob,
        schedule_reorder_prob=schedule_reorder_prob,
    )
    return p, e2e, model, generator


def memory_dim_for_batch(receiver, model, batch):
    inputs = prepare_cgnn_inputs(
        receiver,
        batch["y"][:, 0],
        batch["ls"][:, 0],
        batch["active"][:, 0],
    )
    s = model._initial_state(*[inputs[0], inputs[1], inputs[2], inputs[4]])
    dims = s.shape[2:]
    if any(d is None for d in dims):
        dims = tuple(int(x) for x in tf.shape(s)[2:].numpy())
    else:
        dims = tuple(int(x) for x in dims)
    return int(np.prod(dims)), dims


def make_manager(receiver, model, batch, capacity=4, expiry_slots=8):
    d_mem, state_shape = memory_dim_for_batch(receiver, model, batch)
    manager = DifferentiableUEMemoryManager(
        capacity=int(capacity), d_mem=d_mem, expiry_slots=int(expiry_slots))
    return manager, d_mem, state_shape
