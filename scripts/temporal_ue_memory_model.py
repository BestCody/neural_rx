#!/usr/bin/env python3
"""Temporal UE-memory receiver components shared by training and evaluation.

This module contains architecture and signal-processing code only. It parses no
command-line arguments and starts no training jobs. Keeping it separate makes
``train_temporal_ue_memory_streaming.py`` the single training entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sionna as sn
import tensorflow as tf
from sionna.utils import expand_to_rank, flatten_last_dims
from tensorflow.keras.layers import Dense

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from temporal_compression import build_compressor
from temporal_pooling import build_pooler
from ue_memory_manager import DifferentiableUEMemoryManager
from utils import E2E_Model, Parameters, load_weights


class TemporalUEMemoryCGNN(tf.keras.Model):
    """Pretrained CGNN with identity-owned memory read/write paths."""

    def __init__(
        self,
        base_cgnn,
        d_mem: int,
        d_s: int,
        compression: str,
        pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base = base_cgnn
        self.d_mem = int(d_mem)
        self.d_s = int(d_s)
        self.compression = str(compression)
        self.pooling = str(pooling)

        self.mem_in = Dense(d_s, activation="tanh", name="ue_mem_in")
        self.mem_gate = Dense(
            d_s,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
            bias_initializer=tf.keras.initializers.Constant(-3.0),
            name="ue_mem_gate",
        )
        self.pooler = build_pooler(self.pooling, d_s=self.d_s)
        self.compressor = build_compressor(
            self.compression, d_s=self.d_s, d_mem=self.d_mem
        )

    @property
    def memory_variables(self):
        return (
            self.mem_in.trainable_variables
            + self.mem_gate.trainable_variables
            + self.compressor.trainable_variables
            + self.pooler.trainable_variables
        )

    @property
    def temporal_check_variables(self):
        return (
            list(self.compressor.temporal_check_variables)
            + list(self.pooler.trainable_variables)
        )

    def _initial_state(self, y, pe, h_hat, mcs_ue_mask):
        base = self.base
        norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
        norm = tf.math.divide_no_nan(1.0, tf.sqrt(norm))
        y = y * norm
        if h_hat is not None:
            h_hat = h_hat * tf.expand_dims(norm, axis=1)

        if base._var_mcs_masking:
            return base._s_init[0]((y, pe, h_hat))

        state = base._s_init[0]((y, pe, h_hat)) * expand_to_rank(
            tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1
        )
        for idx in range(1, base._num_mcss_supported):
            state += base._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1
            )
        return state

    def _iterations(self, state, pe, active_tx):
        for idx in range(self.base._num_it):
            state = self.base._iterations[idx]([state, pe, active_tx])
        return state

    def cold_pooled_final(self, inputs):
        """Return a cold K-step state for fitting a frozen PCA basis."""
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        state = self._initial_state(y, pe, h_hat, mcs_ue_mask)
        state = self._iterations(state, pe, active_tx)
        return self.pooler(state, training=False)

    def call(
        self,
        inputs,
        prev_memory=None,
        memory_gap=None,
        memory_valid=None,
        training=None,
    ):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        base = self.base
        state = self._initial_state(y, pe, h_hat, mcs_ue_mask)

        batch_size = tf.shape(state)[0]
        num_tx = tf.shape(state)[1]
        if prev_memory is None:
            prev_memory = tf.zeros(
                [batch_size, num_tx, self.d_mem], dtype=state.dtype
            )
        else:
            prev_memory = tf.cast(prev_memory, state.dtype)
        if memory_valid is None:
            memory_valid = tf.zeros([batch_size, num_tx], tf.bool)
        else:
            memory_valid = tf.cast(memory_valid, tf.bool)
        if memory_gap is None:
            memory_gap = tf.zeros([batch_size, num_tx], tf.int32)
        else:
            memory_gap = tf.cast(memory_gap, tf.int32)

        valid_f = tf.cast(memory_valid, state.dtype)
        safe_memory = tf.where(
            memory_valid[..., None], prev_memory, tf.zeros_like(prev_memory)
        )
        age_feature = tf.math.log1p(
            tf.cast(tf.maximum(memory_gap, 0), state.dtype)
        )
        age_feature *= valid_f

        pooled_init = tf.reduce_mean(state, axis=[2, 3])
        gate_input = tf.concat(
            [
                pooled_init,
                safe_memory,
                age_feature[..., None],
                valid_f[..., None],
            ],
            axis=-1,
        )
        gate = self.mem_gate(gate_input) * valid_f[..., None]
        memory_delta = self.mem_in(safe_memory)
        state += gate[:, :, None, None, :] * memory_delta[:, :, None, None, :]

        state = self._iterations(state, pe, active_tx)
        llr_grid = base._readout_llrs[0](state)
        if base._var_mcs_masking:
            llr_grid = tf.gather(
                llr_grid,
                indices=tf.range(base._num_bits_per_symbol[0]),
                axis=-1,
            )
        h_refined = base._readout_chest(state)

        pooled_final = self.pooler(state, training=training)
        compressed = self.compressor(
            pooled_final,
            safe_memory,
            age_feature,
            memory_valid,
            training=training,
        )
        next_memory = tf.where(
            tf.cast(active_tx, tf.bool)[..., None],
            compressed.memory,
            safe_memory,
        )
        return (
            llr_grid,
            h_refined,
            next_memory,
            compressed.aux_loss,
            compressed.reconstruction_mse,
        )

def demap_llr(ofdm, llr_grid, num_tx, mcs_idx=0):
    llr = tf.cast(llr_grid, tf.float32)
    llr = tf.transpose(llr, [0, 1, 3, 2, 4])
    llr = tf.expand_dims(llr, axis=1)
    llr = ofdm._rg_demapper(llr)
    llr = llr[:, :num_tx]
    llr = flatten_last_dims(llr, 2)
    if ofdm._layer_demappers is None:
        return tf.squeeze(llr, axis=-2)
    return ofdm._layer_demappers[mcs_idx](llr)


def prepare_cgnn_inputs(receiver, y, h_hat, active_tx):
    ofdm = receiver._neural_rx
    num_tx = tf.shape(active_tx)[1]
    y = y[:, 0]
    y = tf.transpose(y, [0, 3, 2, 1])
    y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
    pe = ofdm._nearest_pilot_dist[:num_tx]
    y = tf.cast(y, ofdm._nrx_dtype)
    pe = tf.cast(pe, ofdm._nrx_dtype)
    h_hat = tf.cast(h_hat, ofdm._nrx_dtype)
    active = tf.cast(active_tx, ofdm._nrx_dtype)
    mcs_mask = tf.ones([tf.shape(y)[0], num_tx, 1], tf.float32)
    return [y, pe, h_hat, active, mcs_mask]


def temporal_forward(
    receiver,
    temporal_model,
    y,
    h_hat,
    active_tx,
    memory,
    memory_gap,
    memory_valid,
    training=True,
):
    inputs = prepare_cgnn_inputs(receiver, y, h_hat, active_tx)
    llr_grid, h_ref, next_memory, aux_loss, reconstruction_mse = temporal_model(
        inputs,
        prev_memory=memory,
        memory_gap=memory_gap,
        memory_valid=memory_valid,
        training=training,
    )
    llr = demap_llr(
        receiver._neural_rx, llr_grid, tf.shape(active_tx)[1], 0
    )
    return (
        llr,
        tf.cast(h_ref, tf.float32),
        next_memory,
        aux_loss,
        reconstruction_mse,
    )


def build_backbone(config, num_it, training, num_tx_eval=None):
    """Build and load the shipped cold Neural RX backbone."""
    parameters = Parameters(
        config,
        training=training,
        num_tx_eval=num_tx_eval,
        system="nrx",
    )
    if training:
        e2e = E2E_Model(parameters, training=True)
    else:
        e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    e2e(1, 1.0)
    load_weights(e2e, str(HERE.parent / "weights" / f"{parameters.label}_weights"))
    e2e._receiver._neural_rx._cgnn.num_it = int(num_it)
    return parameters, e2e


def identity_routing_check(d_mem, capacity):
    """Verify that memory follows UE identity rather than receiver position."""
    manager = DifferentiableUEMemoryManager(
        capacity=capacity, d_mem=d_mem, expiry_slots=8
    )
    state = manager.zero_state(1)
    first = tf.ones([d_mem], tf.float32)
    second = tf.ones([d_mem], tf.float32) * 2.0
    ids0 = tf.constant([[0, 1]], tf.int32)
    state = manager.scatter(
        state,
        ids0,
        tf.stack([tf.stack([first, second], axis=0)], axis=0),
        tf.ones([1, 2], tf.float32),
        0,
    )

    if capacity >= 3:
        ids1 = tf.constant([[1, 2]], tf.int32)
        state, gathered, gaps, valid = manager.gather(state, ids1, 1)
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 0.0)
            and valid.numpy().tolist() == [[True, False]]
            and gaps.numpy().tolist() == [[1, 0]]
        )
        state = manager.scatter(
            state,
            ids1,
            tf.ones([1, 2, d_mem], tf.float32) * 3.0,
            tf.ones([1, 2], tf.float32),
            1,
        )
        _, gathered2, gaps2, valid2 = manager.gather(
            state, tf.constant([[0, 1]], tf.int32), 2
        )
        persistence_ok = (
            np.allclose(gathered2.numpy()[0, 0], 1.0)
            and np.allclose(gathered2.numpy()[0, 1], 3.0)
            and valid2.numpy().tolist() == [[True, True]]
            and gaps2.numpy().tolist() == [[2, 1]]
        )
    else:
        _, gathered, gaps, valid = manager.gather(
            state, tf.constant([[1, 0]], tf.int32), 1
        )
        route_ok = (
            np.allclose(gathered.numpy()[0, 0], 2.0)
            and np.allclose(gathered.numpy()[0, 1], 1.0)
            and valid.numpy().tolist() == [[True, True]]
            and gaps.numpy().tolist() == [[1, 1]]
        )
        persistence_ok = route_ok

    expiry = DifferentiableUEMemoryManager(
        capacity=capacity, d_mem=d_mem, expiry_slots=1
    )
    expiry_state = expiry.zero_state(1)
    expiry_state = expiry.scatter(
        expiry_state,
        ids0,
        tf.stack([tf.stack([first, second], axis=0)], axis=0),
        tf.ones([1, 2], tf.float32),
        0,
    )
    _, expired_memory, _, expired_valid = expiry.gather(expiry_state, ids0, 2)
    expiration_ok = (
        not np.any(expired_valid.numpy())
        and np.allclose(expired_memory.numpy(), 0.0)
    )
    return {
        "route_across_positions": bool(route_ok),
        "unscheduled_memory_persists": bool(persistence_ok),
        "expiration_zeroes_stale_memory": bool(expiration_ok),
        "passed": bool(route_ok and persistence_ok and expiration_ok),
    }


def temporal_gradient_check(receiver, model, generator, memory_manager):
    """Verify that TB2 loss differentiates through the TB1 write path."""
    batch_size = 2
    schedule_pair = (
        [[0, 1], [1, 2]]
        if generator.ue_pool_size >= 3
        else [[0, 1], [1, 0]]
    )
    schedule = tf.constant([schedule_pair] * batch_size, tf.int32)
    batch = generator.sample_batch(batch_size, 2, 3.0, ue_ids=schedule)
    check_variables = model.temporal_check_variables
    if not check_variables:
        raise RuntimeError(
            f"No temporal check variables for compression={model.compression}"
        )

    with tf.GradientTape() as tape:
        for variable in check_variables:
            tape.watch(variable)
        state = memory_manager.zero_state(batch_size)
        state, memory0, gap0, valid0 = memory_manager.gather(
            state, batch["ue_ids"][:, 0], 0
        )
        _, _, updated0, _, _ = temporal_forward(
            receiver,
            model,
            batch["y"][:, 0],
            batch["ls"][:, 0],
            batch["active"][:, 0],
            memory0,
            gap0,
            valid0,
            training=True,
        )
        state = memory_manager.scatter(
            state,
            batch["ue_ids"][:, 0],
            updated0,
            batch["active"][:, 0],
            0,
        )
        state, memory1, gap1, valid1 = memory_manager.gather(
            state, batch["ue_ids"][:, 1], 1
        )
        coded1 = receiver._tb_encoders[0](batch["bits"][:, 1])
        llr1, _, _, _, _ = temporal_forward(
            receiver,
            model,
            batch["y"][:, 1],
            batch["ls"][:, 1],
            batch["active"][:, 1],
            memory1,
            gap1,
            valid1,
            training=True,
        )
        future_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(coded1, tf.float32), logits=llr1
            )
        )

    grads = tape.gradient(future_loss, check_variables)
    valid_grads = [gradient for gradient in grads if gradient is not None]
    norm = (
        tf.linalg.global_norm(valid_grads)
        if valid_grads
        else tf.constant(0.0)
    )
    return {
        "compression": model.compression,
        "tb2_only_loss": float(future_loss.numpy()),
        "compression_path_grad_norm": float(norm.numpy()),
        "tb2_memory_valid": valid1.numpy().tolist(),
        "tb2_memory_gap": gap1.numpy().tolist(),
        "forced_schedule": schedule.numpy().tolist(),
        "passed": bool(float(norm.numpy()) > 0.0),
    }


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        sn.config.seed = seed
    except Exception:
        pass


def schedule_change_fraction(ue_ids):
    ids = ue_ids.numpy()
    if ids.shape[1] < 2:
        return 0.0
    changes = [
        np.mean(np.any(ids[:, t] != ids[:, t + 1], axis=-1))
        for t in range(ids.shape[1] - 1)
    ]
    return float(np.mean(changes))
