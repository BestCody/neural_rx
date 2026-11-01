#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from sionna.utils import flatten_last_dims

from utils import E2E_Model, Parameters, load_weights
from artifact_io import file_sha256


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    for variable in model.weights:
        array = np.asarray(variable.numpy())
        digest.update(variable.name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_backbone(
    config,
    num_it,
    training,
    num_tx_eval=None,
    n_size_bwp_override=None,
    checkpoint=None,
    checkpoint_sha256=None,
):
    parameters = Parameters(
        config,
        training=training,
        num_tx_eval=num_tx_eval,
        n_size_bwp_override=n_size_bwp_override,
    )
    root = Path(__file__).resolve().parent.parent
    checkpoint = (root / "models/pretrained" / f"{parameters.label}_weights" if checkpoint is None
                  else Path(checkpoint).expanduser().resolve())
    if checkpoint_sha256 is not None and file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Backbone checkpoint checksum differs")
    if training:
        e2e = E2E_Model(parameters, training=True)
    else:
        e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    e2e(1, 1.0)
    if checkpoint.name.endswith(".weights.h5"):
        e2e._receiver._neural_rx._cgnn.load_weights(str(checkpoint))
    else:
        load_weights(e2e, str(checkpoint))
    if checkpoint_sha256 is not None and file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Backbone checkpoint changed while loading")
    e2e._receiver._neural_rx._cgnn.num_it = int(num_it)
    return parameters, e2e


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


def readout_llr_grid(base_cgnn, state: tf.Tensor) -> tf.Tensor:
    if int(base_cgnn._num_mcss_supported) != 1:
        raise ValueError("the runtime requires one fixed MCS")
    llr = base_cgnn._readout_llrs[0](state)
    if base_cgnn._var_mcs_masking:
        llr = tf.gather(
            llr,
            indices=tf.range(base_cgnn._num_bits_per_symbol[0]),
            axis=-1,
        )
    return llr


def received_to_grid(received) -> tf.Tensor:
    received = tf.convert_to_tensor(received)
    if received.shape.rank != 5 or received.shape[1] != 1:
        raise ValueError("received tensor must have shape [B,1,R,S,F]")
    return tf.transpose(tf.squeeze(received, axis=1), [0, 3, 2, 1])


def exact_contributions(physical_channel, transmitted) -> tf.Tensor:
    physical_channel = tf.convert_to_tensor(physical_channel)
    transmitted = tf.cast(transmitted, physical_channel.dtype)
    if physical_channel.shape.rank != 7:
        raise ValueError("physical channel must have rank seven")
    if transmitted.shape.rank != 5:
        raise ValueError("transmitted grid must have rank five")
    return tf.einsum(
        "bqruasf,buasf->bufsr", physical_channel, transmitted
    )


def expand_noise(no, target) -> tf.Tensor:
    shape = tf.concat([
        tf.shape(no),
        tf.ones([tf.rank(target) - tf.rank(no)], tf.int32),
    ], axis=0)
    return tf.reshape(no, shape)


def make_snr_batch(base, unit_noise, snr, generator, receiver) -> dict:
    batch_size = int(base["bits"].shape[0])
    num_tbs = int(base["bits"].shape[1])
    ebno = tf.fill([batch_size], tf.cast(snr, tf.float32))
    point_no = generator.ebno_to_no(ebno)
    no = tf.repeat(point_no[:, None], num_tbs, axis=1)
    scale = tf.cast(tf.sqrt(expand_noise(no, base["y_clean"])), tf.complex64)
    received = base["y_clean"] + tf.cast(unit_noise, tf.complex64) * scale
    estimates = []
    for tb in range(num_tbs):
        generator._set_slot_pilots(int(base["slot_number"][tb].numpy()))
        estimates.append(receiver.estimate_channel(
            received[:, tb], generator.num_scheduled_tx
        ))
    result = dict(base)
    result["y"] = received
    result["no"] = no
    result["ls"] = tf.stack(estimates, axis=1)
    result["ebno_db"] = tf.repeat(ebno[:, None], num_tbs, axis=1)
    return result
