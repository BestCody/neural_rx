#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf

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


def readout_llr_grid(base_cgnn, state: tf.Tensor) -> tf.Tensor:
    if int(base_cgnn._num_mcss_supported) != 1:
        raise ValueError("the runtime requires one fixed MCS")
    return base_cgnn._readout_llrs[0](state)


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
