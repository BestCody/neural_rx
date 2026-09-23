#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from sionna.utils import expand_to_rank, flatten_last_dims

from pilot_domain_channel import pilot_layout
from utils import E2E_Model, Parameters, load_weights


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    for variable in model.weights:
        array = np.asarray(variable.numpy())
        digest.update(variable.name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_weight_signature(model) -> list[dict]:
    return [
        {
            "index": index,
            "shape": list(np.asarray(variable.numpy()).shape),
            "dtype": np.asarray(variable.numpy()).dtype.str,
        }
        for index, variable in enumerate(model.weights)
    ]


def model_value_checksum(model) -> str:
    digest = hashlib.sha256()
    for record, variable in zip(model_weight_signature(model), model.weights):
        digest.update(str(record["index"]).encode("ascii"))
        digest.update(str(tuple(record["shape"])).encode("ascii"))
        digest.update(record["dtype"].encode("ascii"))
        digest.update(np.asarray(variable.numpy()).tobytes(order="C"))
    return digest.hexdigest()


def build_backbone(
    config,
    num_it,
    training,
    num_tx_eval=None,
    n_size_bwp_override=None,
):
    parameters = Parameters(
        config,
        training=training,
        num_tx_eval=num_tx_eval,
        n_size_bwp_override=n_size_bwp_override,
        system="nrx",
    )
    if training:
        e2e = E2E_Model(parameters, training=True)
    else:
        e2e = E2E_Model(parameters, training=False, mcs_arr_eval_idx=0)
    e2e(1, 1.0)
    path = Path(__file__).resolve().parent.parent
    load_weights(e2e, str(path / "weights" / f"{parameters.label}_weights"))
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


def assign_selected(backbone, checkpoint, variables) -> None:
    with np.load(checkpoint, allow_pickle=False) as archive:
        names = sorted(archive.files)
        if len(names) != len(variables):
            raise ValueError("checkpoint variable count changed")
        for name, variable in zip(names, variables):
            value = archive[name]
            if tuple(value.shape) != tuple(variable.shape):
                raise ValueError("checkpoint variable shape changed")
            variable.assign(value)


def frozen_cold_state(base_cgnn, inputs) -> tf.Tensor:
    y, pe, channel, _, mcs_mask = inputs
    norm = tf.reduce_mean(tf.square(y), axis=(1, 2, 3), keepdims=True)
    norm = tf.math.divide_no_nan(1.0, tf.sqrt(norm))
    y = y * norm
    if channel is not None:
        channel = channel * tf.expand_dims(norm, axis=1)
    if base_cgnn._var_mcs_masking:
        return base_cgnn._s_init[0]((y, pe, channel))
    state = base_cgnn._s_init[0]((y, pe, channel)) * expand_to_rank(
        tf.gather(mcs_mask, indices=0, axis=2), 5, axis=-1
    )
    for index in range(1, base_cgnn._num_mcss_supported):
        state += base_cgnn._s_init[index]((y, pe, channel)) * expand_to_rank(
            tf.gather(mcs_mask, indices=index, axis=2), 5, axis=-1
        )
    return state


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


def run_k2(backbone, inputs) -> tf.Tensor:
    state = frozen_cold_state(backbone, inputs)
    for index in range(2):
        state = backbone._iterations[index]([state, inputs[1], inputs[3]])
    return readout_llr_grid(backbone, state)


def direct_k2_trainable_variables(base_cgnn) -> list:
    layers = list(base_cgnn._s_init)
    for iteration in base_cgnn._iterations[:2]:
        layers.extend([iteration._state_aggreg, iteration._state_update])
    layers.extend(base_cgnn._readout_llrs)
    variables = []
    seen = set()
    for layer in layers:
        for variable in layer.trainable_variables:
            if id(variable) not in seen:
                variables.append(variable)
                seen.add(id(variable))
    return variables


def select_channel_input(
    current_ls,
    posterior_channel,
    history_valid,
    reset_mask=None,
) -> tuple[tf.Tensor, tf.Tensor]:
    current_ls = tf.convert_to_tensor(current_ls)
    posterior_channel = tf.cast(posterior_channel, current_ls.dtype)
    history_valid = tf.cast(history_valid, tf.bool)
    tf.debugging.assert_rank(current_ls, 5)
    tf.debugging.assert_equal(tf.shape(current_ls), tf.shape(posterior_channel))
    tf.debugging.assert_equal(tf.shape(history_valid), tf.shape(current_ls)[:2])
    if reset_mask is None:
        reset_mask = tf.zeros_like(history_valid)
    else:
        reset_mask = tf.cast(reset_mask, tf.bool)
        tf.debugging.assert_equal(tf.shape(reset_mask), tf.shape(history_valid))
    available = tf.logical_and(history_valid, tf.logical_not(reset_mask))
    selected = tf.where(
        available[..., None, None, None], posterior_channel, current_ls
    )
    tf.debugging.assert_all_finite(selected, "channel input is not finite")
    return selected, available


def select_channel_variance(
    posterior_variance,
    observation_variance,
    available,
) -> tf.Tensor:
    observation = tf.convert_to_tensor(observation_variance, tf.float32)
    posterior = tf.cast(posterior_variance, observation.dtype)
    available = tf.cast(available, tf.bool)
    tf.debugging.assert_equal(tf.shape(posterior), tf.shape(observation))
    tf.debugging.assert_equal(tf.shape(available), tf.shape(observation)[:2])
    selected = tf.where(
        available[..., None, None, None], posterior, observation
    )
    tf.debugging.assert_greater_equal(selected, 0.0)
    return selected


def stacked_to_complex_channel(channel) -> tf.Tensor:
    channel = tf.convert_to_tensor(channel, tf.float32)
    width = channel.shape[-1]
    if width is None or int(width) % 2:
        raise ValueError("stacked channel width must be even and static")
    antennas = int(width) // 2
    return tf.complex(channel[..., :antennas], channel[..., antennas:])


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


def derive_cdm_noise_conversion(estimator, num_tx: int) -> dict:
    layout = pilot_layout(estimator, int(num_tx))
    group_width = 2 * int(estimator._num_cdm_groups_without_data)
    pilots = np.asarray(estimator._pilot_pattern.pilots.numpy())[
        :int(num_tx), 0
    ].reshape(
        int(num_tx),
        layout.num_dmrs_symbols,
        layout.pilots_per_dmrs_symbol,
    )
    frequency_counts = []
    energies = []
    for user in range(int(num_tx)):
        for phase in range(layout.num_dmrs_symbols):
            indices = layout.frequency_indices(user, phase)
            _, counts = np.unique(indices // group_width, return_counts=True)
            frequency_counts.extend(counts.tolist())
            energies.extend(np.square(np.abs(
                pilots[user, phase, indices]
            )).tolist())
    unique_counts = np.unique(frequency_counts)
    if unique_counts.size != 1 or int(unique_counts[0]) < 1:
        raise ValueError("CDM frequency averaging is not uniform")
    if not np.allclose(energies, energies[0], rtol=1e-6, atol=1e-7):
        raise ValueError("DMRS pilot energy is not uniform")
    frequency_average = int(unique_counts[0])
    time_average = int(estimator._dmrs_length)
    pilot_energy = float(np.mean(energies))
    factor = float(frequency_average * time_average * pilot_energy)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("invalid CDM observation conversion")
    return {
        "factor": factor,
        "frequency_average": frequency_average,
        "time_average": time_average,
        "pilot_energy": pilot_energy,
        "group_width": group_width,
    }


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
