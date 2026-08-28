# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

COLORMAP = ['#76B900', '#F3BD00','#814B9D','#5C5C5C','#214B9D', '#e48aa7', '#7c4848', '#78A2EB', '#ff1397', '#bee7dd']


import tensorflow as tf
import pickle
import os
import datetime
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import io

import sys
sys.path.append('../')
import sionna as sn
from .e2e_model import E2E_Model
from .parameters import Parameters

from sionna.utils import ebnodb2no, expand_to_rank

def save_weights(system, model_path):
    """Save model weights."""
    weights = system.get_weights()
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(system, model_path):
    """Load model weights."""
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    system.set_weights(weights)

class TriangularDistributionSampler:
    # pylint: disable=line-too-long
    """sampling from a triangular distribution."""

    def __init__(self, minimum, maximum, dtype=tf.float32):
        self._dtype = dtype
        if dtype.is_integer:
            self._dtype_f = tf.float32
            self._a = tf.cast(minimum, tf.float32)
            self._b = tf.cast(maximum, tf.float32)
        else:
            self._dtype_f = dtype
            self._a = tf.cast(minimum, dtype)
            self._b = tf.cast(maximum, dtype)

    def __call__(self, shape):
        u = tf.random.uniform(  shape=shape,
                                minval=0.0,
                                maxval=1.0,
                                dtype=self._dtype_f)

        x = self._a + tf.sqrt(u)*(self._b - self._a)

        if self._dtype.is_integer:
            x = tf.cast(tf.floor(x), self._dtype)

        return x


def plot_to_image(figure):
    """Plot to image."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def training_loop(model, label, filename, training_logdir, training_seed,
                  training_schedule, eval_ebno_db_arr, min_num_tx, max_num_tx,
                  sys_parameters, mcs_arr_training_idx,
                  mcs_training_snr_db_offset=None, mcs_training_probs=None,
                  weight_saving_schedule=None, xla=False):
    # pylint: disable=line-too-long
    """Training loop used to train a system ``model``."""

    print(f"Training with mixed MCS from arr. idx {mcs_arr_training_idx}. Eval EbNo at {eval_ebno_db_arr} dB.")

    tf.random.set_seed(training_seed)

    num_tx_sampler = TriangularDistributionSampler(min_num_tx,
                                                   max_num_tx+1,
                                                   dtype=tf.int64)


    optimizer = tf.keras.optimizers.Adam()

    sn.Config.xla_compat = xla

    if mcs_training_snr_db_offset is not None:
        mcs_training_snr_db_offset = tf.constant(mcs_training_snr_db_offset,
                                                 dtype=tf.float32)

    @tf.function(jit_compile=xla)
    def _compile_step(batch_size, min_snr_db, max_snr_db, double_readout,
                      weighting_double_readout, apply_multiloss, train_tx):

        print("Applying multiloss: ", apply_multiloss)
        model._receiver._neural_rx._cgnn.apply_multiloss = apply_multiloss

        print("Constellation is trainable: ", train_tx)
        for tx_ in model._transmitters:
            tx_._mapper.constellation.trainable = train_tx

        for _ in tf.range(100, dtype=tf.int64):
            num_tx = num_tx_sampler(())

            mcs_arr_training_idx_ = tf.constant(mcs_arr_training_idx,
                                                dtype=tf.int32)
            if mcs_training_probs is None:
                mcs_arr_idx = tf.random.uniform(
                                            (batch_size, max_num_tx),
                                            maxval=len(mcs_arr_training_idx),
                                            dtype=tf.int32)
                mcs_arr_idx = tf.gather(mcs_arr_training_idx_,
                                        indices=mcs_arr_idx)
            else:
                mcs_probs = tf.constant(mcs_training_probs, dtype=tf.float32)
                mcs_probs = tf.gather(mcs_probs,
                                      indices=[num_tx - min_num_tx], axis=0)
                mcs_probs_ = tf.concat([[0.0], tf.squeeze(mcs_probs)], axis=0)
                mcs_cdf = tf.math.cumsum(mcs_probs_ / tf.reduce_sum(mcs_probs_))
                rand_samples = tf.random.uniform((batch_size, max_num_tx),
                                                 maxval=1.0, dtype=tf.float32)
                condition = tf.logical_and(
                    tf.greater_equal(expand_to_rank(rand_samples, 3, axis=-1),
                                     expand_to_rank(mcs_cdf[:-1], 3, axis=0)),
                    tf.less(expand_to_rank(rand_samples, 3, axis=-1),
                            expand_to_rank(mcs_cdf[1:], 3, axis=0)))
                mcs_arr_idx = expand_to_rank(mcs_arr_training_idx, 3, axis=0) * tf.cast(condition, dtype=tf.int32)
                mcs_arr_idx = tf.reduce_sum(mcs_arr_idx, axis=-1)

            mcs_ue_mask = tf.one_hot(mcs_arr_idx,
                                     depth=len(sys_parameters.mcs_index))

            snr_db = tf.random.uniform( shape=[batch_size],
                                        minval=min_snr_db[num_tx - min_num_tx],
                                        maxval=max_snr_db[num_tx - min_num_tx])

            if mcs_training_snr_db_offset is not None:
                _mcs_training_snr_db_offsets = tf.gather(
                                                    mcs_training_snr_db_offset,
                                                    indices=[num_tx-1], axis=0)
                _mcs_training_snr_db_offsets = tf.squeeze(tf.gather(
                                                _mcs_training_snr_db_offsets,
                                                indices=mcs_arr_idx, axis=1))
                active_dmrs = model._active_dmrs_mask(batch_size, num_tx,
                                                      sys_parameters.max_num_tx)
                _mcs_training_snr_db_offsets *= active_dmrs
                _mcs_training_snr_db_offsets = tf.reduce_sum(
                                                _mcs_training_snr_db_offsets,
                                                axis=1)
                snr_db += _mcs_training_snr_db_offsets
            else:
                active_dmrs = None

            with tf.GradientTape() as tape:
                loss_data, loss_chest = model(batch_size, snr_db, num_tx,
                                              mcs_ue_mask=mcs_ue_mask,
                                              active_dmrs=active_dmrs)
                if double_readout:
                    loss = loss_data + weighting_double_readout * loss_chest
                else:
                    loss = loss_data

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_data, loss_chest, loss

    def _training_step(global_iter, num_iterations, batch_size,
                       min_snr_db, max_snr_db, double_readout,
                       weighting_double_readout, apply_multiloss,
                       train_tx):
        for _ in tf.range(int(num_iterations/100), dtype=tf.int64):
            loss_data, loss_chest, loss = _compile_step(batch_size,
                                                       min_snr_db, max_snr_db,
                                                       double_readout,
                                                       weighting_double_readout,
                                                       apply_multiloss,
                                                       train_tx)
            global_iter += 100
            tf.summary.scalar(f"Loss", loss_data, step=global_iter)
            tf.summary.scalar(f"Loss Ch. Est.", loss_chest, step=global_iter)
            tf.summary.scalar(f"Total Loss", loss, step=global_iter)

            if weight_saving_schedule is not None and global_iter in weight_saving_schedule:
                print(f"Saving weights after {global_iter} iterations")
                save_weights(model, filename + f"_{global_iter}_iter")
        return global_iter, loss, loss_data, loss_chest

    @tf.function(jit_compile=xla)
    def eval_model_xla(batch_size, _snr_db, max_num_tx, mcs_arr_idx):
        loss_data_mcs, _ = model(batch_size, _snr_db, num_tx=max_num_tx,
                                 mcs_arr_eval_idx=mcs_arr_idx)
        return loss_data_mcs

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(training_logdir, f"{label}-{current_time}")
    summary_writer = tf.summary.create_file_writer(logdir)

    with summary_writer.as_default():
        tf.summary.text("config",
                        sys_parameters.config_str,
                        step=0)
        global_iter = tf.zeros((), tf.int64)
        for i, num_iterations in enumerate(training_schedule["num_iter"]):

            num_iterations = int(num_iterations)
            lr = training_schedule["learning_rate"][i]
            batch_size = training_schedule["batch_size"][i]
            train_tx = training_schedule["train_tx"][i]
            double_readout = training_schedule["double_readout"][i]
            apply_multiloss = training_schedule["apply_multiloss"][i]
            weighting_double_readout = tf.constant(
                    training_schedule["weighting_double_readout"][i],
                    tf.float32)

            min_snr_db = tf.constant(
                    training_schedule["min_training_snr_db"][i], tf.float32)
            max_snr_db = tf.constant(
                    training_schedule["max_training_snr_db"][i], tf.float32)

            optimizer.learning_rate.assign(lr)
            num_iter_global = num_iterations \
                              // sys_parameters.num_iter_train_save
            for _ in range(num_iter_global):
                global_iter, loss, loss_data, loss_chest = \
                            _training_step(global_iter,
                                           sys_parameters.num_iter_train_save,
                                           batch_size,
                                           min_snr_db, max_snr_db,
                                           double_readout,
                                           weighting_double_readout,
                                           apply_multiloss,
                                           train_tx)
                save_weights(model, filename)
                for mcs_arr_idx in mcs_arr_training_idx:
                    if not sys_parameters.ebno:
                        _no = ebnodb2no(
                            eval_ebno_db_arr[mcs_arr_idx],
                            model._transmitters[mcs_arr_idx]._num_bits_per_symbol,
                            model._transmitters[mcs_arr_idx]._target_coderate,
                            model._transmitters[mcs_arr_idx]._resource_grid)
                        _snr_db = - 10.0 * tf.math.log(_no) / tf.math.log(10.0)
                    else:
                        _snr_db = eval_ebno_db_arr[mcs_arr_idx]
                    loss_data_mcs = eval_model_xla(batch_size, _snr_db,
                                                   max_num_tx, mcs_arr_idx)
                    tf.summary.scalar(
                                f"Eval loss / mcs_arr_idx=" + str(mcs_arr_idx),
                                loss_data_mcs, step=global_iter)
                tf.summary.scalar(f"Loss", loss_data, step=global_iter)
                tf.summary.scalar(f"Loss Ch. Est.", loss_chest,
                                  step=global_iter)
                tf.summary.scalar(f"Total Loss", loss, step=global_iter)

def calculate_goodput(pe, pusch_transmitter, verbose=False):
    """Calculate goodput."""

    num_info_bits = pusch_transmitter._pusch_configs[0].tb_size

    rg_type = pusch_transmitter._resource_grid.build_type_grid()
    num_res = tf.reduce_prod(rg_type.shape[-2:]).numpy()

    eps = 1e-6
    num_pilots = tf.reduce_sum(tf.where(
                    tf.math.abs(pusch_transmitter.pilot_pattern.pilots[0])>eps,
                               1,0)).numpy()

    num_empty_pilots = tf.reduce_sum(tf.where(
                    tf.math.abs(pusch_transmitter.pilot_pattern.pilots[0])<eps,
                             1,0)).numpy()

    gp_baseline = (1 - pe) * num_info_bits / (num_res - num_empty_pilots)
    gp_e2e = (1 - pe)*num_info_bits / (num_res - num_pilots - num_empty_pilots)

    if verbose:
        print(f"------------------------------")
        print(f"Total number of REs: {num_res}")
        print(f"Total number of payload bits: {num_info_bits}")
        print(f"Number of pilots: {num_pilots}")
        print(f"Number of empty pilots: {num_empty_pilots}")
        print(f"Goodput w. pilots: {gp_baseline} [info. bits / RE]")
        print(f"Goodput w.o. pilots: {gp_e2e} [info. bits / RE]")
    return gp_baseline, gp_e2e

def plot_results(config_name, show_ber=False, xlim=None, ylim=None,
                 sim_idx=None, num_tx_eval=None, fig=None, color_offset=0,
                 labels=None, mcs_arr_eval_idx=0):
    # pylint: disable=line-too-long
    """Visualize results"""
    sys_parameters = Parameters(config_name,
                                training=False,
                                system='dummy')
    filename = f"../results/{sys_parameters.label}_results"

    if num_tx_eval is None:
        num_tx_eval = sys_parameters.max_num_tx

    if sim_idx is not None:
        assert isinstance(sim_idx, (int, list, tuple)),\
            "sim_idx must be list of ints."
        if isinstance(sim_idx, int):
            sim_idx = [sim_idx]

    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8));
    else:
        ax = fig.gca()

    if exists(filename):
        with open(filename,'rb') as f:
            snrs, BERs, BLERs = pickle.load(f)

        if show_ber:
            ERs = BERs
        else:
            ERs = BLERs

        if sim_idx is None:
            sim_idx=np.arange(len(ERs))

        idx = 0
        l_idx = 0
        for e in ERs:
            if num_tx_eval == e[1]:
                if idx in sim_idx:
                    if len(e) == 2 or mcs_arr_eval_idx == e[2]:
                        if labels is None:
                            l = e[0]
                        else:
                            l = labels[l_idx]
                            l_idx += 1
                        ax.semilogy(snrs, ERs[e], label=l,
                                    color=COLORMAP[idx+color_offset],
                                    linewidth=3.0)
                        idx += 1
    else:
        print("No results found")

    title = f"5G NR PUSCH {num_tx_eval}x{sys_parameters.num_rx_antennas} "\
            f"MU-MIMO, {sys_parameters.channel_type}-Channel, " \
            f"MCS={sys_parameters.mcs_index[mcs_arr_eval_idx]}, "\
            f"PRBs={sys_parameters.n_size_bwp}"

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, which="both")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("SNR [dB]", fontsize=15)
    if show_ber:
        ax.set_ylabel("BER", fontsize=15)
    else:
        ax.set_ylabel("TBLER", fontsize=15)

    ax.legend(loc="lower left", fontsize=15);

    if xlim is not None:
        ax.set_xlim(xlim);
    else:
        ax.set_xlim([min(snrs),max(snrs)]);

    if ylim is not None:
        ax.set_ylim(ylim);

    return fig

def export_csv(config_name, num_tx_eval):
    """Export results to csv for pgfplots etc."""
    sys_parameters = Parameters(config_name,
                                training=True,
                                system='dummy')
    filename = f"../results/{sys_parameters.label}_results"

    with open(filename,'rb') as f:
        snrs, BERs, BLERs = pickle.load(f)

    a = snrs
    for label in BLERs:
        if num_tx_eval == label[1]:
            a = np.column_stack((snrs, BLERs[label].numpy()))
            if label[0]=="Baseline - LS/lin+LMMSE":
                l = "lslin+lmmse"
            elif label[0]=="Neural Receiver":
                l = "nn"
            elif label[0]=="Baseline - Perf. CSI & K-Best":
                l = "perfcsi"
            elif label[0]=="Baseline - LMMSE+K-Best":
                l = "lmmse+kbest"
            else:
                pass
            df = pd.DataFrame(a, columns=['snr', "bler"])

            df.to_csv(f"{sys_parameters.label}_{num_tx_eval}_{l}.csv",
                      index=False)

def plot_gp(config_name, num_tx_eval=None, xlim=None, ylim=None, fig=None,
            sim_idx=None, color_offset=0, labels=None, verbose=False):
    """Plot gp."""
    sys_parameters = Parameters(config_name,
                                training=False,
                                system='nn')
    filename = f"../results/{sys_parameters.label}_results"

    if num_tx_eval is None:
        num_tx_eval = sys_parameters.max_num_tx

    if sim_idx is not None:
        assert isinstance(sim_idx, (int, list, tuple)),\
            "sim_idx must be list of ints."
        if isinstance(sim_idx, int):
            sim_idx = [sim_idx]

    if fig is None:
        fig, ax = plt.subplots(figsize=(12,8));
    else:
        ax = fig.gca()

    if exists(filename):
        with open(filename,'rb') as f:
            snrs, BERs, BLERs = pickle.load(f)

        if sim_idx is None:
            sim_idx=np.arange(len(BLERs))

        idx = 0
        l_idx = 0
        for e in BLERs:
            if num_tx_eval == e[1]:
                if idx in sim_idx:
                    if labels is None:
                        l = e[0]
                    else:
                        l = labels[l_idx]
                        l_idx += 1
                    gp_bs, gp_e2e = calculate_goodput(
                                            BLERs[e],
                                            sys_parameters.transmitters[0],
                                            verbose=verbose)
                    if e[0]=="Neural Receiver" and sys_parameters.mask_pilots:
                        print("masked pilots detected: "\
                              "ignoring DMRS overhead for NRX results")
                        gp = gp_e2e
                    else:
                        gp = gp_bs
                    ax.plot(snrs, gp,
                            label=l, color=COLORMAP[idx+color_offset],
                            linewidth=3.0)
                idx += 1
    else:
        print("No results found")

    title = f"Goodput: {num_tx_eval}x{sys_parameters.num_rx_antennas} " \
            f"MU-MIMO, {sys_parameters.channel_type}-Channel, "\
            f"MCS={sys_parameters.mcs_index}, PRBs={sys_parameters.n_size_bwp}"

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, which="both")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("SNR [dB]", fontsize=15)
    ax.set_ylabel("[info. bits / RE]", fontsize=15)
    ax.legend(loc="lower left", fontsize=15);

    if xlim is not None:
        ax.set_xlim(xlim);
    else:
        ax.set_xlim([min(snrs),max(snrs)]);

    if ylim is not None:
        ax.set_ylim(ylim);

    return fig


def export_constellation(config_name, fn="custom_constellation"):
    """Export constellation."""

    sys_parameters = Parameters(config_name,
                                training=True,
                                system='nn')

    model = E2E_Model(sys_parameters, training=False)

    model(1,1.);
    filename = f'../weights/{sys_parameters.label}_weights'
    load_weights(model, filename)

    cs = model._transmitters[0]._mapper.constellation.points.numpy()

    m = int(np.log2(len(cs)))
    labels = np.zeros((len(cs), m))
    for idx in range(len(cs)):
        labels[idx,:] = sn.fec.utils.int2bin(idx, m)

    r = {}
    for idx,(l,c) in enumerate(zip(labels,cs)):
        r.update({f"{idx}": {"constellation": c, "label": l}})

    df = pd.DataFrame(r)
    df.to_csv(fn+".csv", index=False)

    return cs, labels

def sample_along_trajectory(waypoints, num_points, velocity):
    """Sample along trajectory."""
    num_segments = len(waypoints) - 1
    waypoints = np.array(waypoints)

    directions = np.roll(waypoints, -1, 0) - waypoints
    distances = np.sqrt(np.sum(np.abs(directions)**2, axis=1, keepdims=True))
    directions /= distances

    distances = distances[:-1,...]
    directions = directions[:-1,...]

    total_distance = np.sum(distances)
    sample_distance = total_distance / num_points

    rx_positions = []
    rx_velocities = []
    for i in range(num_segments):
        num_points_segm = int(np.round(distances[i]/sample_distance))
        p = waypoints[i]
        for _ in range(num_points_segm):
            rx_positions.append(np.copy(p))
            p += directions[i] * sample_distance
            rx_velocities.append(directions[i] * velocity)

    rx_positions = rx_positions[:num_points]
    rx_velocities = rx_velocities[:num_points]
    return rx_positions, rx_velocities, total_distance


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(a, tau):
    a_bytes = tf.io.serialize_tensor(a)
    tau_bytes = tf.io.serialize_tensor(tau)
    feature = {
        'a': _bytes_feature(a_bytes.numpy()),
        'tau': _bytes_feature(tau_bytes.numpy())
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
