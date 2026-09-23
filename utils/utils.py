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
import pickle
import os
import datetime
import sionna as sn

from sionna.utils import ebnodb2no, expand_to_rank

def save_weights(system, model_path):
    weights = system.get_weights()
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)

def load_weights(system, model_path):
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    system.set_weights(weights)

class TriangularDistributionSampler:

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


def training_loop(model, label, filename, training_logdir, training_seed,
                  training_schedule, eval_ebno_db_arr, min_num_tx, max_num_tx,
                  sys_parameters, mcs_arr_training_idx,
                  mcs_training_snr_db_offset=None, mcs_training_probs=None,
                  weight_saving_schedule=None, xla=False):

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
