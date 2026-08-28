#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-config_name", help="config filename", type=str)
parser.add_argument("-num_samples", help="Number of samples",
                    type=int, default=1000000)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx_eval", help="Number of active users",
                    type=int, default=1)

args = parser.parse_args()
config_name = args.config_name
num_tx_eval = args.num_tx_eval


import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
try:
    print('Only GPU number', args.gpu, 'used.')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)

import sys
sys.path.append('../')

import sionna as sn
sn.Config.xla_compat = True
from sionna.channel import GenerateOFDMChannel, gen_single_sector_topology

from utils import Parameters
import numpy as np

parameters = Parameters(config_name,
                        training=False,
                        num_tx_eval=num_tx_eval,
                        system='nrx',
                        compute_cov=True)

batch_size = parameters.batch_size_eval
NUM_SAMPLES = args.num_samples
NUM_IT = int((NUM_SAMPLES//batch_size)+1)

channel_model = parameters.channel_model

gen_ofdm_channel = GenerateOFDMChannel(
                                    channel_model,
                                    parameters.transmitters[0]._resource_grid,
                                    normalize_channel=True)


def sample_channel(batch_size):
    topology = gen_single_sector_topology(batch_size, 1, 'umi',
                                    min_ut_velocity=parameters.min_ut_velocity,
                                    max_ut_velocity=parameters.max_ut_velocity)
    channel_model.set_topology(*topology)

    h_freq = gen_ofdm_channel(batch_size)
    h_freq = h_freq[:,0,:,0,0]

    return h_freq

@tf.function(jit_compile=True)
def estimate_cov_mats(batch_size, num_it):
    rg = parameters.transmitters[0]._resource_grid
    freq_cov_mat = tf.zeros([rg.fft_size, rg.fft_size], tf.complex64)
    time_cov_mat = tf.zeros([rg.num_ofdm_symbols, rg.num_ofdm_symbols],
                             tf.complex64)
    space_cov_mat = tf.zeros([parameters.num_rx_antennas,
                              parameters.num_rx_antennas], tf.complex64)

    for _ in tf.range(num_it):
        h_samples = sample_channel(batch_size)
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        freq_cov_mat += freq_cov_mat_

        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        time_cov_mat += time_cov_mat_

        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(rg.num_ofdm_symbols*num_it, tf.float32),
                               0.0)
    time_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)
    return freq_cov_mat, time_cov_mat, space_cov_mat

freq_cov_mat, time_cov_mat, space_cov_mat = estimate_cov_mats(batch_size,
                                                              NUM_IT)
freq_cov_mat = freq_cov_mat.numpy()
time_cov_mat = time_cov_mat.numpy()
space_cov_mat = space_cov_mat.numpy()

np.save(f'../weights/{parameters.label}_freq_cov_mat', freq_cov_mat)
np.save(f'../weights/{parameters.label}_time_cov_mat', time_cov_mat)
np.save(f'../weights/{parameters.label}_space_cov_mat', space_cov_mat)
