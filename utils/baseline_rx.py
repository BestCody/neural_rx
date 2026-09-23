# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from tensorflow.keras.layers import Layer
import tensorflow as tf

from sionna.ofdm import KBestDetector
from sionna.nr import PUSCHReceiver, TBDecoder


class BaselineReceiver(Layer):
    """Perfect-CSI K-Best control for the temporal benchmark."""

    def __init__(self, sys_parameters, dtype=tf.complex64,
                 return_tb_status=False, mcs_arr_eval_idx=0, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if sys_parameters.system != "baseline_perf_csi_kbest":
            raise ValueError("only the perfect-CSI K-Best control is retained")
        detector = KBestDetector(
            "bit",
            sys_parameters.max_num_tx,
            64,
            sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
            sys_parameters.sm,
            constellation_type="qam",
            num_bits_per_symbol=(
                sys_parameters.transmitters[mcs_arr_eval_idx]
                ._num_bits_per_symbol
            ),
        )
        decoder = TBDecoder(
            sys_parameters.transmitters[mcs_arr_eval_idx]._tb_encoder,
            num_bp_iter=sys_parameters.num_bp_iter,
            cn_type=sys_parameters.cn_type,
        )
        self._receiver = PUSCHReceiver(
            sys_parameters.transmitters[mcs_arr_eval_idx],
            channel_estimator="perfect",
            mimo_detector=detector,
            tb_decoder=decoder,
            stream_management=None,
            input_domain="freq",
            return_tb_crc_status=return_tb_status,
        )

    def call(self, inputs):
        y, h, no = inputs
        return self._receiver([y, h, no])
