#!/usr/bin/env python3
"""Build the frozen v18 final-validation decoder runtime."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_RECEIVER_CONFIG = {
    "learned_iterations": 2,
    "teacher_iterations": 8,
    "ldpc_iterations": 20,
    "bp_first_iterations": 8,
    "bp_remaining_iterations": 12,
    "support_width": 256,
    "candidate_cap": 256,
    "optimized_chunk_size": 6336,
    "num_tx": 2,
    "bits_per_symbol": 4,
    "n_size_bwp": 132,
    "mcs_index": 14,
}


def load_v18_segment_safe_config(path: Path | str) -> dict:
    """Load the segment-safe V18 configuration."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_memory = {
        "type": "segment_safe_pilot_domain_wiener",
        "maximum_history_tbs": 8,
        "reset_threshold_file": "segment_safe_pd_l8_threshold.json",
        "probation_transitions": 1,
        "minimum_prior_innovations": 1,
        "refractory_tbs": 0,
    }
    if value.get("architecture") != (
        "distilled_pd_l8_cjlr_v18_segment_safe"
    ):
        raise ValueError("configuration is not segment-safe V18")
    if value.get("parent_architecture") != "distilled_pd_l8_cjlr_v18":
        raise ValueError("segment-safe V18 parent differs")
    if value.get("receiver") != EXPECTED_RECEIVER_CONFIG:
        raise ValueError("segment-safe V18 receiver settings changed")
    if value.get("memory") != expected_memory:
        raise ValueError("segment-safe V18 memory settings changed")
    if value.get("checkpoint") != {
        "format": "selected_variables_npz",
        "binding": "sealed_candidate_manifest",
    }:
        raise ValueError("segment-safe V18 checkpoint binding changed")
    return value


class FrozenV18Runtime:
    """Hold immutable v18 models and the candidate decode graph."""

    def __init__(
        self,
        tf,
        config_name: str,
        receiver_config: dict,
        checkpoint,
        batch_size: int,
    ):
        from cjlr_optimized_receiver import (
            ExactCartesianKernel,
            UnrolledCodedJointTBDecoder,
        )
        from cjlr_receiver import (
            OneLayerGridCodeMap,
            StatefulSoftTBCode,
            constellation_table,
        )
        from temporal_receiver_core import (
            assign_selected,
            build_backbone,
            derive_cdm_noise_conversion,
            direct_k2_trainable_variables,
            model_checksum,
            prepare_cgnn_inputs,
            received_to_grid,
            run_k2,
            select_channel_input,
            select_channel_variance,
            stacked_to_complex_channel,
        )

        self.tf = tf
        self.config = receiver_config
        self.batch_size = int(batch_size)
        self.users = int(receiver_config["num_tx"])
        self.run_k2 = run_k2
        self.prepare_cgnn_inputs = prepare_cgnn_inputs
        self.select_channel_input = select_channel_input
        self.select_channel_variance = select_channel_variance
        self.received_to_grid = received_to_grid
        self.stacked_to_complex_channel = stacked_to_complex_channel
        self.model_checksum = model_checksum

        parameters, teacher_e2e = build_backbone(
            config_name,
            num_it=int(receiver_config["teacher_iterations"]),
            training=False,
            num_tx_eval=self.users,
            n_size_bwp_override=int(receiver_config["n_size_bwp"]),
        )
        _, student_e2e = build_backbone(
            config_name,
            num_it=int(receiver_config["learned_iterations"]),
            training=False,
            num_tx_eval=self.users,
            n_size_bwp_override=int(receiver_config["n_size_bwp"]),
        )
        self.receiver = teacher_e2e._receiver
        self.teacher = self.receiver._neural_rx._cgnn
        self.student = student_e2e._receiver._neural_rx._cgnn
        assign_selected(
            self.student,
            checkpoint,
            direct_k2_trainable_variables(self.student),
        )
        self.teacher_before = model_checksum(self.teacher)
        self.student_before = model_checksum(self.student)
        self.grid_map = OneLayerGridCodeMap(
            self.receiver._neural_rx,
            self.users,
            int(receiver_config["bits_per_symbol"]),
        )
        points, labels = constellation_table(
            teacher_e2e._transmitters[0]._mapper
        )
        self.kernel = ExactCartesianKernel(
            points.numpy(), labels.numpy(), self.users, "cartesian"
        )
        if self.kernel.support_width != int(receiver_config["support_width"]):
            raise ValueError("exact CJLR support width changed")
        early_decoder = StatefulSoftTBCode(
            self.receiver._tb_encoders[0],
            self.receiver._tb_decoders[0],
            int(receiver_config["bp_first_iterations"]),
        )
        self.cjlr_decoder = UnrolledCodedJointTBDecoder(
            self.receiver._tb_encoders[0],
            self.receiver._tb_decoders[0],
            int(receiver_config["bp_first_iterations"]),
            int(receiver_config["bp_remaining_iterations"]),
            reference_decoder=early_decoder.decoder,
        )
        self.cdm = derive_cdm_noise_conversion(
            self.receiver._ls_est, self.users
        )
        self.symbols = int(self.grid_map.num_data_symbols)
        self.receive_antennas = int(parameters.num_rx_antennas)
        self.bits = int(receiver_config["bits_per_symbol"])
        self.sequential_delta = self._make_delta(self.batch_size)
        self.decode_candidate = self._build_candidate_graph()

    def _make_delta(self, expected_batch: int):
        from cjlr_optimized_receiver import make_chunked_exact_delta

        tf = self.tf
        expected_batch = int(expected_batch)
        flat_length = expected_batch * self.symbols
        chunk_size = int(self.config["optimized_chunk_size"])
        if flat_length % chunk_size:
            raise ValueError("flattened cells do not divide frozen chunk")
        graph = make_chunked_exact_delta(
            self.kernel, chunk_size, jit_compile=True
        )
        concrete = graph.get_concrete_function(
            tf.TensorSpec(
                [flat_length, self.receive_antennas], tf.complex64
            ),
            tf.TensorSpec(
                [flat_length, self.users, self.receive_antennas], tf.complex64
            ),
            tf.TensorSpec(
                [flat_length, self.users, self.receive_antennas], tf.float32
            ),
            tf.TensorSpec(
                [flat_length, self.receive_antennas], tf.float32
            ),
            tf.TensorSpec([flat_length, self.users, self.bits], tf.float32),
        )

        def delta(y, channel, variance, noise, extrinsic):
            value = concrete(
                tf.reshape(y, [flat_length, self.receive_antennas]),
                tf.reshape(
                    channel,
                    [flat_length, self.users, self.receive_antennas],
                ),
                tf.reshape(
                    variance,
                    [flat_length, self.users, self.receive_antennas],
                ),
                tf.reshape(noise, [flat_length, self.receive_antennas]),
                tf.reshape(extrinsic, [flat_length, self.users, self.bits]),
            )
            value = tf.reshape(
                value,
                [expected_batch, self.symbols, self.users, self.bits],
            )
            value = tf.transpose(value, [0, 2, 1, 3])
            return tf.reshape(
                value,
                [expected_batch, self.users, self.symbols * self.bits],
            )

        return delta

    def _received_noise(self, observation_variance, active):
        tf = self.tf
        active_float = tf.cast(active, tf.float32)
        noise_grid = tf.math.divide_no_nan(
            tf.reduce_sum(
                observation_variance
                * active_float[..., None, None, None],
                axis=1,
            ),
            tf.reduce_sum(active_float, axis=1)[..., None, None, None],
        ) * float(self.cdm["factor"])
        return self.grid_map.gather_shared_grid(noise_grid)

    def _cjlr_one(
        self,
        y,
        channel,
        channel_variance,
        observation_variance,
        active,
    ):
        tf = self.tf
        inputs = self.prepare_cgnn_inputs(
            self.receiver, y, channel, active
        )
        coded = self.grid_map.grid_to_coded(
            self.run_k2(self.student, inputs)
        )
        y_data = self.grid_map.gather_shared_grid(
            self.received_to_grid(y)
        )
        pd_channel = self.grid_map.gather_stream_grid(
            self.stacked_to_complex_channel(channel)
        )
        pd_variance = self.grid_map.gather_stream_grid(channel_variance)
        noise = self._received_noise(observation_variance, active)

        def refine(extrinsic):
            data = self.grid_map.coded_to_data(extrinsic)
            return self.sequential_delta(
                y_data,
                tf.transpose(pd_channel, [0, 2, 1, 3]),
                tf.transpose(pd_variance, [0, 2, 1, 3]),
                noise,
                tf.transpose(data, [0, 2, 1, 3]),
            )

        return self.cjlr_decoder(coded, refine)

    def _build_candidate_graph(self):
        tf = self.tf

        @tf.function(reduce_retracing=True, jit_compile=False)
        def decode(
            y,
            current_ls,
            posterior,
            posterior_variance,
            observation_variance,
            history_valid,
            reset_mask,
            active,
        ):
            selected, available = self.select_channel_input(
                current_ls,
                posterior,
                history_valid,
                reset_mask=reset_mask,
            )
            selected_variance = self.select_channel_variance(
                posterior_variance,
                observation_variance,
                available,
            )
            output = self._cjlr_one(
                y,
                selected,
                selected_variance,
                observation_variance,
                active,
            )
            return output.logits, output.hard_bits, output.crc_status

        return decode

    def checksums_unchanged(self) -> bool:
        return bool(
            self.teacher_before == self.model_checksum(self.teacher)
            and self.student_before == self.model_checksum(self.student)
        )
