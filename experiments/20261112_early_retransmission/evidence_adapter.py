"""The proposed NRX's benchmark adapter, also returning per-TB evidence for the early-retransmission controller.

The decoding path is the PilotHistoryBenchmarkAdapter graph, copied unchanged. The evidence is read from tensors that
graph already produces (memory contents, current pilots, the filtered channel estimate and its error covariance, the
NRX LLRs), plus a separate recomputation of the first BP stage (8 iterations) whose unsatisfied parity checks are
counted on the CPU. log_evidence.py checks that every decision equals the stored TBLER sweep.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from neural_rx.benchmark.candidate import checked_decoder_tensors
from neural_rx.receiver.benchmark_adapter import PilotHistoryBenchmarkAdapter
from neural_rx.receiver.channel_filter import power

# Per [SNR point, UE] of each TB. Powers are means over DMRS positions of the squared norm over BS antennas.
FEATURES = (
    "memory_entries",        # valid stored TBs of this UE (0-7), before this TB is written
    "memory_newest_age",     # age of the newest valid entry in TBs (0 if none)
    "memory_newest_power",   # LS channel power of the newest valid entry (0 if none)
    "memory_newest_noise",   # its LS noise variance per antenna
    "memory_mean_power",     # mean LS channel power over valid entries (0 if none)
    "ls_power",              # LS channel power of the current TB
    "ls_noise",              # LS noise variance per antenna of the current TB
    "estimate_power",        # power of the filtered channel estimate
    "estimate_error",        # trace of the estimated error covariance (Sigma)
    "llr_magnitude",         # mean |L| over the UE's coded bits
    "llr_bit_error",         # mean of sigmoid(-|L|), the bit error rate the NRX LLRs imply
    "posterior_bit_error",   # the same after the first BP stage (8 iterations)
    "unsatisfied_checks",    # unsatisfied parity checks after the first BP stage, summed over code blocks
    "failed_code_blocks",    # code blocks with at least one unsatisfied check after the first BP stage
)


class EvidenceAdapter(PilotHistoryBenchmarkAdapter):
    last_evidence = None

    def _parity(self):
        if not hasattr(self, "_parity_matrix"):
            matrix = sparse.csr_matrix(self.decoder.first.tb_decoder._decoder.pcm, dtype=np.int32)
            if matrix.shape[1] != self.decoder.first.cb_length:
                raise ValueError("Parity-check matrix and code-block length differ")
            self._parity_matrix = matrix
        return self._parity_matrix

    def _build_execution(self):
        tf = self.tf
        buffer = self._buffer
        if self.method != "candidate_persistent":
            raise ValueError("Evidence logging is defined for the proposed NRX with its memory")
        parity = self._parity()

        # Copied from PilotHistoryBenchmarkAdapter._build_execution; the decoding path must stay identical.
        @tf.function(reduce_retracing=True, jit_compile=True)
        def prepare(channel, noise, ids, active, step, head, count, tables):
            history = buffer.gather(tables, head, count, ids, active, step)
            observation = self.model.observe(channel, noise)
            preparation = self.model.prepare([None, None, channel, tf.cast(active, tf.float32)],
                                             observation, history, self.layout)
            pending, finite = buffer.prepare_stage(observation, head, count, ids, active)
            return observation, preparation, pending, finite

        @tf.function(reduce_retracing=True, jit_compile=False)
        def execute(y, channel, noise, control, active, head, count):
            ids, step = tf.cast(control[:, :-1], tf.int32), control[0, -1]
            tables = tuple(v.read_value() for v in (buffer.observations, buffer.timestamps, buffer.ports))
            observation, preparation, pending, finite = prepare(
                channel, noise, ids, active, step, head, count, tables)
            inputs = self._prepare_inputs(self.receiver, y, channel, tf.cast(active, tf.float32))
            output, llr, _, posterior_bits, observation, estimate = self.model.decode(
                inputs, noise, ids, None, self.decoder, self.grid_map, observation, preparation)
            logits, hard, crc = checked_decoder_tensors(output)
            with tf.control_dependencies([logits, hard, crc]):
                history = buffer.gather(tables, head, count, ids, active, step)
                evidence = self._graph_evidence(history, observation, estimate, llr, posterior_bits)
                _, _, _, posterior = self.decoder.evidence(llr)
                next_head, next_count, complete = buffer.commit(pending, finite, step)
            with tf.control_dependencies([complete]):
                return (tf.identity(logits), tf.identity(hard), tf.identity(crc), next_head, next_count, complete,
                        evidence, tf.cast(posterior > 0., tf.int8))

        def run(*args):
            *outputs, evidence, hard = execute(*args)
            hard = hard.numpy()
            blocks = hard.reshape(-1, hard.shape[-1]).astype(np.int32)
            unsatisfied = ((parity @ blocks.T) % 2).sum(0).reshape(hard.shape[:-1])
            self.last_evidence = np.concatenate(
                [evidence.numpy(), unsatisfied.sum(-1)[..., None], (unsatisfied > 0).sum(-1)[..., None]],
                -1).astype(np.float64)
            return tuple(outputs)

        return run

    def _graph_evidence(self, history, observation, estimate, llr, posterior_bits):
        tf = self.tf
        valid = history.valid
        entries = tf.reduce_sum(tf.cast(valid, tf.float32), -1)
        ages = tf.where(valid, tf.cast(history.age, tf.float32), 1e9)
        newest = tf.argmin(ages, -1, output_type=tf.int32)
        stored_power = tf.reduce_mean(tf.reduce_sum(tf.square(history.observations[..., :-1]), -1), -1)
        stored_noise = tf.reduce_mean(history.observations[..., -1], -1)
        some = entries > 0
        newest_age = tf.where(some, tf.reduce_min(ages, -1), 0.)
        newest_power = tf.where(some, tf.gather(stored_power, newest, axis=2, batch_dims=2), 0.)
        newest_noise = tf.where(some, tf.gather(stored_noise, newest, axis=2, batch_dims=2), 0.)
        mean_power = tf.reduce_sum(tf.where(valid, stored_power, 0.), -1)/tf.maximum(entries, 1.)
        ls_power = tf.reduce_mean(tf.reduce_sum(tf.square(observation[..., :-1]), -1), -1)
        ls_noise = tf.reduce_mean(observation[..., -1], -1)
        estimate_power = tf.reduce_mean(tf.reduce_sum(power(estimate.coefficients[..., 0, :]), -1), -1)
        estimate_error = tf.reduce_mean(tf.reduce_sum(estimate.variance, -1), -1)
        magnitude = tf.abs(llr)
        posterior_magnitude = tf.abs(posterior_bits)
        return tf.stack([entries, newest_age, newest_power, newest_noise, mean_power, ls_power, ls_noise,
                         estimate_power, estimate_error, tf.reduce_mean(magnitude, -1),
                         tf.reduce_mean(tf.sigmoid(-magnitude), -1),
                         tf.reduce_mean(tf.sigmoid(-posterior_magnitude), -1)], -1)
