from __future__ import annotations

import hashlib
import json

import numpy as np
import tensorflow as tf

from temporal_receiver_core import prepare_cgnn_inputs
from receiver_decoding import OneLayerGridCodeMap, DecoderAssistedTB
from .training_options import DEFAULT_DETECTOR_BATCH_TBS, DEFAULT_DESCRIPTOR_CHUNK_TBS
from .pilot_history import PilotHistory


TARGETS = ("y", "ls", "bits", "coded", "no", "channel_targets")


def routing_plan(ids, active, capacity, expiry, history_tbs, times=None):
    from .pilot_history import validate_memory_metadata
    ids, active = np.asarray(ids), np.asarray(active)
    if ids.ndim != 3 or active.shape != ids.shape:
        raise ValueError("Schedules must be matching batch-by-time-by-user arrays")
    batch, length, users = ids.shape
    validate_memory_metadata(ids.reshape(-1, users), active.reshape(-1, users), capacity)
    if type(history_tbs) is not int or history_tbs < 1:
        raise ValueError("History length must be a positive integer")
    times = np.broadcast_to(np.arange(length, dtype=np.int32), (batch, length)) if times is None else np.asarray(times)
    if times.shape != (batch, length) or not np.issubdtype(times.dtype, np.integer) or np.any(times < 0) or np.any(times >= np.iinfo(np.int32).max) or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("Record timestamps must increase within each episode")
    shape = (batch, length, users, history_tbs)
    indices, age, ports = (np.zeros(shape, np.int32) for _ in range(3))
    valid = np.zeros(shape, bool)
    for lane in range(batch):
        history = [[] for _ in range(capacity)]
        for t in range(length):
            for user in range(users):
                if not active[lane, t, user]:
                    continue
                ue = ids[lane, t, user]
                for j, (old_t, old_user) in enumerate(reversed(history[ue])):
                    gap = times[lane, t]-times[lane, old_t]
                    if expiry is None or gap <= expiry:
                        indices[lane, t, user, j] = (lane*length+old_t)*users+old_user
                        age[lane, t, user, j] = gap
                        ports[lane, t, user, j] = old_user
                        valid[lane, t, user, j] = True
                history[ue].append((t, user))
                history[ue] = history[ue][-history_tbs:]
    return dict(indices=indices, age=age, ports=ports, valid=valid)


def packet_fingerprint(packet):
    digest = hashlib.sha256(json.dumps(packet["metadata"], sort_keys=True).encode())
    for key in ("bits", "ue_ids", "active", "ls"):
        digest.update(memoryview(np.ascontiguousarray(packet[key])).cast("B"))
    return digest.hexdigest()


def flatten_time(value):
    return tf.reshape(value, [-1, *value.shape[2:]])


def signature(value):
    shape = list(value.shape)
    shape[0] = None
    return tf.TensorSpec(shape, value.dtype)


def channel_nll(estimate, targets, layout):
    antennas = estimate.coefficients.shape[-2]
    count = antennas*(antennas-1)//2
    real, imag, diagonal, off_real, off_imag = tf.split(targets, [3*antennas, 3*antennas, antennas, count, count], -1)
    z = tf.reshape(tf.complex(real, imag), [tf.shape(targets)[0], layout.users, layout.count, antennas, 3])
    upper = np.transpose(np.nonzero(np.triu(np.ones((antennas, antennas), bool), 1)))
    basis = np.zeros((count, antennas, antennas), np.complex64)
    for i, (r, c) in enumerate(upper):
        basis[i, r, c] = 1.
    residual = tf.einsum('bucj,jrs->bucrs', tf.complex(off_real, off_imag), tf.constant(basis))
    residual += tf.linalg.adjoint(residual)+tf.linalg.diag(tf.cast(diagonal, tf.complex64))
    delta = tf.linalg.matmul(estimate.coefficients, tf.cast(layout.r_layout, tf.complex64), transpose_b=True)-z
    error = tf.linalg.matmul(delta, delta, adjoint_b=True)+residual
    counts = tf.constant(layout.data_counts, tf.float32)
    present = (counts[None] > 0) & (tf.reduce_sum(estimate.variance, -1) > 0)
    covariance = tf.where(present[..., None, None], estimate.covariance, tf.eye(antennas, dtype=tf.complex64))
    error = tf.where(present[..., None, None], error, 0.)
    factor = tf.linalg.cholesky(covariance)
    trace = tf.math.real(tf.linalg.trace(tf.linalg.cholesky_solve(factor, error)))
    logdet = 2*tf.reduce_sum(tf.math.log(tf.math.real(tf.linalg.diag_part(factor))), -1)
    loss = tf.where(present, (trace+logdet)/antennas, 0.)
    return tf.math.divide_no_nan(tf.reduce_sum(loss*counts[None], -1), tf.reduce_sum(counts, -1)[None])


class BatchedTrainingExecution:
    def __init__(self, trainer, detector_batch_tbs=DEFAULT_DETECTOR_BATCH_TBS,
                 descriptor_chunk_tbs=DEFAULT_DESCRIPTOR_CHUNK_TBS):
        if min(detector_batch_tbs, descriptor_chunk_tbs) < 1:
            raise ValueError("Execution capacities must be positive")
        self.trainer = trainer
        self.detector_batch_tbs = int(detector_batch_tbs)
        self.descriptor_chunk_tbs = int(descriptor_chunk_tbs)
        self.descriptor_functions = {}
        self.gradient_functions = {}

    def settings(self):
        return dict(version="pilot_filter_learned_blend_joint_v1",
                    detector_batch_tbs=self.detector_batch_tbs,
                    descriptor_chunk_tbs=self.descriptor_chunk_tbs,
                    tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled())

    def describe(self, channel, noise):
        shape = channel.shape
        width = shape[3]//12
        if width not in self.descriptor_functions:
            self.descriptor_functions[width] = tf.function(
                self.trainer.model.observe,
                input_signature=[tf.TensorSpec([None, *shape[2:]], tf.float32),
                                 tf.TensorSpec([None], tf.float32)])
        flat = tf.reshape(channel, [-1, *shape[2:]]) if tf.is_tensor(channel) else channel.reshape(-1, *shape[2:])
        noise = tf.reshape(noise, [-1]) if tf.is_tensor(noise) else noise.reshape(-1)
        descriptions = [self.descriptor_functions[width](flat[i:i+self.descriptor_chunk_tbs],
                        noise[i:i+self.descriptor_chunk_tbs])
                        for i in range(0, shape[0]*shape[1], self.descriptor_chunk_tbs)]
        value = tf.concat(descriptions, 0)
        return tf.reshape(value, [*shape[:2], *value.shape[1:]])

    def prepare(self, packets):
        prepared, fingerprints, den = [], [], 0.
        for packet in packets:
            ids, active = np.asarray(packet["ue_ids"]), np.asarray(packet["active"])
            if ids.shape[0] not in (1, 2) or ids.shape[1] < 1:
                raise ValueError("Each base episode needs one history or a paired history")
            if not np.all(ids == ids[:1]) or not np.all(active == active[:1]):
                raise ValueError("Paired histories must use identical schedules")
            if "time_index" in packet and not np.all(np.asarray(packet["time_index"]) == np.asarray(packet["time_index"])[:1]):
                raise ValueError("Paired histories must use identical timestamps")
            for key in TARGETS:
                if tuple(packet[key].shape[:2]) != tuple(ids.shape[:2]):
                    raise ValueError(f"Missing full-episode targets: {key}")
            if ids.shape[0] == 2:
                suffix = int(packet["metadata"]["paired_suffix_tbs"])
                if not 0 < suffix < ids.shape[1]:
                    raise ValueError("Paired histories need a common suffix and distinct earlier observations")
                for key in TARGETS:
                    future = np.asarray(packet[key])[:, -suffix:]
                    if not np.array_equal(future[0], future[1]):
                        raise ValueError(f"Paired histories must have identical future {key}")
            den += float(active[0].sum())
            plan = routing_plan(ids, active, self.trainer.manager.capacity, self.trainer.manager.expiry_slots,
                                self.trainer.model.history_tbs, packet.get("time_index"))
            current = {k: tf.convert_to_tensor(v) for k, v in plan.items()}
            current["observation"] = self.describe(packet["ls"], packet["no"])
            current["active"] = tf.convert_to_tensor(active, tf.float32)
            current["ue_ids"] = tf.convert_to_tensor(ids, tf.int32)
            with tf.device("/CPU:0"):
                current.update({k: tf.convert_to_tensor(packet[k]) for k in TARGETS})
            prepared.append(current)
            fingerprints.append(packet_fingerprint(packet))
        return prepared, fingerprints, den

    def gradient_function(self, width, targets, current, bank):
        if width in self.gradient_functions:
            return self.gradient_functions[width]
        from .training import per_user_bce, per_user_payload_loss
        receiver = self.trainer.contexts[width][1]._receiver
        model, recipe = self.trainer.model, self.trainer.recipe
        users = targets[1].shape[1]
        mapping = OneLayerGridCodeMap(receiver._neural_rx, users, model.base._num_bits_per_symbol[0])
        iterations = int(self.trainer.settings["architecture"]["ldpc_iterations"])
        decoder = DecoderAssistedTB(receiver._tb_encoders[0], receiver._tb_decoders[0], iterations,
                                    self.trainer.settings['architecture']['ldpc_first_iterations'])
        variables = self.trainer.optimization_variables
        specs = [tuple(signature(v) for v in targets), tuple(signature(v) for v in current),
                 signature(bank), tf.TensorSpec([None], tf.float32), tf.TensorSpec([], tf.float32),
                 tuple(tf.TensorSpec(v.shape, v.dtype) for v in variables), tf.TensorSpec([7], tf.float32)]

        @tf.function(input_signature=specs)
        def gradients(targets, current, bank, weights, den, accumulated, totals):
            y, channel, bits, coded, noise, channel_targets = targets
            observation, active, indices, age, ports, valid, ue_ids = current
            previous = tf.gather(bank, indices)
            history = PilotHistory(previous, age, ports, valid)
            layout = model.layout_for(channel)
            inputs = prepare_cgnn_inputs(receiver, y, channel, active)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables)
                both_inputs = [value if i == 1 else tf.concat([value, value], 0)
                               for i, value in enumerate(inputs)]
                reset = model.empty_history(observation)
                both_history = PilotHistory(*(tf.concat([a, b], 0) for a, b in zip(history, reset)))
                output, llr, corrected, posterior, _, estimate = model.decode(both_inputs, tf.concat([noise, noise], 0),
                    tf.concat([ue_ids, ue_ids], 0), both_history, decoder, mapping, tf.concat([observation, observation], 0))
                labels = tf.concat([coded, coded], 0)
                coded_losses = tf.split((per_user_bce(labels, llr)+per_user_bce(labels, posterior)
                                         +per_user_bce(labels, corrected))/3., 2, axis=0)
                payload_losses = tf.split(per_user_payload_loss(tf.concat([bits, bits], 0), output.logits), 2, axis=0)
                mask = active*weights[:, None]/den
                pc, rc = [tf.reduce_sum(mask*value) for value in coded_losses]
                pp, rp = [tf.reduce_sum(mask*value) for value in payload_losses]
                channel_losses = channel_nll(estimate, tf.concat([channel_targets, channel_targets], 0), layout)
                ph, rh = [tf.reduce_sum(mask*value) for value in tf.split(channel_losses, 2, axis=0)]
                cw, pw, rw = recipe["coded_loss_weight"], recipe["payload_loss_weight"], recipe["reset_weight"]
                loss = (cw*(pc+rw*rc) + pw*(pp+rw*rp)
                        + recipe["channel_loss_weight"]*(ph+rw*rh))/((cw+pw)*(1+rw))
            grads = tape.gradient(loss, variables,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            tf.debugging.assert_all_finite(loss, "Nonfinite joint training loss")
            accumulated = tuple(a+g for a, g in zip(accumulated, grads))
            return accumulated, totals+tf.stack([loss, pc, pp, rc, rp, ph, rh])

        self.gradient_functions[width] = gradients
        return gradients

    @tf.function(reduce_retracing=True)
    def finish(self, gradients, totals, memory_rms):
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite joint training gradient")
        norm = tf.linalg.global_norm(gradients)
        tf.debugging.assert_all_finite(norm, "Nonfinite gradient norm")
        count = len(self.trainer.memory_variables)
        metrics = tf.concat([totals, [norm, tf.linalg.global_norm(gradients[:count]), memory_rms]], 0)
        return gradients, metrics

    def run(self, width, packets, denominator=None):
        prepared, fingerprints, local_den = self.prepare(packets)
        den = local_den if denominator is None else float(denominator)
        if den <= 0 or den < local_den:
            raise ValueError("The global denominator must include all active blocks")
        observations = tf.concat([flatten_time(p["observation"]) for p in prepared], 0)
        bank = tf.reshape(observations, [-1, *observations.shape[2:]])
        index_parts, weights, offset = [], [], 0
        for packet in prepared:
            lanes, steps, users = packet["active"].shape
            index_parts.append(flatten_time(packet["indices"])+offset)
            weights.extend([1./lanes]*(lanes*steps))
            offset += lanes*steps*users
        weights = tf.constant(weights, tf.float32)
        with tf.device("/CPU:0"):
            targets = tuple(tf.concat([flatten_time(p[key]) for p in prepared], 0) for key in TARGETS)
        active = tf.concat([flatten_time(p["active"]) for p in prepared], 0)
        metadata = tuple(tf.concat([flatten_time(p[key]) for p in prepared], 0) for key in ("age", "ports", "valid"))
        ids = tf.concat([flatten_time(p["ue_ids"]) for p in prepared], 0)
        current = (observations, active, tf.concat(index_parts, 0), *metadata, ids)
        accumulated = tuple(tf.zeros_like(v) for v in self.trainer.optimization_variables)
        totals = tf.zeros([7])
        function = self.gradient_function(width, targets, current, bank)
        for start in range(0, len(weights), self.detector_batch_tbs):
            end = min(start+self.detector_batch_tbs, len(weights))
            accumulated, totals = function(tuple(v[start:end] for v in targets),
                tuple(v[start:end] for v in current), bank, weights[start:end], tf.constant(den), accumulated, totals)
        memory_rms = tf.sqrt(tf.reduce_mean(tf.square(bank[..., :-1])))
        gradients, metrics = self.finish(accumulated, totals, memory_rms)
        return gradients, metrics, fingerprints, local_den
