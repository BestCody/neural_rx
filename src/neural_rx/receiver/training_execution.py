from __future__ import annotations

import hashlib
import json

import numpy as np
import tensorflow as tf

from neural_rx.backbone.core import prepare_cgnn_inputs
from neural_rx.decoding import OneLayerGridCodeMap
from .training_options import DEFAULT_DETECTOR_BATCH_TBS, DEFAULT_DESCRIPTOR_CHUNK_TBS
from .pilot_history import PilotHistory


TARGETS = ("y", "ls", "bits", "coded", "no", "h")


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
    if (times.shape != (batch, length) or not np.issubdtype(times.dtype, np.integer) or np.any(times < 0)
            or np.any(times >= np.iinfo(np.int32).max) or np.any(np.diff(times, axis=1) <= 0)):
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
        return dict(version="pilot_filter_v1",
                    detector_batch_tbs=self.detector_batch_tbs, detector_batch_scaling="constant_prbs_per_call",
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
            if ids.shape[0] != 1 or ids.shape[1] < 1:
                raise ValueError("Each example is one sequence of TBs")
            for key in TARGETS:
                if tuple(packet[key].shape[:2]) != tuple(ids.shape[:2]):
                    raise ValueError(f"Missing full-episode targets: {key}")
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
        from .training import BASELINE, per_user_bce
        receiver = self.trainer.contexts[width][1]._receiver
        model = self.trainer.model
        users = targets[1].shape[1]
        mapping = OneLayerGridCodeMap(receiver._neural_rx, users, model.base._num_bits_per_symbol[0])
        variables = self.trainer.optimization_variables
        baseline = self.trainer.variant == BASELINE
        specs = [tuple(signature(v) for v in targets), tuple(signature(v) for v in current),
                 signature(bank), tf.TensorSpec([None], tf.float32), tf.TensorSpec([], tf.float32),
                 tuple(tf.TensorSpec(v.shape, v.dtype) for v in variables), tf.TensorSpec([4], tf.float32)]

        channel_weight = float(self.trainer.recipe["channel_loss_weight"])
        distillation = float(self.trainer.recipe["distillation_weight"])
        teacher = self.trainer.teacher

        # Loss of both receivers. Data term: the binary cross-entropy of the LLRs, against the transmitted bits and,
        # with weight distillation_weight, against the soft bits of the shipped eight-iteration network. Channel
        # term, as for the original NRX: the squared error of the network's channel estimate, here on the scale of
        # the network's inputs, because that is the only scale the network knows. The proposed receiver refines the
        # LS estimate with the channel filter and the pilot memory before the two CGNN iterations; the retrained
        # two-iteration network uses the LS estimate. The memory of a TB holds the LS estimates of the UE's earlier TBs
        # in the sequence (bank), and the filter has no trained weights.
        @tf.function(input_signature=specs)
        def gradients(targets, current, bank, weights, den, accumulated, totals):
            y, ls, bits, coded, noise, truth = targets
            observation, active, indices, age, ports, valid, ue_ids = current
            inputs = prepare_cgnn_inputs(receiver, y, ls, active)
            truth *= model.base.input_scale(inputs[0])[:, None]
            share = active*weights[:, None]/den
            if teacher is not None:
                state = teacher.initialize_state(inputs)
                for iteration in teacher._iterations:
                    state = iteration([state, inputs[1], inputs[3]])
                soft = tf.sigmoid(mapping.grid_to_coded(teacher._readout_llrs[0](state)))
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables)
                if baseline:
                    grid, state = model.iterate(inputs)
                else:
                    history = PilotHistory(tf.gather(bank, indices), age, ports, valid)
                    grid, _, state = model.detect_state(inputs, noise, history, observation)
                logits = mapping.grid_to_coded(grid)
                bce = tf.reduce_sum(share*per_user_bce(coded, logits))
                soft_bce = tf.reduce_sum(share*per_user_bce(soft, logits)) if teacher is not None else tf.zeros([])
                error = tf.reduce_sum(share*tf.reduce_mean(tf.square(model.channel_estimate(state)-truth), [2, 3, 4]))
                loss = (1.-distillation)*bce+distillation*soft_bce+channel_weight*error
            grads = tape.gradient(loss, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
            return (tuple(a+g for a, g in zip(accumulated, grads, strict=True)),
                    totals+tf.stack([loss, bce, soft_bce, error]))

        self.gradient_functions[width] = gradients
        return gradients

    @tf.function(reduce_retracing=True)
    def finish(self, gradients, totals):
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite joint training gradient")
        norm = tf.linalg.global_norm(gradients)
        tf.debugging.assert_all_finite(norm, "Nonfinite gradient norm")
        # loss, gradient norm, then the loss's parts: cross-entropy against the bits, against the teacher's soft
        # bits, and the channel-estimate error
        metrics = tf.concat([totals[:1], norm[None], totals[1:]], 0)
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
        totals = tf.zeros([4])
        function = self.gradient_function(width, targets, current, bank)
        # detector_batch_tbs is the number of TBs per gradient call at the widest bandwidth of the recipe; narrower
        # TBs need less memory, so a call takes proportionally more of them (the gradients are sums over the TBs).
        widest = max(self.trainer.recipe.get("prbs", [width]))
        batch = max(self.detector_batch_tbs, self.detector_batch_tbs*widest//width)
        for start in range(0, len(weights), batch):
            end = min(start+batch, len(weights))
            accumulated, totals = function(tuple(v[start:end] for v in targets),
                tuple(v[start:end] for v in current), bank, weights[start:end], tf.constant(den), accumulated, totals)
        gradients, metrics = self.finish(accumulated, totals)
        return gradients, metrics, fingerprints, local_den
