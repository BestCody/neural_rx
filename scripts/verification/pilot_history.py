from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from pilot_history_receiver.pilot_history import PilotHistory
from temporal_receiver_core import prepare_cgnn_inputs


class ReferenceState(NamedTuple):
    observations: tf.Tensor
    timestamps: tf.Tensor
    ports: tf.Tensor
    valid: tf.Tensor


class ReferenceMemoryManager:
    def __init__(self, policy, observations, components):
        self.policy = policy
        self.observations, self.components = observations, components

    def zero_state(self, batch_size):
        shape = [batch_size, self.policy.capacity, self.policy.history_tbs]
        return ReferenceState(tf.zeros([*shape, self.observations, self.components]),
            tf.fill(shape, -1), tf.zeros(shape, tf.int32), tf.zeros(shape, tf.bool))

    def gather(self, state, ids, slot, active):
        observation, timestamp, ports, valid = [tf.gather(v, ids, axis=1, batch_dims=1) for v in state]
        age = tf.cast(slot, tf.int32)-timestamp
        valid &= (age > 0) & tf.cast(active, tf.bool)[..., None]
        if self.policy.expiry_slots is not None:
            valid &= age <= self.policy.expiry_slots
        return PilotHistory(tf.where(valid[..., None, None], observation, 0.),
                            tf.where(valid, age, 0), tf.where(valid, ports, 0), valid)

    def scatter(self, state, ids, observation, active, slot):
        selected = tf.cast(tf.where(tf.cast(active, tf.bool)), tf.int32)
        address = tf.stack([selected[:, 0], tf.gather_nd(ids, selected)], -1)
        old = [tf.gather_nd(v, address) for v in state]
        values = [tf.gather_nd(observation, selected), tf.fill([tf.shape(selected)[0]], tf.cast(slot, tf.int32)),
                  selected[:, 1], tf.ones([tf.shape(selected)[0]], tf.bool)]
        return ReferenceState(*(tf.tensor_scatter_nd_update(table, address,
            tf.concat([new[:, None], previous[:, :-1]], 1))
            for table, new, previous in zip(state, values, old)))


def make_decode_step(receiver, model, memory_manager, decoder):
    from receiver_decoding import OneLayerGridCodeMap
    mapping = OneLayerGridCodeMap(receiver._neural_rx, receiver._tb_encoders[0].num_tx,
                                 model.base._num_bits_per_symbol[0])
    @tf.function(reduce_retracing=True)
    def decode(y, channel, noise, active, ue_ids, slot, state):
        history = memory_manager.gather(state, ue_ids, slot, active)
        inputs = prepare_cgnn_inputs(receiver, y, channel, active)
        output, _, _, _, observation, _ = model.decode(inputs, noise, ue_ids, history, decoder, mapping)
        state = memory_manager.scatter(state, ue_ids, observation, active, slot)
        return output.hard_bits, output.crc_status, state
    return decode
