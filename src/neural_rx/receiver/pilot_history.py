from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf


class PilotHistory(NamedTuple):
    observations: tf.Tensor
    age: tf.Tensor
    ports: tf.Tensor
    valid: tf.Tensor


class RingState(NamedTuple):
    observations: tf.Tensor
    timestamps: tf.Tensor
    ports: tf.Tensor
    head: tf.Tensor
    count: tf.Tensor

    @property
    def valid(self):
        return self.count > 0


class PendingWrite(NamedTuple):
    observation: tf.Tensor
    head: tf.Tensor
    owners: tf.Tensor
    written: tf.Tensor
    next_head: tf.Tensor
    next_count: tf.Tensor


class MemoryPolicy:
    def __init__(self, capacity, history_tbs, expiry_slots):
        for name, value in (("capacity", capacity), ("history_tbs", history_tbs)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if expiry_slots is not None and (type(expiry_slots) is not int or expiry_slots < 1):
            raise ValueError("expiry_slots must be positive or None")
        self.capacity, self.history_tbs, self.expiry_slots = capacity, history_tbs, expiry_slots


def validate_memory_metadata(ids, active, capacity):
    ids, active = np.asarray(ids), np.asarray(active)
    if ids.ndim != 2 or active.shape != ids.shape:
        raise ValueError("UE IDs and activity must be matching batch-by-user matrices")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("UE IDs must be integers")
    if np.any(ids < 0) or np.any(ids >= capacity):
        raise ValueError("UE ID exceeds memory-table capacity")
    if not np.all((active == 0) | (active == 1)):
        raise ValueError("Activity must be finite and binary")
    for row, mask in zip(ids, active.astype(bool, copy=False)):
        if len(np.unique(row[mask])) != len(row[mask]):
            raise ValueError("A physical UE may appear at most once in one scheduled TB")


class PilotMemoryBuffer:
    def __init__(self, policy, batch_size, observations, components):
        from .native_ops import load
        self.native = load()
        self.policy = policy
        self.batch_size = batch_size
        self.slots = policy.history_tbs+1
        shape = [batch_size, policy.capacity, self.slots]
        self.observations = tf.Variable(tf.zeros([*shape, observations, components]), trainable=False)
        with tf.device(self.observations.device):
            self.timestamps = tf.Variable(tf.fill(shape, tf.constant(-1, tf.int64)), trainable=False)
            self.ports = tf.Variable(tf.zeros(shape, tf.int64), trainable=False)

    def empty_metadata(self):
        shape = [self.batch_size, self.policy.capacity]
        with tf.device(self.observations.device):
            return tf.zeros(shape, tf.int64), tf.zeros(shape, tf.int64)

    def read(self, head, count, ids, active, step):
        return self.gather((self.observations, self.timestamps, self.ports), head, count, ids, active, step)

    def gather(self, tables, head, count, ids, active, step):
        ids, step = tf.cast(ids, tf.int64), tf.cast(step, tf.int64)
        next_slot = tf.gather(head, ids, axis=1, batch_dims=1)
        available = tf.gather(count, ids, axis=1, batch_dims=1)
        offsets = tf.range(1, self.policy.history_tbs+1, dtype=tf.int64)
        slot = tf.math.floormod(next_slot[..., None]-offsets, self.slots)
        ue = tf.broadcast_to(ids[..., None], tf.shape(slot))
        batch = tf.broadcast_to(tf.range(self.batch_size, dtype=tf.int64)[:, None, None], tf.shape(slot))
        address = tf.stack([batch, ue, slot], -1)
        observed, timestamp, ports = [tf.gather_nd(table, address) for table in tables]
        age = step-timestamp
        valid = (offsets <= available[..., None]) & (age > 0) & tf.cast(active, tf.bool)[..., None]
        if self.policy.expiry_slots is not None:
            valid &= age <= self.policy.expiry_slots
        return PilotHistory(tf.where(valid[..., None, None], observed, tf.zeros_like(observed)),
                            tf.cast(tf.where(valid, age, 0), tf.int32),
                            tf.cast(tf.where(valid, ports, 0), tf.int32), valid)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def prepare_stage(self, observation, head, count, ids, active):
        matches = ((tf.cast(ids[..., None], tf.int64) == tf.range(self.policy.capacity, dtype=tf.int64)[None, None])
                   & tf.cast(active[..., None], tf.bool))
        written = tf.reduce_any(matches, 1)
        owners = tf.argmax(tf.cast(matches, tf.int32), 1, output_type=tf.int64)
        values = tf.gather(observation, owners, axis=1, batch_dims=1)
        finite = tf.reduce_all(~written[..., None, None] | tf.math.is_finite(values))
        finite &= tf.reduce_all((head >= 0) & (head < self.slots) & (count >= 0) & (count <= self.policy.history_tbs))
        next_head = tf.where(written, (head+1) % self.slots, head)
        next_count = tf.where(written, tf.minimum(count+1, self.policy.history_tbs), count)
        return PendingWrite(observation, head, owners, written, next_head, next_count), finite

    def commit(self, pending, finite, step):
        checked = tf.debugging.assert_equal(finite, True, message="Pilot observation or memory metadata is invalid")
        with tf.control_dependencies([checked]):
            complete = self.native.pilot_memory_append(
                self.observations.handle, self.timestamps.handle, self.ports.handle,
                pending.observation, pending.head, pending.owners, pending.written, tf.cast(step, tf.int64))
        return pending.next_head, pending.next_count, complete

    def stage(self, observation, head, count, ids, active, step):
        pending, finite = self.prepare_stage(observation, head, count, ids, active)
        return self.commit(pending, finite, step)

    def snapshot(self, head, count):
        return RingState(self.observations.read_value(), self.timestamps.read_value(),
                         self.ports.read_value(), head, count)
