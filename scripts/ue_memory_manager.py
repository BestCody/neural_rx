#!/usr/bin/env python3
"""UE-identity-aware memory routing for temporal Neural RX.

The neural receiver learns what to remember; this module owns whose memory it
is. Training uses a differentiable dense TensorFlow table keyed by simulated UE
IDs. Runtime uses a NumPy slot table keyed by arbitrary stable IDs such as
C-RNTIs.
"""

from __future__ import annotations

from typing import Hashable, NamedTuple, Optional, Sequence

import numpy as np
import tensorflow as tf


class TensorMemoryState(NamedTuple):
    memory: tf.Tensor       # [batch, capacity, d_mem]
    valid: tf.Tensor        # [batch, capacity] bool
    last_seen: tf.Tensor    # [batch, capacity] int32; -1 means never seen


class DifferentiableUEMemoryManager:
    """Identity-keyed differentiable memory table used during sequence training."""

    def __init__(self, capacity: int, d_mem: int, expiry_slots: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if d_mem <= 0:
            raise ValueError("d_mem must be positive")
        if expiry_slots is not None and expiry_slots < 1:
            raise ValueError("expiry_slots must be >= 1 or None")
        self.capacity = int(capacity)
        self.d_mem = int(d_mem)
        self.expiry_slots = None if expiry_slots is None else int(expiry_slots)

    def zero_state(self, batch_size, dtype=tf.float32) -> TensorMemoryState:
        return TensorMemoryState(
            tf.zeros([batch_size, self.capacity, self.d_mem], dtype=dtype),
            tf.zeros([batch_size, self.capacity], dtype=tf.bool),
            tf.fill([batch_size, self.capacity], tf.constant(-1, tf.int32)),
        )

    def _checked_ids(self, ue_ids) -> tf.Tensor:
        ids = tf.cast(ue_ids, tf.int32)
        tf.debugging.assert_greater_equal(ids, 0, message="UE IDs must be >= 0")
        tf.debugging.assert_less(
            ids, self.capacity, message="UE ID exceeds memory-table capacity")
        return ids

    def _selection(self, ue_ids, active=None, dtype=tf.float32):
        ids = self._checked_ids(ue_ids)
        one_hot = tf.one_hot(ids, self.capacity, dtype=dtype)
        active_bool = (
            tf.ones(tf.shape(ids), tf.bool)
            if active is None
            else tf.cast(active, tf.bool)
        )
        if active_bool.shape.rank != ids.shape.rank:
            raise ValueError("active must have the same rank as ue_ids")
        selected = one_hot * tf.cast(active_bool[..., None], dtype)
        counts = tf.reduce_sum(selected, axis=1)
        tf.debugging.assert_less_equal(
            counts,
            tf.ones_like(counts),
            message="A physical UE may appear at most once in one scheduled TB",
        )
        return ids, one_hot, active_bool

    def expire(self, state: TensorMemoryState, slot_index) -> TensorMemoryState:
        if self.expiry_slots is None:
            return state
        slot = tf.cast(slot_index, tf.int32)
        distance = slot - state.last_seen
        expired = tf.logical_and(
            state.valid,
            distance > tf.cast(self.expiry_slots, tf.int32),
        )
        return TensorMemoryState(
            tf.where(expired[..., None], tf.zeros_like(state.memory), state.memory),
            tf.logical_and(state.valid, tf.logical_not(expired)),
            tf.where(
                expired,
                tf.fill(tf.shape(state.last_seen), tf.constant(-1, tf.int32)),
                state.last_seen,
            ),
        )

    def gather(self, state: TensorMemoryState, ue_ids, slot_index):
        """Return each scheduled UE's memory, age and validity after expiration."""
        state = self.expire(state, slot_index)
        ids = self._checked_ids(ue_ids)
        memory = tf.gather(state.memory, ids, axis=1, batch_dims=1)
        valid = tf.gather(state.valid, ids, axis=1, batch_dims=1)
        last_seen = tf.gather(state.last_seen, ids, axis=1, batch_dims=1)
        slot = tf.cast(slot_index, tf.int32)
        raw_gap = tf.maximum(slot - last_seen, 0)
        gap = tf.where(valid, raw_gap, tf.zeros_like(raw_gap))
        memory = tf.where(valid[..., None], memory, tf.zeros_like(memory))
        return state, memory, gap, valid

    def scatter(
        self,
        state: TensorMemoryState,
        ue_ids,
        updated_memory,
        active,
        slot_index,
    ) -> TensorMemoryState:
        """Write active scheduled memories back under stable UE identity."""
        updated_memory = tf.cast(updated_memory, state.memory.dtype)
        ids, one_hot, active_bool = self._selection(
            ue_ids, active=active, dtype=updated_memory.dtype)
        del ids
        tf.debugging.assert_equal(
            tf.shape(updated_memory)[:2],
            tf.shape(active_bool),
            message="updated_memory leading dimensions must match ue_ids",
        )
        tf.debugging.assert_equal(
            tf.shape(updated_memory)[-1],
            self.d_mem,
            message="updated_memory last dimension must equal d_mem",
        )

        route = one_hot * tf.cast(active_bool[..., None], updated_memory.dtype)
        scattered = tf.einsum("bsc,bsd->bcd", route, updated_memory)
        selected = tf.reduce_any(
            tf.logical_and(tf.cast(one_hot, tf.bool), active_bool[..., None]),
            axis=1,
        )
        slot = tf.cast(slot_index, tf.int32)
        return TensorMemoryState(
            tf.where(selected[..., None], scattered, state.memory),
            tf.logical_or(state.valid, selected),
            tf.where(
                selected,
                tf.fill(tf.shape(state.last_seen), slot),
                state.last_seen,
            ),
        )

    def remove(self, state: TensorMemoryState, ue_ids) -> TensorMemoryState:
        ids = self._checked_ids(ue_ids)
        one_hot = tf.cast(tf.one_hot(ids, self.capacity), tf.bool)
        selected = tf.reduce_any(one_hot, axis=1)
        return TensorMemoryState(
            tf.where(selected[..., None], tf.zeros_like(state.memory), state.memory),
            tf.where(selected, tf.zeros_like(state.valid), state.valid),
            tf.where(
                selected,
                tf.fill(tf.shape(state.last_seen), tf.constant(-1, tf.int32)),
                state.last_seen,
            ),
        )


class RuntimeUEMemoryManager:
    """Deployment UE-ID -> memory-slot manager.

    Runtime lookup is deliberately transactional: looking up an unseen UE returns
    zero/invalid memory but does *not* allocate a slot. Allocation happens only
    when an active inference result is successfully committed through ``update``.
    This prevents failed inference and inactive new UEs from leaking empty slots.
    """

    def __init__(
        self,
        d_mem: int,
        initial_capacity: int = 16,
        expiry_slots: Optional[int] = 64,
        dtype=np.float32,
    ):
        if d_mem <= 0:
            raise ValueError("d_mem must be positive")
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if expiry_slots is not None and expiry_slots < 1:
            raise ValueError("expiry_slots must be >= 1 or None")
        self.d_mem = int(d_mem)
        self.expiry_slots = None if expiry_slots is None else int(expiry_slots)
        self.dtype = np.dtype(dtype)
        self.memory = np.zeros((int(initial_capacity), self.d_mem), dtype=self.dtype)
        self.slot_to_ue = [None] * int(initial_capacity)
        self.ue_to_slot: dict[Hashable, int] = {}
        self.last_seen = np.full(int(initial_capacity), -1, dtype=np.int64)
        self.free_slots = list(reversed(range(int(initial_capacity))))

    @property
    def capacity(self) -> int:
        return int(self.memory.shape[0])

    def _grow(self) -> None:
        old = self.capacity
        new = max(old * 2, old + 1)
        grown = np.zeros((new, self.d_mem), dtype=self.dtype)
        grown[:old] = self.memory
        self.memory = grown
        self.slot_to_ue.extend([None] * (new - old))
        self.last_seen = np.concatenate(
            [self.last_seen, np.full(new - old, -1, dtype=np.int64)]
        )
        self.free_slots.extend(reversed(range(old, new)))

    def _allocate(self, ue_id: Hashable) -> int:
        if ue_id in self.ue_to_slot:
            return self.ue_to_slot[ue_id]
        if not self.free_slots:
            self._grow()
        slot = self.free_slots.pop()
        self.memory[slot].fill(0)
        self.last_seen[slot] = -1
        self.slot_to_ue[slot] = ue_id
        self.ue_to_slot[ue_id] = slot
        return slot

    def _release_slot(self, slot: int) -> None:
        ue_id = self.slot_to_ue[slot]
        if ue_id is not None:
            self.ue_to_slot.pop(ue_id, None)
        self.memory[slot].fill(0)
        self.last_seen[slot] = -1
        self.slot_to_ue[slot] = None
        if slot not in self.free_slots:
            self.free_slots.append(slot)

    def expire(self, slot_index: int) -> None:
        if self.expiry_slots is None:
            return
        now = int(slot_index)
        for slot, ue_id in enumerate(tuple(self.slot_to_ue)):
            if ue_id is None:
                continue
            seen = int(self.last_seen[slot])
            if seen >= 0 and now - seen > self.expiry_slots:
                self._release_slot(slot)

    @staticmethod
    def _check_unique(ue_ids: Sequence[Hashable]) -> None:
        try:
            unique = len(set(ue_ids)) == len(ue_ids)
        except TypeError as exc:
            raise TypeError("UE IDs must be hashable") from exc
        if not unique:
            raise ValueError("A physical UE may appear at most once in one TB")

    def lookup(self, ue_ids: Sequence[Hashable], slot_index: int):
        """Read previous state without allocating slots for unseen UEs."""
        keys = list(ue_ids)
        self._check_unique(keys)
        self.expire(slot_index)
        now = int(slot_index)
        memories = []
        gaps = []
        valid = []
        zero = np.zeros(self.d_mem, dtype=self.dtype)

        for ue_id in keys:
            slot = self.ue_to_slot.get(ue_id)
            if slot is None:
                memories.append(zero.copy())
                gaps.append(0)
                valid.append(False)
                continue
            seen = int(self.last_seen[slot])
            is_valid = seen >= 0
            memories.append(self.memory[slot].copy() if is_valid else zero.copy())
            gaps.append(max(now - seen, 0) if is_valid else 0)
            valid.append(is_valid)

        if memories:
            memory_out = np.stack(memories, axis=0).astype(self.dtype, copy=False)
        else:
            memory_out = np.zeros((0, self.d_mem), dtype=self.dtype)
        return (
            memory_out,
            np.asarray(gaps, dtype=np.int32),
            np.asarray(valid, dtype=np.bool_),
        )

    def update(
        self,
        ue_ids: Sequence[Hashable],
        updated_memory,
        slot_index: int,
        active: Optional[Sequence[bool]] = None,
    ) -> None:
        """Commit active inference results; this is the allocation point."""
        keys = list(ue_ids)
        self._check_unique(keys)
        values = np.asarray(updated_memory, dtype=self.dtype)
        expected = (len(keys), self.d_mem)
        if values.shape != expected:
            raise ValueError(
                f"updated_memory must have shape {expected}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("updated_memory contains NaN/Inf")

        if active is None:
            active_arr = np.ones(len(keys), dtype=np.bool_)
        else:
            active_arr = np.asarray(active, dtype=np.bool_)
            if active_arr.shape != (len(keys),):
                raise ValueError("active length must match ue_ids")

        now = int(slot_index)
        for ue_id, value, is_active in zip(keys, values, active_arr):
            if not bool(is_active):
                continue
            slot = self._allocate(ue_id)
            self.memory[slot] = value
            self.last_seen[slot] = now

    def remove(self, ue_id: Hashable) -> None:
        slot = self.ue_to_slot.get(ue_id)
        if slot is not None:
            self._release_slot(slot)

    def clear(self) -> None:
        self.memory.fill(0)
        self.slot_to_ue = [None] * self.capacity
        self.ue_to_slot.clear()
        self.last_seen.fill(-1)
        self.free_slots = list(reversed(range(self.capacity)))

    def snapshot(self):
        return {
            "capacity": self.capacity,
            "ue_to_slot": dict(self.ue_to_slot),
            "slot_to_ue": list(self.slot_to_ue),
            "last_seen": self.last_seen.tolist(),
            "free_slots": list(self.free_slots),
        }
