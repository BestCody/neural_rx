#!/usr/bin/env python3
"""UE-identity-aware memory routing for temporal Neural RX.

The neural receiver should learn *what* to remember, while an external memory
manager owns *whose* memory it is.  This module provides two implementations:

1. DifferentiableUEMemoryManager
   TensorFlow-only routing used during sequence training.  Memory is stored in a
   dense [batch, physical_ue_id, d_mem] table so gather/scatter operations remain
   inside the computation graph and future-TB losses can backpropagate through
   earlier writes even when a UE changes input position.

2. RuntimeUEMemoryManager
   A lightweight Python/NumPy manager for deployment-style integration where
   arbitrary stable UE identifiers (for example C-RNTIs) are mapped to reusable
   memory slots with expiration and geometric capacity growth.
"""

from __future__ import annotations

from typing import Hashable, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import tensorflow as tf


class TensorMemoryState(NamedTuple):
    """Differentiable memory-table state carried across a simulated sequence."""

    memory: tf.Tensor       # [batch, capacity, d_mem]
    valid: tf.Tensor        # [batch, capacity] bool
    last_seen: tf.Tensor    # [batch, capacity] int32, -1 means never seen


class DifferentiableUEMemoryManager:
    """Identity-keyed TensorFlow memory table for temporal training.

    Simulated physical UE IDs are dense integers in [0, capacity).  The memory
    tensor itself stays differentiable.  Identity, validity, and age are routing
    metadata and intentionally do not carry gradients.
    """

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
        """Create an empty memory table at a new independent-sequence boundary."""
        memory = tf.zeros([batch_size, self.capacity, self.d_mem], dtype=dtype)
        valid = tf.zeros([batch_size, self.capacity], dtype=tf.bool)
        last_seen = tf.fill([batch_size, self.capacity], tf.constant(-1, tf.int32))
        return TensorMemoryState(memory, valid, last_seen)

    def _checked_ids(self, ue_ids) -> tf.Tensor:
        ids = tf.cast(ue_ids, tf.int32)
        tf.debugging.assert_greater_equal(ids, 0, message="UE IDs must be >= 0")
        tf.debugging.assert_less(
            ids, self.capacity, message="UE ID exceeds memory-table capacity")
        return ids

    def _selection(self, ue_ids, active=None, dtype=tf.float32):
        """Return one-hot [B, scheduled, capacity] and reject active duplicates."""
        ids = self._checked_ids(ue_ids)
        one_hot = tf.one_hot(ids, self.capacity, dtype=dtype)

        if active is None:
            active_bool = tf.ones(tf.shape(ids), tf.bool)
        else:
            active_bool = tf.cast(active, tf.bool)

        selected = one_hot * tf.cast(active_bool[..., None], dtype)
        counts = tf.reduce_sum(selected, axis=1)
        tf.debugging.assert_less_equal(
            counts,
            tf.ones_like(counts),
            message="A physical UE may appear at most once in one scheduled TB",
        )
        return ids, one_hot, active_bool

    def expire(self, state: TensorMemoryState, slot_index) -> TensorMemoryState:
        """Expire memories whose last use is farther away than the configured TTL."""
        if self.expiry_slots is None:
            return state

        slot = tf.cast(slot_index, tf.int32)
        distance = slot - state.last_seen
        expired = tf.logical_and(
            state.valid,
            distance > tf.cast(self.expiry_slots, tf.int32),
        )
        memory = tf.where(
            expired[..., None], tf.zeros_like(state.memory), state.memory)
        valid = tf.logical_and(state.valid, tf.logical_not(expired))
        last_seen = tf.where(
            expired, tf.fill(tf.shape(state.last_seen), -1), state.last_seen)
        return TensorMemoryState(memory, valid, last_seen)

    def gather(self, state: TensorMemoryState, ue_ids, slot_index):
        """Gather each currently scheduled UE's own memory, validity, and age.

        Returns
        -------
        state : TensorMemoryState
            State after applying expiration for the current slot.
        memory : tf.Tensor
            [batch, scheduled, d_mem]. Invalid/new memories are returned as zero.
        gap_slots : tf.Tensor
            [batch, scheduled] int32. Consecutive transmissions have gap=1.
            New/invalid UEs have gap=0.
        valid : tf.Tensor
            [batch, scheduled] bool indicating whether historical memory exists.
        """
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
        """Write scheduled memories back under stable physical UE identity.

        Unscheduled rows are preserved exactly.  This operation uses differentiable
        one-hot routing rather than TensorArray/Python dictionaries, so gradients
        from a later gather can flow into an earlier memory writer.
        """
        ids, one_hot, active_bool = self._selection(
            ue_ids, active=active, dtype=updated_memory.dtype)

        updated_memory = tf.cast(updated_memory, state.memory.dtype)
        active_f = tf.cast(active_bool, updated_memory.dtype)
        route = one_hot * active_f[..., None]

        # Unique active IDs are asserted above, so this is a replacement, not an
        # ambiguous reduction. Shape: [B, capacity, d_mem].
        scattered = tf.einsum("bsc,bsd->bcd", route, updated_memory)
        selected = tf.reduce_any(
            tf.logical_and(
                tf.cast(one_hot, tf.bool),
                active_bool[..., None],
            ),
            axis=1,
        )

        memory = tf.where(selected[..., None], scattered, state.memory)
        valid = tf.logical_or(state.valid, selected)
        slot = tf.cast(slot_index, tf.int32)
        last_seen = tf.where(
            selected,
            tf.fill(tf.shape(state.last_seen), slot),
            state.last_seen,
        )
        return TensorMemoryState(memory, valid, last_seen)

    def remove(self, state: TensorMemoryState, ue_ids) -> TensorMemoryState:
        """Explicitly erase physical UEs and zero their rows before reuse."""
        ids = self._checked_ids(ue_ids)
        one_hot = tf.cast(
            tf.one_hot(ids, self.capacity, dtype=tf.float32), tf.bool)
        selected = tf.reduce_any(one_hot, axis=1)

        memory = tf.where(
            selected[..., None], tf.zeros_like(state.memory), state.memory)
        valid = tf.where(selected, tf.zeros_like(state.valid), state.valid)
        last_seen = tf.where(
            selected, tf.fill(tf.shape(state.last_seen), -1), state.last_seen)
        return TensorMemoryState(memory, valid, last_seen)


class RuntimeUEMemoryManager:
    """Deployment-style UE-ID -> memory-slot manager.

    The key can be any stable hashable identifier.  In a gNB integration the
    intended key is the scheduled UE's C-RNTI.  The neural network never infers
    identity from RF features; the scheduler supplies it.

    This manager is intentionally outside TensorFlow training.  It supports:
      * stable UE-ID -> slot ownership
      * brief absences without losing memory
      * expiration after a configurable scheduling gap
      * explicit UE removal
      * zero-before-reuse
      * free-slot reuse
      * geometric capacity growth
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
        self.expiry_slots = expiry_slots
        self.dtype = np.dtype(dtype)

        self.memory = np.zeros(
            (int(initial_capacity), self.d_mem), dtype=self.dtype)
        self.slot_to_ue = [None] * int(initial_capacity)
        self.ue_to_slot = {}
        self.last_seen = np.full(int(initial_capacity), -1, dtype=np.int64)
        self.free_slots = list(reversed(range(int(initial_capacity))))

    @property
    def capacity(self) -> int:
        return int(self.memory.shape[0])

    def _grow(self):
        old = self.capacity
        new = max(old * 2, old + 1)

        grown = np.zeros((new, self.d_mem), dtype=self.dtype)
        grown[:old] = self.memory
        self.memory = grown

        self.slot_to_ue.extend([None] * (new - old))
        self.last_seen = np.concatenate([
            self.last_seen,
            np.full(new - old, -1, dtype=np.int64),
        ])
        # reverse order so pop() returns the lowest newly available slot
        self.free_slots.extend(reversed(range(old, new)))

    def _allocate(self, ue_id: Hashable) -> int:
        if ue_id in self.ue_to_slot:
            return self.ue_to_slot[ue_id]
        if not self.free_slots:
            self._grow()

        slot = self.free_slots.pop()
        # Defensive zero-before-reuse.
        self.memory[slot].fill(0)
        self.last_seen[slot] = -1
        self.slot_to_ue[slot] = ue_id
        self.ue_to_slot[ue_id] = slot
        return slot

    def _release_slot(self, slot: int):
        ue_id = self.slot_to_ue[slot]
        if ue_id is not None:
            self.ue_to_slot.pop(ue_id, None)
        self.memory[slot].fill(0)
        self.last_seen[slot] = -1
        self.slot_to_ue[slot] = None
        if slot not in self.free_slots:
            self.free_slots.append(slot)

    def expire(self, slot_index: int):
        """Expire all UEs whose last seen slot is older than the TTL."""
        if self.expiry_slots is None:
            return

        now = int(slot_index)
        for slot, ue_id in enumerate(tuple(self.slot_to_ue)):
            if ue_id is None:
                continue
            seen = int(self.last_seen[slot])
            # A newly allocated-but-never-updated UE is kept until update/remove.
            if seen >= 0 and now - seen > int(self.expiry_slots):
                self._release_slot(slot)

    def lookup(self, ue_ids: Sequence[Hashable], slot_index: int):
        """Return memories/gaps for scheduled UEs, allocating new zero rows.

        Returns (memory, gap_slots, valid), where valid=False means this is a
        newly allocated UE with no historical memory.
        """
        self.expire(slot_index)
        now = int(slot_index)

        memories = []
        gaps = []
        valid = []
        for ue_id in ue_ids:
            existed = ue_id in self.ue_to_slot
            slot = self._allocate(ue_id)
            seen = int(self.last_seen[slot])

            memories.append(self.memory[slot].copy())
            valid.append(bool(existed and seen >= 0))
            gaps.append(now - seen if existed and seen >= 0 else 0)

        return (
            np.stack(memories, axis=0).astype(self.dtype, copy=False),
            np.asarray(gaps, dtype=np.int32),
            np.asarray(valid, dtype=np.bool_),
        )

    def update(
        self,
        ue_ids: Sequence[Hashable],
        updated_memory,
        slot_index: int,
        active: Optional[Sequence[bool]] = None,
    ):
        """Write network-produced memories back under the same UE IDs."""
        values = np.asarray(updated_memory, dtype=self.dtype)
        if values.shape != (len(ue_ids), self.d_mem):
            raise ValueError(
                f"updated_memory must have shape {(len(ue_ids), self.d_mem)}, "
                f"got {values.shape}"
            )

        if active is None:
            active = [True] * len(ue_ids)
        if len(active) != len(ue_ids):
            raise ValueError("active length must match ue_ids")

        if len(set(ue_ids)) != len(ue_ids):
            raise ValueError("A physical UE may appear at most once in one TB")

        now = int(slot_index)
        for ue_id, value, is_active in zip(ue_ids, values, active):
            if not is_active:
                continue
            slot = self._allocate(ue_id)
            self.memory[slot] = value
            self.last_seen[slot] = now

    def remove(self, ue_id: Hashable):
        """Immediately erase a UE, e.g. after explicit scheduler/RRC removal."""
        slot = self.ue_to_slot.get(ue_id)
        if slot is not None:
            self._release_slot(slot)

    def clear(self):
        """Reset all memory at an independent run/session boundary."""
        self.memory.fill(0)
        self.slot_to_ue = [None] * self.capacity
        self.ue_to_slot.clear()
        self.last_seen.fill(-1)
        self.free_slots = list(reversed(range(self.capacity)))

    def snapshot(self):
        """Small serializable diagnostic view for tests/logging."""
        return {
            "capacity": self.capacity,
            "ue_to_slot": dict(self.ue_to_slot),
            "slot_to_ue": list(self.slot_to_ue),
            "last_seen": self.last_seen.tolist(),
            "free_slots": list(self.free_slots),
        }
