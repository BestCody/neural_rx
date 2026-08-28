#!/usr/bin/env python3
"""Correctness tests for UE-identity-aware temporal memory routing."""

import json

import numpy as np
import tensorflow as tf

from ue_memory_manager import DifferentiableUEMemoryManager, RuntimeUEMemoryManager


def test_tensor_manager():
    d_mem = 4
    manager = DifferentiableUEMemoryManager(capacity=4, d_mem=d_mem, expiry_slots=3)
    state = manager.zero_state(1)

    ids0 = tf.constant([[0, 1]], tf.int32)
    updates0 = tf.constant([[[1.0] * d_mem, [2.0] * d_mem]], tf.float32)
    state = manager.scatter(state, ids0, updates0, tf.ones([1, 2]), slot_index=0)

    ids1 = tf.constant([[1, 2]], tf.int32)
    state, memory1, gap1, valid1 = manager.gather(state, ids1, 1)
    assert np.allclose(memory1.numpy()[0, 0], 2.0)
    assert np.allclose(memory1.numpy()[0, 1], 0.0)
    assert valid1.numpy().tolist() == [[True, False]]
    assert gap1.numpy().tolist() == [[1, 0]]

    updates1 = tf.constant([[[20.0] * d_mem, [30.0] * d_mem]], tf.float32)
    state = manager.scatter(state, ids1, updates1, tf.ones([1, 2]), slot_index=1)

    ids2 = tf.constant([[0, 1]], tf.int32)
    _, memory2, gap2, valid2 = manager.gather(state, ids2, 2)
    assert np.allclose(memory2.numpy()[0, 0], 1.0)
    assert np.allclose(memory2.numpy()[0, 1], 20.0)
    assert valid2.numpy().tolist() == [[True, True]]
    assert gap2.numpy().tolist() == [[2, 1]]

    exp = DifferentiableUEMemoryManager(capacity=2, d_mem=d_mem, expiry_slots=1)
    exp_state = exp.zero_state(1)
    exp_state = exp.scatter(
        exp_state,
        tf.constant([[0, 1]], tf.int32),
        updates0,
        tf.ones([1, 2]),
        slot_index=0,
    )
    _, memory_exp, _, valid_exp = exp.gather(
        exp_state, tf.constant([[0, 1]], tf.int32), slot_index=2
    )
    assert not np.any(valid_exp.numpy())
    assert np.allclose(memory_exp.numpy(), 0.0)

    return {
        "position_change_routes_by_id": True,
        "unscheduled_memory_persists": True,
        "expiration_zeroes": True,
    }


def test_runtime_manager():
    manager = RuntimeUEMemoryManager(d_mem=3, initial_capacity=2, expiry_slots=2)

    # New lookups are read-only: zero/invalid state is returned without owning a slot.
    mem, gap, valid = manager.lookup([0x4601, 0x4602], slot_index=10)
    assert np.allclose(mem, 0.0)
    assert gap.tolist() == [0, 0]
    assert valid.tolist() == [False, False]
    assert manager.ue_to_slot == {}

    manager.update(
        [0x4601, 0x4602],
        np.asarray([[1, 1, 1], [2, 2, 2]], np.float32),
        slot_index=10,
    )

    mem, gap, valid = manager.lookup([0x4602, 0x4601], slot_index=11)
    assert np.allclose(mem[0], 2.0)
    assert np.allclose(mem[1], 1.0)
    assert gap.tolist() == [1, 1]
    assert valid.tolist() == [True, True]

    # A successful update of a third UE is what allocates and forces growth.
    manager.update(
        [0x4603], np.asarray([[3, 3, 3]], np.float32), slot_index=11
    )
    assert manager.capacity >= 4

    old_slot = manager.ue_to_slot[0x4602]
    manager.remove(0x4602)
    assert 0x4602 not in manager.ue_to_slot
    assert manager.slot_to_ue[old_slot] is None
    assert np.allclose(manager.memory[old_slot], 0.0)

    # Merely looking up a replacement UE still must not allocate or consume the slot.
    before = manager.snapshot()
    mem, _, valid = manager.lookup([0x4700], slot_index=12)
    assert not valid[0]
    assert np.allclose(mem[0], 0.0)
    assert 0x4700 not in manager.ue_to_slot
    assert manager.snapshot()["free_slots"] == before["free_slots"]

    # An inactive update also must not allocate a previously unseen UE.
    manager.update(
        [0x4700],
        np.asarray([[9, 9, 9]], np.float32),
        slot_index=12,
        active=[False],
    )
    assert 0x4700 not in manager.ue_to_slot

    # A later active commit allocates it and reuses a zeroed free row.
    manager.update(
        [0x4700], np.asarray([[4, 4, 4]], np.float32), slot_index=12
    )
    slot_4700 = manager.ue_to_slot[0x4700]
    assert np.allclose(manager.memory[slot_4700], 4.0)

    # UE 0x4601 was last written at 10 and expires beyond TTL=2.
    manager.expire(slot_index=13)
    assert 0x4601 not in manager.ue_to_slot

    duplicate_guard = False
    try:
        manager.lookup([0x4800, 0x4800], slot_index=14)
    except ValueError:
        duplicate_guard = True
    assert duplicate_guard

    finite_guard = False
    try:
        manager.update(
            [0x4800], np.asarray([[np.nan, 0, 0]], np.float32), slot_index=14
        )
    except ValueError:
        finite_guard = True
    assert finite_guard
    assert 0x4800 not in manager.ue_to_slot

    return {
        "new_lookup_is_non_mutating": True,
        "position_independent_lookup": True,
        "allocation_on_successful_commit": True,
        "geometric_growth": True,
        "explicit_remove_zeroes": True,
        "inactive_new_ue_does_not_allocate": True,
        "free_slot_reuse": True,
        "expiration": True,
        "duplicate_guard": True,
        "finite_memory_guard": True,
    }


def main():
    result = {
        "tensor_manager": test_tensor_manager(),
        "runtime_manager": test_runtime_manager(),
    }
    result["passed"] = True
    print("UE_MEMORY_MANAGER_TEST=" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
