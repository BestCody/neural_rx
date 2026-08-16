#!/usr/bin/env python3
"""Correctness tests for UE-identity-aware temporal memory routing."""

import json

import numpy as np
import tensorflow as tf

from ue_memory_manager import (
    DifferentiableUEMemoryManager,
    RuntimeUEMemoryManager,
)


def test_tensor_manager():
    d_mem = 4
    manager = DifferentiableUEMemoryManager(
        capacity=4, d_mem=d_mem, expiry_slots=3)
    state = manager.zero_state(1)

    ids0 = tf.constant([[0, 1]], tf.int32)
    updates0 = tf.constant(
        [[[1.0] * d_mem, [2.0] * d_mem]], tf.float32)
    state = manager.scatter(
        state, ids0, updates0, tf.ones([1, 2]), slot_index=0)

    # B moved positions, C is new.
    ids1 = tf.constant([[1, 2]], tf.int32)
    state, memory1, gap1, valid1 = manager.gather(state, ids1, 1)
    assert np.allclose(memory1.numpy()[0, 0], 2.0)
    assert np.allclose(memory1.numpy()[0, 1], 0.0)
    assert valid1.numpy().tolist() == [[True, False]]
    assert gap1.numpy().tolist() == [[1, 0]]

    updates1 = tf.constant(
        [[[20.0] * d_mem, [30.0] * d_mem]], tf.float32)
    state = manager.scatter(
        state, ids1, updates1, tf.ones([1, 2]), slot_index=1)

    # A was unscheduled at TB2 but must retain its original memory.
    ids2 = tf.constant([[0, 1]], tf.int32)
    _, memory2, gap2, valid2 = manager.gather(state, ids2, 2)
    assert np.allclose(memory2.numpy()[0, 0], 1.0)
    assert np.allclose(memory2.numpy()[0, 1], 20.0)
    assert valid2.numpy().tolist() == [[True, True]]
    assert gap2.numpy().tolist() == [[2, 1]]

    # Expiration must zero before reuse.
    exp = DifferentiableUEMemoryManager(
        capacity=2, d_mem=d_mem, expiry_slots=1)
    exp_state = exp.zero_state(1)
    exp_state = exp.scatter(
        exp_state,
        tf.constant([[0, 1]], tf.int32),
        updates0,
        tf.ones([1, 2]),
        slot_index=0,
    )
    _, memory_exp, _, valid_exp = exp.gather(
        exp_state, tf.constant([[0, 1]], tf.int32), slot_index=2)
    assert not np.any(valid_exp.numpy())
    assert np.allclose(memory_exp.numpy(), 0.0)

    return {
        "position_change_routes_by_id": True,
        "unscheduled_memory_persists": True,
        "expiration_zeroes": True,
    }


def test_runtime_manager():
    manager = RuntimeUEMemoryManager(
        d_mem=3, initial_capacity=2, expiry_slots=2)

    # C-RNTI-like arbitrary stable keys.
    mem, gap, valid = manager.lookup([0x4601, 0x4602], slot_index=10)
    assert np.allclose(mem, 0.0)
    assert gap.tolist() == [0, 0]
    assert valid.tolist() == [False, False]

    manager.update(
        [0x4601, 0x4602],
        np.asarray([[1, 1, 1], [2, 2, 2]], np.float32),
        slot_index=10,
    )

    # Same UE changes scheduled position; lookup order must not matter.
    mem, gap, valid = manager.lookup([0x4602, 0x4601], slot_index=11)
    assert np.allclose(mem[0], 2.0)
    assert np.allclose(mem[1], 1.0)
    assert gap.tolist() == [1, 1]
    assert valid.tolist() == [True, True]

    # Add a third UE to force geometric growth.
    manager.lookup([0x4603], slot_index=11)
    assert manager.capacity >= 4
    manager.update(
        [0x4603], np.asarray([[3, 3, 3]], np.float32), slot_index=11)

    # Explicit removal zeroes and frees a slot.
    old_slot = manager.ue_to_slot[0x4602]
    manager.remove(0x4602)
    assert 0x4602 not in manager.ue_to_slot
    assert manager.slot_to_ue[old_slot] is None
    assert np.allclose(manager.memory[old_slot], 0.0)

    # A new UE should reuse a free zeroed row.
    mem, _, valid = manager.lookup([0x4700], slot_index=12)
    assert not valid[0]
    assert np.allclose(mem[0], 0.0)

    # UE 0x4601 was last seen at 10 and expires when queried beyond TTL.
    manager.expire(slot_index=13)
    assert 0x4601 not in manager.ue_to_slot

    return {
        "arbitrary_stable_ids": True,
        "position_independent_lookup": True,
        "geometric_growth": True,
        "explicit_remove_zeroes": True,
        "free_slot_reuse": True,
        "expiration": True,
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
