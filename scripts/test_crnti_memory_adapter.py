#!/usr/bin/env python3
"""Correctness tests for the runtime C-RNTI -> temporal-memory bridge."""

import json

import numpy as np

from crnti_memory_adapter import CRNTIMemoryAdapter


def main():
    d_mem = 4
    bridge = CRNTIMemoryAdapter(
        d_mem=d_mem,
        initial_capacity=2,
        expiry_slots=2,
    )

    a = 0x4601
    b = 0x4602
    c = 0x4603

    # Slot 0: A/B are new, so the NRX must receive invalid zero memory.
    first = bridge.lookup([a, b], 0)
    new_zero = bool(
        first.crntis == (a, b)
        and not np.any(first.valid)
        and np.allclose(first.memory, 0.0)
        and np.all(first.gap_slots == 0)
    )
    bridge.process_result(
        first,
        np.stack([
            np.ones(d_mem, np.float32),
            np.ones(d_mem, np.float32) * 2.0,
        ]),
        0,
    )

    # Slot 1: scheduler order is B/C. B must follow its C-RNTI from position 1
    # to position 0; C must not inherit A's old row.
    second = bridge.lookup([b, c], 1)
    identity_routing = bool(
        second.crntis == (b, c)
        and second.valid.tolist() == [True, False]
        and second.gap_slots.tolist() == [1, 0]
        and np.allclose(second.memory[0], 2.0)
        and np.allclose(second.memory[1], 0.0)
    )
    bridge.process_result(
        second,
        np.stack([
            np.ones(d_mem, np.float32) * 20.0,
            np.ones(d_mem, np.float32) * 30.0,
        ]),
        1,
    )

    # Slot 2: A was unscheduled for one slot and must retain its own memory.
    third = bridge.lookup([a, b], 2)
    absence_persistence = bool(
        third.valid.tolist() == [True, True]
        and third.gap_slots.tolist() == [2, 1]
        and np.allclose(third.memory[0], 1.0)
        and np.allclose(third.memory[1], 20.0)
    )

    # Explicit RRC/scheduler release immediately destroys B's state.
    bridge.release(b)
    b_after_release = bridge.lookup([b], 3)
    release_zeroes = bool(
        b_after_release.valid.tolist() == [False]
        and np.allclose(b_after_release.memory, 0.0)
    )

    # C last wrote in slot 1. At slot 4 its age is 3 > expiry_slots=2, so it
    # must be expired and reallocated as a new zero state.
    c_after_expiry = bridge.lookup([c], 4)
    expiry_zeroes = bool(
        c_after_expiry.valid.tolist() == [False]
        and np.allclose(c_after_expiry.memory, 0.0)
    )

    # Position-order mismatch is prevented by process_result using immutable
    # lookup keys. The snapshot should contain canonical C-RNTI diagnostics.
    snapshot = bridge.snapshot()
    crnti_diagnostics = bool(
        all(key.startswith("0x") for key in snapshot["crnti_to_slot_hex"])
    )

    duplicate_guard = False
    try:
        bridge.lookup([a, a], 5)
    except ValueError:
        duplicate_guard = True

    invalid_guard = False
    try:
        bridge.lookup([0], 5)
    except ValueError:
        invalid_guard = True

    report = {
        "new_ue_gets_zero_invalid_memory": new_zero,
        "identity_follows_crnti_across_positions": identity_routing,
        "brief_absence_preserves_memory": absence_persistence,
        "explicit_release_zeroes_memory": release_zeroes,
        "expiry_zeroes_memory": expiry_zeroes,
        "crnti_diagnostics": crnti_diagnostics,
        "duplicate_crnti_guard": duplicate_guard,
        "invalid_crnti_guard": invalid_guard,
    }
    report["passed"] = bool(all(report.values()))
    print("CRNTI_MEMORY_ADAPTER_TEST=" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
