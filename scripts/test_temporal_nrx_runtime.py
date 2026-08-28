#!/usr/bin/env python3
"""Pure-NumPy correctness test for the live temporal NRX runtime bridge."""

import json

import numpy as np

from temporal_nrx_runtime import TemporalInferenceOutput, TemporalNRXRuntime


class FakeTemporalInference:
    """Deterministic stand-in for TemporalUEMemoryCGNN."""

    def __init__(self, d_mem):
        self.d_mem = int(d_mem)
        self.calls = []
        self.fail_next = False

    def __call__(
        self,
        *,
        received_grid,
        ls_estimate,
        active_tx,
        prev_memory,
        memory_gap,
        memory_valid,
        **_,
    ):
        del ls_estimate
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic inference failure")

        signal = np.asarray(received_grid, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        if signal.shape != active_tx.shape:
            raise ValueError("fake signal shape must match active_tx")

        self.calls.append(
            {
                "prev_memory": np.asarray(prev_memory).copy(),
                "memory_gap": np.asarray(memory_gap).copy(),
                "memory_valid": np.asarray(memory_valid).copy(),
                "active_tx": np.asarray(active_tx).copy(),
            }
        )
        write = signal[..., None] * np.ones(
            (1, signal.shape[1], self.d_mem), dtype=np.float32
        )
        next_memory = np.asarray(prev_memory, dtype=np.float32) + write
        next_memory = np.where(
            np.asarray(active_tx, dtype=bool)[..., None],
            next_memory,
            np.asarray(prev_memory, dtype=np.float32),
        )
        return TemporalInferenceOutput(
            receiver_output=signal.copy(),
            next_memory=next_memory,
            diagnostics={"fake": True},
        )


def main():
    d_mem = 4
    infer = FakeTemporalInference(d_mem)
    runtime = TemporalNRXRuntime(
        infer, d_mem=d_mem, initial_capacity=2, expiry_slots=8
    )

    a = 0x4601
    b = 0x4602
    c = 0x4603

    first = runtime.process(
        received_grid=np.array([1.0, 2.0], np.float32),
        ls_estimate=None,
        crntis=[a, b],
        slot_index=100,
    )
    cold_start = bool(
        first.crntis == (a, b)
        and not np.any(first.memory_valid)
        and np.all(first.memory_gap == 0)
        and np.allclose(first.previous_memory, 0.0)
        and np.allclose(first.next_memory[0], 1.0)
        and np.allclose(first.next_memory[1], 2.0)
    )

    second = runtime.process(
        received_grid=np.array([10.0, 30.0], np.float32),
        ls_estimate=None,
        crntis=[b, c],
        slot_index=101,
    )
    reorder_routing = bool(
        second.crntis == (b, c)
        and second.memory_valid.tolist() == [True, False]
        and second.memory_gap.tolist() == [1, 0]
        and np.allclose(second.previous_memory[0], 2.0)
        and np.allclose(second.previous_memory[1], 0.0)
        and np.allclose(second.next_memory[0], 12.0)
        and np.allclose(second.next_memory[1], 30.0)
    )

    third = runtime.process(
        received_grid=np.array([4.0, 5.0], np.float32),
        ls_estimate=None,
        crntis=[a, b],
        slot_index=103,
    )
    gap_and_persistence = bool(
        third.memory_valid.tolist() == [True, True]
        and third.memory_gap.tolist() == [3, 2]
        and np.allclose(third.previous_memory[0], 1.0)
        and np.allclose(third.previous_memory[1], 12.0)
        and np.allclose(third.next_memory[0], 5.0)
        and np.allclose(third.next_memory[1], 17.0)
    )

    inactive = runtime.process(
        received_grid=np.array([1.0, 1000.0], np.float32),
        ls_estimate=None,
        crntis=[a, b],
        slot_index=104,
        active=[True, False],
    )
    b_after_inactive = runtime.process(
        received_grid=np.array([0.0], np.float32),
        ls_estimate=None,
        crntis=[b],
        slot_index=105,
    )
    inactive_guard = bool(
        np.allclose(inactive.next_memory[1], 17.0)
        and b_after_inactive.memory_valid.tolist() == [True]
        and b_after_inactive.memory_gap.tolist() == [2]
        and np.allclose(b_after_inactive.previous_memory[0], 17.0)
    )

    # Existing UE: failed inference must not change the stored value.
    before_fail = runtime.process(
        received_grid=np.array([0.0], np.float32),
        ls_estimate=None,
        crntis=[a],
        slot_index=106,
    )
    infer.fail_next = True
    failed = False
    try:
        runtime.process(
            received_grid=np.array([999.0], np.float32),
            ls_estimate=None,
            crntis=[a],
            slot_index=107,
        )
    except RuntimeError:
        failed = True
    after_fail = runtime.process(
        received_grid=np.array([0.0], np.float32),
        ls_estimate=None,
        crntis=[a],
        slot_index=108,
    )
    failure_no_commit = bool(
        failed and np.allclose(after_fail.previous_memory, before_fail.next_memory)
    )

    # Brand-new UE: lookup during a failed inference must not leave an empty slot.
    new_failed_crnti = 0x4610
    infer.fail_next = True
    new_failed = False
    try:
        runtime.process(
            received_grid=np.array([7.0], np.float32),
            ls_estimate=None,
            crntis=[new_failed_crnti],
            slot_index=108,
        )
    except RuntimeError:
        new_failed = True
    snapshot_after_new_failure = runtime.snapshot()
    new_failure_no_allocation = bool(
        new_failed and new_failed_crnti not in snapshot_after_new_failure["ue_to_slot"]
    )

    # A brand-new inactive position also must not acquire a persistent slot.
    inactive_new_crnti = 0x4611
    inactive_new = runtime.process(
        received_grid=np.array([123.0], np.float32),
        ls_estimate=None,
        crntis=[inactive_new_crnti],
        slot_index=108,
        active=[False],
    )
    inactive_new_no_allocation = bool(
        not inactive_new.memory_valid[0]
        and np.allclose(inactive_new.previous_memory, 0.0)
        and inactive_new_crnti not in runtime.snapshot()["ue_to_slot"]
    )

    runtime.release(b)
    b_released = runtime.process(
        received_grid=np.array([0.0], np.float32),
        ls_estimate=None,
        crntis=[b],
        slot_index=109,
    )
    release_zeroes = bool(
        b_released.memory_valid.tolist() == [False]
        and np.allclose(b_released.previous_memory, 0.0)
    )

    shape_contract = bool(
        infer.calls
        and infer.calls[0]["prev_memory"].shape == (1, 2, d_mem)
        and infer.calls[0]["memory_gap"].shape == (1, 2)
        and infer.calls[0]["memory_valid"].shape == (1, 2)
    )

    duplicate_guard = False
    try:
        runtime.process(
            received_grid=np.array([0.0, 0.0], np.float32),
            ls_estimate=None,
            crntis=[a, a],
            slot_index=110,
        )
    except ValueError:
        duplicate_guard = True

    report = {
        "new_ues_receive_zero_invalid_memory": cold_start,
        "receiver_reorder_routes_by_crnti": reorder_routing,
        "scheduling_gap_and_memory_persistence": gap_and_persistence,
        "inactive_existing_position_does_not_overwrite": inactive_guard,
        "failed_existing_inference_does_not_commit_memory": failure_no_commit,
        "failed_new_inference_does_not_allocate": new_failure_no_allocation,
        "inactive_new_ue_does_not_allocate": inactive_new_no_allocation,
        "release_zeroes_memory": release_zeroes,
        "temporal_model_batch_shape_contract": shape_contract,
        "duplicate_crnti_guard": duplicate_guard,
    }
    report["passed"] = bool(all(report.values()))
    print("TEMPORAL_NRX_RUNTIME_TEST=" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
