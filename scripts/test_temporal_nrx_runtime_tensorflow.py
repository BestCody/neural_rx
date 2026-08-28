#!/usr/bin/env python3
"""GPU smoke test: actual K=2 TemporalUEMemoryCGNN through live runtime.

This is a contract/integration test, not an accuracy experiment. It uses the
pretrained cold NRX backbone plus newly initialized temporal-memory layers and
checks that a real TensorFlow temporal forward pass consumes C-RNTI-owned
previous memory on the next correlated TB.
"""

from __future__ import annotations

import json

import numpy as np

import tensorflow as tf

from temporal_nrx_runtime import TensorFlowTemporalInference, TemporalNRXRuntime
from temporal_training_data import TemporalTrainingDataGenerator
from temporal_ue_memory_model import TemporalUEMemoryCGNN, build_backbone


def _as_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def main():
    tf.random.set_seed(20260816)
    np.random.seed(20260816)

    parameters, e2e = build_backbone(
        "nrx_large.cfg", num_it=2, training=True
    )
    temporal_model = TemporalUEMemoryCGNN(
        e2e._receiver._neural_rx._cgnn,
        d_mem=8,
        d_s=parameters.d_s,
        compression="writer",
        pooling="mean",
        name="runtime_contract_temporal_model",
    )
    generator = TemporalTrainingDataGenerator(
        parameters,
        e2e,
        ue_pool_size=4,
        dynamic_scheduling=False,
    )
    batch = generator.sample_batch(batch_size=1, seq_len=3, ebno_db=3.0)

    inference = TensorFlowTemporalInference(
        e2e._receiver,
        temporal_model,
        expected_num_it=2,
    )
    runtime = TemporalNRXRuntime(
        inference,
        d_mem=8,
        initial_capacity=4,
        expiry_slots=8,
    )

    results = []
    crnti_sequences = []
    for t in range(3):
        ue_ids = _as_numpy(batch["ue_ids"])[0, t].astype(np.int64)
        crntis = [0x4601 + int(x) for x in ue_ids]
        active = _as_numpy(batch["active"])[0, t].astype(bool)
        crnti_sequences.append(crntis)

        result = runtime.process(
            received_grid=batch["y"][:, t],
            ls_estimate=batch["ls"][:, t],
            crntis=crntis,
            slot_index=100 + t,
            active=active,
        )
        results.append(result)

    first, second, third = results
    first_cold = bool(
        not np.any(first.memory_valid)
        and np.all(first.memory_gap == 0)
        and np.allclose(first.previous_memory, 0.0)
    )
    second_reuses = bool(
        np.all(second.memory_valid)
        and np.all(second.memory_gap == 1)
        and np.allclose(second.previous_memory, first.next_memory)
    )
    third_reuses = bool(
        np.all(third.memory_valid)
        and np.all(third.memory_gap == 1)
        and np.allclose(third.previous_memory, second.next_memory)
    )

    all_memory_finite = bool(
        all(np.all(np.isfinite(r.next_memory)) for r in results)
    )
    memory_shape = bool(
        all(r.next_memory.shape == (2, 8) for r in results)
    )

    llr_shapes = []
    llrs_finite = True
    for result in results:
        llr = _as_numpy(result.receiver_output)
        llr_shapes.append(list(llr.shape))
        llrs_finite = llrs_finite and bool(np.all(np.isfinite(llr)))

    fixed_identity = bool(
        crnti_sequences[0] == crnti_sequences[1] == crnti_sequences[2]
    )

    report = {
        "actual_tensorflow_temporal_model_ran": True,
        "configured_k": int(temporal_model.base._num_it),
        "first_tb_is_cold_start": first_cold,
        "second_tb_receives_first_tb_memory": second_reuses,
        "third_tb_receives_second_tb_memory": third_reuses,
        "fixed_crnti_identity": fixed_identity,
        "next_memory_shape_is_2x8": memory_shape,
        "next_memory_is_finite": all_memory_finite,
        "llr_is_finite": bool(llrs_finite),
        "llr_shapes": llr_shapes,
        "crntis": crnti_sequences,
    }
    checks = [
        report["configured_k"] == 2,
        first_cold,
        second_reuses,
        third_reuses,
        fixed_identity,
        memory_shape,
        all_memory_finite,
        bool(llrs_finite),
    ]
    report["passed"] = bool(all(checks))

    print("TEMPORAL_NRX_TENSORFLOW_RUNTIME_TEST=" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
