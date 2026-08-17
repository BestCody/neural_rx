#!/usr/bin/env python3
"""PCA trainer with a learned pooler tuned to the target memory capacity.

For Attention/CNN + PCA, each d_mem configuration gets its own pooler
calibration. The calibration proxy uses a learned writer with the SAME d_mem as
the target PCA bottleneck, so the pooler can adapt to the actual memory budget.
After calibration, the pooler is frozen, PCA is fit once on that representation,
and both the pooler and PCA basis stay frozen during the normal temporal run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import train_temporal_ue_memory_v5_pca_pooler_calibrated as core

v3 = core.v3
v4 = core.v4
tf = core.tf
np = core.np


def _capacity_tuned_calibrate_pooler(p, e2e, actual_model, generator):
    """Tune the learned pooler under the same bottleneck width as target PCA."""
    receiver = e2e._receiver
    base = e2e._receiver._neural_rx._cgnn
    target_d_mem = int(v3.ARGS.d_mem)

    calibration_model = v4.PooledTemporalUEMemoryCGNN(
        base,
        d_mem=target_d_mem,
        d_s=int(p.d_s),
        compression="writer",
        name=f"pooler_calibration_{v4.POOLING}_d{target_d_mem}",
    )
    # Share the exact pooler object with the eventual PCA model.
    calibration_model.pooler = actual_model.pooler

    calibration_memory = v3.DifferentiableUEMemoryManager(
        capacity=v3.ARGS.ue_pool_size,
        d_mem=target_d_mem,
        expiry_slots=v3.ARGS.memory_expiry_slots,
    )

    warmup = generator.sample_batch(1, v3.ARGS.seq_len, 3.0)
    _ = v3.make_losses(receiver, calibration_model, calibration_memory, warmup)

    optimizer = tf.keras.optimizers.Adam(v3.ARGS.memory_lr)
    history = []
    start = time.time()
    v3.set_seed(v3.ARGS.seed + 7919)

    for step in range(core.CALIBRATION_STEPS):
        ebno = float(np.random.uniform(v3.ARGS.min_ebno_db, v3.ARGS.max_ebno_db))
        batch = generator.sample_batch(v3.ARGS.batch_size, v3.ARGS.seq_len, ebno)
        with tf.GradientTape() as tape:
            total, loss_data, loss_chest, per_tb, _ = v3.make_losses(
                receiver, calibration_model, calibration_memory, batch
            )

        # NRX base remains frozen. Pooler + temporary same-capacity memory path
        # learn what information is useful when only target_d_mem values survive.
        variables = calibration_model.memory_variables
        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients reached capacity-tuned pooler calibration")
        optimizer.apply_gradients(pairs)

        if step % v3.ARGS.log_every == 0 or step == core.CALIBRATION_STEPS - 1:
            row = {
                "step": step,
                "pooling": v4.POOLING,
                "target_d_mem": target_d_mem,
                "ebno_db": ebno,
                "loss": float(total.numpy()),
                "loss_data": float(loss_data.numpy()),
                "loss_chest": float(loss_chest.numpy()),
                "loss_per_tb": [float(x.numpy()) for x in per_tb],
                "gradient_norm": float(
                    tf.linalg.global_norm([g for g, _ in pairs]).numpy()
                ),
                "seconds": time.time() - start,
            }
            history.append(row)
            print("POOLER_CAPACITY_CALIBRATION=" + json.dumps(row), flush=True)

    actual_model.pooler.trainable = False
    return {
        "method": "same_capacity_temporal_writer_proxy_then_freeze",
        "pooling": v4.POOLING,
        "steps": int(core.CALIBRATION_STEPS),
        "proxy_memory_width": target_d_mem,
        "target_pca_d_mem": target_d_mem,
        "capacity_tuned": True,
        "nrx_base_frozen": True,
        "pooler_frozen_before_pca_fit": True,
        "history": history,
    }


def main():
    core._calibrate_pooler = _capacity_tuned_calibrate_pooler
    core.main()

    out = Path(v3.ARGS.output_dir)
    summary_path = out / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["architecture"] = (
            "ue_identity_aware_temporal_memory_v7_pca_capacity_tuned_pooler"
        )
        protocol = summary.setdefault("pca_protocol", {})
        protocol["pooler_tuned_to_target_d_mem"] = True
        protocol["target_d_mem"] = int(v3.ARGS.d_mem)
        protocol["shared_pooler_across_capacities"] = False
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "V7_CAPACITY_TUNED_PCA_SUMMARY="
            + json.dumps(
                {
                    "pooling": summary.get("pooling"),
                    "d_mem": summary.get("d_mem"),
                    "capacity_tuned": True,
                    "checkpoint": summary.get("checkpoint"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
