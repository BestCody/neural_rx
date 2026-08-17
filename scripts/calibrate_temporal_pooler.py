#!/usr/bin/env python3
"""Calibrate one learned temporal pooler once, independently of PCA capacity.

The output is a small NPZ containing only Attention/CNN pooler weights plus a
JSON metadata sidecar. The exhaustive PCA study reuses the exact same frozen
pooler for d_mem=8,16,32,56, so memory-capacity comparisons do not accidentally
compare different learned pooling representations.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _extract_custom(argv):
    steps = 1000
    output = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--pooler-calibration-steps":
            steps = int(argv[i + 1]); i += 2; continue
        if arg.startswith("--pooler-calibration-steps="):
            steps = int(arg.split("=", 1)[1]); i += 1; continue
        if arg == "--pooler-calibration-output":
            output = argv[i + 1]; i += 2; continue
        if arg.startswith("--pooler-calibration-output="):
            output = arg.split("=", 1)[1]; i += 1; continue
        cleaned.append(arg); i += 1
    if steps <= 0:
        raise SystemExit("--pooler-calibration-steps must be positive")
    if not output:
        raise SystemExit("--pooler-calibration-output is required")
    return steps, Path(output).expanduser(), cleaned


STEPS, OUTPUT, _CLEAN = _extract_custom(sys.argv)
sys.argv[:] = _CLEAN

import train_temporal_ue_memory_v4 as v4

v3 = v4.v3
tf = v4.tf
np = v3.np


def main():
    if v4.POOLING not in {"attention", "cnn"}:
        raise ValueError("Pooler calibration is only needed for attention/cnn")
    if v3.ARGS.compression != "writer":
        raise ValueError("Calibration proxy must use --compression writer")

    # A fixed d_s-wide proxy transport path makes calibration independent of
    # the later PCA memory capacity.
    v3.ARGS.d_mem = 56
    v3.TemporalUEMemoryCGNN = v4.PooledTemporalUEMemoryCGNN
    v3.set_seed(v3.ARGS.seed)
    p, e2e, model, generator, memory_manager = v3.build()
    if int(p.d_s) != 56:
        raise RuntimeError(f"Expected nrx_large d_s=56, got {p.d_s}")

    receiver = e2e._receiver
    warmup = generator.sample_batch(1, v3.ARGS.seq_len, 3.0)
    _ = v3.make_losses(receiver, model, memory_manager, warmup)

    optimizer = tf.keras.optimizers.Adam(v3.ARGS.memory_lr)
    history = []
    start = time.time()
    v3.set_seed(v3.ARGS.seed + 7919)

    for step in range(STEPS):
        ebno = float(np.random.uniform(v3.ARGS.min_ebno_db, v3.ARGS.max_ebno_db))
        batch = generator.sample_batch(v3.ARGS.batch_size, v3.ARGS.seq_len, ebno)
        with tf.GradientTape() as tape:
            total, loss_data, loss_chest, per_tb, _ = v3.make_losses(
                receiver, model, memory_manager, batch
            )
        variables = model.memory_variables
        grads = tape.gradient(total, variables)
        pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        if not pairs:
            raise RuntimeError("No gradients reached pooler calibration")
        optimizer.apply_gradients(pairs)

        if step % v3.ARGS.log_every == 0 or step == STEPS - 1:
            row = {
                "step": step,
                "pooling": v4.POOLING,
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
            print("POOLER_CALIBRATION=" + json.dumps(row), flush=True)

    weights = model.pooler.get_weights()
    if not weights:
        raise RuntimeError("Learned pooler produced no weights")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUTPUT), **{f"w{i}": w for i, w in enumerate(weights)})
    metadata = {
        "protocol": "shared_temporal_pooler_calibration_v1",
        "pooling": v4.POOLING,
        "config": v3.ARGS.config,
        "d_s": int(p.d_s),
        "steps": int(STEPS),
        "seed": int(v3.ARGS.seed),
        "seq_len": int(v3.ARGS.seq_len),
        "batch_size": int(v3.ARGS.batch_size),
        "dynamic_scheduling": not v3.ARGS.fixed_scheduling,
        "proxy_compression": "writer",
        "proxy_memory_width": int(p.d_s),
        "nrx_base_frozen": True,
        "num_weight_arrays": len(weights),
        "weights_file": str(OUTPUT),
        "history": history,
    }
    sidecar = OUTPUT.with_suffix(OUTPUT.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    print("POOLER_CALIBRATION_SUMMARY=" + json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
