#!/usr/bin/env python3
"""Run a small targeted stability check for the repaired temporal autoencoder.

This is deliberately not a research-result run. It trains two configurations
that catastrophically diverged in the original exhaustive sweep long enough to
enter joint fine-tuning, then checks bounded memory, finite losses, auxiliary
loss scale, and temporal gradient connectivity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="/tmp/temporal_autoencoder_stability")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--memory-only-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--gpus", default="0,1")
    return p.parse_args()


def run_case(script_dir, py, out_root, gpu, pooling, d_mem, a):
    name = f"{pooling}_autoencoder_d{d_mem}"
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        str(script_dir / "train_temporal_ue_memory_v4.py"),
        "--gpu", str(gpu),
        "--pooling", pooling,
        "--compression", "autoencoder",
        "--d-mem", str(d_mem),
        "--num-it", "2",
        "--train-steps", str(a.steps),
        "--memory-only-steps", str(a.memory_only_steps),
        "--batch-size", str(a.batch_size),
        "--seq-len", "4",
        "--min-ebno-db", "1.0",
        "--max-ebno-db", "5.0",
        "--memory-lr", "1e-3",
        "--joint-lr", "2e-5",
        "--ae-reconstruction-weight", "0.1",
        "--ue-pool-size", "4",
        "--fixed-scheduling",
        "--seed", str(a.seed),
        "--log-every", "10",
        "--output-dir", str(out),
    ]
    log = out / "train.log"
    with log.open("w") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(script_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode:
        raise RuntimeError(f"{name} failed; see {log}")
    return name, out


def summarize_case(name, out):
    summary = json.loads((out / "training_summary.json").read_text())
    protocol = summary.get("autoencoder_protocol", {})
    rows = summary.get("history", [])
    if not rows:
        raise RuntimeError(f"{name}: no training history")

    memory = [
        float(x)
        for row in rows
        for x in (row.get("memory_norm_per_tb") or [])
    ]
    aux = [float(row["compression_aux_loss"]) for row in rows]
    data = [float(row["loss_data"]) for row in rows]
    grad = summary.get("temporal_compression_gradient_check", {})
    d_mem = int(summary["d_mem"])
    bound = math.sqrt(d_mem) + 1e-4
    finite = all(math.isfinite(x) for x in memory + aux + data)
    last_ratio = 0.1 * aux[-1] / data[-1] if data[-1] else math.inf

    result = {
        "case": name,
        "autoencoder_protocol": protocol,
        "max_memory_norm": max(memory),
        "memory_norm_bound": bound,
        "last_weighted_reconstruction_over_data": last_ratio,
        "last_data_loss": data[-1],
        "last_normalized_reconstruction_aux_loss": aux[-1],
        "finite": finite,
        "gradient_check": grad,
    }
    result["passed"] = bool(
        protocol.get("version") == 2
        and protocol.get("bounded_tanh_bottleneck") is True
        and protocol.get("reconstruction_upstream_state_detached") is True
        and protocol.get("scale_normalized_reconstruction_aux_loss") is True
        and finite
        and result["max_memory_norm"] <= bound
        and last_ratio < 1.0
        and grad.get("passed") is True
    )
    return result


def main():
    a = parse_args()
    if a.steps <= a.memory_only_steps:
        raise SystemExit("--steps must exceed --memory-only-steps to test joint fine-tuning")
    gpus = [int(x) for x in a.gpus.split(",") if x.strip()]
    if len(gpus) < 2:
        raise SystemExit("This smoke expects two GPU IDs")

    script_dir = Path(__file__).resolve().parent
    py = sys.executable
    out_root = Path(a.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    unit = subprocess.run(
        [py, str(script_dir / "test_temporal_compression.py")],
        cwd=str(script_dir),
        text=True,
        capture_output=True,
    )
    (out_root / "unit_test.log").write_text(unit.stdout + unit.stderr)
    if unit.returncode:
        raise RuntimeError("compression regression test failed")

    cases = [(gpus[0], "mean", 8), (gpus[1], "cnn", 16)]
    outputs = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(run_case, script_dir, py, out_root, gpu, pooling, d_mem, a)
            for gpu, pooling, d_mem in cases
        ]
        for fut in futures:
            name, out = fut.result()
            outputs[name] = out

    report = {
        "purpose": "stability_smoke_not_research_result",
        "steps": a.steps,
        "memory_only_steps": a.memory_only_steps,
        "cases": [summarize_case(name, outputs[name]) for name in sorted(outputs)],
    }
    report["passed"] = all(x["passed"] for x in report["cases"])
    path = out_root / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("AUTOENCODER_STABILITY_SMOKE=" + json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("Autoencoder stability smoke failed")


if __name__ == "__main__":
    main()
