#!/usr/bin/env python3
"""Evaluate K=1/K=2 temporal receivers over 64 TBs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ARCHITECTURE = "ue_identity_aware_temporal_memory_streaming_tbptt_v1"
REQUIRED_TRAINING_INVARIANTS = {
    "continuous_channel_per_episode": True,
    "state_carried_across_tbptt_windows": True,
    "gradient_detached_between_tbptt_windows": True,
    "state_reset_between_independent_episodes": True,
    "transport_block_position_input": False,
    "random_valid_ue_reset_at_window_boundaries": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate temporal K=1/K=2 and cold K=1/K=2/K=8 on paired "
            "continuous 64-TB episodes"
        )
    )
    parser.add_argument("--temporal-k1-checkpoint")
    parser.add_argument("--temporal-k2-checkpoint")
    parser.add_argument("--config", default="nrx_large.cfg")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-tbs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=1.5)
    parser.add_argument("--snr-max", type=float, default=4.0)
    parser.add_argument("--snr-step", type=float, default=0.25)
    parser.add_argument("--target-errors", type=int, default=120)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--trajectory-snr", type=float, default=2.75)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--execution-mode", choices=["graph", "eager"], default="graph"
    )
    receiver_group = parser.add_mutually_exclusive_group()
    receiver_group.add_argument(
        "--temporal-only",
        dest="temporal_only",
        action="store_true",
        help="evaluate temporal K=1/K=2 without rerunning cold baselines",
    )
    receiver_group.add_argument(
        "--include-cold-baselines",
        dest="temporal_only",
        action="store_false",
        help="also rerun paired cold K=1/K=2/K=8 baselines",
    )
    parser.set_defaults(temporal_only=True)
    parser.add_argument(
        "--allow-nonstandard-checkpoint",
        action="store_true",
        help="allow short/non-64-TB checkpoints for development smoke tests",
    )
    parser.add_argument(
        "--allow-nonstandard-horizon",
        action="store_true",
        help="allow --num-tbs other than 64 for development smoke tests",
    )
    parser.add_argument(
        "--allow-mismatched-temporal-architecture",
        action="store_true",
        help="allow K=1 and K=2 checkpoints with different memory designs",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def make_snr_grid(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive SNR grid."""
    if step <= 0:
        raise ValueError("snr-step must be positive")
    if stop < start:
        raise ValueError("snr-max must be >= snr-min")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    values = [float(start + index * step) for index in range(count)]
    if values[-1] < stop - 1e-9:
        values.append(float(stop))
    return [float(round(value, 10)) for value in values]


def wilson_interval(errors: int, blocks: int, z: float = 1.959963984540054):
    if blocks <= 0:
        return [None, None]
    p = float(errors) / float(blocks)
    denom = 1.0 + z * z / blocks
    center = (p + z * z / (2.0 * blocks)) / denom
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / blocks + z * z / (4.0 * blocks**2))
        / denom
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def corrected_bler(errors: int, blocks: int) -> float | None:
    """Return a finite corrected BLER estimate."""
    if blocks <= 0:
        return None
    return (float(errors) + 0.5) / (float(blocks) + 1.0)


def log_bler_crossing(points: list[dict], target: float = 0.10):
    """Interpolate a BLER crossing in log space."""
    if target <= 0:
        raise ValueError("target must be positive")
    samples = []
    for point in sorted(points, key=lambda item: item["snr_db"]):
        value = corrected_bler(
            int(point["errors_tb2plus"]), int(point["blocks_tb2plus"])
        )
        if value is not None:
            samples.append((float(point["snr_db"]), value))
    for (x0, y0), (x1, y1) in zip(samples, samples[1:]):
        if (y0 - target) * (y1 - target) > 0:
            continue
        if y0 == y1:
            return float((x0 + x1) / 2.0)
        fraction = (math.log(target) - math.log(y0)) / (
            math.log(y1) - math.log(y0)
        )
        return float(x0 + fraction * (x1 - x0))
    return None


def _summary_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "training_summary.json"


def load_checkpoint_spec(
    checkpoint_value: str,
    expected_k: int,
    expected_config: str,
    allow_nonstandard: bool = False,
) -> dict:
    """Load and validate temporal checkpoint metadata."""
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Temporal K={expected_k} checkpoint: {checkpoint}")
    summary_path = _summary_path(checkpoint)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint provenance is required but missing: {summary_path}"
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    failures = []
    if summary.get("architecture") != ARCHITECTURE:
        failures.append(f"architecture must be {ARCHITECTURE!r}")
    if int(summary.get("num_it", -1)) != int(expected_k):
        failures.append(f"num_it must be {expected_k}")
    if Path(str(summary.get("config", ""))).name != Path(expected_config).name:
        failures.append(f"config must be {Path(expected_config).name!r}")
    method = summary.get("training_method", {})
    for key, expected in REQUIRED_TRAINING_INVARIANTS.items():
        if method.get(key) is not expected:
            failures.append(f"training_method.{key} must be {expected}")

    if not allow_nonstandard:
        standard = {
            "stream_len": 64,
            "tbptt_window": 4,
            "train_steps": 6000,
        }
        for key, expected in standard.items():
            if int(summary.get(key, -1)) != expected:
                failures.append(f"{key} must be {expected}")
        if float(summary.get("memory_reset_prob", 0.0)) <= 0.0:
            failures.append("memory_reset_prob must be positive")

    if failures:
        raise ValueError(
            f"Invalid temporal K={expected_k} checkpoint provenance in "
            f"{summary_path}: " + "; ".join(failures)
        )

    return {
        "checkpoint": str(checkpoint),
        "summary_path": str(summary_path.resolve()),
        "config": Path(str(summary["config"])).name,
        "pooling": str(summary["pooling"]),
        "compression": str(summary["compression"]),
        "d_mem": int(summary["d_mem"]),
        "num_it": int(summary["num_it"]),
        "ue_pool_size": int(summary["ue_pool_size"]),
        "dynamic_scheduling": bool(summary["dynamic_scheduling"]),
        "memory_expiry_slots": summary.get("memory_expiry_slots"),
        "schedule_switch_prob": float(summary["schedule_switch_prob"]),
        "schedule_reorder_prob": float(summary["schedule_reorder_prob"]),
        "stream_len": int(summary["stream_len"]),
        "tbptt_window": int(summary["tbptt_window"]),
        "train_steps": int(summary["train_steps"]),
        "seed": int(summary["seed"]),
    }


def architecture_signature(spec: dict) -> tuple:
    """Return the checkpoint architecture fields."""
    return (
        spec["config"],
        spec["pooling"],
        spec["compression"],
        spec["d_mem"],
        spec["ue_pool_size"],
        spec["dynamic_scheduling"],
        spec["memory_expiry_slots"],
        spec["schedule_switch_prob"],
        spec["schedule_reorder_prob"],
        spec["stream_len"],
        spec["tbptt_window"],
    )


def new_counter(num_tbs: int) -> dict:
    return {
        "errors": np.zeros(num_tbs, dtype=np.int64),
        "blocks": np.zeros(num_tbs, dtype=np.int64),
        "crc_disagreements": np.zeros(num_tbs, dtype=np.int64),
    }


def add_counts(
    counter: dict,
    tb_index: int,
    errors: int,
    blocks: int,
    crc_disagreements: int = 0,
) -> None:
    counter["errors"][tb_index] += int(errors)
    counter["blocks"][tb_index] += int(blocks)
    counter["crc_disagreements"][tb_index] += int(crc_disagreements)


def _ratio(numerator: int, denominator: int):
    return float(numerator) / float(denominator) if denominator else None


def segment_ranges(num_tbs: int) -> list[tuple[str, int, int]]:
    candidates = [
        ("cold_start", 0, min(1, num_tbs)),
        ("early", 1, min(16, num_tbs)),
        ("middle", 16, min(32, num_tbs)),
        ("late", 32, num_tbs),
    ]
    return [entry for entry in candidates if entry[2] > entry[1]]


def finalize_counter(counter: dict, snr_db: float, batches: int) -> dict:
    errors = counter["errors"]
    blocks = counter["blocks"]
    disagreements = counter["crc_disagreements"]
    all_e, all_n = int(errors.sum()), int(blocks.sum())
    warm_e, warm_n = int(errors[1:].sum()), int(blocks[1:].sum())
    per_tb = []
    for index, (error_count, block_count, mismatch_count) in enumerate(
        zip(errors, blocks, disagreements)
    ):
        e, n = int(error_count), int(block_count)
        per_tb.append(
            {
                "tb": index + 1,
                "errors": e,
                "blocks": n,
                "bler": _ratio(e, n),
                "ci95": wilson_interval(e, n),
                "crc_disagreements": int(mismatch_count),
            }
        )
    segments = {}
    for name, start, stop in segment_ranges(len(errors)):
        e, n = int(errors[start:stop].sum()), int(blocks[start:stop].sum())
        segments[name] = {
            "tb_start": start + 1,
            "tb_end": stop,
            "errors": e,
            "blocks": n,
            "bler": _ratio(e, n),
            "ci95": wilson_interval(e, n),
        }
    return {
        "snr_db": float(snr_db),
        "batches": int(batches),
        "errors_all": all_e,
        "blocks_all": all_n,
        "bler_all": _ratio(all_e, all_n),
        "ci95_all": wilson_interval(all_e, all_n),
        "errors_tb2plus": warm_e,
        "blocks_tb2plus": warm_n,
        "bler_tb2plus": _ratio(warm_e, warm_n),
        "ci95_tb2plus": wilson_interval(warm_e, warm_n),
        "crc_disagreements": int(disagreements.sum()),
        "segments": segments,
        "per_tb": per_tb,
    }


def validate_args(args: argparse.Namespace) -> tuple[list[float], dict[int, dict]]:
    if not args.temporal_k1_checkpoint and not args.temporal_k2_checkpoint:
        raise ValueError("Provide at least one temporal K=1 or K=2 checkpoint")
    if args.num_tbs != 64 and not args.allow_nonstandard_horizon:
        raise ValueError("The research evaluator requires --num-tbs 64")
    for name in ("num_tbs", "batch_size", "target_errors", "max_batches"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_tbs < 2:
        raise ValueError("At least two TBs are required for the TB2+ metric")

    snr_grid = make_snr_grid(args.snr_min, args.snr_max, args.snr_step)
    specs = {}
    for k, value in (
        (1, args.temporal_k1_checkpoint),
        (2, args.temporal_k2_checkpoint),
    ):
        if value:
            specs[k] = load_checkpoint_spec(
                value,
                k,
                args.config,
                allow_nonstandard=args.allow_nonstandard_checkpoint,
            )
    if (
        len(specs) == 2
        and architecture_signature(specs[1]) != architecture_signature(specs[2])
        and not args.allow_mismatched_temporal_architecture
    ):
        raise ValueError(
            "K=1 and K=2 checkpoints use different temporal architectures or "
            "scheduling. Pass --allow-mismatched-temporal-architecture only "
            "for an intentionally unmatched comparison."
        )
    return snr_grid, specs


def _comparison_summary(curves: dict[str, list[dict]]) -> dict:
    crossings = {
        method: log_bler_crossing(points) for method, points in curves.items()
    }
    comparisons = {"snr_at_10pct_tb2plus_db": crossings}
    for k in (1, 2):
        temporal = crossings.get(f"temporal_k{k}")
        cold = crossings.get(f"cold_k{k}")
        cold8 = crossings.get("cold_k8")
        if temporal is None:
            continue
        comparisons[f"temporal_k{k}"] = {
            "gain_over_matched_cold_db": (
                None if cold is None else float(cold - temporal)
            ),
            "remaining_gap_to_cold_k8_db": (
                None if cold8 is None else float(temporal - cold8)
            ),
        }
        if cold is not None and cold8 is not None and cold != cold8:
            comparisons[f"temporal_k{k}"]["cold_k8_gap_recovered_fraction"] = (
                float((cold - temporal) / (cold - cold8))
            )
    return comparisons


def _write_csvs(output_dir: Path, curves: dict[str, list[dict]]) -> None:
    with (output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method", "snr_db", "batches", "errors_all", "blocks_all",
            "bler_all", "ci95_all_low", "ci95_all_high", "errors_tb2plus",
            "blocks_tb2plus", "bler_tb2plus", "ci95_tb2plus_low",
            "ci95_tb2plus_high", "crc_disagreements",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, points in curves.items():
            for point in points:
                writer.writerow(
                    {
                        "method": method,
                        "snr_db": point["snr_db"],
                        "batches": point["batches"],
                        "errors_all": point["errors_all"],
                        "blocks_all": point["blocks_all"],
                        "bler_all": point["bler_all"],
                        "ci95_all_low": point["ci95_all"][0],
                        "ci95_all_high": point["ci95_all"][1],
                        "errors_tb2plus": point["errors_tb2plus"],
                        "blocks_tb2plus": point["blocks_tb2plus"],
                        "bler_tb2plus": point["bler_tb2plus"],
                        "ci95_tb2plus_low": point["ci95_tb2plus"][0],
                        "ci95_tb2plus_high": point["ci95_tb2plus"][1],
                        "crc_disagreements": point["crc_disagreements"],
                    }
                )

    with (output_dir / "per_tb.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "method", "snr_db", "tb", "errors", "blocks", "bler",
            "ci95_low", "ci95_high", "crc_disagreements",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, points in curves.items():
            for point in points:
                for row in point["per_tb"]:
                    writer.writerow(
                        {
                            "method": method,
                            "snr_db": point["snr_db"],
                            "tb": row["tb"],
                            "errors": row["errors"],
                            "blocks": row["blocks"],
                            "bler": row["bler"],
                            "ci95_low": row["ci95"][0],
                            "ci95_high": row["ci95"][1],
                            "crc_disagreements": row["crc_disagreements"],
                        }
                    )


def _trailing_mean(values: list[float], window: int = 4) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    for index in range(len(array)):
        result[index] = np.nanmean(array[max(0, index - window + 1): index + 1])
    return result


def _make_plots(
    output_dir: Path,
    curves: dict[str, list[dict]],
    trajectory_snr: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "cold_k1": "Cold K=1",
        "cold_k2": "Cold K=2",
        "cold_k8": "Cold K=8",
        "temporal_k1": "Temporal K=1",
        "temporal_k2": "Temporal K=2",
    }
    colors = {
        "cold_k1": "#9e9e9e",
        "cold_k2": "#4c78a8",
        "cold_k8": "#222222",
        "temporal_k1": "#f58518",
        "temporal_k2": "#54a24b",
    }

    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for method, points in curves.items():
        x = [point["snr_db"] for point in points]
        y = [corrected_bler(point["errors_tb2plus"], point["blocks_tb2plus"])
             for point in points]
        style = "-" if method.startswith("temporal") else "--"
        width = 2.4 if method.startswith("temporal") else 1.7
        axis.semilogy(x, y, style, marker="o", linewidth=width,
                      color=colors[method], label=labels[method])
    axis.axhline(0.10, color="black", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Eb/N0 (dB)")
    axis.set_ylabel("TB2+ TBLER")
    axis.set_title("Persistent temporal Neural RX — 64-TB paired evaluation")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tbler_vs_snr.{suffix}", dpi=220)
    plt.close(fig)

    available_snrs = [point["snr_db"] for points in curves.values() for point in points]
    selected_snr = min(set(available_snrs), key=lambda value: abs(value - trajectory_snr))
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    for method, points in curves.items():
        point = min(points, key=lambda item: abs(item["snr_db"] - selected_snr))
        values = [row["bler"] for row in point["per_tb"]]
        x = np.arange(1, len(values) + 1)
        style = "-" if method.startswith("temporal") else "--"
        width = 2.5 if method.startswith("temporal") else 1.7
        axis.plot(x, _trailing_mean(values), style, linewidth=width,
                  color=colors[method], label=f"{labels[method]} (trailing 4-TB)")
    axis.set_xlabel("Transport-block position")
    axis.set_ylabel("TBLER")
    axis.set_title(f"64-TB memory trajectory at {selected_snr:g} dB")
    axis.set_xlim(1, max(len(point["per_tb"]) for points in curves.values() for point in points))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"tbler_by_position.{suffix}", dpi=220)
    plt.close(fig)


def run_evaluation(args: argparse.Namespace, snr_grid: list[float], specs: dict) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    here = Path(__file__).resolve().parent
    os.chdir(here)
    config_name = Path(args.config).name

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if args.execution_mode == "eager":
        tf.config.run_functions_eagerly(True)

    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent))
    from temporal_training_data import TemporalTrainingDataGenerator
    from temporal_ue_memory_model import (
        TemporalUEMemoryCGNN,
        build_backbone,
        demap_llr,
        prepare_cgnn_inputs,
        set_seed,
        temporal_forward,
    )
    from ue_memory_manager import DifferentiableUEMemoryManager

    def make_cold_decoder(receiver):
        def decode(y, ls, active):
            inputs = prepare_cgnn_inputs(receiver, y, ls, active)
            llr_iterations, _ = receiver._neural_rx._cgnn(inputs)
            llr_grid = llr_iterations[-1][0]
            llr = demap_llr(
                receiver._neural_rx, llr_grid, tf.shape(active)[1], 0
            )
            return receiver._tb_decoders[0](llr)
        return tf.function(decode) if args.execution_mode == "graph" else decode

    def make_temporal_decoder(receiver, model):
        def decode(y, ls, active, memory, gap, valid):
            llr, _, next_memory, _, _ = temporal_forward(
                receiver, model, y, ls, active, memory, gap, valid,
                training=False,
            )
            b_hat, crc = receiver._tb_decoders[0](llr)
            return b_hat, crc, next_memory
        return tf.function(decode) if args.execution_mode == "graph" else decode

    set_seed(args.seed)
    methods = {}
    generator_parameters = None
    generator_e2e = None
    prebuilt_temporal = {}
    if args.temporal_only:
        first_k = min(specs)
        generator_parameters, generator_e2e = build_backbone(
            config_name, first_k, training=False, num_tx_eval=2
        )
        prebuilt_temporal[first_k] = (generator_parameters, generator_e2e)
    else:
        for k in (1, 2, 8):
            parameters, e2e = build_backbone(
                config_name, k, training=False, num_tx_eval=2
            )
            methods[f"cold_k{k}"] = {
                "kind": "cold", "k": k, "receiver": e2e._receiver,
                "decode": make_cold_decoder(e2e._receiver),
            }
            if k == 1:
                generator_parameters, generator_e2e = parameters, e2e

    if int(generator_parameters.n_size_bwp) != 132:
        raise RuntimeError(
            "Research evaluation requires 132 PRBs; config produced "
            f"{generator_parameters.n_size_bwp}"
        )

    first_spec = specs[min(specs)]
    generator = TemporalTrainingDataGenerator(
        generator_parameters,
        generator_e2e,
        ue_pool_size=first_spec["ue_pool_size"],
        dynamic_scheduling=first_spec["dynamic_scheduling"],
        schedule_switch_prob=first_spec["schedule_switch_prob"],
        schedule_reorder_prob=first_spec["schedule_reorder_prob"],
    )

    warm = generator.sample_batch(1, min(args.num_tbs, 2), 3.0)
    for k, spec in sorted(specs.items()):
        if k in prebuilt_temporal:
            parameters, e2e = prebuilt_temporal[k]
        else:
            parameters, e2e = build_backbone(
                config_name, k, training=False, num_tx_eval=2
            )
        model = TemporalUEMemoryCGNN(
            e2e._receiver._neural_rx._cgnn,
            d_mem=spec["d_mem"],
            d_s=parameters.d_s,
            compression=spec["compression"],
            pooling=spec["pooling"],
            name=f"temporal_k{k}_eval",
        )
        manager = DifferentiableUEMemoryManager(
            capacity=spec["ue_pool_size"],
            d_mem=spec["d_mem"],
            expiry_slots=spec["memory_expiry_slots"],
        )
        warm_state = manager.zero_state(1, tf.float32)
        warm_state, memory, gap, valid = manager.gather(
            warm_state, warm["ue_ids"][:, 0], 0
        )
        temporal_forward(
            e2e._receiver, model, warm["y"][:, 0], warm["ls"][:, 0],
            warm["active"][:, 0], memory, gap, valid, training=False,
        )
        model.load_weights(spec["checkpoint"])
        methods[f"temporal_k{k}"] = {
            "kind": "temporal", "k": k, "receiver": e2e._receiver,
            "model": model, "manager": manager,
            "decode": make_temporal_decoder(e2e._receiver, model),
        }

    for method in methods.values():
        if method["kind"] == "cold":
            method["decode"](
                warm["y"][:, 0], warm["ls"][:, 0], warm["active"][:, 0]
            )
        else:
            state = method["manager"].zero_state(1, tf.float32)
            state, memory, gap, valid = method["manager"].gather(
                state, warm["ue_ids"][:, 0], 0
            )
            method["decode"](
                warm["y"][:, 0], warm["ls"][:, 0], warm["active"][:, 0],
                memory, gap, valid,
            )

    def count_decoded(bits, b_hat, crc, active):
        error = tf.reduce_any(
            tf.not_equal(
                tf.cast(bits, tf.int32), tf.cast(tf.round(b_hat), tf.int32)
            ),
            axis=-1,
        )
        mask = tf.cast(active, tf.bool)
        error = tf.logical_and(error, mask)
        crc_error = tf.logical_and(tf.logical_not(tf.cast(crc, tf.bool)), mask)
        disagreement = tf.logical_and(tf.not_equal(error, crc_error), mask)
        return (
            int(tf.reduce_sum(tf.cast(error, tf.int64)).numpy()),
            int(tf.reduce_sum(tf.cast(mask, tf.int64)).numpy()),
            int(tf.reduce_sum(tf.cast(disagreement, tf.int64)).numpy()),
        )

    curves = {name: [] for name in methods}
    for snr_index, snr_db in enumerate(snr_grid):
        set_seed(args.seed + snr_index * 100003)
        counters = {name: new_counter(args.num_tbs) for name in methods}
        batches_run = 0
        for batch_index in range(args.max_batches):
            episode = generator.sample_batch(
                args.batch_size, args.num_tbs, snr_db
            )
            states = {
                name: method["manager"].zero_state(args.batch_size, tf.float32)
                for name, method in methods.items()
                if method["kind"] == "temporal"
            }
            for tb_index in range(args.num_tbs):
                bits = episode["bits"][:, tb_index]
                y = episode["y"][:, tb_index]
                ls = episode["ls"][:, tb_index]
                active = episode["active"][:, tb_index]
                ids = episode["ue_ids"][:, tb_index]
                for name, method in methods.items():
                    if method["kind"] == "cold":
                        b_hat, crc = method["decode"](y, ls, active)
                    else:
                        manager = method["manager"]
                        state, memory, gap, valid = manager.gather(
                            states[name], ids, tb_index
                        )
                        b_hat, crc, next_memory = method["decode"](
                            y, ls, active, memory, gap, valid
                        )
                        states[name] = manager.scatter(
                            state, ids, next_memory, active, tb_index
                        )
                    add_counts(
                        counters[name], tb_index,
                        *count_decoded(bits, b_hat, crc, active),
                    )
            batches_run = batch_index + 1
            progress = {
                name: int(counter["errors"][1:].sum())
                for name, counter in counters.items()
            }
            print(
                "EVAL_PROGRESS=" + json.dumps(
                    {"snr_db": snr_db, "batch": batches_run,
                     "tb2plus_errors": progress}
                ),
                flush=True,
            )
            if all(value >= args.target_errors for value in progress.values()):
                break
        for name in methods:
            curves[name].append(
                finalize_counter(counters[name], snr_db, batches_run)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "methodology": {
            "num_tbs": args.num_tbs,
            "continuous_channel_within_episode": True,
            "same_episode_for_all_methods": True,
            "fresh_payload_and_awgn_per_tb": True,
            "temporal_memory_empty_at_tb1": True,
            "temporal_memory_persists_through_episode": True,
            "temporal_memory_reset_between_independent_episodes": True,
            "cold_receivers_reset_every_tb": True,
            "primary_metric": f"TBLER over TB2 through TB{args.num_tbs}",
            "prbs": 132,
        },
        "config": args.config,
        "cold_baselines": [] if args.temporal_only else [1, 2, 8],
        "temporal_only": bool(args.temporal_only),
        "temporal_checkpoints": specs,
        "snr_grid_db": snr_grid,
        "batch_size": args.batch_size,
        "target_errors": args.target_errors,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "curves": curves,
        "comparisons": _comparison_summary(curves),
    }
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    _write_csvs(output_dir, curves)
    if not args.skip_plots:
        _make_plots(output_dir, curves, args.trajectory_snr)
    print("EVALUATION_SUMMARY=" + json.dumps(result["comparisons"], indent=2))
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    snr_grid, specs = validate_args(args)
    run_evaluation(args, snr_grid, specs)


if __name__ == "__main__":
    main()
