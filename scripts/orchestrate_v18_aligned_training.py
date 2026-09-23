#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "config" / "temporal_benchmark_training_data_v3.json"
RECIPE = ROOT / "config" / "temporal_benchmark_training_recipe_v3.json"
THRESHOLD = ROOT / "config" / "segment_safe_pd_l8_threshold.json"
SELECTION = (
    ROOT / "config" / "temporal_benchmark_checkpoint_selection_v1.json"
)
SEED = 20_262_001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train and register the corrected Segment-Safe screen.')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Run the paired screen using this frozen calibration artifact.",
    )
    parser.add_argument("--candidate-gpu", type=int, default=1)
    parser.add_argument("--reference-gpu", type=int, default=0)
    return parser


def run(marker: Path, command: list[str]) -> None:
    if marker.is_file():
        print(f"ALIGNED_SKIP={marker}", flush=True)
        return
    print("ALIGNED_RUN=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not marker.is_file():
        raise RuntimeError(f"workflow step did not create {marker}")


def run_parallel(steps: list[tuple[Path, list[str]]]) -> None:
    active = []
    for marker, command in steps:
        if marker.is_file():
            print(f"ALIGNED_SKIP={marker}", flush=True)
            continue
        print("ALIGNED_RUN=" + " ".join(command), flush=True)
        active.append((marker, subprocess.Popen(command, cwd=ROOT)))
    failures = []
    for marker, process in active:
        code = process.wait()
        if code != 0 or not marker.is_file():
            failures.append((marker, code))
    if failures:
        raise RuntimeError(f"parallel evaluation failed: {failures}")


def train_command(
    role: str,
    output: Path,
    gpu: int,
    *,
    smoke: bool,
    teacher: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "train_temporal_benchmark.py"),
        "--role",
        role,
        "--stage",
        "screen",
        "--training-seed",
        str(SEED),
        "--gpu",
        str(gpu),
        "--training-data",
        str(DATA),
        "--training-recipe",
        str(RECIPE),
        "--checkpoint-selection",
        str(SELECTION),
        "--threshold-artifact",
        str(THRESHOLD),
        "--output-dir",
        str(output),
    ]
    if smoke:
        command.append("--smoke-test")
    if teacher is not None:
        command.extend(["--teacher-summary", str(teacher)])
    return command


def evaluate_screen(
    base: Path,
    calibration: Path,
    candidate_gpu: int,
    reference_gpu: int,
) -> None:
    registration = base / "registration"
    group = base / "group"
    plan = group / "stage_plan.json"
    run(
        plan,
        [
            sys.executable,
            str(HERE / "prepare_temporal_benchmark.py"),
            "--stage",
            "screen",
            "--candidate",
            str(registration / "candidate.json"),
            "--matched-cold-k3",
            str(registration / "matched_cold_k3.json"),
            "--matched-cold-k8",
            str(registration / "matched_cold_k8.json"),
            "--calibration",
            str(calibration),
            "--output",
            str(plan),
        ],
    )
    artifact_map = registration / "artifact_map.json"
    candidate_output = base / "candidate" / "candidate_output.json"
    reference_output = base / "reference" / "reference_cache.json"
    run_parallel([
        (
            candidate_output,
            [
                sys.executable,
                str(HERE / "run_temporal_benchmark.py"),
                "--mode",
                "candidate",
                "--gpu",
                str(candidate_gpu),
                "--artifact-map",
                str(artifact_map),
                "--candidate",
                str(registration / "candidate.json"),
                "--training-seed",
                str(SEED),
                "--stage-plan",
                str(plan),
                "--calibration",
                str(calibration),
                "--output-dir",
                str(base / "candidate"),
            ],
        ),
        (
            reference_output,
            [
                sys.executable,
                str(HERE / "run_temporal_benchmark.py"),
                "--mode",
                "reference",
                "--gpu",
                str(reference_gpu),
                "--artifact-map",
                str(artifact_map),
                "--matched-cold-k3",
                str(registration / "matched_cold_k3.json"),
                "--matched-cold-k8",
                str(registration / "matched_cold_k8.json"),
                "--training-seed",
                str(SEED),
                "--stage-plan",
                str(plan),
                "--calibration",
                str(calibration),
                "--output-dir",
                str(base / "reference"),
            ],
        ),
    ])
    run_output = group / "runs" / str(SEED) / "benchmark_run.json"
    run(
        run_output,
        [
            sys.executable,
            str(HERE / "compose_temporal_benchmark_run.py"),
            "--candidate-output",
            str(candidate_output),
            "--reference-cache",
            str(reference_output),
            "--stage-plan",
            str(plan),
            "--calibration",
            str(calibration),
            "--output-dir",
            str(run_output.parent),
        ],
    )
    run(
        base / "analysis" / "leaderboard.json",
        [
            sys.executable,
            str(HERE / "analyze_temporal_benchmark.py"),
            "--group",
            str(group),
            "--calibration",
            str(calibration),
            "--output-dir",
            str(base / "analysis"),
        ],
    )


def main() -> None:
    args = build_parser().parse_args()
    base = args.output_dir.expanduser().resolve()
    smoke_k8 = base / "smoke" / "matched_cold_k8"
    smoke_candidate = base / "smoke" / "candidate"
    run(
        smoke_k8 / "training_summary.json",
        train_command(
            "matched_cold_k8", smoke_k8, args.gpu, smoke=True
        ),
    )
    run(
        smoke_candidate / "training_summary.json",
        train_command(
            "candidate",
            smoke_candidate,
            args.gpu,
            smoke=True,
            teacher=smoke_k8 / "training_summary.json",
        ),
    )
    smoke_validation = base / "smoke" / "validation.json"
    run(
        smoke_validation,
        [
            sys.executable,
            str(HERE / "validate_v18_aligned_training.py"),
            "--matched-summary",
            str(smoke_k8 / "training_summary.json"),
            "--candidate-summary",
            str(smoke_candidate / "training_summary.json"),
            "--output",
            str(smoke_validation),
        ],
    )
    if args.smoke_only:
        return

    trained_k8 = base / "train" / "matched_cold_k8"
    trained_candidate = base / "train" / "candidate"
    run(
        trained_k8 / "training_summary.json",
        train_command(
            "matched_cold_k8", trained_k8, args.gpu, smoke=False
        ),
    )
    run(
        trained_candidate / "training_summary.json",
        train_command(
            "candidate",
            trained_candidate,
            args.gpu,
            smoke=False,
            teacher=trained_k8 / "training_summary.json",
        ),
    )
    trained_validation = base / "train" / "validation.json"
    run(
        trained_validation,
        [
            sys.executable,
            str(HERE / "validate_v18_aligned_training.py"),
            "--matched-summary",
            str(trained_k8 / "training_summary.json"),
            "--candidate-summary",
            str(trained_candidate / "training_summary.json"),
            "--output",
            str(trained_validation),
        ],
    )
    registration = base / "registration"
    run(
        registration / "registration.json",
        [
            sys.executable,
            str(HERE / "register_temporal_benchmark_stage.py"),
            "--stage",
            "screen",
            "--candidate-training-dir",
            str(trained_candidate),
            "--matched-k8-training-dir",
            str(trained_k8),
            "--training-data",
            str(DATA),
            "--training-recipe",
            str(RECIPE),
            "--checkpoint-selection",
            str(SELECTION),
            "--output-dir",
            str(registration),
        ],
    )
    if args.calibration is not None:
        calibration = args.calibration.expanduser().resolve()
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
        evaluate_screen(
            base,
            calibration,
            args.candidate_gpu,
            args.reference_gpu,
        )


if __name__ == "__main__":
    main()
