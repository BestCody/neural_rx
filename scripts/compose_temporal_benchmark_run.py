#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_benchmark_cache import compose_run_bundle
from temporal_benchmark_plan import (
    load_calibration_artifact,
    load_stage_plan,
)
from temporal_benchmark_protocol import load_benchmark_protocol


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Join one candidate result with the frozen reference cache.')
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--static-reference-cache", type=Path)
    parser.add_argument("--stage-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    calibration = load_calibration_artifact(args.calibration, protocol)
    plan = load_stage_plan(args.stage_plan, protocol, calibration)
    output = compose_run_bundle(
        args.output_dir,
        protocol=protocol,
        plan=plan,
        candidate_output_path=args.candidate_output,
        reference_cache_path=args.reference_cache,
        static_reference_cache_path=args.static_reference_cache,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
