#!/usr/bin/env python3
"""Create one immutable Temporal Neural RX stage plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_benchmark_candidate import load_candidate_manifest
from temporal_benchmark_io import validate_run_manifests
from temporal_benchmark_plan import (
    build_stage_plan,
    load_calibration_artifact,
)
from temporal_benchmark_protocol import STAGES, load_benchmark_protocol


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--matched-cold-k3", type=Path, required=True)
    parser.add_argument("--matched-cold-k8", type=Path, required=True)
    parser.add_argument("--historical-cold-k8", type=Path)
    parser.add_argument("--perfect-csi-kbest", type=Path)
    parser.add_argument("--episodes-per-condition", type=int)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    candidate = load_candidate_manifest(args.candidate, protocol)
    baselines = {
        "matched_cold_k3": load_candidate_manifest(
            args.matched_cold_k3, protocol
        ),
        "matched_cold_k8": load_candidate_manifest(
            args.matched_cold_k8, protocol
        ),
    }
    if protocol["stages"][args.stage]["include_static_references"]:
        if args.historical_cold_k8 is None or args.perfect_csi_kbest is None:
            raise ValueError("confirmation requires both static references")
        baselines.update({
            "historical_cold_k8": load_candidate_manifest(
                args.historical_cold_k8, protocol
            ),
            "perfect_csi_kbest": load_candidate_manifest(
                args.perfect_csi_kbest, protocol
            ),
        })
    validate_run_manifests(candidate, baselines, protocol, args.stage)
    authorization = None
    if args.authorization is not None:
        authorization = json.loads(
            args.authorization.read_text(encoding="utf-8")
        )
    plan = build_stage_plan(
        protocol,
        calibration,
        args.stage,
        candidate["manifest_sha256"],
        baselines["matched_cold_k3"]["manifest_sha256"],
        baselines["matched_cold_k8"]["manifest_sha256"],
        baselines.get("historical_cold_k8", {}).get("manifest_sha256"),
        baselines.get("perfect_csi_kbest", {}).get("manifest_sha256"),
        authorization=authorization,
        episodes_per_condition=args.episodes_per_condition,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
