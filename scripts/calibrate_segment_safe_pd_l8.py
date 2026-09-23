#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_domain_memory import segment_safe_reset_detector
from temporal_generalization_protocol import (
    build_segment_safe_threshold_artifact,
    conformal_false_reset_threshold,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Calibrate the segment-safe PD-L8 guard.')
    parser.add_argument(
        "--source-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "lapra_reset_calibration_v6_20260902"
            / "pd_dawm_reset_calibration.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(report_path: Path) -> dict:
    report_path = report_path.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    method = report.get("methodology", {})
    if method.get("split_role") != "calibration":
        raise ValueError("source is not calibration data")
    if method.get("controlled_channel_change") is not False:
        raise ValueError("calibration source contains channel changes")
    raw = report.get("raw_episode_metrics", {})
    source = report_path.parent / raw.get("file", "")
    if not source.is_file() or file_sha256(source) != raw.get("sha256"):
        raise ValueError("calibration source checksum differs")
    with np.load(source, allow_pickle=False) as values:
        innovation = np.asarray(
            values["normalized_innovation_mean"], np.float64
        )
        active = np.asarray(values["active"], bool)
    hard_reset = np.zeros_like(active)
    _, score, eligible, _ = segment_safe_reset_detector(
        innovation,
        active,
        hard_reset,
        threshold=np.finfo(np.float64).max,
    )
    calibration = conformal_false_reset_threshold(
        score, eligible, alpha=0.01
    )
    calibration["score"] = "accepted_innovation_median_ratio"
    calibration["eligible_from_prior_innovations"] = 1
    return build_segment_safe_threshold_artifact(
        calibration,
        method["split_manifest_sha256"],
        raw["sha256"],
    )


def main() -> None:
    args = build_parser().parse_args()
    artifact = build_artifact(args.source_report)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
