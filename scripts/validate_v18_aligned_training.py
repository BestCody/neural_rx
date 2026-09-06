#!/usr/bin/env python3
"""Validate one matched-K=8 and aligned-V18 training pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from temporal_benchmark_candidate import file_sha256
from temporal_training_curriculum import STRATA


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def load_summary(path: Path) -> tuple[dict, Path]:
    """Load one training summary from an exact path."""
    path = path.expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8")), path


def artifact_matches(summary: dict, field: str) -> bool:
    """Check one path against its adjacent SHA-256 field."""
    path = Path(summary[field]).expanduser().resolve()
    return path.is_file() and file_sha256(path) == summary[f"{field}_sha256"]


def balanced(summary: dict) -> bool:
    """Require exactly one equal positive count per target stratum."""
    coverage = summary.get("stratum_coverage", {})
    return tuple(coverage) == STRATA and len(set(coverage.values())) == 1 and (
        next(iter(coverage.values()), 0) > 0
    )


def finite_history(summary: dict) -> bool:
    """Require finite positive logged gradients and finite losses."""
    return bool(summary.get("history")) and all(
        math.isfinite(float(row[name]))
        for row in summary["history"]
        for name in ("loss", "truth_loss", "distill_loss", "gradient_norm")
    ) and all(
        float(row["gradient_norm"]) > 0.0 for row in summary["history"]
    )


def validate_pair(matched: dict, candidate: dict, matched_path: Path) -> dict:
    """Return named checks for one dependency-ordered training pair."""
    training_name = matched.get("training")
    version = {
        "temporal_benchmark_training_v2": 2,
        "temporal_benchmark_training_v3": 3,
    }.get(training_name)
    same_fields = (
        "stage",
        "smoke_test",
        "training_seed",
        "training_updates",
        "training_examples",
        "training_data_sha256",
        "training_recipe_sha256",
        "training_source_sha256",
        "benchmark_protocol_sha256",
        "split_manifest_sha256",
        "threshold_artifact_sha256",
    )
    checks = {
        "training_types": (
            version is not None
            and candidate.get("training") == training_name
        ),
        "roles": (
            matched.get("role") == "matched_cold_k8"
            and candidate.get("role") == "candidate"
        ),
        "paired_contract": all(
            matched.get(name) == candidate.get(name) for name in same_fields
        ),
        "paired_data": (
            matched.get("data_fingerprint")
            == candidate.get("data_fingerprint")
        ),
        "paired_generation": (
            matched.get("generated_episodes")
            == candidate.get("generated_episodes")
        ),
        "paired_episode_manifest": (
            matched.get("training_episode_manifest_sha256")
            == candidate.get("training_episode_manifest_sha256")
        ),
        "matched_checkpoint_intact": artifact_matches(
            matched, "full_checkpoint"
        ),
        "candidate_checkpoint_intact": artifact_matches(
            candidate, "full_checkpoint"
        ),
        "candidate_selection_intact": artifact_matches(
            candidate, "selected_checkpoint"
        ),
        "episode_manifest_intact": artifact_matches(
            matched, "training_episode_manifest"
        ),
        "teacher_checkpoint_exact": (
            candidate.get("teacher_checkpoint_sha256")
            == matched.get("full_checkpoint_sha256")
        ),
        "teacher_values_exact": (
            candidate.get("teacher_value_checksum")
            == matched.get("final_value_checksum")
        ),
        "teacher_signature_exact": (
            candidate.get("teacher_weight_signature")
            == matched.get("final_weight_signature")
        ),
        "teacher_summary_exact": (
            candidate.get("teacher_summary_sha256")
            == file_sha256(matched_path)
        ),
        "balanced_matched_strata": balanced(matched),
        "balanced_candidate_strata": balanced(candidate),
        "at_least_three_identities": min(
            int(matched.get("accepted_identity_count_minimum", 0)),
            int(candidate.get("accepted_identity_count_minimum", 0)),
        ) >= 3,
        "candidate_k2_variables": (
            candidate.get("selected_variable_count") == 39
            and candidate.get("selected_scalar_count") == 134_594
        ),
        "finite_matched_training": finite_history(matched),
        "finite_candidate_training": finite_history(candidate),
    }
    if version >= 3:
        matched_power = matched.get("power_coverage", {})
        candidate_power = candidate.get("power_coverage", {})
        checks.update({
            "unique_targets": (
                matched.get("duplicate_target_updates") == 0
                and candidate.get("duplicate_target_updates") == 0
            ),
            "power_coverage_paired": matched_power == candidate_power,
            "weak_users_present": (
                float(matched_power.get(
                    "maximum_assigned_attenuation_db", 0.0
                )) >= 9.0
            ),
            "teacher_warm_start": (
                candidate.get("warm_start_source_checksum")
                == candidate.get("warm_start_target_checksum")
                and candidate.get("warm_start_source_checksum") is not None
            ),
        })
    return checks


def main() -> None:
    """Validate and optionally write the audit."""
    args = build_parser().parse_args()
    matched, matched_path = load_summary(args.matched_summary)
    candidate, _ = load_summary(args.candidate_summary)
    checks = validate_pair(matched, candidate, matched_path)
    result = {
        "schema_version": 1,
        "validation": "v18_aligned_training_pair",
        "all_pass": all(checks.values()),
        "checks": checks,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
