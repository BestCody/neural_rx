#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from temporal_benchmark_candidate import file_sha256
from temporal_benchmark_io import load_run_bundle
from temporal_benchmark_plan import (
    load_calibration_artifact,
    load_stage_plan,
)
from temporal_benchmark_protocol import (
    load_benchmark_protocol,
    stage_accuracy_reference_method_names,
)
from temporal_benchmark_scoring import (
    pareto_frontier,
    rank_frontier,
    score_run_group,
)
from temporal_generalization_protocol import seal_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a comparable Temporal Neural RX leaderboard.')
    parser.add_argument("--group", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "config" / "temporal_benchmark_v1.json",
    )
    return parser


def metric_value(score: dict, metric: str) -> float | None:
    comparisons = score["comparisons"]
    if metric == "mean_k8_gap":
        return comparisons["candidate_gap_to_matched_k8"]["difference"]
    if metric == "worst_core_k8_gap":
        return comparisons["candidate_gap_to_matched_k8"][
            "worst_condition_difference"
        ]
    if metric == "memory_gain":
        return comparisons["memory_gain"]["difference"]
    if metric == "late_drift":
        value = score["late_drift"]
        return None if value is None else value["difference"]
    if metric == "adaptation_excess_area":
        value = score["adaptation"]
        return None if value is None else value["post_change_excess_area"]
    if metric == "episode_outage":
        return score["episode_outage"]["core_mean"]
    if metric == "latency_p95":
        return score["latency"]["p95_ms"]
    raise ValueError("unknown registered metric")


def registration_gate(score: dict, scores: dict[str, dict]) -> dict:
    registration = score["registration"]
    parent_id = registration.get("parent_candidate_id")
    if parent_id is None:
        return {
            "status": "reference_candidate",
            "pass": None,
            "parent_candidate_id": None,
        }
    if parent_id not in scores:
        return {
            "status": "parent_missing",
            "pass": None,
            "parent_candidate_id": parent_id,
        }
    metric = registration["expected_metric"]
    candidate = metric_value(score, metric)
    parent = metric_value(scores[parent_id], metric)
    if candidate is None or parent is None:
        return {
            "status": "metric_unavailable_at_stage",
            "pass": None,
            "parent_candidate_id": parent_id,
            "metric": metric,
        }
    higher_is_better = metric == "memory_gain"
    improvement = candidate - parent if higher_is_better else parent - candidate
    required = float(registration["minimum_useful_improvement"])
    return {
        "status": "evaluated",
        "pass": bool(improvement >= required),
        "parent_candidate_id": parent_id,
        "metric": metric,
        "candidate_value": candidate,
        "parent_value": parent,
        "improvement": improvement,
        "minimum_useful_improvement": required,
    }


def load_group(
    directory: Path,
    protocol: dict,
    calibration: dict,
) -> tuple[dict, list[dict], list[dict]]:
    directory = directory.expanduser().resolve()
    plan = load_stage_plan(
        directory / "stage_plan.json", protocol, calibration
    )
    paths = sorted(directory.glob("runs/*/benchmark_run.json"))
    if not paths:
        raise ValueError(f"no benchmark runs in {directory}")
    payloads = []
    arrays = []
    for path in paths:
        payload, values = load_run_bundle(path, protocol, plan)
        payloads.append(payload)
        arrays.append(values)
    return plan, payloads, arrays


def training_signature(score: dict, payload: dict) -> tuple:
    training = payload["candidate"]["training"]
    return (
        score["stage"],
        training["data_sha256"],
        training["checkpoint_selection_sha256"],
        tuple(training["seeds"]),
        training["examples"],
        training["updates"],
    )


def pairing_signature(
    payloads: list[dict], arrays: list[dict], protocol: dict
) -> dict[str, dict[str, np.ndarray]]:
    signature = {}
    stage = payloads[0]["stage"]
    for payload, values in zip(payloads, arrays):
        seed = str(payload["training_seed"])
        signature[seed] = {
            "snr_db": values["snr_db"],
            "episode_index": values["episode_index"],
            "episode_fingerprint": values["episode_fingerprint"],
            "active": values["active"],
            "ue_ids": values["ue_ids"],
            "channel_change_event": values["channel_change_event"],
            **{
                f"errors_{method}": values[f"errors_{method}"]
                for method in stage_accuracy_reference_method_names(
                    protocol, stage
                )
            },
            **{
                f"latency_ms_{method}": values[f"latency_ms_{method}"]
                for method in protocol["execution"]["latency"][
                    "reference_methods"
                ]
            },
        }
    return signature


def validate_pairing_signatures(reference: dict, current: dict) -> None:
    if set(current) != set(reference):
        raise ValueError("candidate groups use different training seeds")
    for seed, values in current.items():
        for name, value in values.items():
            if not np.array_equal(value, reference[seed][name]):
                raise ValueError(
                    "candidate groups differ in paired array "
                    f"{name} for training seed {seed}"
                )


def write_csv(path: Path, scores: list[dict], frontier: set[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "candidate_id",
            "stage",
            "mean_k8_gap",
            "worst_core_k8_gap",
            "memory_gain",
            "late_drift",
            "episode_outage",
            "latency_p95_ms",
            "latency_p95_over_matched_k3",
            "pareto",
            "registration_gate",
        ])
        for score in scores:
            writer.writerow([
                score["candidate_id"],
                score["stage"],
                metric_value(score, "mean_k8_gap"),
                metric_value(score, "worst_core_k8_gap"),
                metric_value(score, "memory_gain"),
                metric_value(score, "late_drift"),
                metric_value(score, "episode_outage"),
                metric_value(score, "latency_p95"),
                score["latency"]["relative_to"]["matched_cold_k3"][
                    "p95_ratio"
                ],
                score["candidate_id"] in frontier,
                score["registration_gate"]["pass"],
            ])


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_benchmark_protocol(args.protocol)
    calibration = load_calibration_artifact(args.calibration, protocol)
    scores = []
    payload_by_id = {}
    plan_hashes = {}
    pairing_reference = None
    for group in args.group:
        plan, payloads, arrays = load_group(group, protocol, calibration)
        current_pairing = pairing_signature(payloads, arrays, protocol)
        if pairing_reference is None:
            pairing_reference = current_pairing
        else:
            validate_pairing_signatures(pairing_reference, current_pairing)
        score = score_run_group(payloads, arrays, protocol)
        candidate_id = score["candidate_id"]
        if candidate_id in payload_by_id:
            raise ValueError("duplicate leaderboard candidate")
        payload_by_id[candidate_id] = payloads[0]
        plan_hashes[candidate_id] = plan["plan_sha256"]
        scores.append(score)
    signatures = {
        training_signature(score, payload_by_id[score["candidate_id"]])
        for score in scores
    }
    if len(signatures) != 1:
        raise ValueError("incompatible training protocols share a leaderboard")
    score_lookup = {score["candidate_id"]: score for score in scores}
    for score in scores:
        score["registration_gate"] = registration_gate(score, score_lookup)
    frontier = set(pareto_frontier(scores))
    ranking = rank_frontier(scores)
    result = seal_payload({
        "schema_version": 1,
        "leaderboard": "temporal_benchmark_leaderboard_v1",
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "stage": scores[0]["stage"],
        "candidate_plan_sha256": plan_hashes,
        "pareto_frontier": sorted(frontier),
        "frontier_ranking": ranking,
        "scores": scores,
    }, field="leaderboard_sha256")
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "leaderboard.csv"
    write_csv(csv_path, scores, frontier)
    result["csv"] = {
        "file": csv_path.name,
        "sha256": file_sha256(csv_path),
    }
    result = seal_payload(result, field="leaderboard_sha256")
    output = destination / "leaderboard.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "leaderboard": str(output),
        "pareto_frontier": sorted(frontier),
        "frontier_ranking": ranking,
    }))


if __name__ == "__main__":
    main()
