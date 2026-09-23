#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np

from temporal_benchmark_plan import decreasing_isotonic
from temporal_benchmark_protocol import stage_method_names


PERSISTENT = "candidate_persistent"
RESET = "candidate_reset"
MATCHED_K8 = "matched_cold_k8"
def primary_snr_index(protocol: dict, stage: str) -> int:
    name = protocol["statistics"]["primary_snr_point"]
    points = protocol["stages"][stage]["snr_points"]
    if points.count(name) != 1:
        raise ValueError("the primary SNR point is not unique")
    return points.index(name)


def window_slice(bounds: list[int], num_tbs: int) -> slice:
    first, last = (int(value) for value in bounds)
    if not 1 <= first <= last <= int(num_tbs):
        raise ValueError("scoring window is outside the TB horizon")
    return slice(first - 1, last)


def episode_rates(
    errors: np.ndarray,
    active: np.ndarray,
    window: slice,
) -> np.ndarray:
    errors = np.asarray(errors, bool)
    active = np.asarray(active, bool)
    if errors.ndim != 5 or active.ndim != 4:
        raise ValueError("error and activity ranks differ")
    if errors.shape[0] != active.shape[0]:
        raise ValueError("condition axes differ")
    if errors.shape[2:] != active.shape[1:]:
        raise ValueError("episode, TB, or UE axes differ")
    selected_error = errors[:, :, :, window, :]
    selected_active = active[:, None, :, window, :]
    counts = np.sum(selected_error & selected_active, axis=(3, 4))
    blocks = np.sum(selected_active, axis=(3, 4))
    return np.divide(
        counts,
        blocks,
        out=np.zeros_like(counts, dtype=np.float64),
        where=blocks > 0,
    )


def _bootstrap_units(
    values: list[np.ndarray],
    condition_indices: list[int],
    draws: int,
    rng: np.random.RandomState,
) -> tuple[float, np.ndarray, np.ndarray]:
    runs = len(values)
    conditions = len(condition_indices)
    if runs == 0 or conditions == 0:
        raise ValueError("bootstrap requires runs and conditions")
    points = np.zeros((runs, conditions), np.float64)
    sampled = np.zeros((runs, conditions, draws), np.float64)
    for run, value in enumerate(values):
        value = np.asarray(value, np.float64)
        if value.ndim != 3:
            raise ValueError("bootstrap values must be condition, SNR, episode")
        for position, condition in enumerate(condition_indices):
            episodes = np.mean(value[condition], axis=0)
            points[run, position] = float(np.mean(episodes))
            selection = rng.randint(
                0, len(episodes), size=(int(draws), len(episodes))
            )
            sampled[run, position] = np.mean(
                episodes[selection], axis=1
            )
    training_selection = rng.randint(0, runs, size=(int(draws), runs))
    by_draw = np.transpose(sampled, (2, 0, 1))
    chosen = np.take_along_axis(
        by_draw,
        training_selection[:, :, None],
        axis=1,
    )
    mean_samples = np.mean(chosen, axis=(1, 2))
    condition_samples = np.mean(chosen, axis=1)
    return float(np.mean(points)), mean_samples, condition_samples


def paired_summary(
    runs: list[dict[str, np.ndarray]],
    left: str,
    right: str,
    protocol: dict,
    stage: str,
    draws: int,
    rng: np.random.RandomState,
) -> dict:
    tbs = int(protocol["stages"][stage]["num_tbs"])
    window = window_slice(
        protocol["stages"][stage]["scoring_window"], tbs
    )
    condition_names = protocol["stages"][stage]["condition_names"]
    core = [condition_names.index(name) for name in protocol["core_conditions"]]
    differences = []
    snr_index = primary_snr_index(protocol, stage)
    for arrays in runs:
        left_rate = episode_rates(
            arrays[f"errors_{left}"], arrays["active"], window
        )
        right_rate = episode_rates(
            arrays[f"errors_{right}"], arrays["active"], window
        )
        differences.append(
            (left_rate - right_rate)[:, snr_index:snr_index + 1]
        )
    point, samples, condition_samples = _bootstrap_units(
        differences, core, draws, rng
    )
    condition_points = {
        name: float(np.mean([
            np.mean(value[index]) for value in differences
        ]))
        for name, index in zip(protocol["core_conditions"], core)
    }
    worst_samples = np.max(condition_samples, axis=1)
    return {
        "left": left,
        "right": right,
        "difference": point,
        "difference_percentage_points": 100.0 * point,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "condition_differences": condition_points,
        "worst_condition_difference": float(max(condition_points.values())),
        "worst_condition_ci95": [
            float(np.quantile(worst_samples, 0.025)),
            float(np.quantile(worst_samples, 0.975)),
        ],
    }


def method_summary(
    runs: list[dict[str, np.ndarray]],
    method: str,
    protocol: dict,
    stage: str,
) -> dict:
    tbs = int(protocol["stages"][stage]["num_tbs"])
    window = window_slice(
        protocol["stages"][stage]["scoring_window"], tbs
    )
    names = protocol["stages"][stage]["condition_names"]
    snr_index = primary_snr_index(protocol, stage)
    per_condition = {}
    per_condition_all_snr = {}
    for index, name in enumerate(names):
        values = [
            episode_rates(
                arrays[f"errors_{method}"], arrays["active"], window
            )[index]
            for arrays in runs
        ]
        per_condition[name] = float(np.mean([
            value[snr_index] for value in values
        ]))
        per_condition_all_snr[name] = float(np.mean(values))
    core = [per_condition[name] for name in protocol["core_conditions"]]
    return {
        "core_mean_tbler": float(np.mean(core)),
        "per_condition_tbler": per_condition,
        "per_condition_all_snr_mean_tbler": per_condition_all_snr,
    }


def late_drift_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
    draws: int,
    rng: np.random.RandomState,
) -> dict | None:
    tbs = int(protocol["stages"][stage]["num_tbs"])
    if tbs < int(protocol["windows"]["late"][1]):
        return None
    mature = window_slice(protocol["windows"]["mature"], tbs)
    late = window_slice(protocol["windows"]["late"], tbs)
    names = protocol["stages"][stage]["condition_names"]
    stationary = [
        names.index(name) for name in protocol["stationary_conditions"]
    ]
    differences = []
    snr_index = primary_snr_index(protocol, stage)
    for arrays in runs:
        early = episode_rates(
            arrays[f"errors_{PERSISTENT}"], arrays["active"], mature
        )
        final = episode_rates(
            arrays[f"errors_{PERSISTENT}"], arrays["active"], late
        )
        differences.append(
            (final - early)[:, snr_index:snr_index + 1]
        )
    point, samples, _ = _bootstrap_units(
        differences, stationary, draws, rng
    )
    return {
        "definition": "late TBLER minus mature TBLER",
        "difference": point,
        "difference_percentage_points": 100.0 * point,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def episode_outage_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
) -> dict:
    tbs = int(protocol["stages"][stage]["num_tbs"])
    window = window_slice(
        protocol["stages"][stage]["scoring_window"], tbs
    )
    threshold = float(protocol["statistics"]["episode_outage_threshold"])
    names = protocol["stages"][stage]["condition_names"]
    snr_index = primary_snr_index(protocol, stage)
    values = {}
    for index, name in enumerate(names):
        rates = [
            episode_rates(
                arrays[f"errors_{PERSISTENT}"], arrays["active"], window
            )[index]
            for arrays in runs
        ]
        values[name] = float(np.mean([
            np.mean(rate[snr_index] > threshold) for rate in rates
        ]))
    return {
        "threshold": threshold,
        "per_condition": values,
        "core_mean": float(np.mean([
            values[name] for name in protocol["core_conditions"]
        ])),
    }


def snr_crossing(
    snrs: np.ndarray,
    errors: np.ndarray,
    blocks: np.ndarray,
    target: float,
) -> dict:
    snrs = np.asarray(snrs, np.float64)
    errors = np.asarray(errors, np.int64)
    blocks = np.asarray(blocks, np.int64)
    rates = (errors + 0.5) / (blocks + 1.0)
    fitted = decreasing_isotonic(rates, blocks)
    crossing = None
    for index in range(len(snrs) - 1):
        if fitted[index] >= target >= fitted[index + 1]:
            if fitted[index] == fitted[index + 1]:
                crossing = float((snrs[index] + snrs[index + 1]) / 2.0)
            else:
                fraction = (
                    math.log(target) - math.log(fitted[index])
                ) / (math.log(fitted[index + 1]) - math.log(fitted[index]))
                crossing = float(
                    snrs[index]
                    + fraction * (snrs[index + 1] - snrs[index])
                )
            break
    return {
        "snr_at_target_db": crossing,
        "target_bracketed": crossing is not None,
        "raw_tbler": rates.tolist(),
        "isotonic_tbler": fitted.tolist(),
    }


def snr_target_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
) -> dict:
    target = float(protocol["objective"]["target_tbler"])
    tbs = int(protocol["stages"][stage]["num_tbs"])
    window = window_slice(
        protocol["stages"][stage]["scoring_window"], tbs
    )
    result = {}
    for condition_index, name in enumerate(
        protocol["stages"][stage]["condition_names"]
    ):
        snrs = runs[0]["snr_db"][condition_index]
        methods = {}
        for method in (PERSISTENT, MATCHED_K8):
            total_errors = np.zeros(len(snrs), np.int64)
            total_blocks = np.zeros(len(snrs), np.int64)
            for arrays in runs:
                active = arrays["active"][condition_index, :, window]
                for snr_index in range(len(snrs)):
                    error = arrays[f"errors_{method}"][
                        condition_index, snr_index, :, window
                    ]
                    total_errors[snr_index] += int(np.sum(error & active))
                    total_blocks[snr_index] += int(np.sum(active))
            methods[method] = snr_crossing(
                snrs, total_errors, total_blocks, target
            )
        candidate = methods[PERSISTENT]["snr_at_target_db"]
        baseline = methods[MATCHED_K8]["snr_at_target_db"]
        result[name] = {
            "methods": methods,
            "candidate_minus_matched_k8_db": (
                None if candidate is None or baseline is None
                else float(candidate - baseline)
            ),
        }
    return result


def adaptation_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
) -> dict | None:
    tbs = int(next(iter(runs))["active"].shape[2])
    if tbs < 64:
        return None
    name = "controlled_topology_change"
    index = protocol["stages"][stage]["condition_names"].index(name)
    snr_index = primary_snr_index(protocol, stage)
    traces = []
    for arrays in runs:
        active = arrays["active"][index]
        candidate = arrays[f"errors_{PERSISTENT}"][
            index, snr_index:snr_index + 1
        ]
        baseline = arrays[f"errors_{MATCHED_K8}"][
            index, snr_index:snr_index + 1
        ]
        candidate_error = (candidate & active[None]).astype(np.int64)
        baseline_error = (baseline & active[None]).astype(np.int64)
        numerator = np.sum(
            candidate_error - baseline_error, axis=(0, 1, 3)
        )
        denominator = np.sum(active, axis=(0, 2)) * candidate.shape[0]
        traces.append(np.divide(
            numerator,
            denominator,
            out=np.zeros(tbs, np.float64),
            where=denominator > 0,
        ))
    trace = np.mean(traces, axis=0)
    immediate = window_slice(
        protocol["windows"]["immediate_post_change"], tbs
    )
    late = window_slice(protocol["windows"]["late_post_change"], tbs)
    result = {
        "definition": "candidate minus matched-K8 TBLER",
        "per_tb_gap": trace.tolist(),
        "immediate_mean_gap": float(np.mean(trace[immediate])),
        "late_mean_gap": float(np.mean(trace[late])),
        "post_change_excess_area": float(np.sum(trace[immediate.start:])),
    }
    result["schedule_reentry"] = schedule_reentry_summary(
        runs, protocol, stage
    )
    return result


def schedule_reentry_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
    horizon: int = 4,
) -> dict:
    index = protocol["stages"][stage]["condition_names"].index(
        "schedule_gap_reentry"
    )
    snr_index = primary_snr_index(protocol, stage)
    sums = np.zeros(int(horizon), np.float64)
    counts = np.zeros(int(horizon), np.int64)
    event_count = 0
    for arrays in runs:
        active = arrays["active"][index]
        ue_ids = arrays["ue_ids"][index]
        candidate = arrays[f"errors_{PERSISTENT}"][index, snr_index]
        baseline = arrays[f"errors_{MATCHED_K8}"][index, snr_index]
        for episode in range(active.shape[0]):
            seen = set()
            previous = set()
            for tb in range(active.shape[1]):
                current = {
                    int(ue_ids[episode, tb, position])
                    for position in range(active.shape[2])
                    if active[episode, tb, position]
                }
                reentered = (current - previous) & seen
                for ue_id in sorted(reentered):
                    event_count += 1
                    for lag in range(int(horizon)):
                        point_tb = tb + lag
                        if point_tb >= active.shape[1]:
                            break
                        positions = np.flatnonzero(
                            active[episode, point_tb]
                            & (ue_ids[episode, point_tb] == ue_id)
                        )
                        if len(positions) != 1:
                            continue
                        position = int(positions[0])
                        sums[lag] += float(
                            int(candidate[episode, point_tb, position])
                            - int(baseline[episode, point_tb, position])
                        )
                        counts[lag] += 1
                seen |= current
                previous = current
    trace = np.divide(
        sums,
        counts,
        out=np.full(int(horizon), np.nan, np.float64),
        where=counts > 0,
    )
    return {
        "definition": "candidate minus matched-K8 TBLER after UE re-entry",
        "event_count": int(event_count),
        "samples_per_lag": counts.tolist(),
        "per_lag_gap": [None if np.isnan(value) else float(value) for value in trace],
        "excess_area": (
            None if np.all(np.isnan(trace)) else float(np.nansum(trace))
        ),
    }


def latency_summary(
    runs: list[dict[str, np.ndarray]], protocol: dict
) -> dict:
    timing = protocol["execution"]["latency"]
    methods = timing["candidate_methods"] + timing["reference_methods"]
    summaries = {}
    for method in methods:
        values = np.concatenate([
            np.asarray(arrays[f"latency_ms_{method}"], np.float64)
            for arrays in runs
        ])
        summaries[method] = {
            "samples": int(len(values)),
            "p50_ms": float(np.quantile(values, 0.50)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
    candidate = summaries["candidate_persistent"]
    relative = {}
    for method in timing["reference_methods"]:
        reference = summaries[method]
        relative[method] = {
            "p50_ratio": candidate["p50_ms"] / reference["p50_ms"],
            "p95_ratio": candidate["p95_ms"] / reference["p95_ms"],
            "p50_delta_ms": candidate["p50_ms"] - reference["p50_ms"],
            "p95_delta_ms": candidate["p95_ms"] - reference["p95_ms"],
        }
    return {
        **candidate,
        "reference_method": protocol["ranking"]["latency_reference"],
        "methods": summaries,
        "relative_to": relative,
    }


def confirmation_precision_summary(
    protocol: dict,
    stage: str,
    episodes: int,
    k8_gap: dict,
    memory_gain: dict,
) -> dict | None:
    if stage != "confirmation":
        return None
    target = float(protocol["statistics"]["confirmation_ci_half_width"])
    intervals = {
        "worst_core_k8_gap": k8_gap["worst_condition_ci95"],
        "memory_gain": memory_gain["ci95"],
    }
    half_widths = {
        name: 0.5 * (float(bounds[1]) - float(bounds[0]))
        for name, bounds in intervals.items()
    }
    observed = max(half_widths.values())
    schedule = [
        int(value)
        for value in protocol["stages"][stage]["episode_schedule"]
    ]
    remaining = [value for value in schedule if value > int(episodes)]
    target_met = observed <= target
    return {
        "episodes_per_condition": int(episodes),
        "target_ci_half_width": target,
        "half_widths": half_widths,
        "target_met": bool(target_met),
        "maximum_reached": not remaining,
        "next_episode_count": (
            None if target_met or not remaining else remaining[0]
        ),
    }


def validate_run_group(payloads: list[dict], arrays: list[dict], protocol: dict):
    if not payloads or len(payloads) != len(arrays):
        raise ValueError("run group is empty or incomplete")
    manifest = payloads[0]["candidate"]
    stage = payloads[0]["stage"]
    manifest_hash = manifest["manifest_sha256"]
    expected_seeds = set(manifest["training"]["seeds"])
    actual_seeds = set()
    reference_fingerprints = arrays[0]["episode_fingerprint"]
    reference_snr = arrays[0]["snr_db"]
    for payload, values in zip(payloads, arrays):
        if payload["stage"] != stage:
            raise ValueError("candidate stages differ")
        if payload["candidate"]["manifest_sha256"] != manifest_hash:
            raise ValueError("candidate manifests differ")
        seed = int(payload["training_seed"])
        if seed in actual_seeds:
            raise ValueError("duplicate candidate training seed")
        actual_seeds.add(seed)
        if not np.array_equal(
            values["episode_fingerprint"], reference_fingerprints
        ):
            raise ValueError("training seeds saw different episodes")
        if not np.array_equal(values["snr_db"], reference_snr):
            raise ValueError("training seeds saw different SNRs")
    required = int(protocol["stages"][stage]["required_training_seeds"])
    if actual_seeds != expected_seeds or len(actual_seeds) != required:
        raise ValueError("training-seed coverage differs")
    return manifest, stage


def score_run_group(
    payloads: list[dict],
    arrays: list[dict[str, np.ndarray]],
    protocol: dict,
) -> dict:
    manifest, stage = validate_run_group(payloads, arrays, protocol)
    draws = int(protocol["statistics"]["bootstrap_samples"])
    rng = np.random.RandomState(20_261_220)
    k8_gap = paired_summary(
        arrays, PERSISTENT, MATCHED_K8, protocol, stage, draws, rng
    )
    memory_gain = paired_summary(
        arrays, RESET, PERSISTENT, protocol, stage, draws, rng
    )
    result = {
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "stage": stage,
        "training_seeds": sorted(payload["training_seed"] for payload in payloads),
        "methods": {
            method: method_summary(arrays, method, protocol, stage)
            for method in stage_method_names(protocol, stage)
        },
        "comparisons": {
            "candidate_gap_to_matched_k8": k8_gap,
            "memory_gain": memory_gain,
        },
        "late_drift": late_drift_summary(
            arrays, protocol, stage, draws, rng
        ),
        "episode_outage": episode_outage_summary(
            arrays, protocol, stage
        ),
        "adaptation": adaptation_summary(arrays, protocol, stage),
        "latency": latency_summary(arrays, protocol),
        "snr_at_target": (
            None if len(protocol["stages"][stage]["snr_points"]) < 3
            else snr_target_summary(arrays, protocol, stage)
        ),
        "registration": manifest["registration"],
    }
    result["confirmation_precision"] = confirmation_precision_summary(
        protocol,
        stage,
        int(arrays[0]["active"].shape[1]),
        k8_gap,
        memory_gain,
    )
    return result


def pareto_frontier(scores: list[dict]) -> list[str]:
    metrics = {}
    for score in scores:
        gap = score["comparisons"]["candidate_gap_to_matched_k8"]
        metrics[score["candidate_id"]] = (
            gap["worst_condition_difference"],
            gap["difference"],
            -score["comparisons"]["memory_gain"]["difference"],
            _latency_rank_value(score),
        )
    frontier = []
    for candidate, point in metrics.items():
        dominated = False
        for other, comparison in metrics.items():
            if other == candidate:
                continue
            no_worse = all(a <= b for a, b in zip(comparison, point))
            strictly = any(a < b for a, b in zip(comparison, point))
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier)


def rank_frontier(scores: list[dict]) -> list[str]:
    frontier = set(pareto_frontier(scores))

    def key(score):
        gap = score["comparisons"]["candidate_gap_to_matched_k8"]
        return (
            gap["worst_condition_difference"],
            gap["difference"],
            -score["comparisons"]["memory_gain"]["difference"],
            _latency_rank_value(score),
            score["candidate_id"],
        )

    return [
        score["candidate_id"]
        for score in sorted(
            (value for value in scores if value["candidate_id"] in frontier),
            key=key,
        )
    ]


def _latency_rank_value(score: dict) -> float:
    latency = score["latency"]
    reference = latency["reference_method"]
    return float(latency["relative_to"][reference]["p95_ratio"])
