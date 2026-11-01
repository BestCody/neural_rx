from __future__ import annotations

import numpy as np

PERSISTENT = "candidate_persistent"
REFERENCE_K8 = "frozen_cold_k8"
TBLER_DEFINITION = "failed_active_user_blocks / evaluated_active_user_blocks"


def payload_error_flags(hard_bits, crc_status, truth, active):
    hard, truth = np.asarray(hard_bits, bool), np.asarray(truth, bool)
    crc, active = np.asarray(crc_status, bool), np.asarray(active, bool)
    if hard.ndim < 1 or hard.shape != truth.shape:
        raise ValueError("Decoded and transmitted payload shapes differ")
    if crc.shape != hard.shape[:-1] or active.shape != crc.shape:
        raise ValueError("CRC and activity shapes must match payload blocks")
    errors = np.any(hard != truth, axis=-1) & active
    disagreement = np.logical_xor(errors, ~crc) & active
    return errors, disagreement


def crc_diagnostics(errors, disagreement):
    errors, disagreement = np.asarray(errors, bool), np.asarray(disagreement, bool)
    if errors.shape != disagreement.shape:
        raise ValueError("CRC and payload-error shapes differ")
    return {
        "undetected_payload_errors": int(np.sum(errors & disagreement)),
        "crc_failures_with_correct_payload": int(np.sum(~errors & disagreement)),
    }


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


def block_error_rate(errors, blocks, *, allow_empty=False):
    errors, blocks = np.broadcast_arrays(np.asarray(errors, np.float64), np.asarray(blocks))
    if np.any(blocks < 0) or (not allow_empty and np.any(blocks == 0)):
        raise ValueError("Scoring window contains no active user blocks")
    return np.divide(errors, blocks, out=np.full(errors.shape, np.nan), where=blocks > 0)


def episode_counts(
    errors: np.ndarray,
    active: np.ndarray,
    window: slice,
) -> tuple[np.ndarray, np.ndarray]:
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
    return counts, np.broadcast_to(blocks, counts.shape)


def pooled_tbler(errors, active, window):
    counts, blocks = episode_counts(errors, active, window)
    return block_error_rate(counts.sum(axis=-1), blocks.sum(axis=-1))


def _bootstrap_counts(
    values: list[tuple[np.ndarray, np.ndarray]],
    condition_indices: list[int],
    draws: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    runs = len(values)
    conditions = len(condition_indices)
    if runs == 0 or conditions == 0 or draws < 1:
        raise ValueError("bootstrap requires runs, conditions, and draws")
    selected = []
    for counts, blocks in values:
        if counts.ndim != 3 or blocks.shape != counts.shape or counts.shape[-1] == 0:
            raise ValueError("bootstrap counts must be term, condition, episode")
        selected.append((counts[:, condition_indices], blocks[:, condition_indices]))
    counts = sum(value.sum(axis=-1) for value, _ in selected)
    blocks = sum(value.sum(axis=-1) for _, value in selected)
    points = block_error_rate(counts, blocks).sum(axis=0)
    sampled_counts = np.zeros((int(draws), *counts.shape), np.float64)
    sampled_blocks = np.zeros_like(sampled_counts)
    for _ in range(runs):
        chosen_runs = rng.randint(0, runs, size=int(draws))
        for run, (numerator, denominator) in enumerate(selected):
            rows = np.flatnonzero(chosen_runs == run)
            if len(rows) == 0:
                continue
            episodes = numerator.shape[-1]
            for condition in range(conditions):
                selection = rng.randint(0, episodes, size=(len(rows), episodes))
                sampled_counts[rows, :, condition] += numerator[:, condition][:, selection].sum(axis=-1).T
                sampled_blocks[rows, :, condition] += denominator[:, condition][:, selection].sum(axis=-1).T
    valid = np.all(sampled_blocks > 0, axis=(1, 2))
    if not np.any(valid):
        raise ValueError("No bootstrap samples contain active blocks in every scoring window")
    samples = block_error_rate(sampled_counts[valid], sampled_blocks[valid]).sum(axis=1)
    pooled = float(block_error_rate(counts.sum(axis=-1), blocks.sum(axis=-1)).sum())
    pooled_samples = block_error_rate(sampled_counts[valid].sum(axis=-1),
                                     sampled_blocks[valid].sum(axis=-1)).sum(axis=-1)
    return points, samples, pooled, pooled_samples


def _difference_counts(arrays, left, right, window, snr_index):
    counts, blocks = episode_counts(arrays[f"errors_{left}"], arrays["active"], window)
    if right is not None:
        other, _ = episode_counts(arrays[f"errors_{right}"], arrays["active"], window)
        counts = counts - other
    return counts[:, snr_index], blocks[:, snr_index]


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
        counts, blocks = _difference_counts(arrays, left, right, window, snr_index)
        differences.append((counts[None], blocks[None]))
    points, condition_samples, point, samples = _bootstrap_counts(
        differences, core, draws, rng
    )
    condition_points = dict(zip(protocol["core_conditions"], map(float, points)))
    worst_samples = np.max(condition_samples, axis=1)
    return {
        "left": left,
        "right": right,
        "difference": point,
        "difference_percentage_points": 100.0 * point,
        "bootstrap_valid_samples": len(samples),
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
    counts = [episode_counts(arrays[f"errors_{method}"], arrays["active"], window)
              for arrays in runs]
    failures = sum(value.sum(axis=-1) for value, _ in counts)
    blocks = sum(value.sum(axis=-1) for _, value in counts)
    values = block_error_rate(failures, blocks)
    for index, name in enumerate(names):
        per_condition[name] = float(values[index, snr_index])
    core = [names.index(name) for name in protocol["core_conditions"]]
    core_errors = int(failures[core, snr_index].sum())
    core_blocks = int(blocks[core, snr_index].sum())
    return {
        "core_tbler": float(block_error_rate(core_errors, core_blocks)),
        "core_errors": core_errors,
        "core_blocks": core_blocks,
        "per_condition_tbler": per_condition,
    }


def late_drift_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
    draws: int,
    rng: np.random.RandomState,
    reference: str | None = None,
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
    if not stationary:
        return None
    differences = []
    snr_index = primary_snr_index(protocol, stage)
    for arrays in runs:
        early, early_blocks = _difference_counts(arrays, PERSISTENT, reference, mature, snr_index)
        final, final_blocks = _difference_counts(arrays, PERSISTENT, reference, late, snr_index)
        differences.append((np.stack((final, -early)), np.stack((final_blocks, early_blocks))))
    _, _, point, samples = _bootstrap_counts(
        differences, stationary, draws, rng
    )
    return {
        "definition": "late TBLER minus mature TBLER" if reference is None else f"late minus mature persistent-vs-{reference} TBLER gap",
        "difference": point,
        "difference_percentage_points": 100.0 * point,
        "bootstrap_valid_samples": len(samples),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def adaptation_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
    reference: str = REFERENCE_K8,
) -> dict | None:
    tbs = int(next(iter(runs))["active"].shape[2])
    if tbs < 64:
        return None
    name = "controlled_topology_change"
    names = protocol["stages"][stage]["condition_names"]
    if name not in names:
        return None
    index = names.index(name)
    snr_index = primary_snr_index(protocol, stage)
    failures = np.zeros(tbs, np.int64)
    blocks = np.zeros(tbs, np.int64)
    for arrays in runs:
        active = arrays["active"][index]
        candidate = arrays[f"errors_{PERSISTENT}"][
            index, snr_index:snr_index + 1
        ]
        baseline = arrays[f"errors_{reference}"][
            index, snr_index:snr_index + 1
        ]
        candidate_error = (candidate & active[None]).astype(np.int64)
        baseline_error = (baseline & active[None]).astype(np.int64)
        numerator = np.sum(
            candidate_error - baseline_error, axis=(0, 1, 3)
        )
        denominator = np.sum(active, axis=(0, 2)) * candidate.shape[0]
        failures += numerator
        blocks += denominator
    trace = block_error_rate(failures, blocks, allow_empty=True)
    immediate = window_slice(
        protocol["windows"]["immediate_post_change"], tbs
    )
    late = window_slice(protocol["windows"]["late_post_change"], tbs)
    result = {
        "definition": f"candidate minus {reference} TBLER",
        "per_tb_gap": [None if np.isnan(value) else float(value) for value in trace],
        "immediate_mean_gap": float(block_error_rate(failures[immediate].sum(), blocks[immediate].sum())),
        "late_mean_gap": float(block_error_rate(failures[late].sum(), blocks[late].sum())),
        "post_change_excess_area": (float(np.sum(trace[immediate.start:]))
                                    if np.all(blocks[immediate.start:] > 0) else None),
    }
    result["schedule_reentry"] = schedule_reentry_summary(
        runs, protocol, stage, reference=reference
    )
    return result


def schedule_reentry_summary(
    runs: list[dict[str, np.ndarray]],
    protocol: dict,
    stage: str,
    horizon: int = 4,
    reference: str = REFERENCE_K8,
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
        baseline = arrays[f"errors_{reference}"][index, snr_index]
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
    trace = block_error_rate(sums, counts, allow_empty=True)
    return {
        "definition": f"candidate minus {reference} TBLER after UE re-entry",
        "event_count": int(event_count),
        "samples_per_lag": counts.tolist(),
        "per_lag_gap": [None if np.isnan(value) else float(value) for value in trace],
        "excess_area": (
            None if np.all(np.isnan(trace)) else float(np.nansum(trace))
        ),
    }
