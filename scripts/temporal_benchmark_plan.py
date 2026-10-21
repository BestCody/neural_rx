from __future__ import annotations

import copy
import math

import numpy as np

from temporal_benchmark_scoring import block_error_rate


def decreasing_isotonic(values, weights) -> np.ndarray:
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("isotonic inputs must be matching vectors")
    if len(values) == 0 or np.any(weights <= 0.0):
        raise ValueError("isotonic weights must be positive")
    levels = []
    masses = []
    lengths = []
    for value, weight in zip(values, weights):
        levels.append(float(value))
        masses.append(float(weight))
        lengths.append(1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            mass = masses[-2] + masses[-1]
            level = (
                levels[-2] * masses[-2] + levels[-1] * masses[-1]
            ) / mass
            length = lengths[-2] + lengths[-1]
            levels[-2:] = [level]
            masses[-2:] = [mass]
            lengths[-2:] = [length]
    return np.concatenate([
        np.full(length, level, np.float64)
        for level, length in zip(levels, lengths)
    ])


def snr_at_target(snrs, rates, target):
    snrs, rates = np.asarray(snrs, np.float64), np.asarray(rates, np.float64)
    if snrs.ndim != 1 or snrs.shape != rates.shape or len(snrs) < 2:
        raise ValueError("Threshold interpolation requires matching SNR and TBLER vectors")
    if (not 0 < target < 1 or not np.all(np.isfinite(snrs))
            or not np.all(np.isfinite(rates)) or np.any(np.diff(snrs) <= 0)
            or np.any((rates < 0) | (rates > 1))):
        raise ValueError("Invalid SNR or TBLER values for threshold interpolation")
    for index, (high, low) in enumerate(zip(rates, rates[1:])):
        if high >= target >= low:
            if high == low:
                return float((snrs[index] + snrs[index + 1]) / 2.0)
            fraction = ((math.log(target) - math.log(high)) / (math.log(low) - math.log(high))
                        if low > 0 else (high - target) / (high - low))
            return float(snrs[index] + fraction * (snrs[index + 1] - snrs[index]))
    return None


def calibrate_condition(points: list[dict], target: float) -> dict:
    if len(points) < 3:
        raise ValueError("calibration requires at least three SNR points")
    points = sorted(points, key=lambda value: float(value["snr_db"]))
    snrs = np.asarray([float(value["snr_db"]) for value in points])
    if len(np.unique(snrs)) != len(snrs):
        raise ValueError("calibration SNR values must be unique")
    errors = np.asarray([int(value["errors"]) for value in points])
    blocks = np.asarray([int(value["blocks"]) for value in points])
    if np.any(errors < 0) or np.any(errors > blocks):
        raise ValueError("invalid error count")
    raw = block_error_rate(errors, blocks)
    fitted = decreasing_isotonic(raw, blocks)
    eligible = np.arange(1, len(snrs) - 1)
    distance = np.abs(np.log(np.maximum(fitted[eligible], 1e-12) / target))
    anchor_index = int(eligible[np.argmin(distance)])
    crossing = snr_at_target(snrs, fitted, target)
    return {
        "lower_snr_db": float(snrs[anchor_index - 1]),
        "anchor_snr_db": float(snrs[anchor_index]),
        "upper_snr_db": float(snrs[anchor_index + 1]),
        "anchor_index": anchor_index,
        "target_bracketed": crossing is not None,
        "snr_at_target_db": crossing,
        "interpolation": "logarithmic TBLER; linear when a bracket endpoint is zero",
        "raw_tbler": raw.tolist(),
        "isotonic_tbler": fitted.tolist(),
        "points": copy.deepcopy(points),
    }
