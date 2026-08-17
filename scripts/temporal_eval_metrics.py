#!/usr/bin/env python3
"""Shared statistical helpers for temporal Neural RX evaluation."""

from __future__ import annotations

import math


def make_snr_grid(snr_min: float, snr_max: float, snr_step: float) -> list[float]:
    """Inclusive SNR grid that never overshoots snr_max."""
    snr_min = float(snr_min)
    snr_max = float(snr_max)
    snr_step = float(snr_step)
    if not math.isfinite(snr_min) or not math.isfinite(snr_max):
        raise ValueError("SNR bounds must be finite")
    if not math.isfinite(snr_step) or snr_step <= 0:
        raise ValueError("snr_step must be finite and positive")
    if snr_max < snr_min:
        raise ValueError("snr_max must be >= snr_min")

    out = []
    i = 0
    tol = max(1.0, abs(snr_min), abs(snr_max)) * 1e-10
    while True:
        x = snr_min + i * snr_step
        if x > snr_max + tol:
            break
        out.append(float(round(min(x, snr_max), 10)))
        i += 1
    if not out:
        out = [float(round(snr_min, 10))]
    if out[-1] < snr_max - tol:
        out.append(float(round(snr_max, 10)))
    # Protect against a floating-point duplicate of the endpoint.
    deduped = []
    for x in out:
        if not deduped or abs(x - deduped[-1]) > tol:
            deduped.append(x)
    return deduped


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054):
    """Wilson 95% binomial interval by default."""
    errors = int(errors)
    total = int(total)
    if total <= 0:
        return [None, None]
    if errors < 0 or errors > total:
        raise ValueError("errors must be in [0, total]")
    p = errors / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _interpolation_bler(point: dict, bler_key: str) -> float | None:
    """Positive BLER estimate suitable for log-domain interpolation.

    A measured BLER of exactly zero cannot be log-transformed. Instead of
    dropping that point (which can hide a real target crossing), use a small
    Jeffreys-style continuity correction based on the observed error/block
    counts: (errors + 0.5) / (blocks + 1).
    """
    raw = point.get(bler_key)
    if raw is None:
        return None
    raw = float(raw)
    if not math.isfinite(raw) or raw < 0:
        raise ValueError(f"invalid {bler_key}={raw}")
    if raw > 0:
        return raw

    if bler_key == "bler_tb2plus":
        e_key, n_key = "errors_tb2plus", "blocks_tb2plus"
    elif bler_key == "bler_all":
        e_key, n_key = "errors_all", "blocks_all"
    else:
        return None

    errors = int(point.get(e_key, 0))
    blocks = int(point.get(n_key, 0))
    if blocks <= 0:
        return None
    return (errors + 0.5) / (blocks + 1.0)


def log_bler_crossing(
    points: list[dict],
    target: float = 0.1,
    bler_key: str = "bler_tb2plus",
) -> float | None:
    """Interpolate SNR where a BLER curve crosses target in log-BLER space."""
    target = float(target)
    if not 0.0 < target < 1.0:
        raise ValueError("target BLER must be in (0, 1)")

    prepared = []
    for point in points:
        y = _interpolation_bler(point, bler_key)
        if y is not None:
            prepared.append((float(point["snr_db"]), y))
    prepared.sort()

    for (x0, y0), (x1, y1) in zip(prepared[:-1], prepared[1:]):
        if (y0 - target) * (y1 - target) > 0:
            continue
        if y0 == target:
            return x0
        if y1 == target:
            return x1
        if y0 == y1:
            return (x0 + x1) / 2.0
        frac = (
            math.log10(target) - math.log10(y0)
        ) / (
            math.log10(y1) - math.log10(y0)
        )
        return float(x0 + frac * (x1 - x0))
    return None
