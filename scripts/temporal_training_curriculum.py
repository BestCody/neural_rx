#!/usr/bin/env python3
"""Select balanced receiver states for temporal training."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


STRATA = ("cold", "return", "partner_change", "rebuild", "mature")


class TrainingTarget(NamedTuple):
    """Identify one event-relative optimizer update."""

    stratum: str
    tb: int


def _active_set(ids: np.ndarray, active: np.ndarray, tb: int) -> frozenset:
    return frozenset(
        int(value)
        for value, enabled in zip(ids[tb], active[tb])
        if bool(enabled)
    )


def event_relative_candidates(
    ue_ids: np.ndarray,
    active: np.ndarray,
    reset_mask: np.ndarray,
    history_valid: np.ndarray,
) -> dict[str, tuple[int, ...]]:
    """Classify causal TBs without exposing labels to the receiver."""
    ue_ids = np.asarray(ue_ids, np.int64)
    active = np.asarray(active, bool)
    reset_mask = np.asarray(reset_mask, bool)
    history_valid = np.asarray(history_valid, bool)
    if ue_ids.ndim != 2 or active.shape != ue_ids.shape:
        raise ValueError("schedule arrays must have shape [TB, position]")
    if reset_mask.shape != ue_ids.shape or history_valid.shape != ue_ids.shape:
        raise ValueError("memory masks must match the schedule")
    if np.any(ue_ids < 0):
        raise ValueError("physical UE IDs cannot be negative")
    if any(len(np.unique(row)) != len(row) for row in ue_ids):
        raise ValueError("scheduled identities must be unique")

    candidates = {name: [] for name in STRATA}
    seen = set()
    previous = frozenset()
    for tb in range(ue_ids.shape[0]):
        current = _active_set(ue_ids, active, tb)
        new = current - seen
        returning = (current & seen) - previous
        continuing = current & previous
        if new:
            candidates["cold"].append(tb)
        if returning and continuing:
            candidates["return"].append(tb)
        if (
            tb > 0
            and len(current) >= 2
            and continuing
            and current != previous
        ):
            candidates["partner_change"].append(tb)
        if tb > 0 and np.any(reset_mask[tb] & active[tb]):
            candidates["rebuild"].append(tb)
        if (
            tb > 0
            and len(current) >= 2
            and current == previous
            and all(
                history_valid[tb, position]
                for position, identity in enumerate(ue_ids[tb])
                if identity in current and active[tb, position]
            )
        ):
            candidates["mature"].append(tb)
        seen.update(current)
        previous = current
    return {name: tuple(values) for name, values in candidates.items()}


def select_event_relative_targets(
    ue_ids: np.ndarray,
    active: np.ndarray,
    reset_mask: np.ndarray,
    history_valid: np.ndarray,
    rng,
) -> tuple[TrainingTarget, ...] | None:
    """Choose one deterministic random target from every stratum."""
    candidates = event_relative_candidates(
        ue_ids, active, reset_mask, history_valid
    )
    if any(not candidates[name] for name in STRATA):
        return None
    observed = {
        int(identity)
        for row, mask in zip(np.asarray(ue_ids), np.asarray(active, bool))
        for identity, enabled in zip(row, mask)
        if enabled
    }
    if len(observed) < 3:
        return None
    return tuple(
        TrainingTarget(name, int(rng.choice(candidates[name])))
        for name in STRATA
    )


def select_unique_event_relative_targets(
    ue_ids: np.ndarray,
    active: np.ndarray,
    reset_mask: np.ndarray,
    history_valid: np.ndarray,
    rng,
    difficulty_scores: np.ndarray | None = None,
) -> tuple[TrainingTarget, ...] | None:
    """Choose one distinct TB for every receiver stratum."""
    candidates = event_relative_candidates(
        ue_ids, active, reset_mask, history_valid
    )
    if any(not candidates[name] for name in STRATA):
        return None
    observed = {
        int(identity)
        for row, mask in zip(np.asarray(ue_ids), np.asarray(active, bool))
        for identity, enabled in zip(row, mask)
        if enabled
    }
    if len(observed) < 3:
        return None
    if difficulty_scores is not None:
        scores = np.asarray(difficulty_scores, np.float64)
        if scores.shape != (ue_ids.shape[0],):
            raise ValueError("difficulty scores must match TB count")
        if not np.all(np.isfinite(scores)):
            raise ValueError("difficulty scores must be finite")
    else:
        scores = None
    shuffled = {}
    for name in STRATA:
        values = [int(value) for value in candidates[name]]
        tie_break = rng.random_sample(len(values))
        if scores is None:
            order = np.argsort(tie_break)
        else:
            order = np.lexsort((tie_break, -scores[values]))
        shuffled[name] = tuple(values[index] for index in order)
    order = sorted(STRATA, key=lambda name: len(shuffled[name]))
    assignment = {}

    def assign(index, used):
        if index == len(order):
            return True
        name = order[index]
        for tb in shuffled[name]:
            if tb in used:
                continue
            assignment[name] = tb
            if assign(index + 1, used | {tb}):
                return True
        assignment.pop(name, None)
        return False

    if not assign(0, set()):
        return None
    return tuple(
        TrainingTarget(name, assignment[name]) for name in STRATA
    )
