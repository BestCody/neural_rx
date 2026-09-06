#!/usr/bin/env python3
"""Causal UE-routed PD-L8 channel memory."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from pilot_domain_channel import (
    cdm_average_raw_pilots,
    estimate_cdm_noise_variance,
    extract_pilot_statistics,
    extract_raw_pilot_estimates,
    interpolate_pilot_statistics,
    interpolate_pilot_sequence,
    pilot_domain_wiener_estimate,
)


class PilotMemorySequence(NamedTuple):
    """Hold causal PD-L8 estimates and reset metadata."""

    posterior_channel: np.ndarray
    posterior_variance: np.ndarray
    observation_variance: np.ndarray
    reset_mask: np.ndarray
    history_valid: np.ndarray
    innovation: np.ndarray
    innovation_score: np.ndarray
    innovation_score_valid: np.ndarray
    pilots: np.ndarray
    pilot_error: np.ndarray


class PilotMemoryPoint(NamedTuple):
    """Hold one online PD-L8 estimate and its reset state."""

    posterior_channel: np.ndarray
    posterior_variance: np.ndarray
    observation_variance: np.ndarray
    reset_mask: np.ndarray
    history_valid: np.ndarray


class SegmentSafeDecision(NamedTuple):
    """Describe one causal guard transition."""

    reset: bool
    history_valid: bool
    score: float
    score_valid: bool
    reason: str


class SegmentSafeDMRSGuard:
    """Guard history before each use."""

    def __init__(self, threshold: float, history_window: int = 8):
        self.threshold = float(threshold)
        self.history_window = int(history_window)
        if self.threshold <= 0.0:
            raise ValueError("reset threshold must be positive")
        if self.history_window < 1:
            raise ValueError("history window must be positive")
        self.clear()

    def clear(self) -> None:
        """Clear the local innovation baseline."""
        self.innovations = deque(maxlen=self.history_window)

    def update(
        self,
        innovation: float | None,
        *,
        active: bool,
        consecutive: bool,
        external_reset: bool = False,
    ) -> SegmentSafeDecision:
        """Advance one observable guard transition."""
        if not active:
            self.clear()
            return SegmentSafeDecision(False, False, 0.0, False, "inactive")
        if external_reset:
            self.clear()
            return SegmentSafeDecision(True, False, 0.0, False, "external")
        if not consecutive:
            self.clear()
            return SegmentSafeDecision(False, False, 0.0, False, "cold")
        if innovation is None or not np.isfinite(float(innovation)):
            raise ValueError("consecutive transition needs finite innovation")
        current = float(innovation)
        if current < 0.0:
            raise ValueError("innovation cannot be negative")
        if not self.innovations:
            self.innovations.append(current)
            return SegmentSafeDecision(True, False, 0.0, False, "probation")
        baseline = max(float(np.median(self.innovations)), 1e-12)
        score = current / baseline
        if not np.isfinite(score):
            raise FloatingPointError("DMRS guard score is not finite")
        if score > self.threshold:
            self.clear()
            return SegmentSafeDecision(True, False, score, True, "change")
        self.innovations.append(current)
        return SegmentSafeDecision(False, True, score, True, "accepted")


class SegmentSafeTrackPoint(NamedTuple):
    """Expose one identity-safe memory transition."""

    pilots: tuple[np.ndarray, ...]
    errors: tuple[np.ndarray, ...]
    decision: SegmentSafeDecision
    innovation: float


def normalized_pilot_innovation(
    current: np.ndarray,
    current_error: np.ndarray,
    previous: np.ndarray,
    previous_error: np.ndarray,
) -> float:
    """Measure one normalized change between DMRS observations."""
    difference = np.square(np.abs(current - previous))
    noise = np.maximum(current_error + previous_error, 1e-12)
    value = float(np.mean(difference / noise))
    if not np.isfinite(value):
        raise FloatingPointError("DMRS innovation is not finite")
    if value < 0.0:
        raise FloatingPointError("DMRS innovation cannot be negative")
    return value


class SegmentSafePilotBank:
    """Apply the shared Segment-Safe transition by opaque UE ID."""

    def __init__(
        self,
        threshold: float,
        maximum_history_tbs: int = 8,
        guard_history_tbs: int = 8,
    ):
        self.threshold = float(threshold)
        self.maximum_history_tbs = int(maximum_history_tbs)
        self.guard_history_tbs = int(guard_history_tbs)
        if self.threshold <= 0.0:
            raise ValueError("reset threshold must be positive")
        if self.maximum_history_tbs < 1 or self.guard_history_tbs < 1:
            raise ValueError("memory windows must be positive")
        self.clear()

    def clear(self) -> None:
        """Clear all physical-UE tracks."""
        self.tracks = {}
        self.seen_ids = set()

    def state_is_empty(self) -> bool:
        """Report whether the bank contains an active track."""
        return not self.tracks

    def _new_track(self) -> dict:
        return {
            "pilots": deque(maxlen=self.maximum_history_tbs),
            "errors": deque(maxlen=self.maximum_history_tbs),
            "guard": SegmentSafeDMRSGuard(
                self.threshold, self.guard_history_tbs
            ),
            "last_seen": -1,
        }

    def update(
        self,
        tb_index: int,
        ue_id: int,
        pilot: np.ndarray,
        error: np.ndarray,
        *,
        active: bool,
        external_reset: bool = False,
    ) -> SegmentSafeTrackPoint:
        """Advance one scheduled identity without exposing its ID."""
        tb_index = int(tb_index)
        ue_id = int(ue_id)
        pilot = np.asarray(pilot)
        error = np.asarray(error)
        if tb_index < 0 or ue_id < 0:
            raise ValueError("TB indices and UE IDs cannot be negative")
        if pilot.shape != error.shape:
            raise ValueError("pilot and error shapes must match")
        if not np.all(np.isfinite(pilot)):
            raise FloatingPointError("pilot observation is not finite")
        if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
            raise FloatingPointError("pilot error must be finite and positive")
        if not active:
            self.tracks.pop(ue_id, None)
            self.seen_ids.add(ue_id)
            decision = SegmentSafeDecision(
                False, False, 0.0, False, "inactive"
            )
            return SegmentSafeTrackPoint((pilot,), (error,), decision, 0.0)

        track = self.tracks.get(ue_id)
        consecutive = bool(
            track is not None and int(track["last_seen"]) == tb_index - 1
        )
        reentry = bool(ue_id in self.seen_ids and not consecutive)
        if not consecutive:
            track = self._new_track()
            innovation = 0.0
            decision = track["guard"].update(
                None,
                active=True,
                consecutive=False,
                external_reset=bool(external_reset or reentry),
            )
        else:
            innovation = normalized_pilot_innovation(
                pilot,
                error,
                track["pilots"][-1],
                track["errors"][-1],
            )
            decision = track["guard"].update(
                innovation,
                active=True,
                consecutive=True,
                external_reset=bool(external_reset),
            )
        if decision.reset:
            track["pilots"].clear()
            track["errors"].clear()
        track["pilots"].append(pilot.copy())
        track["errors"].append(error.copy())
        track["last_seen"] = tb_index
        self.tracks[ue_id] = track
        self.seen_ids.add(ue_id)
        return SegmentSafeTrackPoint(
            tuple(track["pilots"]),
            tuple(track["errors"]),
            decision,
            innovation,
        )


class _OnlinePilotDomainMemoryBase:
    """Share online pilot extraction and posterior construction."""

    def __init__(
        self,
        receiver,
        parameters,
        threshold: float,
        tf,
        maximum_history_tbs: int = 8,
    ):
        self.receiver = receiver
        self.parameters = parameters
        self.threshold = float(threshold)
        self.tf = tf
        self.maximum_history_tbs = int(maximum_history_tbs)
        if self.threshold <= 0.0:
            raise ValueError("reset threshold must be positive")
        if self.maximum_history_tbs < 1:
            raise ValueError("memory horizon must be positive")
        self.start_episode()

    def start_episode(self) -> None:
        """Clear identities and restart episode position."""
        self._tracks = {}
        self._seen_ids = set()
        self._tb_index = -1
        self.last_point = None

    def reset_memory(self) -> None:
        """Clear identities without changing episode position."""
        self._tracks = {}
        self._seen_ids = set()
        self.last_point = None

    def state_is_empty(self) -> bool:
        """Report whether any UE history remains."""
        return not self._tracks

    def _set_slot_pilots(self, slot_number: int) -> None:
        pilots = self.tf.gather(
            self.parameters.pilots, int(slot_number), axis=0
        )
        self.receiver._ls_est._pilot_pattern.pilots = pilots

    def update(self, inputs: dict) -> PilotMemoryPoint:
        """Consume one observable TB and return scheduled PD-L8 inputs."""
        self._tb_index += 1
        y = np.asarray(inputs["received_grid"])
        active = np.asarray(inputs["active"], bool)
        ue_ids = np.asarray(inputs["ue_ids"], np.int64)
        if y.ndim != 4 or active.ndim != 1 or ue_ids.shape != active.shape:
            raise ValueError("online PD-L8 input shapes differ")
        if len(set(ue_ids.tolist())) != len(ue_ids):
            raise ValueError("scheduled UE identities must be unique")
        self._set_slot_pilots(int(np.asarray(inputs["slot_number"])))
        raw, layout = extract_raw_pilot_estimates(
            self.receiver._ls_est,
            self.tf.convert_to_tensor(y[None]),
            len(active),
            self.tf,
        )
        pilots = cdm_average_raw_pilots(raw)
        errors, _ = estimate_cdm_noise_variance(raw)
        posterior = np.empty_like(pilots)
        posterior_error = np.empty_like(errors)
        reset_mask = np.zeros(len(active), bool)
        history_valid = np.zeros(len(active), bool)
        for position, ue_id in enumerate(ue_ids.tolist()):
            history, history_error, reset, valid = self._update_track(
                int(ue_id),
                pilots[0, position],
                errors[0, position],
                bool(active[position]),
            )
            sequence = np.asarray(history)[None, :, None]
            sequence_error = np.asarray(history_error)[None, :, None]
            estimate, variance, _ = pilot_domain_wiener_estimate(
                sequence,
                sequence_error,
                max_history=min(self.maximum_history_tbs, len(history)),
            )
            posterior[0, position] = estimate[0, -1, 0]
            posterior_error[0, position] = variance[0, -1, 0]
            reset_mask[position] = reset
            history_valid[position] = valid
        posterior_grid, posterior_grid_error = interpolate_pilot_statistics(
            self.receiver._ls_est,
            posterior,
            posterior_error,
            layout,
            self.tf,
        )
        _, observation_grid_error = interpolate_pilot_statistics(
            self.receiver._ls_est,
            pilots,
            errors,
            layout,
            self.tf,
        )
        self.last_point = PilotMemoryPoint(
            complex_to_stacked(posterior_grid),
            posterior_grid_error.astype(np.float32),
            observation_grid_error.astype(np.float32),
            reset_mask[None],
            history_valid[None],
        )
        return self.last_point


class OnlineSegmentSafePilotDomainMemory(_OnlinePilotDomainMemoryBase):
    """Maintain segment-safe online PD-L8 state."""

    def start_episode(self) -> None:
        """Start one episode with the shared transition bank."""
        self._bank = SegmentSafePilotBank(
            self.threshold, self.maximum_history_tbs
        )
        self._tracks = self._bank.tracks
        self._seen_ids = self._bank.seen_ids
        self._tb_index = -1
        self.last_point = None

    def reset_memory(self) -> None:
        """Clear all identity tracks at the current episode position."""
        self._bank.clear()
        self._tracks = self._bank.tracks
        self._seen_ids = self._bank.seen_ids
        self.last_point = None

    def state_is_empty(self) -> bool:
        """Report whether the shared transition bank is empty."""
        return self._bank.state_is_empty()

    def _update_track(
        self, ue_id, pilot, error, active, external_reset=False
    ):
        point = self._bank.update(
            self._tb_index,
            ue_id,
            pilot,
            error,
            active=bool(active),
            external_reset=bool(external_reset),
        )
        return (
            list(point.pilots),
            list(point.errors),
            bool(point.decision.reset),
            bool(point.decision.history_valid),
        )


def complex_to_stacked(value: np.ndarray) -> np.ndarray:
    """Stack real and imaginary channel components."""
    value = np.asarray(value)
    if not np.iscomplexobj(value):
        raise ValueError("channel grid must be complex")
    return np.concatenate([value.real, value.imag], axis=-1).astype(
        np.float32
    )


def extract_pilot_sequence(receiver, generator, batch, tf):
    """Extract exact pilot estimates for every TB."""
    pilots = []
    errors = []
    layout = None
    for tb in range(int(batch["y"].shape[1])):
        slot = int(batch["slot_number"][tb].numpy())
        generator._set_slot_pilots(slot)
        pilot, error, current_layout = extract_pilot_statistics(
            receiver._ls_est,
            batch["y"][:, tb],
            batch["no"][:, tb],
            generator.num_scheduled_tx,
            tf,
        )
        if layout is None:
            layout = current_layout
        elif layout != current_layout:
            raise ValueError("pilot layout changed within an episode")
        pilots.append(pilot)
        errors.append(error)
    return np.stack(pilots, axis=1), np.stack(errors, axis=1), layout


def extract_pilot_sequence_with_noise(
    receiver, generator, batch, noise_source: str, tf
):
    """Extract pilots with exact or measured CDM noise."""
    if noise_source == "exact":
        pilots, errors, layout = extract_pilot_sequence(
            receiver, generator, batch, tf
        )
        return pilots, errors, layout, []
    if noise_source != "cdm":
        raise ValueError(f"unsupported noise source: {noise_source}")
    pilots = []
    errors = []
    diagnostics = []
    layout = None
    for tb in range(int(batch["y"].shape[1])):
        slot = int(batch["slot_number"][tb].numpy())
        generator._set_slot_pilots(slot)
        raw, current_layout = extract_raw_pilot_estimates(
            receiver._ls_est,
            batch["y"][:, tb],
            generator.num_scheduled_tx,
            tf,
        )
        if layout is None:
            layout = current_layout
        elif layout != current_layout:
            raise ValueError("pilot layout changed within an episode")
        pilots.append(cdm_average_raw_pilots(raw))
        error, diagnostic = estimate_cdm_noise_variance(raw)
        errors.append(error)
        diagnostics.append(diagnostic)
    return (
        np.stack(pilots, axis=1),
        np.stack(errors, axis=1),
        layout,
        diagnostics,
    )


def active_dmrs_innovation(
    pilots: np.ndarray,
    error_variance: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Measure normalized DMRS change for active UEs."""
    pilots = np.asarray(pilots)
    error_variance = np.asarray(error_variance, np.float64)
    active = np.asarray(active, bool)
    if pilots.shape != error_variance.shape or pilots.ndim != 6:
        raise ValueError("pilot arrays must be matching rank-six tensors")
    if active.shape != pilots.shape[:3]:
        raise ValueError("activity does not match pilot sequence")
    if np.any(error_variance <= 0.0):
        raise ValueError("pilot error variance must be positive")
    innovation = np.zeros(active.shape, np.float64)
    valid = active[:, 1:] & active[:, :-1]
    difference = np.square(np.abs(pilots[:, 1:] - pilots[:, :-1]))
    noise = error_variance[:, 1:] + error_variance[:, :-1]
    values = np.mean(
        difference / np.maximum(noise, 1e-12), axis=(3, 4, 5)
    )
    innovation[:, 1:] = np.where(valid, values, 0.0)
    if not np.all(np.isfinite(innovation)):
        raise FloatingPointError("DMRS innovation is not finite")
    return innovation


def segment_safe_reset_detector(
    innovation: np.ndarray,
    active: np.ndarray,
    hard_reset: np.ndarray,
    threshold: float,
    history_window: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the shared segment-safe guard."""
    innovation = np.asarray(innovation, np.float64)
    active = np.asarray(active, bool)
    hard_reset = np.asarray(hard_reset, bool)
    if innovation.shape != active.shape or active.shape != hard_reset.shape:
        raise ValueError("guard inputs must share [batch, TB, UE] shape")
    if history_window < 1 or float(threshold) <= 0.0:
        raise ValueError("invalid segment-safe guard settings")
    reset = np.zeros_like(active)
    score = np.zeros_like(innovation)
    eligible = np.zeros_like(active)
    history_valid = np.zeros_like(active)
    for batch in range(active.shape[0]):
        for user in range(active.shape[2]):
            guard = SegmentSafeDMRSGuard(threshold, history_window)
            for tb in range(active.shape[1]):
                is_active = bool(active[batch, tb, user])
                consecutive = bool(
                    tb > 0
                    and is_active
                    and active[batch, tb - 1, user]
                )
                reentry = bool(tb > 0 and is_active and not consecutive)
                external = bool(
                    is_active
                    and (hard_reset[batch, tb, user] or reentry)
                )
                current = (
                    float(innovation[batch, tb, user])
                    if consecutive
                    else None
                )
                decision = guard.update(
                    current,
                    active=is_active,
                    consecutive=consecutive,
                    external_reset=external,
                )
                reset[batch, tb, user] = decision.reset
                score[batch, tb, user] = decision.score
                eligible[batch, tb, user] = decision.score_valid
                history_valid[batch, tb, user] = decision.history_valid
    return reset, score, eligible, history_valid


def segmented_wiener_estimate(
    pilots: np.ndarray,
    error_variance: np.ndarray,
    reset_mask: np.ndarray,
    max_history: int,
    history_source: np.ndarray | None = None,
    history_error_variance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PD filtering independently between memory resets."""
    pilots = np.asarray(pilots)
    error_variance = np.asarray(error_variance)
    reset_mask = np.asarray(reset_mask, dtype=bool)
    if pilots.shape != error_variance.shape or pilots.ndim != 6:
        raise ValueError("pilot arrays must have matching six-rank shapes")
    if reset_mask.shape != pilots.shape[:3]:
        raise ValueError("reset mask does not match the pilot sequence")
    source = pilots if history_source is None else np.asarray(history_source)
    source_error = (
        error_variance
        if history_error_variance is None
        else np.asarray(history_error_variance)
    )
    if source.shape != pilots.shape or source_error.shape != pilots.shape:
        raise ValueError("history inputs do not match pilot observations")
    estimate = np.empty_like(pilots)
    posterior = np.empty_like(error_variance)
    for episode in range(pilots.shape[0]):
        for user in range(pilots.shape[2]):
            starts = [0]
            starts.extend(
                int(value)
                for value in np.flatnonzero(reset_mask[episode, 1:, user]) + 1
            )
            stops = starts[1:] + [pilots.shape[1]]
            for start, stop in zip(starts, stops):
                slc = np.s_[
                    episode:episode + 1,
                    start:stop,
                    user:user + 1,
                ]
                filtered, variance, _ = pilot_domain_wiener_estimate(
                    pilots[slc],
                    error_variance[slc],
                    max_history=min(max_history, stop - start),
                    history_source=source[slc],
                    history_error_variance=source_error[slc],
                )
                estimate[slc] = filtered
                posterior[slc] = variance
    if not np.all(np.isfinite(estimate)):
        raise FloatingPointError("segmented estimate is not finite")
    if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
        raise FloatingPointError("segmented variance is invalid")
    return estimate, posterior


def build_segment_safe_pd_l8_sequence_from_pilots(
    receiver,
    batch: dict,
    pilots: np.ndarray,
    errors: np.ndarray,
    layout,
    threshold: float,
    tf,
    shuffle_history: bool = False,
) -> PilotMemorySequence:
    """Build sparse Segment-Safe memory by opaque physical UE ID."""
    if shuffle_history:
        raise ValueError("sparse Segment-Safe memory cannot shuffle history")
    pilots = np.asarray(pilots)
    errors = np.asarray(errors)
    ue_ids = np.asarray(batch["ue_ids"].numpy(), np.int64)
    active = np.asarray(batch["active"].numpy(), bool)
    hard_reset = np.asarray(batch["hard_reset"].numpy(), bool)
    if pilots.shape != errors.shape or pilots.ndim != 6:
        raise ValueError("pilot sequences must have matching rank-six shapes")
    if ue_ids.shape != pilots.shape[:3]:
        raise ValueError("scheduled UE IDs do not match pilot positions")
    if active.shape != ue_ids.shape or hard_reset.shape != ue_ids.shape:
        raise ValueError("activity and reset masks must match UE IDs")
    if np.any(ue_ids < 0):
        raise ValueError("physical UE IDs cannot be negative")
    for row in ue_ids.reshape(-1, ue_ids.shape[-1]):
        if len(np.unique(row)) != len(row):
            raise ValueError("scheduled UE identities must be unique")

    posterior = np.empty_like(pilots)
    posterior_error = np.empty_like(errors)
    reset = np.zeros_like(active)
    history_valid = np.zeros_like(active)
    innovation = np.zeros(active.shape, np.float64)
    score = np.zeros(active.shape, np.float64)
    score_valid = np.zeros_like(active)
    for episode in range(pilots.shape[0]):
        bank = SegmentSafePilotBank(float(threshold), 8)
        for tb in range(pilots.shape[1]):
            for position in range(pilots.shape[2]):
                point = bank.update(
                    tb,
                    int(ue_ids[episode, tb, position]),
                    pilots[episode, tb, position],
                    errors[episode, tb, position],
                    active=bool(active[episode, tb, position]),
                    external_reset=bool(
                        hard_reset[episode, tb, position]
                    ),
                )
                history = np.asarray(point.pilots)[None, :, None]
                history_error = np.asarray(point.errors)[None, :, None]
                estimate, variance, _ = pilot_domain_wiener_estimate(
                    history,
                    history_error,
                    max_history=min(8, len(point.pilots)),
                )
                posterior[episode, tb, position] = estimate[0, -1, 0]
                posterior_error[episode, tb, position] = variance[0, -1, 0]
                reset[episode, tb, position] = point.decision.reset
                history_valid[episode, tb, position] = (
                    point.decision.history_valid
                )
                innovation[episode, tb, position] = point.innovation
                score[episode, tb, position] = point.decision.score
                score_valid[episode, tb, position] = (
                    point.decision.score_valid
                )
    posterior_grid, posterior_grid_error = interpolate_pilot_sequence(
        receiver._ls_est,
        posterior,
        posterior_error,
        layout,
        tf,
    )
    _, observation_grid_error = interpolate_pilot_sequence(
        receiver._ls_est,
        pilots,
        errors,
        layout,
        tf,
    )
    return PilotMemorySequence(
        complex_to_stacked(posterior_grid),
        posterior_grid_error,
        observation_grid_error,
        reset,
        history_valid,
        innovation,
        score,
        score_valid,
        pilots,
        errors,
    )


def build_segment_safe_pd_l8_sequence(
    receiver,
    generator,
    batch: dict,
    threshold: float,
    tf,
    shuffle_history: bool = False,
) -> PilotMemorySequence:
    """Build one segment-safe PD-L8 episode."""
    pilots, errors, layout, _ = extract_pilot_sequence_with_noise(
        receiver, generator, batch, "cdm", tf
    )
    return build_segment_safe_pd_l8_sequence_from_pilots(
        receiver,
        batch,
        pilots,
        errors,
        layout,
        threshold,
        tf,
        shuffle_history=shuffle_history,
    )
