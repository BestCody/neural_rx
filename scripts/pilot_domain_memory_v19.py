#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from pilot_domain_channel import (
    cdm_average_raw_pilots,
    estimate_cdm_noise_variance,
)
from pilot_domain_channel_v19 import (
    extract_raw_pilot_estimates_cached,
    interpolate_pilot_bundle,
    pilot_domain_wiener_latest,
)
from pilot_domain_memory import (
    OnlineSegmentSafePilotDomainMemory,
    PilotMemoryPoint,
    complex_to_stacked,
)


class OnlineStreamedSegmentSafePilotDomainMemory(
    OnlineSegmentSafePilotDomainMemory
):

    def update(self, inputs: dict) -> PilotMemoryPoint:
        self._tb_index += 1
        y = np.asarray(inputs["received_grid"])
        active = np.asarray(inputs["active"], bool)
        ue_ids = np.asarray(inputs["ue_ids"], np.int64)
        if y.ndim != 4 or active.ndim != 1 or ue_ids.shape != active.shape:
            raise ValueError("online PD-L8 input shapes differ")
        if len(set(ue_ids.tolist())) != len(ue_ids):
            raise ValueError("scheduled UE identities must be unique")
        self._set_slot_pilots(int(np.asarray(inputs["slot_number"])))
        raw, layout = extract_raw_pilot_estimates_cached(
            self.receiver._ls_est,
            self.tf.convert_to_tensor(y[None]),
            len(active),
            self.tf,
            layout=getattr(self, "_pilot_layout", None),
        )
        self._pilot_layout = layout
        pilots = cdm_average_raw_pilots(raw)
        errors, _ = estimate_cdm_noise_variance(raw)
        posterior = np.empty_like(pilots)
        posterior_error = np.empty_like(errors)
        reset_mask = np.zeros(len(active), bool)
        history_valid = np.zeros(len(active), bool)
        groups = {}
        for position, ue_id in enumerate(ue_ids.tolist()):
            history, history_error, reset, valid = self._update_track(
                int(ue_id),
                pilots[0, position],
                errors[0, position],
                bool(active[position]),
            )
            length = min(self.maximum_history_tbs, len(history))
            record = (
                position,
                tuple(history[-length:]),
                tuple(history_error[-length:]),
            )
            groups.setdefault(length, []).append(record)
            reset_mask[position] = reset
            history_valid[position] = valid
        systems = 0
        projection_iterations = 0
        for length, records in groups.items():
            sequence = np.stack([
                np.asarray(record[1]) for record in records
            ], axis=1)[None]
            sequence_error = np.stack([
                np.asarray(record[2]) for record in records
            ], axis=1)[None]
            estimate, variance, diagnostics = pilot_domain_wiener_latest(
                sequence,
                sequence_error,
                max_history=length,
            )
            systems += int(diagnostics["total_updates"])
            projection_iterations = max(
                projection_iterations,
                int(diagnostics["max_projection_iterations"]),
            )
            for member, record in enumerate(records):
                position = int(record[0])
                posterior[0, position] = estimate[0, member]
                posterior_error[0, position] = variance[0, member]
        (
            posterior_grid,
            posterior_grid_error,
            observation_grid_error,
        ) = interpolate_pilot_bundle(
            self.receiver._ls_est,
            posterior,
            posterior_error,
            pilots,
            errors,
            layout,
            self.tf,
        )
        self.last_diagnostics = {
            "history_groups": tuple(sorted(groups)),
            "newest_posterior_systems": systems,
            "discarded_prefix_systems": 0,
            "interpolation_calls": 1,
            "max_projection_iterations": projection_iterations,
        }
        self.last_point = PilotMemoryPoint(
            complex_to_stacked(posterior_grid),
            posterior_grid_error.astype(np.float32),
            observation_grid_error.astype(np.float32),
            reset_mask[None],
            history_valid[None],
        )
        return self.last_point
