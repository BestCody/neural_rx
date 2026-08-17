#!/usr/bin/env python3
"""Live srsRAN <-> temporal Neural Receiver tensor adapter.

The C++ srsRAN plugin passes one PUSCH as frequency-domain complex tensors:

    rx_grid          [R, S, F]
    channel_estimate [U, R, S, F]

This module converts them to the tensor contract used by the temporal NRX:

    received_grid [1, R, S, F] complex64
    ls_estimate   [U, F, S, 2R] float32
    pilot PE      [U, F, S, 2] float32

The persistent state transaction remains owned by TemporalNRXRuntime and is
keyed only by the scheduler-provided C-RNTI and slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from temporal_nrx_runtime import (
    TemporalInferenceOutput,
    TemporalNRXRuntime,
    TensorFlowTemporalInference,
)


_TYPE1_PORT0_PILOT_OFFSETS = (0, 2, 4, 6, 8, 10)


def _complex64(name: str, value, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {array.shape}")
    if not np.iscomplexobj(array):
        raise TypeError(f"{name} must be complex-valued, got dtype {array.dtype}")
    array = np.asarray(array, dtype=np.complex64)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array)


def received_grid_to_nrx(rx_grid) -> np.ndarray:
    """Convert srsRAN [R,S,F] complex grid to temporal-runtime input."""
    rx = _complex64("rx_grid", rx_grid, 3)
    return np.ascontiguousarray(rx[None, ...])


def channel_estimate_to_nrx(channel_estimate) -> np.ndarray:
    """Convert srsRAN [U,R,S,F] complex CE to [U,F,S,2R] float features."""
    h = _complex64("channel_estimate", channel_estimate, 4)
    h = np.transpose(h, (0, 3, 2, 1))
    return np.ascontiguousarray(
        np.concatenate((h.real, h.imag), axis=-1), dtype=np.float32
    )


def _normalize_feature(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.mean(x, dtype=np.float64)
    std = float(np.std(x, dtype=np.float64))
    if std > 0.0:
        x = x / np.float32(std)
    return x


def build_type1_pilot_positional_encoding(
    *,
    num_ues: int,
    num_subcarriers: int,
    num_symbols: int,
    dmrs_symbols: Iterable[int],
    pilot_offsets_by_ue: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Build the NRX pilot-distance encoding for contiguous Type-1 PUSCH PRBs.

    This mirrors NRPreprocessing._calculate_nn_indices: distances are computed
    within one 12-subcarrier PRB, normalized independently in time/frequency,
    then tiled over the contiguous PRB allocation.
    """
    num_ues = int(num_ues)
    num_subcarriers = int(num_subcarriers)
    num_symbols = int(num_symbols)
    if num_ues <= 0:
        raise ValueError("num_ues must be positive")
    if num_subcarriers <= 0 or num_subcarriers % 12 != 0:
        raise ValueError("num_subcarriers must be a positive multiple of 12")
    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")

    dmrs = tuple(sorted({int(x) for x in dmrs_symbols}))
    if not dmrs:
        raise ValueError("at least one DMRS symbol is required")
    if dmrs[0] < 0 or dmrs[-1] >= num_symbols:
        raise ValueError(f"DMRS symbols {dmrs} outside [0,{num_symbols})")

    if pilot_offsets_by_ue is None:
        pilot_offsets_by_ue = [_TYPE1_PORT0_PILOT_OFFSETS] * num_ues
    if len(pilot_offsets_by_ue) != num_ues:
        raise ValueError("pilot_offsets_by_ue length must equal num_ues")

    subcarrier = np.arange(12, dtype=np.int32)
    symbol = np.arange(num_symbols, dtype=np.int32)
    kk, ll = np.meshgrid(subcarrier, symbol)
    re_pos = np.stack((kk, ll), axis=-1).reshape(-1, 1, 2)

    per_ue = []
    for offsets in pilot_offsets_by_ue:
        offsets = tuple(int(x) for x in offsets)
        if not offsets or min(offsets) < 0 or max(offsets) >= 12:
            raise ValueError(f"invalid Type-1 pilot offsets: {offsets}")
        pilot_pos = np.array(
            [(k, l) for l in dmrs for k in offsets], dtype=np.int32
        )[None, :, :]
        diff = np.abs(re_pos - pilot_pos)
        pe = np.min(diff, axis=1).reshape(num_symbols, 12, 2)
        pe = np.transpose(pe, (1, 0, 2)).astype(np.float32)

        time_distance = _normalize_feature(pe[..., 1:2])
        freq_distance = _normalize_feature(pe[..., 0:1])
        pe = np.concatenate((time_distance, freq_distance), axis=-1)
        pe = np.tile(pe, (num_subcarriers // 12, 1, 1))
        per_ue.append(pe)

    return np.ascontiguousarray(np.stack(per_ue, axis=0), dtype=np.float32)


class LiveTensorFlowTemporalInference(TensorFlowTemporalInference):
    """TensorFlow temporal inference with live pilot PE and raw-grid output."""

    def __call__(
        self,
        *,
        received_grid,
        ls_estimate,
        active_tx,
        prev_memory,
        memory_gap,
        memory_valid,
        pilot_positional_encoding=None,
        output_mode: str = "grid",
        **_,
    ) -> TemporalInferenceOutput:
        tf = self.tf
        ofdm = self.receiver._neural_rx

        y = self._batched(received_grid, unbatched_rank=4)
        h_hat = self._batched(ls_estimate, unbatched_rank=4)
        active = tf.convert_to_tensor(active_tx)
        prev_memory = tf.convert_to_tensor(prev_memory)
        memory_gap = tf.convert_to_tensor(memory_gap)
        memory_valid = tf.convert_to_tensor(memory_valid)

        if y.shape.rank != 5:
            raise ValueError("received_grid must be [B,1,R,S,F] or [1,R,S,F]")
        if h_hat.shape.rank != 5:
            raise ValueError("ls_estimate must be [B,U,F,S,2R] or [U,F,S,2R]")

        num_tx = tf.shape(active)[1]
        y2 = y[:, 0]
        y2 = tf.transpose(y2, [0, 3, 2, 1])
        y2 = tf.concat([tf.math.real(y2), tf.math.imag(y2)], axis=-1)

        if pilot_positional_encoding is None:
            pe = ofdm._nearest_pilot_dist[:num_tx]
        else:
            pe = tf.convert_to_tensor(pilot_positional_encoding)
            if pe.shape.rank != 4:
                raise ValueError("pilot_positional_encoding must be [U,F,S,2]")

        y2 = tf.cast(y2, ofdm._nrx_dtype)
        pe = tf.cast(pe, ofdm._nrx_dtype)
        h_hat = tf.cast(h_hat, ofdm._nrx_dtype)
        active_model = tf.cast(active, ofdm._nrx_dtype)
        mcs_mask = tf.ones([tf.shape(y2)[0], num_tx, 1], tf.float32)

        raw = self.temporal_model(
            [y2, pe, h_hat, active_model, mcs_mask],
            prev_memory=prev_memory,
            memory_gap=memory_gap,
            memory_valid=memory_valid,
            training=False,
        )
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise TypeError(
                "Temporal model must return at least "
                "(llr_grid, h_refined, next_memory)"
            )

        llr_grid, h_refined, next_memory = raw[:3]
        if output_mode == "grid":
            receiver_output = tf.cast(llr_grid, tf.float32)
        elif output_mode == "demapped":
            receiver_output = self._demap_llr(ofdm, llr_grid, num_tx, 0)
        else:
            raise ValueError("output_mode must be 'grid' or 'demapped'")

        diagnostics = None
        if len(raw) >= 5:
            diagnostics = {
                "aux_loss": raw[3],
                "reconstruction_mse": raw[4],
            }

        return TemporalInferenceOutput(
            receiver_output=receiver_output,
            channel_estimate=tf.cast(h_refined, tf.float32),
            next_memory=next_memory,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class SrsranInferenceResult:
    """Live output in srsRAN sign/order plus temporal-runtime metadata."""

    llr_grid: np.ndarray
    next_memory: np.ndarray
    previous_memory: np.ndarray
    memory_gap: np.ndarray
    memory_valid: np.ndarray


class SrsranTemporalNRXAdapter:
    """Convert native PUSCH tensors, run temporal NRX, and commit C-RNTI memory."""

    def __init__(self, runtime: TemporalNRXRuntime):
        self.runtime = runtime

    def infer(
        self,
        *,
        rx_grid,
        channel_estimate,
        crntis: Sequence[int],
        slot_index: int,
        dmrs_symbols: Iterable[int],
        pilot_offsets_by_ue: Sequence[Sequence[int]] | None = None,
        active: Sequence[bool] | np.ndarray | None = None,
    ) -> SrsranInferenceResult:
        rx = _complex64("rx_grid", rx_grid, 3)
        h = _complex64("channel_estimate", channel_estimate, 4)
        num_ues, num_rx, num_symbols, num_subcarriers = h.shape
        if rx.shape != (num_rx, num_symbols, num_subcarriers):
            raise ValueError(
                "rx_grid/channel_estimate geometry mismatch: "
                f"rx={rx.shape}, h={h.shape}"
            )
        if len(tuple(crntis)) != num_ues:
            raise ValueError("number of C-RNTIs must equal channel-estimate UE dimension")

        y = received_grid_to_nrx(rx)
        h_nrx = channel_estimate_to_nrx(h)
        pe = build_type1_pilot_positional_encoding(
            num_ues=num_ues,
            num_subcarriers=num_subcarriers,
            num_symbols=num_symbols,
            dmrs_symbols=dmrs_symbols,
            pilot_offsets_by_ue=pilot_offsets_by_ue,
        )

        result = self.runtime.process(
            received_grid=y,
            ls_estimate=h_nrx,
            crntis=crntis,
            slot_index=int(slot_index),
            active=active,
            pilot_positional_encoding=pe,
            output_mode="grid",
        )

        llr = result.receiver_output
        if hasattr(llr, "numpy"):
            llr = llr.numpy()
        llr = np.asarray(llr, dtype=np.float32)
        if llr.ndim != 5 or llr.shape[0] != 1 or llr.shape[1] != num_ues:
            raise ValueError(
                "temporal NRX grid output must be [1,U,F,S,Q], got "
                f"{llr.shape}"
            )
        if llr.shape[2] != num_subcarriers or llr.shape[3] != num_symbols:
            raise ValueError(
                "temporal NRX output geometry does not match live PUSCH: "
                f"output={llr.shape}, expected F={num_subcarriers}, S={num_symbols}"
            )
        if not np.all(np.isfinite(llr)):
            raise ValueError("temporal NRX LLR grid contains NaN/Inf")

        llr_srsran = -np.transpose(llr[0], (0, 2, 1, 3))
        return SrsranInferenceResult(
            llr_grid=np.ascontiguousarray(llr_srsran, dtype=np.float32),
            next_memory=result.next_memory,
            previous_memory=result.previous_memory,
            memory_gap=result.memory_gap,
            memory_valid=result.memory_valid,
        )
