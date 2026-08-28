#!/usr/bin/env python3
"""Live C-RNTI-keyed runtime wrapper for temporal Neural RX.

This module closes the receiver-side half of the live bridge:

    srsRAN scheduled PUSCH context (C-RNTI + slot)
        -> CRNTIMemoryAdapter.lookup(...)
        -> temporal K-step NRX with prev_memory/gap/valid
        -> CRNTIMemoryAdapter.process_result(...)

UE identity is never inferred from RF samples. The caller must supply C-RNTIs
in exactly the same UE/receiver-position order as the signal tensors passed to
the neural receiver.

The generic TemporalNRXRuntime is deliberately independent of TensorFlow so the
identity/memory transaction can be tested without a GPU. TensorFlowTemporalInference
adapts the existing TemporalUEMemoryCGNN interface used by the training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from crnti_memory_adapter import (
    CRNTIMemoryAdapter,
    RuntimeMemoryInput,
    normalize_crntis,
)


@dataclass(frozen=True)
class TemporalInferenceOutput:
    """Normalized output returned by a live temporal-NRX inference callable."""

    receiver_output: Any
    next_memory: Any
    channel_estimate: Any = None
    diagnostics: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class TemporalRuntimeResult:
    """Result plus the exact memory metadata consumed for this inference."""

    receiver_output: Any
    channel_estimate: Any
    next_memory: np.ndarray
    crntis: tuple[int, ...]
    slot_index: int
    previous_memory: np.ndarray
    memory_gap: np.ndarray
    memory_valid: np.ndarray
    diagnostics: Optional[Mapping[str, Any]] = None


def _to_numpy(value) -> np.ndarray:
    """Convert NumPy/TensorFlow-like values without importing TensorFlow."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_inference_output(value) -> TemporalInferenceOutput:
    if isinstance(value, TemporalInferenceOutput):
        return value

    if isinstance(value, Mapping):
        if "next_memory" not in value:
            raise ValueError("inference mapping must contain next_memory")
        return TemporalInferenceOutput(
            receiver_output=value.get("receiver_output"),
            next_memory=value["next_memory"],
            channel_estimate=value.get("channel_estimate"),
            diagnostics=value.get("diagnostics"),
        )

    raise TypeError(
        "inference_fn must return TemporalInferenceOutput or a mapping with "
        "next_memory"
    )


class TemporalNRXRuntime:
    """C-RNTI-keyed live temporal receiver transaction.

    Parameters
    ----------
    inference_fn:
        Callable invoked with keyword arguments:

            received_grid
            ls_estimate
            active_tx       [1, U] bool
            prev_memory     [1, U, d_mem] float32
            memory_gap      [1, U] int32
            memory_valid    [1, U] bool

        It must return TemporalInferenceOutput (or an equivalent mapping).
    d_mem:
        Persistent memory width per UE. This must match the trained model.

    Notes
    -----
    A single lock covers lookup -> inference -> commit. This is the conservative
    correctness-first deployment policy: two calls cannot interleave and update
    the same UE out of slot order. Once the live receiver batches all UEs for a
    slot in one invocation, this lock is normally uncontended. A future highly
    parallel runtime can replace it with per-UE ordering once needed.
    """

    def __init__(
        self,
        inference_fn: Callable[..., TemporalInferenceOutput],
        d_mem: int,
        initial_capacity: int = 16,
        expiry_slots: int | None = 64,
        adapter: CRNTIMemoryAdapter | None = None,
    ):
        if not callable(inference_fn):
            raise TypeError("inference_fn must be callable")
        self.inference_fn = inference_fn
        self.d_mem = int(d_mem)
        self.memory = adapter or CRNTIMemoryAdapter(
            d_mem=self.d_mem,
            initial_capacity=initial_capacity,
            expiry_slots=expiry_slots,
        )
        if int(self.memory.d_mem) != self.d_mem:
            raise ValueError(
                f"adapter d_mem={self.memory.d_mem} does not match d_mem={self.d_mem}"
            )
        self._lock = RLock()

    def _normalize_active(
        self, active: Sequence[bool] | np.ndarray | None, num_ues: int
    ) -> np.ndarray:
        if active is None:
            return np.ones(num_ues, dtype=np.bool_)
        active_arr = np.asarray(active, dtype=np.bool_)
        if active_arr.shape == (1, num_ues):
            active_arr = active_arr[0]
        if active_arr.shape != (num_ues,):
            raise ValueError(
                f"active must have shape {(num_ues,)} or {(1, num_ues)}, "
                f"got {active_arr.shape}"
            )
        return active_arr

    def _normalize_next_memory(self, value, num_ues: int) -> np.ndarray:
        memory = _to_numpy(value).astype(np.float32, copy=False)
        if memory.shape == (1, num_ues, self.d_mem):
            memory = memory[0]
        expected = (num_ues, self.d_mem)
        if memory.shape != expected:
            raise ValueError(
                f"inference next_memory must have shape {expected} or "
                f"{(1,) + expected}, got {memory.shape}"
            )
        if not np.all(np.isfinite(memory)):
            raise ValueError("inference next_memory contains NaN/Inf")
        return memory

    def process(
        self,
        received_grid,
        ls_estimate,
        crntis: Sequence[int],
        slot_index: int,
        active: Sequence[bool] | np.ndarray | None = None,
        **inference_kwargs,
    ) -> TemporalRuntimeResult:
        """Run one live temporal inference and persist the resulting UE memory.

        `crntis` must use the exact UE ordering represented by received_grid,
        ls_estimate and active. The immutable keys returned by lookup are used
        for commit so receiver-position reorder cannot redirect memory.
        """
        keys = normalize_crntis(crntis)
        if not keys:
            raise ValueError("at least one scheduled C-RNTI is required")
        slot = int(slot_index)
        active_arr = self._normalize_active(active, len(keys))

        with self._lock:
            lookup = self.memory.lookup(keys, slot)

            # TemporalUEMemoryCGNN is trained with a batch dimension. Live gNB
            # processing is batch-size one, so the runtime inserts it here.
            output = _normalize_inference_output(
                self.inference_fn(
                    received_grid=received_grid,
                    ls_estimate=ls_estimate,
                    active_tx=active_arr[None, :],
                    prev_memory=lookup.memory[None, ...],
                    memory_gap=lookup.gap_slots[None, ...],
                    memory_valid=lookup.valid[None, ...],
                    **inference_kwargs,
                )
            )
            next_memory = self._normalize_next_memory(
                output.next_memory, len(keys)
            )

            # Commit only after inference and memory validation succeed.
            self.memory.process_result(
                lookup,
                next_memory,
                slot,
                active=active_arr,
            )

            return TemporalRuntimeResult(
                receiver_output=output.receiver_output,
                channel_estimate=output.channel_estimate,
                next_memory=next_memory.copy(),
                crntis=lookup.crntis,
                slot_index=slot,
                previous_memory=lookup.memory.copy(),
                memory_gap=lookup.gap_slots.copy(),
                memory_valid=lookup.valid.copy(),
                diagnostics=output.diagnostics,
            )

    def release(self, crnti: int) -> None:
        with self._lock:
            self.memory.release(crnti)

    def handover(self, crnti: int) -> None:
        with self._lock:
            self.memory.handover(crnti)

    def reestablishment(self, old_crnti: int) -> None:
        with self._lock:
            self.memory.reestablishment(old_crnti)

    def expire(self, slot_index: int) -> None:
        with self._lock:
            self.memory.expire(int(slot_index))

    def clear(self) -> None:
        with self._lock:
            self.memory.clear()

    def snapshot(self):
        with self._lock:
            return self.memory.snapshot()


class TensorFlowTemporalInference:
    """Adapter from the trained TemporalUEMemoryCGNN to TemporalNRXRuntime.

    This reproduces the preprocessing/demapping used by
    ``temporal_ue_memory_model.py`` while keeping live memory ownership outside
    the neural model. Only the first three model outputs are required for
    deployment.
    """

    def __init__(self, receiver, temporal_model, expected_num_it: int | None = 2):
        # Lazy imports keep the generic runtime/test path NumPy-only.
        import tensorflow as tf
        from sionna.utils import flatten_last_dims

        self.tf = tf
        self.flatten_last_dims = flatten_last_dims
        self.receiver = receiver
        self.temporal_model = temporal_model

        if expected_num_it is not None:
            base = getattr(temporal_model, "base", None)
            actual = getattr(base, "_num_it", None)
            if actual is not None and int(actual) != int(expected_num_it):
                raise ValueError(
                    f"temporal model is configured for K={actual}, "
                    f"expected K={expected_num_it}"
                )

    def _batched(self, value, unbatched_rank: int):
        x = self.tf.convert_to_tensor(value)
        if x.shape.rank == unbatched_rank:
            x = x[None, ...]
        return x

    def _demap_llr(self, ofdm, llr_grid, num_tx, mcs_idx=0):
        tf = self.tf
        llr = tf.cast(llr_grid, tf.float32)
        llr = tf.transpose(llr, [0, 1, 3, 2, 4])
        llr = tf.expand_dims(llr, axis=1)
        llr = ofdm._rg_demapper(llr)
        llr = llr[:, :num_tx]
        llr = self.flatten_last_dims(llr, 2)
        if ofdm._layer_demappers is None:
            llr = tf.squeeze(llr, axis=-2)
        else:
            llr = ofdm._layer_demappers[mcs_idx](llr)
        return llr

    def __call__(
        self,
        *,
        received_grid,
        ls_estimate,
        active_tx,
        prev_memory,
        memory_gap,
        memory_valid,
        **_,
    ) -> TemporalInferenceOutput:
        tf = self.tf
        ofdm = self.receiver._neural_rx

        # Per-TB training shapes are:
        #   y     [B, 1, RxAnt, Symbols, Subcarriers]
        #   h_hat [B, U, Subcarriers, Symbols, Features]
        # Accept the same tensors without B and insert B=1 when necessary.
        y = self._batched(received_grid, unbatched_rank=4)
        h_hat = self._batched(ls_estimate, unbatched_rank=4)
        active = tf.convert_to_tensor(active_tx)
        prev_memory = tf.convert_to_tensor(prev_memory)
        memory_gap = tf.convert_to_tensor(memory_gap)
        memory_valid = tf.convert_to_tensor(memory_valid)

        if y.shape.rank != 5:
            raise ValueError(
                "received_grid must be [B,1,RxAnt,Symbols,Subcarriers] "
                "or the same tensor without B"
            )
        if h_hat.shape.rank != 5:
            raise ValueError(
                "ls_estimate must be [B,U,Subcarriers,Symbols,Features] "
                "or the same tensor without B"
            )

        num_tx = tf.shape(active)[1]
        y2 = y[:, 0]
        y2 = tf.transpose(y2, [0, 3, 2, 1])
        y2 = tf.concat([tf.math.real(y2), tf.math.imag(y2)], axis=-1)
        pe = ofdm._nearest_pilot_dist[:num_tx]

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
                "TemporalUEMemoryCGNN must return at least "
                "(llr_grid, h_refined, next_memory)"
            )

        llr_grid, h_refined, next_memory = raw[:3]
        llr = self._demap_llr(ofdm, llr_grid, num_tx, 0)

        diagnostics = None
        if len(raw) >= 5:
            diagnostics = {
                "aux_loss": raw[3],
                "reconstruction_mse": raw[4],
            }

        return TemporalInferenceOutput(
            receiver_output=llr,
            channel_estimate=tf.cast(h_refined, tf.float32),
            next_memory=next_memory,
            diagnostics=diagnostics,
        )
