#!/usr/bin/env python3
"""Embedded-Python entrypoint used by libtemporal_nrx_python_plugin.so.

Set TEMPORAL_NRX_FACTORY to ``module:function``. The callable must return a
configured ``SrsranTemporalNRXAdapter``. This keeps checkpoint/architecture
selection outside srsRAN and lets the exact trained winner be installed without
rebuilding the gNB.
"""

from __future__ import annotations

import importlib
import os
from threading import RLock

import numpy as np

_adapter = None
_lock = RLock()


def install_adapter(adapter) -> None:
    global _adapter
    if not hasattr(adapter, "infer"):
        raise TypeError("adapter must provide infer(...)")
    with _lock:
        _adapter = adapter


def _load_adapter():
    global _adapter
    with _lock:
        if _adapter is not None:
            return _adapter
        spec = os.environ.get("TEMPORAL_NRX_FACTORY", "").strip()
        if not spec or ":" not in spec:
            raise RuntimeError(
                "TEMPORAL_NRX_FACTORY must be module:function returning a "
                "configured SrsranTemporalNRXAdapter"
            )
        module_name, function_name = spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        adapter = factory()
        if not hasattr(adapter, "infer"):
            raise TypeError("TEMPORAL_NRX_FACTORY did not return an adapter")
        _adapter = adapter
        return adapter


def infer_pusch(
    rx_grid,
    channel_estimate,
    crnti: int,
    slot_index: int,
    dmrs_symbol_mask: int,
    bits_per_symbol: int,
):
    """Run one synchronous single-layer PUSCH and return [S,F,Q] float LLRs."""
    rx_grid = np.asarray(rx_grid, dtype=np.complex64)
    channel_estimate = np.asarray(channel_estimate, dtype=np.complex64)
    if rx_grid.ndim != 3:
        raise ValueError(f"rx_grid must be [R,S,F], got {rx_grid.shape}")
    if channel_estimate.ndim == 3:
        channel_estimate = channel_estimate[None, ...]
    if channel_estimate.ndim != 4 or channel_estimate.shape[0] != 1:
        raise ValueError(
            "embedded srsRAN v1 ABI currently supports one PUSCH layer/UE"
        )

    num_symbols = int(rx_grid.shape[1])
    dmrs_symbols = [
        symbol for symbol in range(num_symbols)
        if (int(dmrs_symbol_mask) >> symbol) & 1
    ]
    if not dmrs_symbols:
        raise ValueError("PUSCH has no DMRS symbols")

    out = _load_adapter().infer(
        rx_grid=rx_grid,
        channel_estimate=channel_estimate,
        crntis=[int(crnti)],
        slot_index=int(slot_index),
        dmrs_symbols=dmrs_symbols,
    )
    llr = np.asarray(out.llr_grid[0], dtype=np.float32)
    if llr.ndim != 3:
        raise ValueError(f"adapter returned invalid LLR shape {llr.shape}")
    if llr.shape[-1] < int(bits_per_symbol):
        raise ValueError(
            f"NRX produced {llr.shape[-1]} LLRs/RE but PUSCH needs "
            f"{bits_per_symbol}"
        )
    return np.ascontiguousarray(llr[..., : int(bits_per_symbol)], dtype=np.float32)
