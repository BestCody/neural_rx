#!/usr/bin/env python3
"""C-RNTI boundary adapter for runtime temporal UE memory.

The neural receiver must never infer UE identity from RF samples. The gNB
scheduler already knows the UE associated with each scheduled PUSCH. This
adapter converts those scheduler-supplied C-RNTIs into the stable keys used by
RuntimeUEMemoryManager and exposes explicit lifecycle hooks.

Intended production call sequence per slot:

    crntis from scheduled PUSCH PDUs
        -> lookup(crntis, slot_index)
        -> NRX(prev_memory, gap, valid, current signal)
        -> commit(crntis, next_memory, slot_index)

On UE release, re-establishment with a new identity, handover, or any event that
invalidates temporal state, call release(crnti). Expiration handles UEs that
simply stop being scheduled for longer than the configured TTL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ue_memory_manager import RuntimeUEMemoryManager


# C-RNTI is carried by the gNB as a 16-bit RNTI value. Keep validation at the
# integration boundary so the generic memory manager can remain ID-agnostic.
_MIN_CRNTI = 0x0001
_MAX_CRNTI = 0xFFFE


def normalize_crnti(value) -> int:
    """Return a canonical integer C-RNTI suitable as a stable dictionary key."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"C-RNTI must be an integer, got {type(value).__name__}")
    value = int(value)
    if not _MIN_CRNTI <= value <= _MAX_CRNTI:
        raise ValueError(
            f"C-RNTI must be in 0x{_MIN_CRNTI:04X}..0x{_MAX_CRNTI:04X}, "
            f"got 0x{value & 0xFFFF:04X} ({value})"
        )
    return value


def normalize_crntis(values: Iterable[int]) -> list[int]:
    crntis = [normalize_crnti(v) for v in values]
    if len(set(crntis)) != len(crntis):
        raise ValueError("The same C-RNTI may appear at most once in one PUSCH batch")
    return crntis


@dataclass(frozen=True)
class RuntimeMemoryInput:
    """State handed from the C-RNTI adapter to the neural receiver."""

    crntis: tuple[int, ...]
    memory: np.ndarray
    gap_slots: np.ndarray
    valid: np.ndarray


class CRNTIMemoryAdapter:
    """Scheduler C-RNTI -> identity-owned temporal-memory bridge."""

    def __init__(
        self,
        d_mem: int,
        initial_capacity: int = 16,
        expiry_slots: int | None = 64,
        dtype=np.float32,
        manager: RuntimeUEMemoryManager | None = None,
    ):
        if manager is not None:
            if int(manager.d_mem) != int(d_mem):
                raise ValueError(
                    f"manager d_mem={manager.d_mem} does not match d_mem={d_mem}")
            self.manager = manager
        else:
            self.manager = RuntimeUEMemoryManager(
                d_mem=d_mem,
                initial_capacity=initial_capacity,
                expiry_slots=expiry_slots,
                dtype=dtype,
            )
        self.d_mem = int(d_mem)

    def lookup(self, crntis: Sequence[int], slot_index: int) -> RuntimeMemoryInput:
        """Resolve scheduled PUSCH C-RNTIs to previous memory/gap/valid tensors."""
        keys = normalize_crntis(crntis)
        memory, gap, valid = self.manager.lookup(keys, int(slot_index))
        return RuntimeMemoryInput(
            crntis=tuple(keys),
            memory=memory,
            gap_slots=gap,
            valid=valid,
        )

    def commit(
        self,
        crntis: Sequence[int],
        next_memory,
        slot_index: int,
        active: Sequence[bool] | None = None,
    ) -> None:
        """Write NRX-produced memory back under the exact scheduled C-RNTIs."""
        keys = normalize_crntis(crntis)
        self.manager.update(
            keys,
            next_memory,
            int(slot_index),
            active=active,
        )

    def process_result(
        self,
        lookup: RuntimeMemoryInput,
        next_memory,
        slot_index: int,
        active: Sequence[bool] | None = None,
    ) -> None:
        """Commit using the immutable keys returned by lookup.

        Runtime integrations should prefer this method because it prevents a
        caller from accidentally looking up one UE order and committing another.
        """
        self.commit(
            lookup.crntis,
            next_memory,
            slot_index,
            active=active,
        )

    def release(self, crnti: int) -> None:
        """Erase temporal state immediately on scheduler/RRC UE removal."""
        self.manager.remove(normalize_crnti(crnti))

    def handover(self, crnti: int) -> None:
        """Conservative handover hook: old cell-specific state is invalidated."""
        self.release(crnti)

    def reestablishment(self, old_crnti: int) -> None:
        """Invalidate the old identity before a re-established UE is re-keyed."""
        self.release(old_crnti)

    def expire(self, slot_index: int) -> None:
        self.manager.expire(int(slot_index))

    def clear(self) -> None:
        self.manager.clear()

    def snapshot(self):
        snap = self.manager.snapshot()
        # JSON/log-friendly canonical hexadecimal view alongside integer keys.
        snap["crnti_to_slot_hex"] = {
            f"0x{int(crnti):04X}": int(slot)
            for crnti, slot in self.manager.ue_to_slot.items()
        }
        return snap
