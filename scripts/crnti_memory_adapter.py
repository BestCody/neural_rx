#!/usr/bin/env python3
"""Route temporal memory by C-RNTI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ue_memory_manager import RuntimeUEMemoryManager


_MIN_CRNTI = 0x0001
_MAX_CRNTI = 0xFFFF


def normalize_crnti(value) -> int:
    """Return a canonical non-zero 16-bit C-RNTI key."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"C-RNTI must be an integer, got {type(value).__name__}")
    value = int(value)
    if not _MIN_CRNTI <= value <= _MAX_CRNTI:
        raise ValueError(
            f"C-RNTI key must be a non-zero 16-bit value, got {value}"
        )
    return value


def normalize_crntis(values: Iterable[int]) -> list[int]:
    crntis = [normalize_crnti(v) for v in values]
    if len(set(crntis)) != len(crntis):
        raise ValueError("The same C-RNTI may appear at most once in one PUSCH batch")
    return crntis


@dataclass(frozen=True)
class RuntimeMemoryInput:
    """Memory lookup result for one receiver call."""

    crntis: tuple[int, ...]
    slot_index: int
    memory: np.ndarray
    gap_slots: np.ndarray
    valid: np.ndarray


class CRNTIMemoryAdapter:
    """Bind C-RNTI lookups to successful commits."""

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
        """Lookup."""
        keys = normalize_crntis(crntis)
        slot = int(slot_index)
        memory, gap, valid = self.manager.lookup(keys, slot)
        return RuntimeMemoryInput(
            crntis=tuple(keys),
            slot_index=slot,
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
        """Commit."""
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
        """Process result."""
        slot = int(slot_index)
        if slot != int(lookup.slot_index):
            raise ValueError(
                f"lookup was for slot {lookup.slot_index}, cannot commit as slot {slot}"
            )
        self.commit(
            lookup.crntis,
            next_memory,
            slot,
            active=active,
        )

    def release(self, crnti: int) -> None:
        """Release."""
        self.manager.remove(normalize_crnti(crnti))

    def handover(self, crnti: int) -> None:
        """Handover."""
        self.release(crnti)

    def reestablishment(self, old_crnti: int) -> None:
        """Reestablishment."""
        self.release(old_crnti)

    def expire(self, slot_index: int) -> None:
        self.manager.expire(int(slot_index))

    def clear(self) -> None:
        self.manager.clear()

    def snapshot(self):
        snap = self.manager.snapshot()
        snap["crnti_to_slot_hex"] = {
            f"0x{int(crnti):04X}": int(slot)
            for crnti, slot in self.manager.ue_to_slot.items()
        }
        return snap
