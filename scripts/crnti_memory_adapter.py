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
        -> process_result(lookup, next_memory, same slot_index)

On UE release, re-establishment with a new identity, handover, or any event that
invalidates temporal state, call release(crnti). Expiration handles UEs that
simply stop being scheduled for longer than the configured TTL.
"""

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
    """Immutable state handed from the C-RNTI adapter to the neural receiver."""

    crntis: tuple[int, ...]
    slot_index: int
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
        """Commit using the immutable keys and slot returned by lookup.

        This prevents both UE-order mismatches and accidental cross-slot commits.
        A result computed from slot N's previous memory must be committed as slot N.
        """
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
        snap["crnti_to_slot_hex"] = {
            f"0x{int(crnti):04X}": int(slot)
            for crnti, slot in self.manager.ue_to_slot.items()
        }
        return snap
