#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass

from artifact_io import canonical_json


TDL_MODELS = ("A", "B", "C", "D", "E", "A30", "B100", "C300")
FIXED_TDL_DELAYS = {
    "A30": 30e-9,
    "B100": 100e-9,
    "C300": 300e-9,
}


@dataclass(frozen=True)
class ChannelProfileSpec:

    name: str
    family: str
    model: str
    min_delay_spread_s: float | None
    max_delay_spread_s: float | None
    min_doppler_hz: float
    max_doppler_hz: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("channel profile name cannot be empty")
        if self.family not in ("tdl", "umi"):
            raise ValueError(f"unsupported channel family: {self.family}")
        if self.family == "tdl":
            if self.model not in TDL_MODELS:
                raise ValueError(f"unsupported TDL model: {self.model}")
            if self.min_delay_spread_s is None:
                raise ValueError("TDL delay spread is required")
            if self.max_delay_spread_s is None:
                raise ValueError("TDL delay spread is required")
            if self.min_delay_spread_s <= 0.0:
                raise ValueError("TDL delay spread must be positive")
            if self.max_delay_spread_s < self.min_delay_spread_s:
                raise ValueError("invalid TDL delay-spread range")
            fixed = FIXED_TDL_DELAYS.get(self.model)
            if fixed is not None and not (
                math.isclose(
                    self.min_delay_spread_s, fixed, rel_tol=0.0, abs_tol=1e-18
                )
                and math.isclose(
                    self.max_delay_spread_s, fixed, rel_tol=0.0, abs_tol=1e-18
                )
            ):
                raise ValueError(f"{self.model} has a fixed delay spread")
        else:
            if self.model != "UMi":
                raise ValueError("UMi family requires model UMi")
            if self.min_delay_spread_s is not None:
                raise ValueError("UMi delay spread comes from topology")
            if self.max_delay_spread_s is not None:
                raise ValueError("UMi delay spread comes from topology")
        if self.min_doppler_hz < 0.0:
            raise ValueError("Doppler cannot be negative")
        if self.max_doppler_hz < self.min_doppler_hz:
            raise ValueError("invalid Doppler range")


    @classmethod
    def from_dict(cls, value: dict) -> "ChannelProfileSpec":
        profile = cls(**value)
        profile.validate()
        return profile


def payload_sha256(value: dict, field: str = "manifest_sha256") -> str:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def seal_payload(value: dict, field: str = "manifest_sha256") -> dict:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    payload[field] = payload_sha256(payload, field)
    return payload


def verify_payload(value: dict, field: str = "manifest_sha256") -> None:
    expected = value.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing or invalid {field}")
    actual = payload_sha256(value, field)
    if actual != expected:
        raise ValueError(f"{field} does not match payload")
