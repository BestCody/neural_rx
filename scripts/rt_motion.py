# SPDX-FileCopyrightText: Copyright (c) 2026 Zhouyou Gu
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Motion:
    start: tuple[float, float, float]
    velocity: tuple[float, float, float]
    x_bounds: tuple[float, float] | None = None
    waypoints: tuple[tuple[float, float, float], ...] = ()
    route_mode: str = 'pingpong'
    speed_mps: float = 0.0

    def _route(self) -> tuple[tuple[tuple[float, float, float], ...], list[float], float]:
        points = list(self.waypoints)
        if self.route_mode == 'loop' and points[0] != points[-1]:
            points.append(points[0])
        lengths = [0.0]
        for previous, current in zip(points, points[1:]):
            lengths.append(lengths[-1] + math.dist(previous, current))
        return (tuple(points), lengths, lengths[-1])

    def _travelled(self, elapsed_seconds: float, total: float) -> tuple[float, float]:
        if total <= 0.0 or self.speed_mps == 0.0:
            return (0.0, 1.0)
        distance = self.speed_mps * elapsed_seconds
        if self.route_mode == 'loop':
            return (distance % total, 1.0)
        phase = distance % (2.0 * total)
        return (phase, 1.0) if phase <= total else (2.0 * total - phase, -1.0)

    def position_at(self, elapsed_seconds: float) -> tuple[float, float, float]:
        if len(self.waypoints) >= 2:
            points, lengths, total = self._route()
            travelled, _ = self._travelled(elapsed_seconds, total)
            for index in range(len(points) - 1):
                if travelled <= lengths[index + 1] or index == len(points) - 2:
                    segment = lengths[index + 1] - lengths[index]
                    ratio = 0.0 if segment <= 0.0 else (travelled - lengths[index]) / segment
                    ratio = min(1.0, max(0.0, ratio))
                    return tuple((a + (b - a) * ratio for a, b in zip(points[index], points[index + 1])))
            return points[-1]
        position = tuple((origin + elapsed_seconds * speed for origin, speed in zip(self.start, self.velocity)))
        if self.x_bounds is None or self.velocity[0] == 0.0:
            return position
        low, high = self.x_bounds
        span = high - low
        phase = (self.start[0] - low + elapsed_seconds * self.velocity[0]) % (2.0 * span)
        x = low + phase if phase <= span else high - (phase - span)
        return (x, position[1], position[2])

    def velocity_at(self, elapsed_seconds: float) -> tuple[float, float, float]:
        if len(self.waypoints) >= 2:
            points, lengths, total = self._route()
            travelled, direction = self._travelled(elapsed_seconds, total)
            for index in range(len(points) - 1):
                if travelled <= lengths[index + 1] or index == len(points) - 2:
                    segment = lengths[index + 1] - lengths[index]
                    if segment <= 0.0:
                        return (0.0, 0.0, 0.0)
                    return tuple(((b - a) / segment * self.speed_mps * direction for a, b in zip(points[index], points[index + 1])))
            return (0.0, 0.0, 0.0)
        if self.x_bounds is None or self.velocity[0] == 0.0:
            return self.velocity
        low, high = self.x_bounds
        span = high - low
        phase = (self.start[0] - low + elapsed_seconds * self.velocity[0]) % (2.0 * span)
        direction = 1.0 if phase < span else -1.0
        return (self.velocity[0] * direction, self.velocity[1], self.velocity[2])
