"""Shared physical diagnostics for G1 motion-tracking leaderboards."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np


def quaternion_geodesic_degrees(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    q2 = np.asarray(q2_wxyz, dtype=np.float64)
    q1 /= max(np.linalg.norm(q1), 1e-12)
    q2 /= max(np.linalg.norm(q2), 1e-12)
    cosine = np.clip(abs(float(np.dot(q1, q2))), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(cosine)))


class TrackingMetricAccumulator:
    """Aggregate only simulator-observable, backend-comparable diagnostics."""

    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def update(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            scalar = float(value)
            if np.isfinite(scalar):
                self.values[name].append(scalar)

    def summarize(self, *, completed_steps: int, total_steps: int, success: bool) -> dict[str, float]:
        result = {
            name: float(np.mean(items))
            for name, items in sorted(self.values.items())
            if items
        }
        result.update(
            {
                "survival_rate": float(completed_steps / max(total_steps, 1)),
                "success_rate": float(bool(success)),
                "completed_steps": int(completed_steps),
                "total_steps": int(total_steps),
            }
        )
        return result
