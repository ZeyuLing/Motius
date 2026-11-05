"""Qpos-level G1 style features and nearest-neighbour style cost.

The PhysFlow generator already emits Unitree G1 qpos references.  This module
keeps the style signal in that same representation: it compares generated qpos
statistics against a precomputed bank of G1/HYMotion references.  Lower cost
means the motion is closer to the robot-style bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LOWER_BODY_DOF = np.arange(0, 12)
WAIST_DOF = np.arange(12, 15)
UPPER_BODY_DOF = np.arange(15, 29)


def categorize_style_text(text: str) -> str:
    """Map a caption/path to a coarse action class for same-action lookup."""
    t = (text or "").lower()
    if any(k in t for k in ("run", "jog", "walk", "step", "stroll", "march", "pace")):
        return "locomotion"
    if any(k in t for k in ("jump", "hop", "kick", "squat", "crouch", "lunge", "spin", "cartwheel")):
        return "dynamic"
    if any(k in t for k in ("wave", "arm", "hand", "clap", "point", "reach", "gesture", "punch")):
        return "upper_body"
    if any(k in t for k in ("dance", "turn", "ballet", "ballerina")):
        return "dance"
    if any(k in t for k in ("stand", "idle", "static", "pose", "balance")):
        return "standing"
    return "general"


def _safe_stats(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.percentile(x, 90)),
            float(np.max(x)),
        ],
        dtype=np.float32,
    )


def _quat_yaw_wxyz(q: np.ndarray) -> np.ndarray:
    """Return yaw angle from wxyz quaternions."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def qpos_style_feature(qpos: np.ndarray, length: Optional[int] = None, fps: float = 30.0) -> np.ndarray:
    """Extract a fixed qpos-level style vector from ``[T, 36]`` G1 qpos.

    Features are deliberately cheap: root speed/height/yaw-rate, per-joint
    position/velocity statistics, lower/upper-body energy balance, and action
    amplitude.  They require no simulator rollout or differentiable FK.
    """
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"Expected qpos with shape (T, 36), got {arr.shape}")
    if length is not None:
        arr = arr[: max(1, int(length))]
    if arr.shape[0] < 2:
        arr = np.concatenate([arr, arr[-1:]], axis=0)

    root = arr[:, :3]
    quat = arr[:, 3:7]
    dof = arr[:, 7:]
    dt = 1.0 / float(fps)
    root_vel = np.diff(root, axis=0) / dt
    dof_vel = np.diff(dof, axis=0) / dt
    yaw = _quat_yaw_wxyz(quat)
    yaw_vel = np.diff(np.unwrap(yaw)) / dt

    lower_energy = np.mean(np.abs(dof_vel[:, LOWER_BODY_DOF]))
    waist_energy = np.mean(np.abs(dof_vel[:, WAIST_DOF]))
    upper_energy = np.mean(np.abs(dof_vel[:, UPPER_BODY_DOF]))
    total_energy = lower_energy + waist_energy + upper_energy + 1e-6

    chunks = [
        _safe_stats(np.linalg.norm(root_vel[:, :2], axis=-1)),
        _safe_stats(root[:, 2]),
        _safe_stats(np.abs(yaw_vel)),
        np.asarray(
            [
                float(np.linalg.norm(root[-1, :2] - root[0, :2])),
                float(lower_energy / total_energy),
                float(waist_energy / total_energy),
                float(upper_energy / total_energy),
            ],
            dtype=np.float32,
        ),
        np.mean(dof, axis=0).astype(np.float32),
        np.std(dof, axis=0).astype(np.float32),
        np.mean(np.abs(dof_vel), axis=0).astype(np.float32),
        np.percentile(np.abs(dof_vel), 90, axis=0).astype(np.float32),
    ]
    return np.concatenate(chunks, axis=0).astype(np.float32)


@dataclass
class G1StyleBank:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_features(
        cls,
        features: np.ndarray,
        labels: Optional[Iterable[str]] = None,
        paths: Optional[Iterable[str]] = None,
    ) -> "G1StyleBank":
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features, got {feats.shape}")
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        n = feats.shape[0]
        label_arr = np.asarray(list(labels) if labels is not None else ["general"] * n, dtype=object)
        path_arr = np.asarray(list(paths) if paths is not None else [""] * n, dtype=object)
        if len(label_arr) != n or len(path_arr) != n:
            raise ValueError("features, labels, and paths must have the same length")
        return cls(feats, label_arr, path_arr, mean.astype(np.float32), std)

    @classmethod
    def load(cls, path: str | Path) -> "G1StyleBank":
        data = np.load(path, allow_pickle=True)
        return cls(
            features=np.asarray(data["features"], dtype=np.float32),
            labels=np.asarray(data.get("labels", np.array([], dtype=object)), dtype=object),
            paths=np.asarray(data.get("paths", np.array([], dtype=object)), dtype=object),
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features.astype(np.float32),
            labels=self.labels.astype(object),
            paths=self.paths.astype(object),
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
        )

    def style_cost(self, qpos: np.ndarray, length: Optional[int] = None, category: Optional[str] = None) -> float:
        feat = qpos_style_feature(qpos, length=length)
        z = (feat - self.mean) / self.std
        bank = (self.features - self.mean[None]) / self.std[None]
        mask = np.ones(len(bank), dtype=bool)
        if category and len(self.labels) == len(bank):
            same = self.labels == category
            if np.any(same):
                mask = same
        dist = np.mean((bank[mask] - z[None]) ** 2, axis=1)
        return float(np.min(dist))
