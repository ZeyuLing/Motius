"""Canonical coordinate transforms for SMPL-22 joint trajectories."""

from __future__ import annotations

import numpy as np

from .names import SMPL22_FOOT_JOINTS


def canonicalize_smpl22_joints(
    joints,
    *,
    floor_align: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Place the first pelvis at XZ origin and make the body face ``+Z``.

    The first-frame left-to-right hip/shoulder axis defines canonical ``+X``;
    the corresponding forward axis is ``cross(+X, +Y)``. The transform is one
    rigid yaw and translation per clip, so velocity, acceleration, and jerk
    magnitudes are preserved.
    """

    value = np.asarray(joints, dtype=np.float64)
    flattened = value.ndim == 2 and value.shape[1] == 66
    if flattened:
        value = value.reshape(len(value), 22, 3)
    if value.ndim != 3 or value.shape[1:] != (22, 3) or not len(value):
        raise ValueError(f"SMPL-22 joints must have shape (T,22,3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("SMPL-22 joints contain non-finite values")

    output = value.copy()
    output[..., 0] -= output[0, 0, 0]
    output[..., 2] -= output[0, 0, 2]
    right = (output[0, 2] - output[0, 1]) + (output[0, 17] - output[0, 16])
    right[1] = 0.0
    norm = float(np.linalg.norm(right))
    if norm <= eps:
        right = np.asarray([1.0, 0.0, 0.0])
    else:
        right /= norm
    up = np.asarray([0.0, 1.0, 0.0])
    forward = np.cross(right, up)
    basis = np.stack((right, up, forward), axis=-1)
    output = output @ basis
    if floor_align:
        output[..., 1] -= float(output[:, SMPL22_FOOT_JOINTS, 1].min())
    result = output.astype(np.float32)
    return result.reshape(len(result), 66) if flattened else result


__all__ = ["canonicalize_smpl22_joints"]
