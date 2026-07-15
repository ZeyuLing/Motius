"""BABEL/FlowMDM 135-dimensional SMPL body representation.

This representation is numerically 135-dimensional but is not Motius
``motion135``.  BABEL stores root height, planar root velocity, and 22 local
rotations using the first two *rows* of each rotation matrix.  The helpers in
this module keep that distinction explicit and provide the canonical bridge to
Motius' Y-up SMPL-22 ``motion135`` and joints.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from motius.motion.skeleton.names import SMPL22_PARENTS


BABEL135_DIM = 135

_Z_UP_TO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def _as_float_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_rows(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def babel_rows6d_to_matrix(rot6d) -> np.ndarray:
    """Decode BABEL's ``R[:2, :].reshape(6)`` rotation layout."""

    value = _as_float_array(rot6d, name="rot6d")
    if value.shape[-1] != 6:
        raise ValueError(f"BABEL rotation 6D must end in 6 channels, got {value.shape}")
    row0 = _normalize_rows(value[..., :3])
    row1_raw = value[..., 3:]
    row1 = _normalize_rows(row1_raw - (row0 * row1_raw).sum(-1, keepdims=True) * row0)
    row2 = np.cross(row0, row1, axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def matrix_to_babel_rows6d(matrix) -> np.ndarray:
    """Encode matrices as BABEL's first-two-rows rotation 6D layout."""

    value = _as_float_array(matrix, name="matrix")
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in (3,3), got {value.shape}")
    return value[..., :2, :].reshape(*value.shape[:-2], 6)


def _stats(features: np.ndarray, mean, std) -> tuple[np.ndarray, np.ndarray]:
    mean_array = _as_float_array(mean, name="mean").reshape(-1)
    std_array = _as_float_array(std, name="std").reshape(-1)
    if mean_array.shape != (BABEL135_DIM,) or std_array.shape != (BABEL135_DIM,):
        raise ValueError(
            f"BABEL stats must both have shape ({BABEL135_DIM},), got "
            f"{mean_array.shape} and {std_array.shape}"
        )
    if np.any(std_array <= 0):
        raise ValueError("BABEL std must be strictly positive")
    return mean_array, std_array


def encode_babel135(
    poses,
    trans,
    *,
    mean=None,
    std=None,
    canonicalize: bool = True,
    facing_offset: float = np.pi / 2,
) -> np.ndarray:
    """Encode SMPL-H body poses and Z-up translations as BABEL-135.

    The canonicalization reproduces FlowMDM's ``Globalvelandy`` transform:
    the first-frame root yaw (plus ``facing_offset``) is removed, root planar
    translation is represented as velocity, and absolute Z height is retained.
    ``poses`` accepts ``(T, >=66)`` axis-angle channels or ``(T, >=22, 3)``.
    """

    pose = _as_float_array(poses, name="poses")
    translation = _as_float_array(trans, name="trans")
    if pose.ndim == 2:
        pose = pose.reshape(len(pose), -1, 3)
    if pose.ndim != 3 or pose.shape[1] < 22 or pose.shape[2] != 3:
        raise ValueError(f"poses must have shape (T,>=22,3), got {pose.shape}")
    if translation.shape != (len(pose), 3):
        raise ValueError(f"trans must have shape ({len(pose)},3), got {translation.shape}")
    if len(pose) == 0:
        return np.empty((0, BABEL135_DIM), dtype=np.float32)

    rotations = axis_angle_to_matrix(pose[:, :22].reshape(-1, 3)).reshape(
        len(pose), 22, 3, 3
    )
    planar_velocity = np.diff(translation[:, :2], axis=0, prepend=translation[[0], :2])
    if canonicalize:
        root_axis_angle = matrix_to_axis_angle(rotations[0, 0])
        yaw = float(root_axis_angle[2]) + float(facing_offset)
        canonicalizer = axis_angle_to_matrix(np.asarray([0.0, 0.0, yaw]))
        rotations = rotations.copy()
        rotations[:, 0] = canonicalizer.T @ rotations[:, 0]
        planar_velocity = planar_velocity @ canonicalizer[:2, :2]

    features = np.concatenate(
        (
            translation[:, 2:3],
            planar_velocity,
            matrix_to_babel_rows6d(rotations).reshape(len(pose), 132),
        ),
        axis=-1,
    )
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")
    if mean is not None:
        mean_array, std_array = _stats(features, mean, std)
        features = (features - mean_array) / std_array
    return features.astype(np.float32)


def decode_babel135(
    features,
    *,
    mean=None,
    std=None,
    target_up_axis: Literal["z", "y"] = "y",
) -> dict[str, np.ndarray]:
    """Decode BABEL-135 into local rotations and root translation."""

    value = _as_float_array(features, name="features")
    if value.ndim != 2 or value.shape[1] != BABEL135_DIM:
        raise ValueError(f"BABEL features must have shape (T,135), got {value.shape}")
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")
    if mean is not None:
        mean_array, std_array = _stats(value, mean, std)
        value = value * std_array + mean_array

    trajectory = np.cumsum(value[:, 1:3], axis=0)
    if len(trajectory):
        trajectory -= trajectory[[0]]
    translation = np.concatenate((trajectory, value[:, :1]), axis=-1)
    rotations = babel_rows6d_to_matrix(value[:, 3:].reshape(len(value), 22, 6))

    if target_up_axis == "y":
        # AMASS/BABEL is Z-up, while the SMPL rest skeleton is Y-up.  Only the
        # root rotation maps body coordinates into the world frame; all other
        # rotations remain parent-local and must not be basis-conjugated.
        translation = translation @ _Z_UP_TO_Y_UP.T
        rotations = rotations.copy()
        rotations[:, 0] = _Z_UP_TO_Y_UP @ rotations[:, 0]
    elif target_up_axis != "z":
        raise ValueError(f"target_up_axis must be 'z' or 'y', got {target_up_axis!r}")

    return {
        "translation": translation.astype(np.float32),
        "local_rotations": rotations.astype(np.float32),
    }


def babel135_to_motion135(features, **kwargs) -> np.ndarray:
    """Convert BABEL-135 to Motius Y-up ``motion135``."""

    decoded = decode_babel135(features, **kwargs)
    rotation = matrix_to_rotation_6d(
        decoded["local_rotations"], convention="row"
    ).reshape(len(decoded["translation"]), 132)
    return np.concatenate((decoded["translation"], rotation), axis=-1).astype(np.float32)


def babel135_to_joints(features, *, bone_offsets, **kwargs) -> np.ndarray:
    """Decode BABEL-135 to canonical SMPL-22 joints with explicit offsets.

    ``bone_offsets`` must use the native SMPL body frame, as returned by
    :func:`motius.motion.skeleton.smpl22_rest_offsets`.  For Y-up output, the
    root model-origin offset is converted to the output world basis while the
    parent-local child offsets remain unchanged.
    """

    import torch

    from motius.motion.skeleton.fk import forward_kinematics

    decoded = decode_babel135(features, **kwargs)
    offsets = _as_float_array(bone_offsets, name="bone_offsets")
    if offsets.shape != (22, 3):
        raise ValueError(f"bone_offsets must have shape (22,3), got {offsets.shape}")
    offsets = offsets.copy()
    if kwargs.get("target_up_axis", "y") == "y":
        offsets[0] = _Z_UP_TO_Y_UP @ offsets[0]
    joints, _ = forward_kinematics(
        torch.from_numpy(decoded["local_rotations"]),
        torch.from_numpy(decoded["translation"]),
        torch.from_numpy(offsets.astype(np.float32)),
    )
    return joints.numpy().astype(np.float32)


def infer_smpl22_offsets(
    poses,
    trans,
    joint_positions,
    *,
    target_up_axis: Literal["z", "y"] = "y",
    parents: Sequence[int] = SMPL22_PARENTS,
) -> np.ndarray:
    """Recover fixed SMPL-22 rest offsets from a processed BABEL sequence."""

    pose = _as_float_array(poses, name="poses")
    translation = _as_float_array(trans, name="trans")
    joints = _as_float_array(joint_positions, name="joint_positions")
    if pose.ndim == 2:
        pose = pose.reshape(len(pose), -1, 3)
    if pose.ndim != 3 or pose.shape[1] < 22 or pose.shape[2] != 3:
        raise ValueError(f"poses must have shape (T,>=22,3), got {pose.shape}")
    if translation.shape != (len(pose), 3):
        raise ValueError(f"trans must have shape ({len(pose)},3), got {translation.shape}")
    if joints.ndim != 3 or joints.shape[0] != len(pose) or joints.shape[1] < 22:
        raise ValueError(f"joint_positions must have shape (T,>=22,3), got {joints.shape}")

    local = axis_angle_to_matrix(pose[:, :22].reshape(-1, 3)).reshape(
        len(pose), 22, 3, 3
    )
    world = np.empty_like(local)
    offsets = np.empty((22, 3), dtype=np.float64)
    for joint, parent in enumerate(parents):
        if parent < 0:
            world[:, joint] = local[:, joint]
            offsets[joint] = np.median(joints[:, joint] - translation, axis=0)
        else:
            world[:, joint] = world[:, parent] @ local[:, joint]
            delta = joints[:, joint] - joints[:, parent]
            local_delta = (world[:, parent].transpose(0, 2, 1) @ delta[..., None])[..., 0]
            offsets[joint] = np.median(local_delta, axis=0)

    if target_up_axis == "y":
        # Child offsets are parent-local SMPL quantities.  Only the root joint
        # offset is expressed directly in the source world basis.
        offsets[0] = _Z_UP_TO_Y_UP @ offsets[0]
    elif target_up_axis != "z":
        raise ValueError(f"target_up_axis must be 'z' or 'y', got {target_up_axis!r}")
    return offsets.astype(np.float32)


__all__ = [
    "BABEL135_DIM",
    "babel_rows6d_to_matrix",
    "matrix_to_babel_rows6d",
    "encode_babel135",
    "decode_babel135",
    "babel135_to_motion135",
    "babel135_to_joints",
    "infer_smpl22_offsets",
]
