"""InterHuman native 262D motion representation.

Each person stores 22 global joint positions, 22 global joint velocities,
21 non-root local rotations, and four foot-contact channels. Two-person clips
must share one canonical frame; independently canonicalizing both people would
destroy their relative placement and facing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


INTERHUMAN_JOINTS = 22
INTERHUMAN_DIM = 262
POSITION_SLICE = slice(0, 66)
VELOCITY_SLICE = slice(66, 132)
ROTATION_SLICE = slice(132, 258)
FOOT_CONTACT_SLICE = slice(258, 262)

_RIGHT_HIP = 2
_LEFT_HIP = 1
_LEFT_FOOT = (7, 10)
_RIGHT_FOOT = (8, 11)
_RAW_TO_Y_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class InterHumanCanonicalTransform:
    """First-frame transform returned by the official preprocessing route."""

    root_quaternion: np.ndarray
    root_position_xz: np.ndarray


def _as_motion(motion) -> np.ndarray:
    value = np.asarray(motion, dtype=np.float32)
    if value.ndim < 2 or value.shape[-1] != INTERHUMAN_DIM:
        raise ValueError(f"InterHuman motion must end in 262 channels, got {value.shape}")
    return value


def interhuman262_to_joints(motion) -> np.ndarray:
    """Decode stored global SMPL-22 joint positions."""

    value = _as_motion(motion)
    return value[..., POSITION_SLICE].reshape(value.shape[:-1] + (INTERHUMAN_JOINTS, 3)).copy()


def interhuman262_to_joint_velocities(motion) -> np.ndarray:
    """Decode stored global SMPL-22 per-frame displacements."""

    value = _as_motion(motion)
    return value[..., VELOCITY_SLICE].reshape(value.shape[:-1] + (INTERHUMAN_JOINTS, 3)).copy()


def interhuman262_to_local_rot6d(motion, include_root: bool = False) -> np.ndarray:
    """Decode the 21 non-root local 6D rotations.

    InterHuman does not store a root rotation. With ``include_root=True`` an
    identity root is prepended so the result can enter an SMPL-22 bridge.
    """

    value = _as_motion(motion)
    rotations = value[..., ROTATION_SLICE].reshape(value.shape[:-1] + (21, 6)).copy()
    if not include_root:
        return rotations
    identity = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    root = np.broadcast_to(identity, rotations.shape[:-2] + (1, 6)).copy()
    return np.concatenate([root, rotations], axis=-2)


def interhuman262_to_local_rotmat(motion, include_root: bool = True) -> np.ndarray:
    """Decode local rotation matrices using InterHuman's 6D convention."""

    import torch

    from motius.motion.representation.rotation import rotation_6d_to_matrix

    rotations = interhuman262_to_local_rot6d(motion, include_root=include_root)
    matrices = rotation_6d_to_matrix(torch.from_numpy(rotations), convention="row")
    return matrices.numpy()


def interhuman262_to_foot_contacts(motion) -> np.ndarray:
    """Return the four official foot-contact channels."""

    return _as_motion(motion)[..., FOOT_CONTACT_SLICE].copy()


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < eps):
        raise ValueError("Cannot canonicalize a degenerate hip-facing vector")
    return vector / norm


def _qinv(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float32).copy()
    result[..., 1:] *= -1
    return result


def _qmul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(q, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(r, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _qrot(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    qvec = q[..., 1:]
    uv = np.cross(qvec, vectors)
    uuv = np.cross(qvec, uv)
    return vectors + 2.0 * (q[..., :1] * uv + uuv)


def _qbetween(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    xyz = np.cross(source, target)
    w = np.sqrt(np.sum(source * source, axis=-1, keepdims=True) * np.sum(target * target, axis=-1, keepdims=True))
    w = w + np.sum(source * target, axis=-1, keepdims=True)
    quat = np.concatenate([w, xyz], axis=-1)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    antiparallel = norm[..., 0] < 1e-8
    if np.any(antiparallel):
        source_unit = _normalize(source[antiparallel])
        basis = np.zeros_like(source_unit)
        use_y = np.abs(source_unit[:, 0]) > 0.9
        basis[:, 0] = 1.0
        basis[use_y] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        axis = _normalize(np.cross(source_unit, basis))
        quat[antiparallel, 0] = 0.0
        quat[antiparallel, 1:] = axis
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / norm


def _canonicalize_positions(
    joints,
    *,
    reference_frame: int = 0,
    source_coordinates: str = "interhuman_raw",
) -> Tuple[np.ndarray, InterHumanCanonicalTransform]:
    positions = np.asarray(joints, dtype=np.float32).copy()
    if positions.ndim != 3 or positions.shape[1:] != (INTERHUMAN_JOINTS, 3):
        raise ValueError(f"joints must have shape (T,22,3), got {positions.shape}")
    if len(positions) < 2:
        raise ValueError("InterHuman encoding requires at least two frames")
    if not 0 <= reference_frame < len(positions):
        raise ValueError(f"reference_frame {reference_frame} is outside a {len(positions)}-frame clip")

    source_coordinates = source_coordinates.lower()
    if source_coordinates in {"interhuman_raw", "smpl_z_up", "z_up"}:
        positions = np.einsum("mn,tjn->tjm", _RAW_TO_Y_UP, positions)
    elif source_coordinates not in {"interhuman", "interhuman_y_up", "y_up"}:
        raise ValueError("source_coordinates must be interhuman_raw/z_up or interhuman_y_up/y_up")

    positions[..., 1] -= positions[..., 1].min()
    root_pose = positions[reference_frame].copy()
    root_xz = root_pose[0] * np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    positions -= root_xz

    across = _normalize(root_pose[_RIGHT_HIP] - root_pose[_LEFT_HIP])
    forward = _normalize(np.cross(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), across)[None])[0]
    root_quaternion = _qbetween(forward[None], np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))[0]
    quat = np.broadcast_to(root_quaternion, positions.shape[:-1] + (4,))
    positions = _qrot(quat, positions)
    transform = InterHumanCanonicalTransform(root_quaternion, root_xz[None])
    return positions.astype(np.float32), transform


def _foot_contacts(positions: np.ndarray, threshold: float) -> np.ndarray:
    def detect(indices, heights):
        delta = positions[1:, indices] - positions[:-1, indices]
        speed2 = np.sum(delta * delta, axis=-1)
        height = positions[:-1, indices, 1]
        return ((speed2 < threshold) & (height < np.asarray(heights))).astype(np.float32)

    left = detect(_LEFT_FOOT, (0.12, 0.05))
    right = detect(_RIGHT_FOOT, (0.12, 0.05))
    return np.concatenate([left, right], axis=-1)


def _pack_interhuman262(
    positions: np.ndarray,
    local_rot6d,
    *,
    feet_threshold: float,
) -> np.ndarray:
    rotations = np.asarray(local_rot6d, dtype=np.float32)
    if rotations.shape == (len(positions), 126):
        rotations = rotations.reshape(len(positions), 21, 6)
    if rotations.shape != (len(positions), 21, 6):
        raise ValueError(
            "local_rot6d must contain the 21 non-root joints with shape "
            f"(T,21,6) or (T,126), got {rotations.shape}"
        )
    velocities = positions[1:] - positions[:-1]
    return np.concatenate(
        [
            positions[:-1].reshape(len(positions) - 1, 66),
            velocities.reshape(len(positions) - 1, 66),
            rotations[:-1].reshape(len(positions) - 1, 126),
            _foot_contacts(positions, feet_threshold),
        ],
        axis=-1,
    ).astype(np.float32)


def joints_to_interhuman262(
    joints,
    local_rot6d,
    *,
    feet_threshold: float = 0.001,
    reference_frame: int = 0,
    source_coordinates: str = "interhuman_raw",
    return_transform: bool = False,
):
    """Encode one person using the official InterHuman preprocessing."""

    positions, transform = _canonicalize_positions(
        joints,
        reference_frame=reference_frame,
        source_coordinates=source_coordinates,
    )
    motion = _pack_interhuman262(positions, local_rot6d, feet_threshold=feet_threshold)
    return (motion, transform) if return_transform else motion


def _place_second_person(
    motion: np.ndarray,
    first: InterHumanCanonicalTransform,
    second: InterHumanCanonicalTransform,
) -> np.ndarray:
    relative_rotation = _qmul(second.root_quaternion[None], _qinv(first.root_quaternion[None]))
    half_angle = np.arctan2(relative_rotation[:, 2:3], relative_rotation[:, 0:1])
    offset = _qrot(first.root_quaternion[None], second.root_position_xz - first.root_position_xz)[:, [0, 2]]
    relative = np.concatenate([half_angle, offset], axis=-1)[0]

    output = motion.copy()
    positions = interhuman262_to_joints(output)
    velocities = interhuman262_to_joint_velocities(output)
    quat = np.zeros(positions.shape[:-1] + (4,), dtype=np.float32)
    quat[..., 0] = np.cos(relative[0])
    quat[..., 2] = np.sin(relative[0])
    positions = _qrot(_qinv(quat), positions)
    positions[..., [0, 2]] += relative[1:3]
    velocities = _qrot(_qinv(quat), velocities)
    output[..., POSITION_SLICE] = positions.reshape(output.shape[:-1] + (66,))
    output[..., VELOCITY_SLICE] = velocities.reshape(output.shape[:-1] + (66,))
    return output


def joints_pair_to_interhuman262(
    joints,
    local_rot6d,
    *,
    feet_threshold: float = 0.001,
    reference_frame: int = 0,
    source_coordinates: str = "interhuman_raw",
) -> np.ndarray:
    """Encode a pair as ``(T-1,2,262)`` in person one's canonical frame."""

    positions = np.asarray(joints, dtype=np.float32)
    rotations = np.asarray(local_rot6d, dtype=np.float32)
    if positions.ndim != 4 or positions.shape[1:] != (2, 22, 3):
        raise ValueError(f"paired joints must have shape (T,2,22,3), got {positions.shape}")
    if rotations.shape[:2] != positions.shape[:2]:
        raise ValueError("paired joints and rotations must share T and person dimensions")

    first, first_transform = joints_to_interhuman262(
        positions[:, 0],
        rotations[:, 0],
        feet_threshold=feet_threshold,
        reference_frame=reference_frame,
        source_coordinates=source_coordinates,
        return_transform=True,
    )
    second, second_transform = joints_to_interhuman262(
        positions[:, 1],
        rotations[:, 1],
        feet_threshold=feet_threshold,
        reference_frame=reference_frame,
        source_coordinates=source_coordinates,
        return_transform=True,
    )
    second = _place_second_person(second, first_transform, second_transform)
    return np.stack([first, second], axis=1)


__all__ = [
    "FOOT_CONTACT_SLICE",
    "INTERHUMAN_DIM",
    "INTERHUMAN_JOINTS",
    "InterHumanCanonicalTransform",
    "POSITION_SLICE",
    "ROTATION_SLICE",
    "VELOCITY_SLICE",
    "interhuman262_to_foot_contacts",
    "interhuman262_to_joint_velocities",
    "interhuman262_to_joints",
    "interhuman262_to_local_rot6d",
    "interhuman262_to_local_rotmat",
    "joints_pair_to_interhuman262",
    "joints_to_interhuman262",
]
