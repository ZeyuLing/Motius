"""HumanML3D (HML263) representation helpers.

This is the public entry point for decoding the HumanML3D-263 feature vector to
3D joint positions.

The core HML263 -> 22-joint decoder (:func:`recover_from_ric`) is implemented
natively here (pure torch, no external repository checkout), matching the canonical
HumanML3D / MoMask reference. See
:class:`motius.motion.representation.specs.HML263` for the channel layout.

The inverse path is implemented here as a first-class protocol API. It follows
the official HumanML3D canonicalization, skeleton retargeting, IK, velocity, and
foot-contact extraction recipe without requiring another repository checkout.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import Tensor

from motius.motion.representation.specs import HML263


HML263_RAW_OFFSETS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

HML263_KINEMATIC_CHAINS = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)

# Bone lengths extracted from the official HumanML3D target sequence 000021.
# Directions come from HML263_RAW_OFFSETS, exactly as in the official pipeline.
HML263_TARGET_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.1030739695, 0.0, 0.0],
        [-0.1098833680, 0.0, 0.0],
        [0.0, 0.1315682381, 0.0],
        [0.0, -0.3936232626, 0.0],
        [0.0, -0.3901882172, 0.0],
        [0.0, 0.1431902200, 0.0],
        [0.0, -0.4324331284, 0.0],
        [0.0, -0.4256435037, 0.0],
        [0.0, 0.0573647358, 0.0],
        [0.0, 0.0, 0.1433817595],
        [0.0, 0.0, 0.1494190246],
        [0.0, 0.2193600535, 0.0],
        [0.1374867707, 0.0, 0.0],
        [-0.1433828324, 0.0, 0.0],
        [0.0, 0.0, 0.1030392274],
        [0.0, -0.1316139847, 0.0],
        [0.0, -0.1229843721, 0.0],
        [0.0, -0.2568399906, 0.0],
        [0.0, -0.2630918622, 0.0],
        [0.0, -0.2660117149, 0.0],
        [0.0, -0.2698763907, 0.0],
    ],
    dtype=np.float32,
)

_HML263_FACE_JOINTS = (2, 1, 17, 16)
_HML263_LEFT_FEET = (7, 10)
_HML263_RIGHT_FEET = (8, 11)
_HML263_LEG_SCALE_JOINTS = (5, 8)


def _normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, eps)


def _qinv_np(quaternion: np.ndarray) -> np.ndarray:
    return quaternion * np.asarray([1, -1, -1, -1], dtype=quaternion.dtype)


def _qmul_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_t = torch.from_numpy(np.ascontiguousarray(left)).float()
    right_t = torch.from_numpy(np.ascontiguousarray(right)).float()
    original_shape = left_t.shape
    terms = torch.bmm(right_t.reshape(-1, 4, 1), left_t.reshape(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).reshape(original_shape).numpy()


def _qrot_np(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(np.ascontiguousarray(quaternion)).float().reshape(-1, 4)
    v = torch.from_numpy(np.ascontiguousarray(vector)).float().reshape(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).reshape(vector.shape).numpy()


def _qbetween_np(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_t = torch.from_numpy(np.ascontiguousarray(source)).float()
    target_t = torch.from_numpy(np.ascontiguousarray(target)).float()
    xyz = torch.cross(source_t, target_t, dim=-1)
    real = torch.sqrt(
        (source_t.square()).sum(dim=-1, keepdim=True)
        * (target_t.square()).sum(dim=-1, keepdim=True)
    ) + (source_t * target_t).sum(dim=-1, keepdim=True)
    quaternion = torch.cat([real, xyz], dim=-1)
    quaternion = quaternion / torch.linalg.norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return quaternion.numpy()


def _quaternion_to_cont6d_np(quaternion: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(np.ascontiguousarray(quaternion)).float()
    real, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * real),
            two_s * (i * k + j * real),
            two_s * (i * j + k * real),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * real),
            two_s * (i * k - j * real),
            two_s * (j * k + i * real),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1).numpy()


def _parents_from_chains() -> tuple[int, ...]:
    parents = [0] * len(HML263_RAW_OFFSETS)
    parents[0] = -1
    for chain in HML263_KINEMATIC_CHAINS:
        for parent, child in zip(chain[:-1], chain[1:]):
            parents[child] = parent
    return tuple(parents)


_HML263_PARENTS = _parents_from_chains()


def _offsets_from_joints(joints: np.ndarray) -> np.ndarray:
    offsets = HML263_RAW_OFFSETS.copy()
    for joint, parent in enumerate(_HML263_PARENTS[1:], start=1):
        length = np.linalg.norm(joints[joint] - joints[parent])
        offsets[joint] *= length
    return offsets


def _inverse_kinematics_np(
    joints: np.ndarray, *, smooth_forward: bool = False
) -> np.ndarray:
    first_hip, second_hip, right_shoulder, left_shoulder = _HML263_FACE_JOINTS
    across = (
        joints[:, second_hip]
        - joints[:, first_hip]
        + joints[:, right_shoulder]
        - joints[:, left_shoulder]
    )
    across = _normalize_np(across)
    forward = np.cross(np.asarray([[0.0, 1.0, 0.0]]), across, axis=-1)
    if smooth_forward:
        forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = _normalize_np(forward)
    target = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), len(joints), axis=0)
    root_quaternion = _qbetween_np(forward, target)
    root_quaternion[0] = np.asarray([1.0, 0.0, 0.0, 0.0])

    quaternion = np.zeros(joints.shape[:-1] + (4,), dtype=np.float64)
    quaternion[:, 0] = root_quaternion
    for chain in HML263_KINEMATIC_CHAINS:
        rotation = root_quaternion
        for parent, child in zip(chain[:-1], chain[1:]):
            source = np.repeat(HML263_RAW_OFFSETS[child][None], len(joints), axis=0)
            target_direction = _normalize_np(joints[:, child] - joints[:, parent])
            source_to_target = _qbetween_np(source, target_direction)
            local_rotation = _qmul_np(_qinv_np(rotation), source_to_target)
            quaternion[:, child] = local_rotation
            rotation = _qmul_np(rotation, local_rotation)
    return quaternion


def _forward_kinematics_np(
    quaternion: np.ndarray, root_position: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    joints = np.zeros(quaternion.shape[:-1] + (3,), dtype=np.float64)
    joints[:, 0] = root_position
    repeated_offsets = np.repeat(offsets[None], len(quaternion), axis=0)
    for chain in HML263_KINEMATIC_CHAINS:
        rotation = quaternion[:, 0]
        for parent, child in zip(chain[:-1], chain[1:]):
            rotation = _qmul_np(rotation, quaternion[:, child])
            joints[:, child] = (
                _qrot_np(rotation, repeated_offsets[:, child]) + joints[:, parent]
            )
    return joints


def _uniform_skeleton(
    joints: np.ndarray, target_offsets: np.ndarray
) -> np.ndarray:
    source_offsets = _offsets_from_joints(joints[0])
    upper_leg, lower_leg = _HML263_LEG_SCALE_JOINTS
    source_leg = np.abs(source_offsets[upper_leg]).max() + np.abs(
        source_offsets[lower_leg]
    ).max()
    target_leg = np.abs(target_offsets[upper_leg]).max() + np.abs(
        target_offsets[lower_leg]
    ).max()
    if source_leg <= 0:
        raise ValueError("cannot canonicalize a skeleton with zero leg length")
    root_position = joints[:, 0] * (target_leg / source_leg)
    quaternion = _inverse_kinematics_np(joints)
    return _forward_kinematics_np(quaternion, root_position, target_offsets)


def _canonicalize_hml263_joints(
    joints: np.ndarray, target_offsets: np.ndarray
) -> np.ndarray:
    positions = _uniform_skeleton(joints, target_offsets)
    positions[:, :, 1] -= positions[:, :, 1].min()
    initial = positions[0].copy()
    positions -= initial[0] * np.asarray([1.0, 0.0, 1.0])

    right_hip, left_hip, right_shoulder, left_shoulder = _HML263_FACE_JOINTS
    across = (
        initial[right_hip]
        - initial[left_hip]
        + initial[right_shoulder]
        - initial[left_shoulder]
    )
    across = _normalize_np(across)
    forward = _normalize_np(
        np.cross(np.asarray([[0.0, 1.0, 0.0]]), across[None], axis=-1)
    )
    root_rotation = _qbetween_np(forward, np.asarray([[0.0, 0.0, 1.0]]))
    root_rotation = np.broadcast_to(
        root_rotation, positions.shape[:-1] + (4,)
    ).copy()
    return _qrot_np(root_rotation, positions)


def _foot_contacts(
    positions: np.ndarray, feet: tuple[int, int], threshold: float
) -> np.ndarray:
    velocity_squared = (
        (positions[1:, feet, :] - positions[:-1, feet, :]) ** 2
    ).sum(axis=-1)
    return (velocity_squared < threshold).astype(np.float64)


def joints_to_hml263(
    joints,
    *,
    feet_threshold: float = 0.002,
    target_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Encode 20-fps SMPL-22 joints with the official HumanML3D protocol.

    The input must already use HumanML3D coordinates: Y up, metric units, and
    the standard first-22 SMPL joint order. This function does not resample or
    guess a source coordinate system.
    """

    positions = np.asarray(joints, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"joints must have shape (T,22,3), got {positions.shape}")
    if len(positions) < 2:
        raise ValueError("HumanML3D encoding requires at least two frames")
    if not np.isfinite(positions).all():
        raise ValueError("joints contain NaN or infinite values")
    offsets = (
        HML263_TARGET_OFFSETS
        if target_offsets is None
        else np.asarray(target_offsets, dtype=np.float32)
    )
    if offsets.shape != (22, 3):
        raise ValueError(f"target_offsets must have shape (22,3), got {offsets.shape}")

    canonical = _canonicalize_hml263_joints(positions, offsets)
    global_positions = canonical.copy()
    left_contact = _foot_contacts(canonical, _HML263_LEFT_FEET, feet_threshold)
    right_contact = _foot_contacts(canonical, _HML263_RIGHT_FEET, feet_threshold)

    quaternion = _inverse_kinematics_np(canonical, smooth_forward=True)
    cont6d = _quaternion_to_cont6d_np(quaternion)
    root_rotation = quaternion[:, 0].copy()
    root_velocity = canonical[1:, 0] - canonical[:-1, 0]
    root_velocity = _qrot_np(root_rotation[1:], root_velocity)
    root_angular_velocity = _qmul_np(
        root_rotation[1:], _qinv_np(root_rotation[:-1])
    )

    local_positions = canonical.copy()
    local_positions[..., 0] -= local_positions[:, 0:1, 0]
    local_positions[..., 2] -= local_positions[:, 0:1, 2]
    local_positions = _qrot_np(
        np.repeat(root_rotation[:, None], 22, axis=1), local_positions
    )

    root_height = local_positions[:, 0, 1:2]
    root_data = np.concatenate(
        [
            np.arcsin(root_angular_velocity[:, 2:3]),
            root_velocity[:, [0, 2]],
            root_height[:-1],
        ],
        axis=-1,
    )
    ric_data = local_positions[:, 1:].reshape(len(local_positions), -1)
    rotation_data = cont6d[:, 1:].reshape(len(cont6d), -1)
    local_velocity = _qrot_np(
        np.repeat(root_rotation[:-1, None], 22, axis=1),
        global_positions[1:] - global_positions[:-1],
    ).reshape(len(global_positions) - 1, -1)

    features = np.concatenate(
        [
            root_data,
            ric_data[:-1],
            rotation_data[:-1],
            local_velocity,
            left_contact,
            right_contact,
        ],
        axis=-1,
    )
    if features.shape[-1] != HML263.dim:
        raise RuntimeError(f"internal HML263 layout error: got {features.shape}")
    return features.astype(np.float32)


def joints_to_humanml263(joints, **kwargs) -> np.ndarray:
    """Descriptive alias of :func:`joints_to_hml263`."""

    return joints_to_hml263(joints, **kwargs)


def linear_resample_joints(
    joints: np.ndarray, src_fps: float, dst_fps: float
) -> np.ndarray:
    """Phase-aligned linear resampling for a time-major joint sequence."""

    positions = np.asarray(joints)
    if len(positions) < 2 or abs(src_fps - dst_fps) < 1e-9:
        return positions.copy()
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be positive")
    duration = (len(positions) - 1) / src_fps
    output_frames = max(int(round(duration * dst_fps)) + 1, 2)
    source_time = np.arange(len(positions), dtype=np.float64) / src_fps
    target_time = np.clip(
        np.arange(output_frames, dtype=np.float64) / dst_fps,
        source_time[0],
        source_time[-1],
    )
    flat = positions.reshape(len(positions), -1)
    output = np.empty((output_frames, flat.shape[1]), dtype=np.float64)
    for channel in range(flat.shape[1]):
        output[:, channel] = np.interp(target_time, source_time, flat[:, channel])
    return output.reshape((output_frames,) + positions.shape[1:]).astype(positions.dtype)


# --------------------------------------------------------------------------- #
# Quaternion helpers (HumanML3D root recovery uses yaw quaternions)
# --------------------------------------------------------------------------- #
def _qinv(q: Tensor) -> Tensor:
    """Quaternion inverse (w,x,y,z) for unit quaternions."""
    assert q.shape[-1] == 4
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def _qrot(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector(s) ``v`` (...,3) by quaternion(s) ``q`` (...,4), w-first."""
    assert q.shape[-1] == 4 and v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data: Tensor) -> Tuple[Tensor, Tensor]:
    """Recover root yaw quaternion and root position from HML263 features.

    Args:
        data: ``(..., T, 263)`` HumanML3D features.

    Returns:
        ``(r_rot_quat (...,T,4), r_pos (...,T,3))``.
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    # integrate angular velocity over time (yaw)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device, dtype=data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # rotate planar velocities back to world, then integrate
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]  # absolute root height
    return r_rot_quat, r_pos


def recover_from_ric(data: Tensor, joints_num: int = 22) -> Tensor:
    """Decode HumanML3D-263 features to world-space joint positions.

    Native re-implementation of the canonical HumanML3D / MoMask
    ``recover_from_ric``.

    Args:
        data: ``(..., T, 263)`` HumanML3D features (un-normalized).
        joints_num: number of joints (22 for the SMPL body subset).

    Returns:
        ``(..., T, joints_num, 3)`` world-space joint positions.
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # rotate non-root joints from heading-aligned frame back to world
    positions = _qrot(
        _qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions,
    )
    # add root planar position
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    # prepend root joint
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def hml263_to_joints(m263, joints_num: int = 22):
    """Decode HML263 -> ``(T, joints_num, 3)`` joints (accepts numpy or torch).

    Returns the same array type as the input.
    """
    import numpy as np

    is_np = isinstance(m263, np.ndarray)
    t = torch.as_tensor(m263, dtype=torch.float32) if is_np else m263
    if t.shape[-1] != HML263.dim:
        raise ValueError(f"expected last dim {HML263.dim}, got {t.shape[-1]}")
    joints = recover_from_ric(t, joints_num)
    return joints.detach().cpu().numpy() if is_np else joints


__all__ = [
    "HML263_RAW_OFFSETS",
    "HML263_KINEMATIC_CHAINS",
    "HML263_TARGET_OFFSETS",
    "recover_root_rot_pos",
    "recover_from_ric",
    "hml263_to_joints",
    "joints_to_hml263",
    "joints_to_humanml263",
    "linear_resample_joints",
]
