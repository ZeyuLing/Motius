"""HumanML3D (HML263) representation helpers.

This is the public entry point for decoding the HumanML3D-263 feature vector to
3D joint positions.

The core HML263 -> 22-joint decoder (:func:`recover_from_ric`) is implemented
natively here (pure torch, no external repository checkout), matching the canonical
HumanML3D / MoMask reference. See
:class:`motius.motion.representation.specs.HML263` for the channel layout.

Encoding arbitrary joints back to HML263 is intentionally not hidden behind a
generic converter: the official path performs canonicalization, skeleton
retargeting, IK, and contact extraction with HumanML3D assets. Motius keeps that
dataset-production protocol separate from this deterministic decoder.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from motius.motion.representation.specs import HML263


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
    "recover_root_rot_pos",
    "recover_from_ric",
    "hml263_to_joints",
]
