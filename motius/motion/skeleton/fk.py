"""Forward kinematics for SMPL-style skeletons (single canonical implementation).

This consolidates the previously-duplicated FK code (``pipelines/motion/
differentiable_fk.py``, ``components/utils/geometry/matrix.forward_kinematics``,
``retarget/smpl_soma.differentiable_fk``) onto one implementation that uses the
unified rotation module.

Conventions
-----------
- ``rot6d`` here is **ROW-major** (the HyMotion / training-data convention,
  ``motion_135``). Internally we call
  ``rotation_6d_to_matrix(..., convention="row")`` which is numerically identical
  to the legacy ``hymotion_m2m.network.geometry.rot6d_to_rotation_matrix``.
- ``bone_offsets[j]`` is the rest offset of joint ``j`` relative to its parent
  (``offsets[0]`` = root rest position, usually ~origin).

All functions are torch-only and differentiable.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from motius.motion.representation.rotation import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
from motius.motion.skeleton.names import SMPL22_PARENTS

NUM_JOINTS = 22


# --------------------------------------------------------------------------- #
# rot6d <-> matrix (row-major) — thin, explicit wrappers over the unified module
# --------------------------------------------------------------------------- #
def rot6d_to_rotmat_row_major(rot6d: Tensor) -> Tensor:
    """Row-major 6D ``(*,6)`` -> rotation matrix ``(*,3,3)``."""
    return rotation_6d_to_matrix(rot6d, convention="row")


def rotmat_to_rot6d_row_major(rotmat: Tensor) -> Tensor:
    """Rotation matrix ``(*,3,3)`` -> row-major 6D ``(*,6)``."""
    return matrix_to_rotation_6d(rotmat, convention="row")


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def forward_kinematics(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
    parents: Sequence[int] = SMPL22_PARENTS,
) -> Tuple[Tensor, Tensor]:
    """Differentiable FK from local rotation matrices + root translation.

    Args:
        local_rotmat: ``(*, J, 3, 3)`` local (parent-relative) rotation matrices.
        translation: ``(*, 3)`` root translation.
        bone_offsets: ``(J, 3)`` rest offsets relative to parent.
        parents: parent index per joint (root = -1). Defaults to SMPL-22.

    Returns:
        world_positions: ``(*, J, 3)``.
        world_rotations: ``(*, J, 3, 3)``.
    """
    num_joints = len(parents)
    world_rot_list: list = [None] * num_joints
    world_pos_list: list = [None] * num_joints

    for j in range(num_joints):
        parent = parents[j]
        if parent < 0:
            world_rot_list[j] = local_rotmat[..., j, :, :]
            world_pos_list[j] = translation + bone_offsets[j]
        else:
            world_rot_list[j] = world_rot_list[parent] @ local_rotmat[..., j, :, :]
            offset_rotated = (
                world_rot_list[parent] @ bone_offsets[j].unsqueeze(-1)
            ).squeeze(-1)
            world_pos_list[j] = world_pos_list[parent] + offset_rotated

    world_pos = torch.stack(world_pos_list, dim=-2)
    world_rot = torch.stack(world_rot_list, dim=-3)
    return world_pos, world_rot


def differentiable_fk(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
) -> Tuple[Tensor, Tensor]:
    """SMPL-22 FK (back-compat alias of :func:`forward_kinematics`)."""
    return forward_kinematics(local_rotmat, translation, bone_offsets, SMPL22_PARENTS)


# --------------------------------------------------------------------------- #
# Local <-> global rotation propagation (row-major rot6d)
# --------------------------------------------------------------------------- #
def local_to_global_rot6d(
    rot6d_row: Tensor, parents: Sequence[int] = SMPL22_PARENTS
) -> Tensor:
    """Propagate local rotations to global along the tree. ``(*, J, 6)`` row-major.

    ``global[j] = global[parent] @ local[j]``.
    """
    mats = rot6d_to_rotmat_row_major(rot6d_row)  # (*, J, 3, 3)
    num_joints = len(parents)
    g: list = [None] * num_joints
    for j in range(num_joints):
        p = parents[j]
        g[j] = mats[..., j, :, :] if p < 0 else g[p] @ mats[..., j, :, :]
    gmat = torch.stack(g, dim=-3)
    return rotmat_to_rot6d_row_major(gmat)


def global_to_local_rot6d(
    rot6d_row: Tensor, parents: Sequence[int] = SMPL22_PARENTS
) -> Tensor:
    """Inverse of :func:`local_to_global_rot6d`. ``(*, J, 6)`` row-major.

    ``local[j] = global[parent]^T @ global[j]``.
    """
    gmat = rot6d_to_rotmat_row_major(rot6d_row)  # (*, J, 3, 3)
    num_joints = len(parents)
    out: list = [None] * num_joints
    for j in range(num_joints):
        p = parents[j]
        if p < 0:
            out[j] = gmat[..., j, :, :]
        else:
            out[j] = gmat[..., p, :, :].transpose(-1, -2) @ gmat[..., j, :, :]
    lmat = torch.stack(out, dim=-3)
    return rotmat_to_rot6d_row_major(lmat)


# --------------------------------------------------------------------------- #
# motion_135 <-> FK
# --------------------------------------------------------------------------- #
def motion135_to_fk(
    motion_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = "local",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Parse denormalized ``motion_135`` and run FK.

    Args:
        motion_denorm: ``(*, 135)`` = ``[transl(3), 22*rot6d_row(132)]``.
        bone_offsets: ``(22, 3)`` rest offsets.
        rotation_space: ``"local"`` (default) or ``"global"``.

    Returns:
        ``(world_pos (*,22,3), world_rot (*,22,3,3), transl (*,3), local_rotmat (*,22,3,3))``.
    """
    leading = motion_denorm.shape[:-1]
    translation = motion_denorm[..., :3]
    rot6d = motion_denorm[..., 3:135].reshape(*leading, 22, 6)

    if rotation_space == "global":
        rot6d = global_to_local_rot6d(rot6d)
    elif rotation_space != "local":
        raise ValueError(f"rotation_space must be 'local'/'global', got {rotation_space!r}")

    local_rotmat = rot6d_to_rotmat_row_major(rot6d)
    world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)
    return world_pos, world_rot, translation, local_rotmat


def fk_to_motion135(
    local_rotmat: Tensor,
    translation: Tensor,
    rotation_space: str = "local",
) -> Tensor:
    """Encode local rotation matrices + translation back to ``motion_135``."""
    leading = local_rotmat.shape[:-3]
    rot6d = rotmat_to_rot6d_row_major(local_rotmat)  # (*, 22, 6)
    if rotation_space == "global":
        rot6d = local_to_global_rot6d(rot6d)
    elif rotation_space != "local":
        raise ValueError(f"rotation_space must be 'local'/'global', got {rotation_space!r}")
    rot6d_flat = rot6d.reshape(*leading, 132)
    return torch.cat([translation, rot6d_flat], dim=-1)


__all__ = [
    "NUM_JOINTS",
    "rot6d_to_rotmat_row_major",
    "rotmat_to_rot6d_row_major",
    "forward_kinematics",
    "differentiable_fk",
    "local_to_global_rot6d",
    "global_to_local_rot6d",
    "motion135_to_fk",
    "fk_to_motion135",
]
