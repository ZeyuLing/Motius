# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo motion completion, adapted from the upstream implementation."""

from typing import Dict
import torch

from motius.motion.body_models.kimodo import SkeletonBase
from motius.motion.representation.kimodo.feature_utils import compute_heading_angle, compute_vel_xyz
from motius.motion.representation.kimodo.feet import foot_detect_from_pos_and_vel
from motius.motion.representation.kimodo.smooth_root import get_smooth_root_pos

def _coerce_time_local_root(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize to shapes (T, J, 3, 3) and (T, 3)."""
    if local_rot_mats.dim() == 5:
        if int(local_rot_mats.shape[0]) != 1:
            raise ValueError(f"local_rot_mats batch size must be 1 for single clip; got {local_rot_mats.shape[0]}")
        local_rot_mats = local_rot_mats[0]
    if root_positions.dim() == 3:
        if int(root_positions.shape[0]) != 1:
            raise ValueError(f"root_positions batch size must be 1; got {root_positions.shape[0]}")
        root_positions = root_positions[0]
    if local_rot_mats.dim() != 4:
        raise ValueError(f"local_rot_mats must be (T,J,3,3); got {tuple(local_rot_mats.shape)}")
    if root_positions.dim() != 2 or int(root_positions.shape[-1]) != 3:
        raise ValueError(f"root_positions must be (T,3); got {tuple(root_positions.shape)}")
    if int(local_rot_mats.shape[0]) != int(root_positions.shape[0]):
        raise ValueError("local_rot_mats and root_positions must have the same number of frames")
    return local_rot_mats, root_positions


def complete_motion_dict(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SkeletonBase,
    fps: float,
) -> Dict[str, torch.Tensor]:
    """Build the Kimodo motion output dict from local rotations and root positions.

    Matches keys written by CLI generation (see docs/source/user_guide/output_formats.md).

    Args:
        local_rot_mats: (T, J, 3, 3) or (1, T, J, 3, 3) local rotation matrices.
        root_positions: (T, 3) or (1, T, 3) root / pelvis world positions (meters).
        skeleton: Skeleton instance (SOMA77, G1, SMPL-X, etc.).
        fps: Sampling rate (Hz).

    Returns:
        Dict with tensors ``posed_joints``, ``global_rot_mats``, ``local_rot_mats``,
        ``foot_contacts``, ``smooth_root_pos``, ``root_positions``, ``global_root_heading``.
    """
    device = local_rot_mats.device
    dtype = local_rot_mats.dtype
    local_rot_mats, root_positions = _coerce_time_local_root(
        local_rot_mats.to(device=device, dtype=dtype),
        root_positions.to(device=device, dtype=dtype),
    )

    global_rot_mats, posed_joints, _ = skeleton.fk(local_rot_mats, root_positions)

    smooth_root_pos = get_smooth_root_pos(root_positions.unsqueeze(0)).squeeze(0)

    lengths = torch.tensor([posed_joints.shape[0]], device=device)
    velocities = compute_vel_xyz(posed_joints.unsqueeze(0), fps, lengths=lengths).squeeze(0)

    heading_angle = compute_heading_angle(posed_joints.unsqueeze(0), skeleton).squeeze(0)
    global_root_heading = torch.stack([torch.cos(heading_angle), torch.sin(heading_angle)], dim=-1)

    foot_contacts = foot_detect_from_pos_and_vel(
        posed_joints.unsqueeze(0),
        velocities.unsqueeze(0),
        skeleton,
        0.15,
        0.10,
    ).squeeze(0)

    return {
        "posed_joints": posed_joints,
        "global_rot_mats": global_rot_mats,
        "local_rot_mats": local_rot_mats,
        "foot_contacts": foot_contacts,
        "smooth_root_pos": smooth_root_pos,
        "root_positions": root_positions,
        "global_root_heading": global_root_heading,
    }
