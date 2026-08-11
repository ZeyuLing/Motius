# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMR motion representation: global root, global joints, velocities, and foot contacts."""

from typing import Optional

import einops
import torch

from ...skeleton.kinematics import fk
from ...tools import ensure_batched, to_numpy
from ..feature_utils import RotateFeatures, compute_heading_angle, compute_vel_xyz
from ..feet import foot_detect_from_pos_and_vel
from .base import MotionRepBase


class TMRMotionRep(MotionRepBase):
    """Motion representation with global root and global joint positions.

    Feature layout:
    - root position ``(x, y, z)``
    - root heading as ``(cos(theta), sin(theta))``
    - local joint positions (root removed, ground-referenced)
    - global joint velocities
    - binary foot contacts
    """

    def __init__(
        self,
        skeleton,
        fps,
        stats_path: Optional[str] = None,
    ):
        nbjoints = skeleton.nbjoints

        self.size_dict = {
            "root_pos": torch.Size([3]),
            "global_root_heading": torch.Size([2]),
            "local_joints_positions": torch.Size([nbjoints - 1, 3]),
            "velocities": torch.Size([nbjoints, 3]),
            "foot_contacts": torch.Size([4]),
        }
        self.last_root_feature = "global_root_heading"
        self.local_root_size_dict = {
            "local_root_rot_vel": torch.Size([1]),
            "local_root_vel": torch.Size([2]),
            "global_root_y": torch.Size([1]),
        }
        super().__init__(skeleton, fps, stats_path)

    @ensure_batched(local_joint_rots=5, root_positions=3, posed_joints=4, lengths=1)
    def __call__(
        self,
        local_joint_rots: Optional[torch.Tensor] = None,
        root_positions: Optional[torch.Tensor] = None,
        posed_joints: Optional[torch.Tensor] = None,
        *,
        to_normalize: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert motion inputs to this feature representation.

        Args:
            local_joint_rots: Local joint rotation matrices ``[B, T, J, 3, 3]``.
                Required when ``posed_joints`` is not provided.
            root_positions: Root translations ``[B, T, 3]``. Required when
                ``posed_joints`` is not provided.
            posed_joints: Optional precomputed global joint positions
                ``[B, T, J, 3]``. If passed, FK is skipped.
            to_normalize: Whether to normalize output features.
            lengths: Optional valid lengths for variable-length batches.

        Returns:
            Motion features with shape ``[B, T, motion_rep_dim]``.
        """
        if posed_joints is not None:
            device = posed_joints.device
            nbatch, nbframes, nbjoints = posed_joints.shape[:3]
        else:
            device = local_joint_rots.device
            nbatch, nbframes, nbjoints = local_joint_rots.shape[:3]

        if lengths is None:
            assert nbatch == 1, "If lenghts is not provided, the input should not be batched."
            lengths = torch.tensor([nbframes], device=device)

        if posed_joints is None:
            _, global_positions, local_joints_positions_origin_is_pelvis = fk(
                local_joint_rots, root_positions, self.skeleton
            )
        else:
            global_positions = posed_joints
            root_positions = posed_joints[:, :, 0]
            local_joints_positions_origin_is_pelvis = posed_joints - root_positions[:, :, None]

        root_heading_angle = compute_heading_angle(global_positions, self.skeleton)
        global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)

        ground_offset = 0 * root_positions
        ground_offset[..., 1] = root_positions[..., 1]
        local_joints_positions = local_joints_positions_origin_is_pelvis[:, :, 1:] + ground_offset[:, :, None]
        velocities = compute_vel_xyz(global_positions, self.fps, lengths=lengths)
        foot_contacts = foot_detect_from_pos_and_vel(global_positions, velocities, self.skeleton, 0.15, 0.10)

        features, _ = einops.pack(
            [
                root_positions,
                global_root_heading,
                local_joints_positions,
                velocities,
                foot_contacts,
            ],
            "batch time *",
        )

        if to_normalize:
            features = self.normalize(features)
        return features

    @ensure_batched(features=3, angle=1)
    def rotate(self, features: torch.Tensor, angle: torch.Tensor):
        """Rotate all spatial features by a heading delta (radians)."""
        # rotate by the angle
        # it add the angle to the current features
        # assume it is not normalized
        bs = features.shape[0]
        device = features.device
        [
            root_pos,
            global_root_heading,
            local_joints_positions,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=device)
        if len(angle.shape) == 0:
            angle = angle.repeat(bs)

        RF = RotateFeatures(angle)
        new_features, _ = einops.pack(
            [
                RF.rotate_positions(root_pos),
                RF.rotate_2d_positions(global_root_heading),
                RF.rotate_positions(local_joints_positions),
                RF.rotate_positions(velocities),
                foot_contacts,
            ],
            "batch time *",
        )
        return new_features

    @ensure_batched(features=3, translation_2d=2)
    def translate_2d(
        self,
        features: torch.Tensor,
        translation_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Translate root planar position by ``(dx, dz)``."""
        # only move on the ground
        # For 3D, we should not forget to move the local_joints_positions as well
        bs = features.shape[0]
        if len(translation_2d.shape) == 1:
            translation_2d = translation_2d.repeat(bs, 1)

        new_features = features.clone()
        new_root_pos = new_features[:, :, self.slice_dict["root_pos"]]
        new_root_pos[:, :, 0] += translation_2d[:, 0]
        new_root_pos[:, :, 2] += translation_2d[:, 1]
        return new_features

    @ensure_batched(features=3)
    def inverse(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        posed_joints_from="positions",
        return_numpy: bool = False,
    ) -> torch.Tensor:
        """Decode features back to a motion dictionary.

        Args:
            features: Feature tensor ``[B, T, D]``.
            is_normalized: Whether input features are normalized.
            posed_joints_from: Must be ``"positions"`` for this representation.
            return_numpy: Whether to convert tensors to numpy arrays.

        Returns:
            Dictionary containing reconstructed positions and auxiliary data.
        """
        assert posed_joints_from == "positions"
        if is_normalized:
            features = self.unnormalize(features)

        [
            root_positions,
            global_root_heading,
            local_joints_positions,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        dummy_root = 0 * local_joints_positions[:, :, [0]]
        posed_joints_from_pos = torch.stack([dummy_root, local_joints_positions], axis=2)
        posed_joints_from_pos[..., 0] += root_positions[..., None, 0]
        posed_joints_from_pos[..., 2] += root_positions[..., None, 2]
        root_positions = posed_joints_from_pos[..., self.skeleton.root_idx, :]
        foot_contacts = foot_contacts > 0.5
        posed_joints = posed_joints_from_pos

        output_tensor_dict = {
            "local_rot_mats": None,
            "global_rot_mats": None,
            "posed_joints": posed_joints,
            "root_positions": root_positions,
            "foot_contacts": foot_contacts,
            "global_root_heading": global_root_heading,
        }
        if return_numpy:
            return to_numpy(output_tensor_dict)
        return output_tensor_dict
