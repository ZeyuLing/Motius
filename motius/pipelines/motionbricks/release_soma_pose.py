# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA capture/standard pose conversion in the inference pipeline."""

import torch
import torch as t
from motius.motion.representation.motionbricks.soma_rotations import (
    cont6d_to_matrix,
    matrix_to_cont6d,
    change_t_pose_global_mats_onnx_compatible,
)


def replace_contiguous_features(
    values: t.Tensor, replacement: t.Tensor, feature_slice: slice
) -> t.Tensor:
    """Replace a contiguous feature range along the last axis.

    A slice with a step or an open end raises ValueError.
    """
    if (
        feature_slice.start is None
        or feature_slice.stop is None
        or feature_slice.step not in (None, 1)
        or (feature_slice.start < 0)
        or (feature_slice.stop < feature_slice.start)
    ):
        raise ValueError(f"Invalid contiguous feature slice: {feature_slice}.")
    return torch.cat(
        [
            values[..., : feature_slice.start],
            replacement,
            values[..., feature_slice.stop :],
        ],
        dim=-1,
    )


def generate_soma(
    self,
    global_root_values: t.Tensor,
    has_global_root_values: t.Tensor,
    local_root_values: t.Tensor,
    has_local_root_values: t.Tensor,
    local_poses: t.Tensor,
    has_local_poses: t.Tensor,
    allowed_pred_num_tokens: t.Tensor,
):
    rotation_start = self.local_rotation_slice.start
    rotation_end = self.local_rotation_slice.stop
    global_joints_ric_rot6d = local_poses[:, :, rotation_start:rotation_end]
    global_rot_6d = global_joints_ric_rot6d.reshape(
        local_poses.shape[0],
        local_poses.shape[1],
        self.inference.global_motion_rep.num_joints,
        6,
    )
    global_rot_mats = cont6d_to_matrix(global_rot_6d)
    global_rot_mats = change_t_pose_global_mats_onnx_compatible(
        global_rot_mats, self.capture_offsets, self.standard_offsets
    )
    retargeted_local_poses = torch.cat(
        [
            local_poses[..., :rotation_start],
            matrix_to_cont6d(global_rot_mats).reshape(
                local_poses.shape[0],
                local_poses.shape[1],
                self.inference.global_motion_rep.num_joints * 6,
            ),
            local_poses[..., rotation_end:],
        ],
        dim=-1,
    )
    pred_global_motions, pred_num_tokens = self.inference.predict(
        global_root_values,
        has_global_root_values,
        local_root_values,
        has_local_root_values,
        retargeted_local_poses,
        has_local_poses,
        num_tokens=None,
        allowed_pred_num_tokens=allowed_pred_num_tokens,
        config=self.config,
    )
    batch_size, num_frames, _ = pred_global_motions.shape
    global_joints_ric_rot6d = pred_global_motions[:, :, self.global_rotation_slice]
    global_rot_6d = global_joints_ric_rot6d.reshape(
        batch_size, num_frames, self.inference.global_motion_rep.num_joints, 6
    )
    global_rot_mats = cont6d_to_matrix(global_rot_6d)
    global_rot_mats = change_t_pose_global_mats_onnx_compatible(
        global_rot_mats, self.standard_offsets, self.capture_offsets
    )
    retargeted_global_rotations = matrix_to_cont6d(global_rot_mats).reshape(
        pred_global_motions.shape[0],
        pred_global_motions.shape[1],
        self.inference.global_motion_rep.num_joints * 6,
    )
    pred_global_motions = replace_contiguous_features(
        pred_global_motions, retargeted_global_rotations, self.global_rotation_slice
    )
    return [pred_global_motions, pred_num_tokens]
