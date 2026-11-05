# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Blend and place Unitree G1 locomotion target poses."""

from __future__ import annotations

import warp as wp

from motius.pipelines.motionbricks.scene_blend_spaces import (
    BlendSample,
    directional_2d_dof_blend,
    linear_1d_dof_blend,
)
from motius.pipelines.motionbricks.scene_target_pose import (
    adjust_target_dof_speed,
    place_target_dof,
)
from motius.pipelines.motionbricks.scene_target_pose_blender import (
    BaseTargetPoseBlender,
)


class G1TargetPoseBlender(BaseTargetPoseBlender):
    """Blend G1 joint coordinates around the Idle target of each 2D directional blend space."""

    def blend_linear(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        linear_1d_dof_blend(target_poses, sample, output, stream=stream)

    def blend_directional(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        directional_2d_dof_blend(
            target_poses, sample, target_poses[0][0], output, stream=stream
        )

    def adjust_speed(self, poses: wp.array, speed: float, stream) -> None:
        adjust_target_dof_speed(poses, speed, stream=stream)

    def place(
        self,
        ground_position: wp.vec3,
        facing_direction: wp.vec3,
        blended_poses: wp.array,
        target_poses: wp.array,
        stream,
    ) -> None:
        place_target_dof(
            ground_position,
            facing_direction,
            blended_poses,
            target_poses,
            stream=stream,
        )


__all__ = ["G1TargetPoseBlender"]
