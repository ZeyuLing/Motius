# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Blend and place SOMA locomotion target poses."""

from __future__ import annotations

import warp as wp

from motius.pipelines.motionbricks.scene_blend_spaces import (
    BlendSample,
    directional_2d_pose_blend,
    linear_1d_pose_blend,
)
from motius.pipelines.motionbricks.scene_target_pose import (
    adjust_target_pose_speed,
    place_target_poses,
)
from motius.pipelines.motionbricks.scene_target_pose_blender import (
    BaseTargetPoseBlender,
)


class SOMATargetPoseBlender(BaseTargetPoseBlender):
    """Blend SOMA local joint transforms around the skeleton's reference pose."""

    def __init__(self, reference_pose: wp.array) -> None:
        """Keep the skeleton's reference pose for directional blending."""
        self.reference_pose = reference_pose

    def blend_linear(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        linear_1d_pose_blend(target_poses, sample, output, stream=stream)

    def blend_directional(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        directional_2d_pose_blend(
            target_poses, sample, self.reference_pose, output, stream=stream
        )

    def adjust_speed(self, poses: wp.array, speed: float, stream) -> None:
        adjust_target_pose_speed(poses, speed, stream=stream)

    def place(
        self,
        ground_position: wp.vec3,
        facing_direction: wp.vec3,
        blended_poses: wp.array,
        target_poses: wp.array,
        stream,
    ) -> None:
        place_target_poses(
            ground_position,
            facing_direction,
            blended_poses,
            target_poses,
            stream=stream,
        )


__all__ = ["SOMATargetPoseBlender"]
