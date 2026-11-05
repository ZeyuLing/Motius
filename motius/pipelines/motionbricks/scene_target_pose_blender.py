# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Blend a locomotion mode's target poses and place them ahead of the character."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import warp as wp

from motius.pipelines.motionbricks.scene_blend_spaces import BlendSample, BlendSpaceType
from motius.pipelines.motionbricks.scene_motion import (
    LocomotionModeRuntime,
    directional_blend_space_value,
)


class BaseTargetPoseBlender(ABC):
    """Turn a locomotion mode and the character's desired velocity into target poses.

    evaluate() runs the shared steps. Each rig overrides the blend, speed, and
    placement steps for its pose layout.
    """

    def evaluate(
        self,
        character,
        mode_runtime: LocomotionModeRuntime,
        blended_poses: wp.array,
        target_poses: wp.array,
    ) -> None:
        """Blend the mode's targets for the desired velocity and place them in target_poses."""
        mode = mode_runtime.definition
        stream = character.engine.stream
        movement = character.movement
        speed = float(np.linalg.norm(movement.desired_velocity))
        if mode.blend_space_type is BlendSpaceType.LINEAR_1D:
            sample = mode_runtime.blend_space.evaluate(
                wp.vec2(speed / mode.max_speed, 0.0)
            )
            self.blend_linear(mode_runtime.target_poses, sample, blended_poses, stream)
        else:
            sample = mode_runtime.blend_space.evaluate(
                directional_blend_space_value(
                    movement.current_facing, movement.desired_velocity, mode.max_speed
                )
            )
            self.blend_directional(
                mode_runtime.target_poses, sample, blended_poses, stream
            )
        self.adjust_speed(blended_poses, speed * mode.velocity_scale, stream)
        ground_position = (
            character.character_ground_position
            + movement.desired_velocity * mode.velocity_scale * mode.position_scale
        )
        ground_position[2] = 0.0
        self.place(
            wp.vec3(*ground_position),
            wp.vec3(*movement.current_facing),
            blended_poses,
            target_poses,
            stream,
        )

    @abstractmethod
    def blend_linear(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        """Blend the targets of a 1D blend space into output."""

    @abstractmethod
    def blend_directional(
        self, target_poses: wp.array, sample: BlendSample, output: wp.array, stream
    ) -> None:
        """Blend the targets of a 2D blend space into output.

        When the blend weights add up to less than one, a base pose fills the rest. SOMA
        uses its reference pose, and G1 uses the Idle target.
        """

    @abstractmethod
    def adjust_speed(self, poses: wp.array, speed: float, stream) -> None:
        """Scale the root's motion along the ground to a mean speed."""

    @abstractmethod
    def place(
        self,
        ground_position: wp.vec3,
        facing_direction: wp.vec3,
        blended_poses: wp.array,
        target_poses: wp.array,
        stream,
    ) -> None:
        """Copy the blended poses into target_poses, placed on the ground and turned to face a direction.

        One placement moves every root, so the roots keep their offsets from one another.
        Other joint values stay as they are.
        """


__all__ = ["BaseTargetPoseBlender"]
