# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Locomotion modes, movement smoothing, and clip data for the characters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    ModelRig,
)
from motius.motion.runtime.motionbricks.math_utils import (
    critical_spring_angle,
    critical_spring_damper_exact,
    project_to_plane,
)
from motius.pipelines.motionbricks.scene_blend_spaces import (
    BlendSpace,
    BlendSpaceType,
    prepare_directional_poses,
)

if TYPE_CHECKING:
    from motius.motion.io.motionbricks.smart_primitive_document import (
        SmartPrimitiveAssetPool,
    )
    from motius.motion.runtime.motionbricks.animation_buffer import AnimationBuffer
    from motius.motion.runtime.motionbricks.dof_animation_buffer import (
        DOFAnimationBuffer,
    )
MOVEMENT_EPSILON = 1e-06
LINEAR_LOCOMOTION_MODE = "linear_1d"
DIRECTIONAL_LOCOMOTION_MODE = "directional_2d"
CROUCH_LOCOMOTION_MODE = "crouch_directional_2d"
DIRECTIONAL_TARGET_VALUES = (
    ("Idle", (0.0, 0.0)),
    ("Forward", (1.0, 0.0)),
    ("Backward", (-1.0, 0.0)),
    ("Left", (0.0, 1.0)),
    ("Right", (0.0, -1.0)),
)
CLIP_FACING_YAW = -0.5 * math.pi


class PlaybackSource(Enum):
    """Where the character's current pose comes from."""

    GENERATED = "generated"
    CLIP_TRANSITION = "clip_transition"
    CLIP = "clip"


@dataclass(frozen=True, slots=True)
class LocomotionCommand:
    """What the controls ask the character to do this frame.

    Attributes:
        facing_direction: Ground direction the character faces in the 2D modes, in world
            space. Scene Preview passes the camera direction.
        movement: Movement input along the ground in world space. Its length is the input
            strength from 0 to 1. None, or a length of about zero, means no movement.
        use_max_speed: Base the speed on the mode's maximum speed instead of its default
            speed.
        requested_mode: Name of the mode to switch to. None keeps the current mode.
    """

    facing_direction: np.ndarray
    movement: np.ndarray | None
    use_max_speed: bool = False
    requested_mode: str | None = None


@dataclass(frozen=True, slots=True)
class LocomotionSettings:
    """How quickly movement responds to input and when to generate new motion."""

    facing_halflife: float = 0.15
    velocity_direction_halflife: float = 0.2
    speed_acceleration_halflife: float = 0.6
    speed_deceleration_halflife: float = 0.4
    speed_regeneration_fraction: float = 0.3
    minimum_speed_regeneration_fraction: float = 0.1
    direction_regeneration_dot: float = 0.92
    maximum_direction_change_speed_reduction: float = 0.25


@dataclass(frozen=True, slots=True)
class LocomotionMode:
    """Blend targets, blend space settings, and speed limits for one locomotion mode.

    Attributes:
        name: Mode name, such as linear_1d.
        smart_primitive_path: Smart Primitive document that holds the blend targets.
        blend_space_type: 1D blend space or 2D directional blend space.
        target_values: Name and blend value of each blend target.
        max_speed: Maximum speed in m/s.
        default_speed: Speed when input does not request maximum speed, in m/s.
        minimum_speed: Slowest speed while there is movement input, in m/s.
        transition_early_by_frames: Transition Early By for each generation.
        allowed_token_counts: Token counts the model may choose.
        velocity_scale: Scale on the desired velocity for target speed and placement.
        position_scale: How far ahead the targets sit, in seconds of travel at the scaled
            desired velocity.
        label: Name shown in the apps.
    """

    name: str
    smart_primitive_path: Path
    blend_space_type: BlendSpaceType
    target_values: tuple[tuple[str, tuple[float, float]], ...]
    max_speed: float
    default_speed: float
    minimum_speed: float = 0.0
    transition_early_by_frames: int = 8
    allowed_token_counts: tuple[int, ...] = tuple(range(8, 15))
    velocity_scale: float = 1.0
    position_scale: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Locomotion mode name cannot be empty.")
        if not self.target_values:
            raise ValueError("Locomotion mode target_values cannot be empty.")
        if self.max_speed <= 0.0:
            raise ValueError("Locomotion mode max_speed must be positive.")
        if not 0.0 < self.default_speed <= self.max_speed:
            raise ValueError(
                "Locomotion mode default_speed must be positive and at most max_speed."
            )
        if not 0.0 <= self.minimum_speed <= self.default_speed:
            raise ValueError(
                "Locomotion mode minimum_speed must be 0 to default_speed."
            )
        if self.transition_early_by_frames < 0:
            raise ValueError(
                "Locomotion mode transition_early_by_frames cannot be negative."
            )
        if self.velocity_scale < 0.0:
            raise ValueError("Locomotion mode velocity_scale cannot be negative.")
        if self.position_scale < 0.0:
            raise ValueError("Locomotion mode position_scale cannot be negative.")
        if not self.allowed_token_counts:
            raise ValueError("Locomotion mode allowed_token_counts cannot be empty.")
        if not all(
            MODEL_MIN_TOKEN_COUNT <= count <= MODEL_MAX_TOKEN_COUNT
            for count in self.allowed_token_counts
        ):
            raise ValueError(
                f"Locomotion mode allowed_token_counts must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}."
            )


def style_locomotion_mode(
    target: str,
    smart_primitive_path: Path,
    minimum_speed: float,
    max_speed: float,
    *,
    velocity_scale: float = 1.0,
    position_scale: float = 1.5,
    transition_early_by_frames: int = 2,
    allowed_token_counts: tuple[int, ...] = tuple(range(6, 16)),
) -> LocomotionMode:
    """Return a 1D mode that blends from the Idle target at 0 to the style target at 1.

    Args:
        target: Name of the style's blend target, such as Skipping. It also names the
            mode, as in skipping_1d.
        smart_primitive_path: Smart Primitive document that holds Idle and the style
            target.
        minimum_speed: Slowest speed while there is movement input, in m/s.
        max_speed: Maximum speed and default speed, in m/s. The style pose shows in full
            at this speed.
        velocity_scale: Scale on the desired velocity for target speed and placement.
        position_scale: How far ahead the targets sit, in seconds of travel at the scaled
            desired velocity. The default keeps each style close to the speed of its clip.
        transition_early_by_frames: Transition Early By. The default lets the motion come
            close to the style pose.
        allowed_token_counts: Token counts the model may choose.
    """
    return LocomotionMode(
        name=f"{target.lower().replace(' ', '_')}_1d",
        smart_primitive_path=smart_primitive_path,
        blend_space_type=BlendSpaceType.LINEAR_1D,
        target_values=(("Idle", (0.0, 0.0)), (target, (1.0, 0.0))),
        max_speed=max_speed,
        default_speed=max_speed,
        minimum_speed=minimum_speed,
        transition_early_by_frames=transition_early_by_frames,
        allowed_token_counts=allowed_token_counts,
        velocity_scale=velocity_scale,
        position_scale=position_scale,
        label=target,
    )


@dataclass(slots=True)
class AnimationClip:
    """A loaded clip and the data used to place it in world space."""

    animation: AnimationBuffer | DOFAnimationBuffer
    source_ground: wp.transform
    duration: float
    root_offset: wp.transform
    root_positions: np.ndarray


def make_animation_clip(
    animation: AnimationBuffer | DOFAnimationBuffer, model_to_runtime: wp.transform
) -> AnimationClip:
    """Describe a clip and where it starts on the ground."""
    roots = animation.root_transforms()
    first_root = wp.transform(*roots[0])
    return AnimationClip(
        animation=animation,
        source_ground=project_to_plane(
            wp.mul(model_to_runtime, first_root), wp.vec3(0.0, 0.0, 1.0)
        ),
        duration=max(animation.num_frames - 1, 1) / animation.sample_rate,
        root_offset=wp.transform_identity(),
        root_positions=roots[:, :3].copy(),
    )


@dataclass(frozen=True, slots=True)
class LocomotionModeRuntime:
    """The blend space and target poses used by a locomotion mode."""

    definition: LocomotionMode
    blend_space: BlendSpace
    target_poses: wp.array


def load_locomotion_mode(
    mode: LocomotionMode, asset_pool: SmartPrimitiveAssetPool
) -> LocomotionModeRuntime:
    """Load a mode's target poses from its Smart Primitive and build its blend space."""
    smart_primitive = asset_pool.instantiate(mode.smart_primitive_path)
    reference_pose = asset_pool.reference_pose
    target_poses = wp.empty(
        (len(mode.target_values), MODEL_POSES_PER_TOKEN, *reference_pose.shape),
        dtype=reference_pose.dtype,
        device=asset_pool.device,
    )
    blend_space = BlendSpace(mode.blend_space_type, device=asset_pool.device)
    for data_index, (node_name, value) in enumerate(mode.target_values):
        node_index = smart_primitive.node_index(node_name)
        if node_index < 0 or not smart_primitive.has_target_pose(node_index):
            raise ValueError(
                f"The {mode.name!r} Smart Primitive has no target named {node_name!r}."
            )
        params = smart_primitive.get_local_target_poses(
            node_index, target_poses[data_index]
        )
        if params.num_poses_to_use != MODEL_POSES_PER_TOKEN:
            raise ValueError(
                f"The {mode.name!r} target {node_name!r} must use {MODEL_POSES_PER_TOKEN} poses."
            )
        blend_space.add_target(wp.vec2(*value), data_index)
    blend_space.compile()
    if (
        smart_primitive.rig is ModelRig.SOMA
        and mode.blend_space_type is BlendSpaceType.DIRECTIONAL_2D
    ):
        prepare_directional_poses(target_poses, reference_pose)
    return LocomotionModeRuntime(mode, blend_space, target_poses)


def ground_direction(direction: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return a ground direction of length one, using fallback when needed."""
    for candidate in (direction, fallback):
        ground = np.array(
            [float(candidate[0]), float(candidate[1]), 0.0], dtype=np.float32
        )
        length = float(np.linalg.norm(ground))
        if np.isfinite(length) and length > MOVEMENT_EPSILON:
            return ground / length
    raise ValueError("direction and fallback have no valid ground direction.")


def character_start_position_and_facing(
    transform: wp.transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ground position and facing of a scene's CharacterStart."""
    ground_transform = project_to_plane(transform, wp.vec3(0.0, 0.0, 1.0))
    position = np.array(ground_transform.p, dtype=np.float32)
    world_facing = wp.transform_vector(ground_transform, wp.vec3(1.0, 0.0, 0.0))
    facing = ground_direction(
        np.array(world_facing, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    return (position, facing)


def directional_blend_space_value(
    facing_direction: np.ndarray, desired_velocity: np.ndarray, max_speed: float
) -> wp.vec2:
    """Express movement as a forward and left blend value relative to facing."""
    normalized_velocity = desired_velocity / max_speed
    facing_left = np.array(
        [-facing_direction[1], facing_direction[0], 0.0], dtype=np.float32
    )
    return wp.vec2(
        float(np.clip(np.dot(normalized_velocity, facing_direction), -1.0, 1.0)),
        float(np.clip(np.dot(normalized_velocity, facing_left), -1.0, 1.0)),
    )


def _speed_reduction_scale(
    speed: float, max_speed: float, maximum_reduction: float
) -> float:
    speed_fraction = min(max(speed / max_speed, 0.0), 1.0)
    return 1.0 - maximum_reduction * speed_fraction


class MovementControl:
    """Smooth movement commands into the desired velocity and facing.

    Critically damped springs ease the speed, the velocity direction, and the
    facing. Locomotion places its targets from desired_velocity and current_facing.
    Directions lie on the ground in world space. Speeds are in meters per second,
    and angles in radians.

    Attributes:
        settings: Spring half lives, and the speed reduction on sharp turns.
        input_speed: Speed the movement input asks for. It is 0 without movement input.
        input_velocity_direction: Direction of the latest movement input.
        desired_facing: Facing that current_facing turns toward.
        last_input_facing: Facing at the latest movement input. Idle clips face
            this way.
        current_facing: Smoothed facing. current_facing_angle and current_facing_rate
            hold its angle and turn rate.
        desired_velocity_direction: Smoothed velocity direction.
            desired_velocity_angle and desired_velocity_direction_rate hold its angle
            and turn rate.
        desired_speed: Smoothed speed the character aims for. desired_speed_rate holds
            its rate of change.
        desired_velocity: desired_velocity_direction times desired_speed.
    """

    def __init__(
        self, facing: np.ndarray, settings: LocomotionSettings = LocomotionSettings()
    ) -> None:
        """Start at rest, facing along the ground direction facing."""
        self.settings = settings
        angle = math.atan2(float(facing[1]), float(facing[0]))
        self.input_speed = 0.0
        self.input_velocity_direction = facing.copy()
        self.desired_facing = facing.copy()
        self.last_input_facing = facing.copy()
        self.current_facing = facing.copy()
        self.current_facing_angle = angle
        self.current_facing_rate = 0.0
        self.desired_velocity_direction = facing.copy()
        self.desired_velocity_angle = angle
        self.desired_velocity_direction_rate = 0.0
        self.desired_speed = 0.0
        self.desired_speed_rate = 0.0
        self.desired_velocity = np.zeros(3, dtype=np.float32)

    @property
    def input_velocity(self) -> np.ndarray:
        """Input velocity on the ground before smoothing, in meters per second."""
        return self.input_velocity_direction * self.input_speed

    def update(
        self,
        command: LocomotionCommand,
        mode: LocomotionMode,
        delta_time: float,
        previous_mode: LocomotionMode | None = None,
    ) -> None:
        """Apply one frame of movement input for the active mode.

        previous_mode is the mode this frame switched away from, or None. The switch
        eases in.
        """
        self._read_input(command, mode)
        if previous_mode is not None:
            self._ease_mode_switch(mode, previous_mode)
        self._smooth(mode, delta_time)

    def realign(self, velocity: np.ndarray, mode: LocomotionMode) -> None:
        """Resume locomotion from the character's actual velocity. The input stays as it is."""
        speed = float(np.linalg.norm(velocity[:2]))
        direction = ground_direction(velocity, self.current_facing)
        angle = math.atan2(direction[1], direction[0])
        self.desired_speed = speed
        self.desired_speed_rate = 0.0
        self.desired_velocity_direction[:] = direction
        self.desired_velocity_angle = angle
        self.desired_velocity_direction_rate = 0.0
        np.multiply(direction, speed, out=self.desired_velocity)
        if mode.blend_space_type is BlendSpaceType.LINEAR_1D:
            self._set_current_facing(direction, angle, 0.0)

    def face(self, direction: np.ndarray) -> None:
        """Set every facing to direction at once."""
        self.desired_facing = direction.copy()
        self.last_input_facing = direction.copy()
        self._set_current_facing(
            direction, math.atan2(float(direction[1]), float(direction[0])), 0.0
        )

    def reset_desired_velocity(self) -> None:
        """Reset the desired velocity and the desired speed to zero."""
        self.desired_speed = 0.0
        self.desired_speed_rate = 0.0
        self.desired_velocity.fill(0.0)

    def _set_current_facing(
        self, direction: np.ndarray, angle: float, rate: float
    ) -> None:
        self.current_facing[:] = direction
        self.current_facing_angle = angle
        self.current_facing_rate = rate

    def _read_input(self, command: LocomotionCommand, mode: LocomotionMode) -> None:
        facing_direction = ground_direction(
            command.facing_direction, self.desired_facing
        )
        movement_amount = (
            0.0
            if command.movement is None
            else float(np.linalg.norm(command.movement[:2]))
        )
        if not movement_amount > MOVEMENT_EPSILON:
            if (
                mode.blend_space_type is BlendSpaceType.DIRECTIONAL_2D
                or self.desired_speed <= 0.0
            ):
                self.desired_facing = facing_direction
            self.input_speed = 0.0
            return
        movement_direction = ground_direction(
            command.movement, self.input_velocity_direction
        )
        self.input_velocity_direction = movement_direction
        self.desired_facing = (
            movement_direction
            if mode.blend_space_type is BlendSpaceType.LINEAR_1D
            else facing_direction
        )
        self.last_input_facing = self.desired_facing.copy()
        requested_speed = (
            mode.max_speed if command.use_max_speed else mode.default_speed
        )
        self.input_speed = max(
            requested_speed * min(movement_amount, 1.0), mode.minimum_speed
        )

    def _ease_mode_switch(
        self, mode: LocomotionMode, previous_mode: LocomotionMode
    ) -> None:
        if mode.blend_space_type is BlendSpaceType.DIRECTIONAL_2D:
            facing = self.desired_facing
            self._set_current_facing(facing, math.atan2(facing[1], facing[0]), 0.0)
        if mode.max_speed <= previous_mode.max_speed:
            speed_scale = _speed_reduction_scale(
                self.desired_speed,
                mode.max_speed,
                self.settings.maximum_direction_change_speed_reduction,
            )
            self.desired_speed = min(self.desired_speed * speed_scale, mode.max_speed)
            self.desired_speed_rate *= speed_scale

    def _smooth(self, mode: LocomotionMode, delta_time: float) -> None:
        settings = self.settings
        self.desired_velocity_angle, self.desired_velocity_direction_rate = (
            critical_spring_angle(
                self.desired_velocity_angle,
                self.desired_velocity_direction_rate,
                self.input_velocity_direction,
                settings.velocity_direction_halflife,
                delta_time,
            )
        )
        self.desired_velocity_direction[0] = math.cos(self.desired_velocity_angle)
        self.desired_velocity_direction[1] = math.sin(self.desired_velocity_angle)
        if mode.blend_space_type is BlendSpaceType.LINEAR_1D:
            self._set_current_facing(
                self.desired_velocity_direction,
                self.desired_velocity_angle,
                self.desired_velocity_direction_rate,
            )
        else:
            self.current_facing_angle, self.current_facing_rate = critical_spring_angle(
                self.current_facing_angle,
                self.current_facing_rate,
                self.desired_facing,
                settings.facing_halflife,
                delta_time,
            )
            self.current_facing[0] = math.cos(self.current_facing_angle)
            self.current_facing[1] = math.sin(self.current_facing_angle)
        speed_goal = self.input_speed
        turn_dot = float(
            np.dot(self.desired_velocity_direction, self.input_velocity_direction)
        )
        if turn_dot < settings.direction_regeneration_dot:
            speed_goal *= _speed_reduction_scale(
                self.desired_speed,
                mode.max_speed,
                settings.maximum_direction_change_speed_reduction,
            )
        speed_halflife = (
            settings.speed_acceleration_halflife
            if speed_goal > self.desired_speed
            else settings.speed_deceleration_halflife
        )
        self.desired_speed, self.desired_speed_rate = critical_spring_damper_exact(
            self.desired_speed,
            self.desired_speed_rate,
            speed_goal,
            speed_halflife,
            delta_time,
        )
        if self.desired_speed > mode.max_speed:
            self.desired_speed = mode.max_speed
            self.desired_speed_rate = min(self.desired_speed_rate, 0.0)
        np.multiply(
            self.desired_velocity_direction,
            self.desired_speed,
            out=self.desired_velocity,
        )


__all__ = [
    "CLIP_FACING_YAW",
    "CROUCH_LOCOMOTION_MODE",
    "DIRECTIONAL_LOCOMOTION_MODE",
    "DIRECTIONAL_TARGET_VALUES",
    "LINEAR_LOCOMOTION_MODE",
    "MOVEMENT_EPSILON",
    "AnimationClip",
    "LocomotionCommand",
    "LocomotionMode",
    "LocomotionModeRuntime",
    "LocomotionSettings",
    "MovementControl",
    "PlaybackSource",
    "character_start_position_and_facing",
    "directional_blend_space_value",
    "ground_direction",
    "load_locomotion_mode",
    "make_animation_clip",
    "style_locomotion_mode",
]
