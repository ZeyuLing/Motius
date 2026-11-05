# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Idle, locomotion, and interaction states shared by both rigs."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import MODEL_POSES_PER_TOKEN
from motius.pipelines.motionbricks.scene_motion import (
    MOVEMENT_EPSILON,
    AnimationClip,
    LocomotionMode,
    LocomotionModeRuntime,
    LocomotionSettings,
    MovementControl,
)
from motius.pipelines.motionbricks.scene_smart_primitive import CharacterProperties
from motius.pipelines.motionbricks.scene_target_pose import TargetPoseInferenceParams

if TYPE_CHECKING:
    from motius.pipelines.motionbricks.scene_smart_primitive_behaviors import (
        SmartPrimitiveBehavior,
    )
    from motius.pipelines.motionbricks.scene_state_machine import StateMachine


def _resolve_clip_path(character, path):
    path = Path(path)
    if not path.is_absolute():
        path = (
            getattr(getattr(character, "asset_pool", None), "assets_root", Path("."))
            / path
        )
    return path.resolve()


class State:
    """Common entry, exit, and update methods for states."""

    name = "State"

    def enter(self, character, state_machine: StateMachine) -> None:
        """Run when the state machine switches to this state."""

    def exit(self, character, state_machine: StateMachine) -> None:
        """Run when the state machine leaves this state."""

    def update(self, character, state_machine: StateMachine, delta_time: float) -> None:
        """Advance the state by delta_time (in seconds)."""
        raise NotImplementedError


class IdleState(State):
    """Generate a transition into an idle clip, then loop the clip."""

    def __init__(self, name: str, clips: tuple[AnimationClip, ...]) -> None:
        """Create an idle state that picks from the given looping clips."""
        if not clips:
            raise ValueError("Idle state requires at least one clip.")
        self.name = name
        self.clips = clips
        self._random = random.Random()

    def start(self, character) -> None:
        """Start in an idle loop, without a generated transition."""
        character.movement.reset_desired_velocity()
        character.initialize_clip_loop(self.choose_clip(character))

    def enter(self, character, state_machine: StateMachine) -> None:
        character.movement.reset_desired_velocity()
        character.generate_clip_transition(self.choose_clip(character))

    def update(self, character, state_machine: StateMachine, delta_time: float) -> None:
        if character.movement.input_speed > 0.0:
            state_machine.transition_to_locomotion(character)
            return
        if character.clip_transition_ends_within(delta_time):
            character.start_clip(character.current_clip, loop=True)

    def choose_clip(self, character) -> AnimationClip:
        """Pick a random idle clip, preferring one other than the current clip."""
        clips = self.clips
        candidates = tuple(clip for clip in clips if clip is not character.current_clip)
        return self._random.choice(candidates or clips)


class LocomotionState(State):
    """Place target poses ahead along the desired velocity, and generate toward them.

    The character's MovementControl smooths the input into that velocity.
    """

    def __init__(
        self,
        name: str,
        mode_runtimes: tuple[LocomotionModeRuntime, ...],
        initial_mode: str,
        pose: wp.array,
        settings: LocomotionSettings = LocomotionSettings(),
    ) -> None:
        """Create a locomotion state with its loaded modes, initial mode, and regeneration settings.

        pose has the layout of the character's poses. It sizes the target pose buffers.
        """
        self.name = name
        self.settings = settings
        self.mode_runtimes = {
            runtime.definition.name: runtime for runtime in mode_runtimes
        }
        self.active_mode_runtime = self.mode_runtimes[initial_mode]
        self._blended_target_poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, *pose.shape), dtype=pose.dtype, device=pose.device
        )
        self._target_poses = wp.empty_like(self._blended_target_poses)
        self._previous_generation_desired_velocity = np.zeros(3, dtype=np.float32)
        self._force_regeneration = True

    @property
    def active_mode(self) -> LocomotionMode:
        """Definition of the active mode."""
        return self.active_mode_runtime.definition

    def select_mode(self, name: str) -> None:
        """Make a mode active. A new mode generates new motion on the next update."""
        mode_runtime = self.mode_runtimes[name]
        if mode_runtime is not self.active_mode_runtime:
            self.active_mode_runtime = mode_runtime
            self._force_regeneration = True

    def enter(self, character, state_machine: StateMachine) -> None:
        character.movement.realign(character.actual_velocity, self.active_mode)
        self._generate(character)

    def update(self, character, state_machine: StateMachine, delta_time: float) -> None:
        if character.movement.input_speed <= 0.0:
            state_machine.transition_to_idle(character)
            return
        if (
            self._force_regeneration
            or character.time_to_transition() <= delta_time
            or self._needs_regeneration(character.movement)
        ):
            self._generate(character)

    def evaluate_current_target_poses(
        self, character, blended_poses: wp.array, target_poses: wp.array
    ) -> None:
        """Blend and place the active mode's targets for generation or debug display."""
        character.target_pose_blender.evaluate(
            character, self.active_mode_runtime, blended_poses, target_poses
        )

    def _generate(self, character) -> None:
        self.evaluate_current_target_poses(
            character, self._blended_target_poses, self._target_poses
        )
        character.generate_to_target_poses(
            self._target_poses,
            TargetPoseInferenceParams(
                transition_early_by_frames=self.active_mode.transition_early_by_frames,
                num_poses_to_use=MODEL_POSES_PER_TOKEN,
                allowed_token_counts=self.active_mode.allowed_token_counts,
            ),
        )
        self._previous_generation_desired_velocity[:] = (
            character.movement.desired_velocity
        )
        self._force_regeneration = False

    def _needs_regeneration(self, movement: MovementControl) -> bool:
        settings = self.settings
        previous_speed = float(
            np.linalg.norm(self._previous_generation_desired_velocity)
        )
        max_speed = self.active_mode.max_speed
        locomotion_started = (
            movement.desired_speed > 0.0 and previous_speed <= MOVEMENT_EPSILON
        )
        speed_delta = movement.desired_speed - previous_speed
        remaining_speed_range = (
            max_speed - previous_speed if speed_delta > 0.0 else previous_speed
        )
        regeneration_threshold = max(
            remaining_speed_range * settings.speed_regeneration_fraction,
            max_speed * settings.minimum_speed_regeneration_fraction,
        )
        if locomotion_started or abs(speed_delta) > regeneration_threshold:
            return True
        if previous_speed <= MOVEMENT_EPSILON:
            return False
        previous_direction = self._previous_generation_desired_velocity / previous_speed
        return (
            float(np.dot(movement.desired_velocity_direction, previous_direction))
            < settings.direction_regeneration_dot
        )


class InteractionState(State):
    """Run a Smart Primitive's sequences and clips until the behavior finishes.

    The behavior chooses the next sequence or clip. This state generates motion
    toward each target. The character plays that motion and records its recent poses.
    """

    name = "Interaction"

    def __init__(self) -> None:
        self.primitive: SmartPrimitiveBehavior | None = None
        self._character = None
        self._state_machine: StateMachine | None = None
        self._target_poses: wp.array | None = None
        self._inference_params: tuple[TargetPoseInferenceParams, ...] = ()
        self._target_index = -1
        self._playing_clip = False
        self._clip_loop = False
        self._clips: dict[tuple[Path, bool], AnimationClip] = {}
        self._on_complete: Callable[[], None] | None = None

    @property
    def character(self):
        """Character in the active interaction."""
        if self._character is None:
            raise RuntimeError(
                "Cannot access the character without an active interaction."
            )
        return self._character

    @property
    def is_looping(self) -> bool:
        """Whether a looping clip is playing."""
        return self._playing_clip and self._clip_loop

    def prepare(self, character, primitive: SmartPrimitiveBehavior) -> None:
        """Load each clip and its looping version before an interaction needs them."""
        for path in primitive.clip_paths:
            path = _resolve_clip_path(character, path)
            for loop in (False, True):
                if (path, loop) not in self._clips:
                    self._clips[path, loop] = character.load_clip(path, loop=loop)

    def configure(self, character, primitive: SmartPrimitiveBehavior) -> None:
        """Prepare a Smart Primitive behavior for the next interaction.

        Load its clips and size the target pose buffers for it.
        """
        pose = character.current_pose
        self.prepare(character, primitive)
        capacity = max(1, primitive.target_pose_count, primitive.max_sequence_length)
        required_frames = capacity * MODEL_POSES_PER_TOKEN
        if self._target_poses is None or self._target_poses.shape[0] < required_frames:
            self._target_poses = wp.empty(
                (required_frames, *pose.shape),
                dtype=pose.dtype,
                device=character.engine.device,
            )
        self.primitive = primitive

    def enter(self, character, state_machine: StateMachine) -> None:
        self._character = character
        self._state_machine = state_machine
        self.primitive.bind(self)
        self.primitive.enter()

    def exit(self, character, state_machine: StateMachine) -> None:
        self._stop_target_pose_sequence(character)
        character.face_pose()
        self.primitive.unbind()
        self.primitive = None
        self._character = None
        self._state_machine = None
        self._inference_params = ()
        self._playing_clip = False

    def play_target_pose_sequence(
        self, name: str | None = None, *, on_complete: Callable[[], None] | None = None
    ) -> None:
        """Generate motion through a target pose sequence, then call on_complete.

        Without a name, use the targets returned by the behavior's get_targets().
        Without a callback, return to locomotion or idle when the sequence ends.
        """
        character = self.character
        primitive = self.primitive
        properties = CharacterProperties(
            position=character.character_ground_position,
            velocity=character.actual_velocity,
            facing=character.movement.current_facing,
            desired_velocity=character.movement.desired_velocity,
            probe_segment=character.probe_segment,
        )
        if name is None:
            params = primitive.get_targets(
                properties, self._target_poses, stream=character.engine.stream
            )
        else:
            index = primitive.sequence_index(name)
            if index < 0:
                raise ValueError(f"Smart Primitive has no sequence named {name!r}.")
            primitive.align_sequence_pivot(
                index, properties, stream=character.engine.stream
            )
            params = primitive.get_world_target_poses_for_sequence(
                index, self._target_poses, stream=character.engine.stream
            )
        params = tuple(params)
        if not params:
            raise ValueError("The Smart Primitive behavior returned no targets.")
        self._stop_target_pose_sequence(character)
        self._playing_clip = False
        self._inference_params = params
        self._target_index = 0
        self._on_complete = on_complete if on_complete is not None else self.finish
        self._generate_current_target(character)

    def play_clip(
        self,
        path: Path,
        *,
        loop: bool = False,
        align_to_current_pose: bool = True,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Play a clip once, or keep repeating it when loop=True.

        Add the path to the behavior's clip_paths so the clip loads with the scene.
        By default, the clip's first root pose lines up with the character's
        current root, in both position and rotation. When a clip played once ends,
        on_complete runs. Without a callback, the character returns to locomotion
        or idle.
        """
        character = self.character
        path = _resolve_clip_path(character, path)
        if (path, loop) not in self._clips:
            raise ValueError(
                f"Cannot play {path}. Declare it in the behavior's clip_paths first."
            )
        self._stop_target_pose_sequence(character)
        character.start_clip(
            self._clips[path, loop],
            loop=loop,
            align_to_current_pose=align_to_current_pose,
        )
        self._playing_clip = True
        self._clip_loop = loop
        self._on_complete = on_complete if on_complete is not None else self.finish

    def finish(self) -> None:
        """End the interaction and return to locomotion or idle."""
        self._state_machine.resume_movement(self.character)

    def update(self, character, state_machine: StateMachine, delta_time: float) -> None:
        if self._playing_clip:
            if character.clip_ends_within(delta_time):
                self._complete_playback(character)
            return
        if self._target_index < 0 or character.time_to_transition() > delta_time:
            return
        self._target_index += 1
        if self._target_index < len(self._inference_params):
            self._generate_current_target(character)
        else:
            self._complete_playback(character)

    def _complete_playback(self, character) -> None:
        on_complete = self._on_complete
        self._playing_clip = False
        self._stop_target_pose_sequence(character)
        on_complete()

    def _stop_target_pose_sequence(self, character) -> None:
        self._target_index = -1
        self._on_complete = None
        character.clear_ik_pins()

    def _generate_current_target(self, character) -> None:
        frame_offset = self._target_index * MODEL_POSES_PER_TOKEN
        character.generate_to_target_poses(
            self._target_poses[frame_offset : frame_offset + MODEL_POSES_PER_TOKEN],
            self._inference_params[self._target_index],
        )


__all__ = ["IdleState", "InteractionState", "LocomotionState", "State"]
