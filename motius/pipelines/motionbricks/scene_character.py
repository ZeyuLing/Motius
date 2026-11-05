# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Play generated motion and clips for a character, and keep recent poses for the next generation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
)
from motius.motion.runtime.motionbricks.foot_contacts import FOOT_CONTACT_COUNT
from motius.motion.runtime.motionbricks.pose_utils import sample_root_position
from motius.pipelines.motionbricks.scene_interaction import InteractionProbe
from motius.pipelines.motionbricks.scene_motion import (
    CLIP_FACING_YAW,
    AnimationClip,
    LocomotionCommand,
    LocomotionMode,
    MovementControl,
    PlaybackSource,
    character_start_position_and_facing,
    ground_direction,
    load_locomotion_mode,
    make_animation_clip,
)
from motius.pipelines.motionbricks.scene_smart_primitive import InteractionBox
from motius.pipelines.motionbricks.scene_state_machine import StateMachine
from motius.pipelines.motionbricks.scene_target_pose import TargetPoseInferenceParams

if TYPE_CHECKING:
    from motius.motion.io.motionbricks.smart_primitive_document import (
        SmartPrimitiveAssetPool,
    )
    from motius.pipelines.motionbricks.scene_target_pose import TargetPoseIKCorrector
    from motius.pipelines.motionbricks.scene_target_pose_blender import (
        BaseTargetPoseBlender,
    )
_CLIP_TRANSITION_INFERENCE_PARAMS = TargetPoseInferenceParams(
    transition_early_by_frames=0, num_poses_to_use=MODEL_POSES_PER_TOKEN
)
_TARGET_POSE_IK_BLEND_START_FRACTION = 0.5


def _root_transform(pose: wp.array) -> wp.transform:
    """Read the root of a pose of local joint transforms or joint coordinates."""
    values = pose.numpy().reshape(-1)
    return wp.transform(*values[:7])


class BaseCharacter(ABC):
    """Animate a character with generated motion and clips through its own state machine.

    Each rig subclass supplies its motion player and target pose blender. It also sets
    root_forward_axis. Its constructor passes its presets as defaults. A rig with target
    pose IK also supplies an IK corrector.
    """

    root_forward_axis: wp.vec3
    target_pose_blender: BaseTargetPoseBlender

    def __init__(
        self,
        engine,
        *,
        asset_pool: SmartPrimitiveAssetPool,
        initial_transform: wp.transform,
        modes: tuple[LocomotionMode, ...],
        initial_mode: str,
        idle_paths: tuple[str | Path, ...],
        crouch_idle_paths: tuple[str | Path, ...],
        interaction_probe: InteractionProbe,
        target_pose_ik_corrector: TargetPoseIKCorrector | None = None,
        current_pose: wp.array | None = None,
    ) -> None:
        """Start the character in an idle clip at initial_transform.

        Args:
            engine: Engine that generates the character's motion.
            asset_pool: Clips and Smart Primitive assets of the rig.
            initial_transform: Where the character starts, from the scene's CharacterStart.
            modes: Locomotion modes, standing and crouching.
            initial_mode: Name of the mode the character starts in.
            idle_paths: Standing idle clips.
            crouch_idle_paths: Crouching idle clips.
            interaction_probe: Size and reach of the probe box.
            target_pose_ik_corrector: Target pose IK for pinned hands and feet. None
                applies no IK.
            current_pose: Pose buffer the character plays into, such as one robot's
                view of a Newton joint_q. None creates a copy of the reference pose.
        """
        self.engine = engine
        self.asset_pool = asset_pool
        self.clip_registry = asset_pool.clip_registry
        self.model_to_runtime = asset_pool.model_to_runtime
        self.interaction_probe = interaction_probe
        self._current_clip: AnimationClip | None = None
        self._playback_source = PlaybackSource.GENERATED
        self._clip_playback_time = 0.0
        self._clip_loop = False
        initial_position, initial_facing = character_start_position_and_facing(
            initial_transform
        )
        self.movement = MovementControl(initial_facing)
        self.character_ground_position = initial_position
        self.actual_velocity = np.zeros(3, dtype=np.float32)
        self.current_pose = (
            wp.clone(asset_pool.reference_pose)
            if current_pose is None
            else current_pose
        )
        self.current_foot_contacts = wp.zeros(
            FOOT_CONTACT_COUNT, dtype=wp.float32, device=engine.device
        )
        self.player = self._create_player()
        self._target_pose_ik_corrector = target_pose_ik_corrector
        self._clip_target_poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, *self.current_pose.shape),
            dtype=self.current_pose.dtype,
            device=engine.device,
        )
        self.probe_box = wp.empty(1, dtype=InteractionBox, device=engine.device)
        self.state_machine = StateMachine(
            tuple(load_locomotion_mode(mode, asset_pool) for mode in modes),
            initial_mode,
            standing_idle_clips=tuple(
                self.load_clip(path, loop=True) for path in idle_paths
            ),
            crouching_idle_clips=tuple(
                self.load_clip(path, loop=True) for path in crouch_idle_paths
            ),
            pose=self.current_pose,
        )
        seed = getattr(asset_pool, "random_seed", None)
        if seed is not None:
            self.state_machine.standing_idle_state._random.seed(seed)
            self.state_machine.crouching_idle_state._random.seed(seed)
        self.state_machine.start(self)
        self._update_probe_box()

    def load_clip(self, path: Path, *, loop: bool) -> AnimationClip:
        """Load a clip from the registry. A looping clip gets its own copy with its ends joined."""
        path = Path(path)
        if not path.is_absolute():
            path = getattr(self.asset_pool, "assets_root", Path(".")) / path
        animation = (
            self.clip_registry.load_looping(path)
            if loop
            else self.clip_registry.load(path).animation
        )
        return make_animation_clip(animation, self.model_to_runtime)

    @abstractmethod
    def _create_player(self):
        """Return this rig's motion player, which samples generated motion into current_pose."""

    @property
    def current_clip(self) -> AnimationClip | None:
        """Latest clip that played or that a clip transition leads into."""
        return self._current_clip

    @property
    def probe_segment(self) -> tuple[np.ndarray, np.ndarray]:
        """The near and far ends of the character's probe box on the ground."""
        return self.interaction_probe.segment(
            self.character_ground_position,
            self.movement.desired_velocity_direction,
            float(np.linalg.norm(self.movement.desired_velocity[:2])),
        )

    def clear_ik_pins(self) -> None:
        """Stop applying the current target's hand and foot pins."""
        if self._target_pose_ik_corrector is not None:
            self._target_pose_ik_corrector.clear_pins()

    def clip_transition_ends_within(self, delta_time: float) -> bool:
        """Return whether a clip transition reaches its transition point within delta_time (in seconds)."""
        return (
            self._playback_source is PlaybackSource.CLIP_TRANSITION
            and self.time_to_transition() <= delta_time
        )

    def clip_ends_within(self, delta_time: float) -> bool:
        """Return whether the current clip ends within delta_time. Looping clips never end."""
        return (
            self._playback_source is PlaybackSource.CLIP
            and (not self._clip_loop)
            and (self._clip_playback_time + delta_time >= self._current_clip.duration)
        )

    def time_to_transition(self) -> float:
        """Return the seconds left until the transition point. Clip playback returns 0."""
        if self._playback_source is PlaybackSource.CLIP:
            return 0.0
        return self.player.time_to_transition

    def step(self, delta_time: float, command: LocomotionCommand) -> None:
        """Advance the state machine and playback by delta_time (in seconds).

        The pose that is shown is then recorded for the next generation.
        """
        self.state_machine.update(self, delta_time, command)
        previous_ground_position = self.character_ground_position.copy()
        self._sample_playback(delta_time)
        self._update_character_velocity(previous_ground_position, delta_time)
        self.player.record_pose(delta_time, self.current_pose)
        self._update_probe_box()

    def _update_probe_box(self) -> None:
        self.interaction_probe.write_box(
            self.probe_segment,
            self.state_machine.probe_box_height_scale,
            self.probe_box,
            self.engine.stream,
        )

    def _sample_playback(self, delta_time: float) -> None:
        """Advance playback by the full frame time and sample the resulting pose."""
        if self._playback_source is PlaybackSource.CLIP:
            clip = self._current_clip
            self._clip_playback_time += delta_time
            if self._clip_loop:
                self._clip_playback_time %= clip.duration
            clip.animation.sample(
                self._clip_playback_time,
                self.current_pose,
                root_offset=clip.root_offset,
                stream=self.engine.stream,
            )
        else:
            self.player.sample(
                delta_time, self.current_pose, self.current_foot_contacts
            )
            self._apply_target_pose_ik()
        self._update_character_position()

    def generate_to_target_poses(
        self,
        target_poses: wp.array,
        inference_params: TargetPoseInferenceParams,
        playback_source: PlaybackSource = PlaybackSource.GENERATED,
    ) -> None:
        """Generate from recent motion toward target poses placed in the scene.

        Target pose IK pins the hands and feet in the target's ik_pin_mask to its
        first pose. Other targets clear the pins. playback_source is CLIP_TRANSITION
        when the motion leads into a clip.
        """
        if self._target_pose_ik_corrector is not None:
            self._target_pose_ik_corrector.capture_pins(
                target_poses[0], inference_params.ik_pin_mask, stream=self.engine.stream
            )
        self.player.generate(target_poses, inference_params)
        self._playback_source = playback_source
        self.player.sample(0.0, self.current_pose, self.current_foot_contacts)
        self._apply_target_pose_ik()
        self._update_character_position()

    def generate_clip_transition(self, clip: AnimationClip) -> None:
        """Place a clip ahead of the character and generate motion to its first frame."""
        self._current_clip = clip
        clip_ground_position = self.character_ground_position + self.actual_velocity
        self._align_clip(clip, clip_ground_position, self.movement.last_input_facing)
        for frame in range(MODEL_POSES_PER_TOKEN):
            clip.animation.sample(
                frame / MODEL_SAMPLE_RATE,
                self._clip_target_poses[frame],
                root_offset=clip.root_offset,
                stream=self.engine.stream,
            )
        self.generate_to_target_poses(
            self._clip_target_poses,
            _CLIP_TRANSITION_INFERENCE_PARAMS,
            PlaybackSource.CLIP_TRANSITION,
        )

    def _target_pose_ik_weight(self) -> float:
        playback_duration = self.player.playback_duration
        if playback_duration <= 0.0:
            return 1.0
        progress = 1.0 - self.time_to_transition() / playback_duration
        return wp.smoothstep(_TARGET_POSE_IK_BLEND_START_FRACTION, 1.0, progress)

    def _apply_target_pose_ik(self) -> None:
        if self._target_pose_ik_corrector is None:
            return
        self._target_pose_ik_corrector.apply(
            self.current_pose, self._target_pose_ik_weight(), stream=self.engine.stream
        )

    def _align_clip(
        self,
        clip: AnimationClip,
        ground_position: np.ndarray,
        facing_direction: np.ndarray,
    ) -> None:
        yaw = math.atan2(facing_direction[1], facing_direction[0]) - CLIP_FACING_YAW
        facing_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        target_ground = wp.transform(
            wp.vec3(float(ground_position[0]), float(ground_position[1]), 0.0),
            wp.mul(facing_rotation, clip.source_ground.q),
        )
        alignment = wp.mul(target_ground, wp.transform_inverse(clip.source_ground))
        clip.root_offset = wp.mul(alignment, self.model_to_runtime)

    def start_clip(
        self,
        clip: AnimationClip,
        *,
        loop: bool = False,
        align_to_current_pose: bool = False,
    ) -> None:
        """Play a clip, optionally looping it or aligning its first root with the current root.

        The motion history is kept for the next generation. Aligning the roots also
        converts source coordinates to world space.
        """
        if align_to_current_pose:
            first_pose = wp.empty_like(self.current_pose)
            clip.animation.sample(0.0, first_pose, stream=self.engine.stream)
            clip.root_offset = wp.mul(
                _root_transform(self.current_pose),
                wp.transform_inverse(_root_transform(first_pose)),
            )
        self._current_clip = clip
        self._clip_playback_time = 0.0
        self._clip_loop = loop
        self._playback_source = PlaybackSource.CLIP
        clip.animation.sample(
            self._clip_playback_time,
            self.current_pose,
            root_offset=clip.root_offset,
            stream=self.engine.stream,
        )
        self.current_foot_contacts.zero_()
        self._update_character_position()

    def initialize_clip_loop(self, clip: AnimationClip) -> None:
        """Start in an idle clip and fill the player's history with the starting pose."""
        self._align_clip(
            clip, self.character_ground_position, self.movement.last_input_facing
        )
        self.start_clip(clip, loop=True)
        self.player.seed_history_from_pose(self.current_pose)

    def pose_facing(self) -> np.ndarray:
        """Return the current pose's heading, from its root rotation."""
        forward = wp.transform_vector(
            _root_transform(self.current_pose), self.root_forward_axis
        )
        return ground_direction(
            np.array([forward[0], forward[1], 0.0], dtype=np.float32),
            self.movement.current_facing,
        )

    def face_pose(self) -> None:
        """Set every facing to the current pose's heading."""
        self.movement.face(self.pose_facing())

    def _update_character_velocity(
        self, previous_ground_position: np.ndarray, delta_time: float
    ) -> None:
        np.subtract(
            self.character_ground_position,
            previous_ground_position,
            out=self.actual_velocity,
        )
        self.actual_velocity /= delta_time
        self.actual_velocity[2] = 0.0

    def _update_character_position(self) -> None:
        if self._playback_source is PlaybackSource.CLIP:
            clip = self._current_clip
            position = sample_root_position(
                clip.root_positions,
                self._clip_playback_time,
                clip.animation.sample_rate,
                clip.root_offset,
            )
        else:
            position = self.player.current_root_position()
        self.character_ground_position[:] = position
        self.character_ground_position[2] = 0.0


__all__ = ["BaseCharacter"]
