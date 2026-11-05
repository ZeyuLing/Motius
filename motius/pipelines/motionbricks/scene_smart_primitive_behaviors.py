# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Example behaviors that scenes choose for each Smart Primitive by class name."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import ModelRig
from motius.pipelines.motionbricks.scene_assets import ASSETS_ROOT
from motius.pipelines.motionbricks.scene_smart_primitive import (
    CharacterProperties,
    SmartPrimitive,
    SmartPrimitiveAsset,
)
from motius.pipelines.motionbricks.scene_target_pose import TargetPoseInferenceParams

if TYPE_CHECKING:
    from motius.pipelines.motionbricks.scene_states import InteractionState
_MOTION_ROOT = ASSETS_ROOT / "motions"
_SITTING_CLIPS = {
    ModelRig.SOMA: _MOTION_ROOT / "soma/sitting/sit_on_chair_loop.bvh",
    ModelRig.UNITREE_G1: _MOTION_ROOT / "unitree_g1/sitting/sit_on_chair_loop.csv",
}


class SmartPrimitiveBehavior(SmartPrimitive):
    """Define how a character interacts with this Smart Primitive.

    Scenes choose a subclass by its class name. Every subclass appears in the
    Scene Editor's Behavior list once its module is imported.

    Override enter() to start a target pose sequence or a clip.
    Override interact() to respond when the user interacts again.
    Use self.interaction_state to play sequences and clips or finish the interaction.

    By default, the character follows the targets from get_targets(),
    then returns to locomotion or idle.
    """

    required_sequences: tuple[str, ...] = ()
    clip_paths: tuple[Path, ...] = ()

    def __init__(self, asset: SmartPrimitiveAsset) -> None:
        """Create the runtime instance, and check that the required sequences exist."""
        super().__init__(asset)
        self._interaction_state: InteractionState | None = None
        for name in self.required_sequences:
            if self.sequence_index(name) < 0:
                raise ValueError(f"{type(self).__name__} requires sequence {name!r}.")

    @property
    def interaction_state(self) -> InteractionState:
        """Interaction state that runs this behavior while it is active."""
        if self._interaction_state is None:
            raise RuntimeError(
                "Cannot access the interaction state without an active interaction."
            )
        return self._interaction_state

    def bind(self, interaction_state: InteractionState) -> None:
        """Give the behavior the interaction state that runs it. Interactions call this on entry."""
        self._interaction_state = interaction_state

    def unbind(self) -> None:
        """Release the interaction state. Interactions call this on exit."""
        self._interaction_state = None

    def get_targets(
        self,
        character_properties: CharacterProperties,
        output_target_poses: wp.array,
        stream: wp.Stream | None = None,
    ) -> tuple[TargetPoseInferenceParams, ...]:
        """Write an interaction's targets and return their generation settings.

        The base class returns no targets. Subclasses select their targets, update the
        hierarchy, and write the poses in the same order as the returned settings.

        Args:
            character_properties: The approaching character.
            output_target_poses: Array that receives four poses per target.
            stream: Stream for the work. None uses the current stream.
        """
        return ()

    def enter(self) -> None:
        """Start the interaction.

        The default plays the sequence that get_targets() chooses.
        """
        self.interaction_state.play_target_pose_sequence()

    def interact(self) -> None:
        """Respond when the user interacts again before this behavior has finished."""


class VelocityMatchingSequenceBehavior(SmartPrimitiveBehavior):
    """Pick the sequence closest to the character's speed, then align its pivot."""

    def __init__(self, asset: SmartPrimitiveAsset) -> None:
        """Create the runtime instance. The document needs at least one sequence."""
        super().__init__(asset)
        if self.sequence_count == 0:
            raise ValueError(
                "Velocity Matching Sequence behavior requires at least one Target Pose sequence."
            )

    def get_targets(
        self,
        character_properties: CharacterProperties,
        output_target_poses: wp.array,
        stream: wp.Stream | None = None,
    ) -> tuple[TargetPoseInferenceParams, ...]:
        character_speed = float(np.linalg.norm(character_properties.velocity))
        sequence_index = min(
            range(self.sequence_count),
            key=lambda index: abs(self.sequence_start_speed(index) - character_speed),
        )
        self.align_sequence_pivot(sequence_index, character_properties, stream)
        return self.get_world_target_poses_for_sequence(
            sequence_index, output_target_poses, stream
        )


class SittingBehavior(SmartPrimitiveBehavior):
    """Sit down, stay seated, and stand up when the user interacts again."""

    required_sequences = ("Sit", "Stand")

    @property
    def clip_paths(self) -> tuple[Path, ...]:
        """The rig's looping sitting clip."""
        return (_SITTING_CLIPS[self.rig],)

    def enter(self) -> None:
        self.interaction_state.play_target_pose_sequence(
            "Sit", on_complete=self._start_sitting
        )

    def _start_sitting(self) -> None:
        self.interaction_state.play_clip(
            _SITTING_CLIPS[self.rig], loop=True, align_to_current_pose=True
        )

    def interact(self) -> None:
        if self.interaction_state.is_looping:
            self.interaction_state.play_target_pose_sequence("Stand")


def behavior_types(
    base: type[SmartPrimitiveBehavior] = SmartPrimitiveBehavior,
) -> dict[str, type[SmartPrimitiveBehavior]]:
    """Find every subclass of base by class name, in definition order.

    A subclass is found once its module has been imported.
    """
    types = {}
    for behavior in base.__subclasses__():
        types[behavior.__name__] = behavior
        types.update(behavior_types(behavior))
    return types


__all__ = [
    "SittingBehavior",
    "SmartPrimitiveBehavior",
    "VelocityMatchingSequenceBehavior",
    "behavior_types",
]
