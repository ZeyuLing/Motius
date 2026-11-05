# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""State machines that switch a character between idle, locomotion, and interactions."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from motius.pipelines.motionbricks.scene_motion import (
    CROUCH_LOCOMOTION_MODE,
    DIRECTIONAL_LOCOMOTION_MODE,
    LINEAR_LOCOMOTION_MODE,
    AnimationClip,
    LocomotionCommand,
    LocomotionModeRuntime,
)
from motius.pipelines.motionbricks.scene_states import (
    IdleState,
    InteractionState,
    LocomotionState,
    State,
)

_CROUCH_PROBE_HEIGHT_SCALE = 0.6


@dataclass(frozen=True, slots=True)
class _MovementStatePair:
    idle: IdleState
    locomotion: LocomotionState
    probe_box_height_scale: float = 1.0


class StateMachine:
    """Switch between idle, locomotion, and interactions while standing or crouching.

    The caller loads the locomotion modes and the idle clips. The states generate
    motion, and this machine decides which one runs. A state change exits the old
    state before it enters the new one.
    """

    def __init__(
        self,
        mode_runtimes: tuple[LocomotionModeRuntime, ...],
        initial_mode: str = LINEAR_LOCOMOTION_MODE,
        *,
        standing_idle_clips: tuple[AnimationClip, ...],
        crouching_idle_clips: tuple[AnimationClip, ...],
        pose: wp.array,
    ) -> None:
        """Create the idle and locomotion states for standing and crouching.

        The interaction state is shared by both. The state machine starts in the idle state
        of initial_mode. Call start() to begin its idle loop.

        Args:
            mode_runtimes: Loaded locomotion modes. They must include Forward, Strafing,
                and Crouching. Any other mode is a standing style mode, listed in
                style_modes in the order given.
            initial_mode: Name of the mode to start in.
            standing_idle_clips: Looping clips for the standing idle state.
            crouching_idle_clips: Looping clips for the crouching idle state.
            pose: Pose with the layout of the character's poses. It sizes the target
                pose buffers.
        """
        modes_by_name = {runtime.definition.name: runtime for runtime in mode_runtimes}
        if len(modes_by_name) != len(mode_runtimes):
            raise ValueError("Locomotion mode names must be unique.")
        core_names = (
            LINEAR_LOCOMOTION_MODE,
            DIRECTIONAL_LOCOMOTION_MODE,
            CROUCH_LOCOMOTION_MODE,
        )
        missing_names = set(core_names).difference(modes_by_name)
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise ValueError(f"Required locomotion modes are missing: {missing}.")
        if initial_mode not in modes_by_name:
            raise ValueError(f"Unknown initial locomotion mode {initial_mode!r}.")
        self.style_modes = tuple(
            name for name in modes_by_name if name not in core_names
        )
        standing_names = (
            LINEAR_LOCOMOTION_MODE,
            DIRECTIONAL_LOCOMOTION_MODE,
            *self.style_modes,
        )
        self.standing_idle_state = IdleState("Idle", standing_idle_clips)
        self.standing_locomotion_state = LocomotionState(
            "Locomotion",
            tuple(modes_by_name[name] for name in standing_names),
            initial_mode
            if initial_mode != CROUCH_LOCOMOTION_MODE
            else LINEAR_LOCOMOTION_MODE,
            pose,
        )
        self.crouching_idle_state = IdleState("Crouching Idle", crouching_idle_clips)
        self.crouching_locomotion_state = LocomotionState(
            "Crouching Locomotion",
            (modes_by_name[CROUCH_LOCOMOTION_MODE],),
            CROUCH_LOCOMOTION_MODE,
            pose,
        )
        self.interaction_state = InteractionState()
        self._standing_pair = _MovementStatePair(
            self.standing_idle_state, self.standing_locomotion_state
        )
        self._crouching_pair = _MovementStatePair(
            self.crouching_idle_state,
            self.crouching_locomotion_state,
            probe_box_height_scale=_CROUCH_PROBE_HEIGHT_SCALE,
        )
        self._movement_pairs_by_mode = {
            name: self._standing_pair for name in standing_names
        }
        self._movement_pairs_by_mode[CROUCH_LOCOMOTION_MODE] = self._crouching_pair
        self._active_movement_pair = self._movement_pairs_by_mode[initial_mode]
        self._state: State = self._active_movement_pair.idle

    @property
    def state(self) -> State:
        """Current state."""
        return self._state

    @property
    def active_locomotion_state(self) -> LocomotionState:
        """Locomotion state of the active movement pair, standing or crouching."""
        return self._active_movement_pair.locomotion

    @property
    def probe_box_height_scale(self) -> float:
        """Scale of the probe box height for the active movement pair, standing or crouching."""
        return self._active_movement_pair.probe_box_height_scale

    @property
    def is_locomoting(self) -> bool:
        """Whether a locomotion state is current."""
        return self.state in (
            self.standing_locomotion_state,
            self.crouching_locomotion_state,
        )

    @property
    def is_interacting(self) -> bool:
        """Whether the interaction state is current."""
        return self.state is self.interaction_state

    def start(self, character) -> None:
        """Start the character in the initial idle loop."""
        self._active_movement_pair.idle.start(character)

    def transition_to(self, character, next_state: State) -> None:
        """Exit the current state and enter next_state, unless next_state is already current."""
        if next_state is self._state:
            return
        self._state.exit(character, self)
        self._state = next_state
        next_state.enter(character, self)

    def update(self, character, delta_time: float, command: LocomotionCommand) -> None:
        """Switch modes as the command asks, smooth the movement, then update the current state.

        Movement smooths in every state, so it carries into the next one.
        """
        previous_pair = self._active_movement_pair
        previous_mode = previous_pair.locomotion.active_mode
        if command.requested_mode is not None:
            self._active_movement_pair = self._movement_pairs_by_mode[
                command.requested_mode
            ]
            self._active_movement_pair.locomotion.select_mode(command.requested_mode)
        mode = self.active_locomotion_state.active_mode
        character.movement.update(
            command,
            mode,
            delta_time,
            previous_mode if mode is not previous_mode else None,
        )
        movement_pair_changed = previous_pair is not self._active_movement_pair
        if movement_pair_changed and self.state in (
            previous_pair.idle,
            previous_pair.locomotion,
        ):
            self.resume_movement(character)
        self.state.update(character, self, delta_time)

    def transition_to_idle(self, character) -> None:
        """Switch to the idle state of the active movement pair."""
        self.transition_to(character, self._active_movement_pair.idle)

    def transition_to_locomotion(self, character) -> None:
        """Switch to the locomotion state of the active movement pair."""
        self.transition_to(character, self._active_movement_pair.locomotion)

    def resume_movement(self, character) -> None:
        """Switch to locomotion when there is movement input, or to idle otherwise."""
        if character.movement.input_speed > 0.0:
            self.transition_to_locomotion(character)
        else:
            self.transition_to_idle(character)

    def start_interaction(self, character, primitive) -> None:
        """Let a Smart Primitive control the character until it calls finish()."""
        if self.is_interacting:
            raise RuntimeError("Cannot start an interaction while another is active.")
        self.interaction_state.configure(character, primitive)
        try:
            self.transition_to(character, self.interaction_state)
        except Exception:
            self.resume_movement(character)
            raise

    def interact(self) -> None:
        """Pass an interaction press to the active Smart Primitive behavior."""
        if self.is_interacting:
            self.interaction_state.primitive.interact()


__all__ = ["StateMachine"]
