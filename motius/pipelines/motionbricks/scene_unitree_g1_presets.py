# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Default locomotion modes, idle clips, and settings for the G1 character."""

from motius.pipelines.motionbricks.scene_assets import ASSETS_ROOT
from motius.pipelines.motionbricks.scene_blend_spaces import BlendSpaceType
from motius.pipelines.motionbricks.scene_interaction import InteractionProbe
from motius.pipelines.motionbricks.scene_motion import (
    CROUCH_LOCOMOTION_MODE,
    DIRECTIONAL_LOCOMOTION_MODE,
    DIRECTIONAL_TARGET_VALUES,
    LINEAR_LOCOMOTION_MODE,
    LocomotionMode,
    style_locomotion_mode,
)

_SMART_PRIMITIVE_ROOT = ASSETS_ROOT / "smart_primitives/unitree_g1"
_MOTION_ROOT = ASSETS_ROOT / "motions/unitree_g1"
_IDLE_ROOT = _MOTION_ROOT / "locomotion/idle"
DEFAULT_IDLE_CLIPS = (
    _IDLE_ROOT / "idle_loop_001__A268.csv",
    _IDLE_ROOT / "neutral_looking_around_R_001__A542.csv",
    _IDLE_ROOT / "legs_relax_002__A407.csv",
)
DEFAULT_CROUCH_IDLE_CLIPS = (_IDLE_ROOT / "crouch_idle_001__A529.csv",)
LINEAR_LOCOMOTION = LocomotionMode(
    name=LINEAR_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs1_linear.smart_primitive.json",
    blend_space_type=BlendSpaceType.LINEAR_1D,
    target_values=(
        ("Idle", (0.0, 0.0)),
        ("Walk", (1.3 / 3.4, 0.0)),
        ("Jog", (2.0 / 3.4, 0.0)),
        ("Run", (1.0, 0.0)),
    ),
    max_speed=3.4,
    default_speed=1.3,
    minimum_speed=1.3,
    transition_early_by_frames=10,
    label="Forward",
)
DIRECTIONAL_LOCOMOTION = LocomotionMode(
    name=DIRECTIONAL_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs2_directional.smart_primitive.json",
    blend_space_type=BlendSpaceType.DIRECTIONAL_2D,
    target_values=DIRECTIONAL_TARGET_VALUES,
    max_speed=2.3,
    default_speed=1.3,
    minimum_speed=1.3,
    transition_early_by_frames=10,
    label="Strafing",
)
CROUCH_LOCOMOTION = LocomotionMode(
    name=CROUCH_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs2_directional_crouch.smart_primitive.json",
    blend_space_type=BlendSpaceType.DIRECTIONAL_2D,
    target_values=DIRECTIONAL_TARGET_VALUES,
    max_speed=1.5,
    default_speed=0.8,
    minimum_speed=0.8,
    transition_early_by_frames=10,
    label="Crouching",
)
_STYLE_PATH = (
    _SMART_PRIMITIVE_ROOT
    / "locomotion_style_variations_bs1_linear.smart_primitive.json"
)
STYLE_LOCOMOTION_MODES = (
    style_locomotion_mode(
        "Skipping",
        _STYLE_PATH,
        minimum_speed=1.3,
        max_speed=1.45,
        transition_early_by_frames=0,
    ),
    style_locomotion_mode(
        "Vogue", _STYLE_PATH, minimum_speed=0.8, max_speed=0.9, position_scale=1.3
    ),
    style_locomotion_mode(
        "Injured Torso", _STYLE_PATH, minimum_speed=0.4, max_speed=0.45
    ),
    style_locomotion_mode("Injured Leg", _STYLE_PATH, minimum_speed=0.4, max_speed=0.6),
    style_locomotion_mode("Zombie", _STYLE_PATH, minimum_speed=0.4, max_speed=0.45),
    style_locomotion_mode("Combat", _STYLE_PATH, minimum_speed=0.65, max_speed=0.75),
)
DEFAULT_LOCOMOTION_MODES = (
    LINEAR_LOCOMOTION,
    DIRECTIONAL_LOCOMOTION,
    CROUCH_LOCOMOTION,
    *STYLE_LOCOMOTION_MODES,
)
INTERACTION_PROBE = InteractionProbe(
    box_width=0.79,
    box_min_depth=0.59,
    box_height=1.26,
    near_distance=0.12,
    lookahead_time=1.0,
    extra_lookahead_min_speed=1.3,
    extra_lookahead_time=0.25,
)
__all__ = [
    "CROUCH_LOCOMOTION",
    "DEFAULT_CROUCH_IDLE_CLIPS",
    "DEFAULT_IDLE_CLIPS",
    "DEFAULT_LOCOMOTION_MODES",
    "DIRECTIONAL_LOCOMOTION",
    "INTERACTION_PROBE",
    "LINEAR_LOCOMOTION",
    "STYLE_LOCOMOTION_MODES",
]
