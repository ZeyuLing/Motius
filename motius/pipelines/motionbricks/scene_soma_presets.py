# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Default locomotion modes, idle clips, and settings for the SOMA character."""

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

_SMART_PRIMITIVE_ROOT = ASSETS_ROOT / "smart_primitives" / "soma"
_IDLE_ROOT = ASSETS_ROOT / "motions" / "soma" / "locomotion" / "idle"
DEFAULT_IDLE_CLIPS = (
    _IDLE_ROOT / "idle_loop_001__A268.bvh",
    _IDLE_ROOT / "neutral_looking_around_R_001__A542.bvh",
    _IDLE_ROOT / "warm_up_neck_001__A359.bvh",
    _IDLE_ROOT / "legs_relax_002__A407.bvh",
)
DEFAULT_CROUCH_IDLE_CLIPS = (_IDLE_ROOT / "crouch_idle_001__A529.bvh",)
_NORMAL_DEFAULT_SPEED = 2.3
_NORMAL_MAX_SPEED = 4.5
_NORMAL_MIN_SPEED = 0.8 * _NORMAL_DEFAULT_SPEED
_STRAFE_DEFAULT_SPEED = 2.3
_STRAFE_MAX_SPEED = 3.0
_STRAFE_MIN_SPEED = 0.8 * _STRAFE_DEFAULT_SPEED
_CROUCH_DEFAULT_SPEED = 1.5
_CROUCH_MAX_SPEED = 2.5
_CROUCH_MIN_SPEED = _CROUCH_DEFAULT_SPEED * 0.8
_LINEAR_TARGET_VALUES = (
    ("Idle", (0.0, 0.0)),
    ("Walk", (_NORMAL_DEFAULT_SPEED / _NORMAL_MAX_SPEED, 0.0)),
    ("Jog", (3.0 / _NORMAL_MAX_SPEED, 0.0)),
    ("Run", (1.0, 0.0)),
)
LINEAR_LOCOMOTION = LocomotionMode(
    name=LINEAR_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs1_linear.smart_primitive.json",
    blend_space_type=BlendSpaceType.LINEAR_1D,
    target_values=_LINEAR_TARGET_VALUES,
    max_speed=_NORMAL_MAX_SPEED,
    default_speed=_NORMAL_DEFAULT_SPEED,
    minimum_speed=_NORMAL_MIN_SPEED,
    velocity_scale=0.97,
    transition_early_by_frames=10,
    label="Forward",
)
DIRECTIONAL_LOCOMOTION = LocomotionMode(
    name=DIRECTIONAL_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs2_directional.smart_primitive.json",
    blend_space_type=BlendSpaceType.DIRECTIONAL_2D,
    target_values=DIRECTIONAL_TARGET_VALUES,
    max_speed=_STRAFE_MAX_SPEED,
    default_speed=_STRAFE_DEFAULT_SPEED,
    minimum_speed=_STRAFE_MIN_SPEED,
    velocity_scale=0.95,
    position_scale=0.9,
    label="Strafing",
)
CROUCH_LOCOMOTION = LocomotionMode(
    name=CROUCH_LOCOMOTION_MODE,
    smart_primitive_path=_SMART_PRIMITIVE_ROOT
    / "locomotion_bs2_directional_crouch.smart_primitive.json",
    blend_space_type=BlendSpaceType.DIRECTIONAL_2D,
    target_values=DIRECTIONAL_TARGET_VALUES,
    max_speed=_CROUCH_MAX_SPEED,
    default_speed=_CROUCH_DEFAULT_SPEED,
    minimum_speed=_CROUCH_MIN_SPEED,
    velocity_scale=0.9,
    position_scale=0.8,
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
        position_scale=1.32,
        allowed_token_counts=tuple(range(6, 14)),
    ),
    style_locomotion_mode(
        "Vogue", _STYLE_PATH, minimum_speed=1.2, max_speed=1.3, position_scale=1.3
    ),
    style_locomotion_mode(
        "Injured Torso",
        _STYLE_PATH,
        minimum_speed=0.7,
        max_speed=0.8,
        position_scale=1.2,
        transition_early_by_frames=6,
        allowed_token_counts=tuple(range(8, 17)),
    ),
    style_locomotion_mode(
        "Injured Leg", _STYLE_PATH, minimum_speed=0.7, max_speed=0.8, position_scale=1.2
    ),
    style_locomotion_mode(
        "Zombie", _STYLE_PATH, minimum_speed=0.7, max_speed=0.85, position_scale=1.3
    ),
    style_locomotion_mode(
        "Combat", _STYLE_PATH, minimum_speed=0.7, max_speed=1.0, position_scale=1.7
    ),
)
DEFAULT_LOCOMOTION_MODES = (
    LINEAR_LOCOMOTION,
    DIRECTIONAL_LOCOMOTION,
    CROUCH_LOCOMOTION,
    *STYLE_LOCOMOTION_MODES,
)
INTERACTION_PROBE = InteractionProbe(
    box_width=1.0,
    box_min_depth=0.75,
    box_height=1.6,
    near_distance=0.15,
    lookahead_time=1.0,
    extra_lookahead_min_speed=2.3,
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
