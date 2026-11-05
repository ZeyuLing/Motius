# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Termination compute kernels for environments.

Pure tensor functions (kernels) for computing terminations.
Use MdpComponent in experiment configs to bind kernels to context paths.

Organized into:
- base: Core termination functions (fall, height, contact, episode length)
- tracking: Motion tracking terminations (tracking error, BeyondMimic style)
- task: Task-specific terminations (path following)
"""
from motius.simulators.isaaclab.protomotions.mdp.context_views import EnvContext
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    check_fall_contact_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    check_height_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    check_max_length_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    combine_fall_termination,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    threshold_termination,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    fall_termination,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    height_termination,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.base import (
    contact_termination,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    compute_tracking_error,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    compute_anchor_pos_error_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    compute_anchor_ori_error_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    compute_relative_body_pos_error_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    compute_anchor_height_error_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    motion_clip_done,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    mean_body_pos_error,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    max_body_pos_error,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    mean_body_rot_error,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    anchor_pos_error_value,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    anchor_ori_error_value,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    anchor_height_error_value,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.tracking import (
    relative_body_pos_max_error,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.task import (
    check_path_distance_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.task import (
    check_path_height_term,
)
from motius.simulators.isaaclab.protomotions.mdp.terminations.task import (
    check_steering_velocity_error,
)

__all__ = [
    "EnvContext",
    "check_fall_contact_term",
    "check_height_term",
    "check_max_length_term",
    "combine_fall_termination",
    "threshold_termination",
    "fall_termination",
    "height_termination",
    "contact_termination",
    "compute_tracking_error",
    "compute_anchor_pos_error_term",
    "compute_anchor_ori_error_term",
    "compute_relative_body_pos_error_term",
    "compute_anchor_height_error_term",
    "motion_clip_done",
    "mean_body_pos_error",
    "max_body_pos_error",
    "mean_body_rot_error",
    "anchor_pos_error_value",
    "anchor_ori_error_value",
    "anchor_height_error_value",
    "relative_body_pos_max_error",
    "check_path_distance_term",
    "check_path_height_term",
    "check_steering_velocity_error",
]
