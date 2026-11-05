# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 terminations components; see licenses/sonic."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from motius.simulators.sonic.mdp.commands import TrackingCommand
from motius.simulators.sonic.mdp.commands import _get_body_indexes


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = None
    anchor_pos = None
    anchor_ori = None
    anchor_ori_full = None
    ee_body_pos = None
    anchor_pos_xy = None
    foot_pos_xyz = None
    cumm_body_pos_error = None
    cumm_body_ori_error = None
    cumm_body_pos_error_local = None
    cumm_body_ori_error_local = None
    object_pos_deviation = None
    object_z_pos_deviation = None
    object_not_lifted = None
    robot_table_contact_before_object = None
    hand_table_contact_termination = None
    grasp_failure_after_contact = None


def exceeded_anchor_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if anchor Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``
    (e.g. crouching or sitting motions).

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    height_diff = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_diff, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
        return height_diff.gt(thresh)
    return height_diff.gt(threshold)


def exceeded_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate if the squared anchor orientation error exceeds a threshold.

    Compute the full quaternion error magnitude between reference and robot
    anchor orientations, then compare the squared value against ``threshold``.

    Args:
        env: The manager-based RL environment.
        asset_cfg: Scene entity config (unused but required by termination API).
        command_name: Name of the tracking command term.
        threshold: Maximum allowed squared orientation error (radians^2).

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    return angular_err.square().gt(threshold)


def exceeded_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Terminate if any tracked body position error exceeds a distance threshold.

    Compare per-body world-frame positions between the reference motion and the
    robot, and terminate if *any* body exceeds ``threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Maximum allowed position error per body in meters.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = (
        command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    )
    return pos_diff.norm(dim=-1).gt(threshold).any(dim=-1)


def exceeded_body_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    body_names: list[str] | None = None,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate if any tracked body Z-height error exceeds a threshold.

    When ``threshold_adaptive`` is True, use a looser ``down_threshold`` for
    environments whose reference root height is below ``root_height_threshold``.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.
        threshold: Default maximum allowed height error per body in meters.
        threshold_adaptive: Enable per-env adaptive thresholding.
        down_threshold: Looser threshold applied when the reference root is low.
        body_names: Optional list of body names to check. If ``None``, all
            tracked bodies from the command are used.
        root_height_threshold: Height below which ``down_threshold`` is used.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    height_err = (
        command.body_pos_relative_w[:, tracked, 2]
        - command.robot_body_pos_w[:, tracked, 2]
    ).abs()
    if threshold_adaptive:
        thresh = torch.full_like(height_err, threshold)
        thresh[command.running_ref_root_height < root_height_threshold] = down_threshold
        return height_err.gt(thresh).any(dim=-1)
    return height_err.gt(threshold).any(dim=-1)


def tracking_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the motion clip has been fully played.

    Compare the elapsed simulation steps (including the motion start offset)
    against the total length of each environment's assigned motion clip.

    Args:
        env: The manager-based RL environment.
        command_name: Name of the tracking command term.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    elapsed = command.time_steps + command.motion_start_time_steps + 1
    total = command.motion_lib.get_time_step_total(command.motion_ids)
    return elapsed >= total
