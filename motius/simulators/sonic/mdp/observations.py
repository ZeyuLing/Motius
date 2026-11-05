# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 observations components; see licenses/sonic."""
from __future__ import annotations
from typing import TYPE_CHECKING
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from isaaclab.utils.math import subtract_frame_transforms
import torch
import motius.simulators.sonic.mdp.commands as commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_heading_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    grab_contact_flag = None
    hand_object_transform_6d = None
    object_pos_b = None
    object_ori_b_6d = None
    table_pos_b = None
    table_ori_b = None
    last_meta_action = None
    height_map_flat = None


@configclass
class PolicyAtmCfg(ObsGroup):
    """Observations for action_transform_module (ATM).

    This observation group provides body-only observations (29 DOF) for use with
    pre-trained action_transform_module. When using a 43 DOF robot (29 body + 14 hand),
    this group extracts only the body joint observations matching ATM's expected input.

    NOTE: Order must match the observations the pretrained ATM was trained with.
    """

    base_ang_vel = None
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    gravity_dir = None


@configclass
class TeacherCfg(ObsGroup):
    """Teacher observations for distillation.

    This provides privileged state observations that the teacher policy uses.
    The student policy learns to imitate the teacher's actions using vision instead.
    """

    command = None
    command_vel = None
    command_max = None
    command_multi_future = None
    command_multi_future_joint_pos = None
    command_multi_future_joint_body_pos = None
    command_multi_future_joint_body_diff_pos = None
    command_multi_future_joint_body_abs_pos = None
    command_multi_future_lower_body = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    root_pos_multi_future = None
    root_quat_multi_future = None
    joint_pos_multi_future = None
    smpl_pose = None
    smpl_body_pose = None
    smpl_joints_multi_future = None
    smpl_joints_lower_multi_future = None
    smpl_root_ori_b = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_pos_b_xy = None
    motion_anchor_ori_w_mf = None
    motion_anchor_ori_b = None
    motion_anchor_ori_w = None
    motion_anchor_yaw_b = None
    robot_anchor_ori_w = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    joint_pos_wo_hand = None
    joint_vel_wo_hand = None
    actions_wo_hand = None
    vr_3point_target = None
    vr_3point_target_compliant = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    head_orn_target_multi_future = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    body_pos = None
    body_pos_diff_l = None
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    grab_contact_flag = None
    last_meta_action = None


@configclass
class PrivilegedCfg(ObsGroup):
    """Privileged observations for the critic network (asymmetric actor-critic)."""

    command = None
    command_max = None
    command_multi_future = None
    command_multi_future_lower_body = None
    command_multi_future_lower_body_joint_pos = None
    command_max_multi_future = None
    command_max_diff_w = None
    command_max_diff_w_multi_future = None
    command_max_diff_l = None
    command_max_diff_l_multi_future = None
    command_z_multi_future = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b = None
    motion_anchor_ori_b_mf = None
    motion_anchor_ori_heading_mf = None
    body_pos = None
    body_ori = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    vr_3point_target = None
    vr_3point_target_compliant_multi_future = None
    vr_3point_target_multi_future = None
    vr_3point_orn_target_multi_future = None
    head_orn_target_multi_future = None
    vr_3point_local_target = None
    vr_3point_local_target_compliant = None
    vr_3point_local_target_multi_future = None
    vr_3point_local_orn_target = None
    vr_wrists_local_pos_target = None
    vr_wrists_local_orn_target = None
    vr_head_local_orn_target = None
    gravity_dir = None
    compliance = None
    ext_forces = None
    motion_anchor_gravity_dir = None
    target_object_pos = None
    hand_object_transform = None
    finger_tips_force = None
    object_pos_b_multi_future = None
    object_ori_b_multi_future = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    hand_object_transform_6d = None
    object_pos_b = None
    object_ori_b_6d = None
    table_pos_b = None
    table_ori_b = None
    task_stage = None
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    grab_contact_flag = None
    height_map_flat = None


@configclass
class DiscriminatorCfg:
    """Observation specifications for the discriminator."""

    disc_obs = None
    ref_disc_obs = None


@configclass
class TokenizerCfg(ObsGroup):
    """Observations for the tokenizer (SONIC/ATM encoder input)."""

    encoder_index = None
    command_multi_future_nonflat = None
    motion_anchor_ori_w = None
    command_z_multi_future_nonflat = None
    command_z = None
    motion_anchor_ori_b = None
    motion_anchor_ori_heading_b = None
    motion_anchor_ori_b_nonflat = None
    motion_anchor_ori_b_mf_nonflat = None
    motion_anchor_ori_w_mf_nonflat = None
    command_multi_future_egocentric_joint_transforms = None
    command_multi_future_egocentric_joint_transforms_nonflat = None
    command_multi_future_egocentric_joint_positions = None
    command_multi_future_egocentric_joint_positions_nonflat = None
    command_multi_future_egocentric_joint_rotations = None
    command_multi_future_egocentric_joint_rotations_nonflat = None
    command_multi_future_root_transforms = None
    command_multi_future_root_transforms_nonflat = None
    motion_anchor_ori_heading_mf_nonflat = None
    motion_anchor_ori_refheading_mf_nonflat = None
    heading_diff_robot_ref = None
    motion_anchor_ori_refheading = None
    motion_anchor_ori_heading = None
    command_multi_future_lower_body = None
    vr_3point_local_target = None
    vr_3point_local_orn_target = None
    vr_3point_local_target_compliant = None
    smpl_joints_multi_future_nonflat = None
    smpl_joints_multi_future_local_nonflat = None
    smpl_joints_multi_future_local_flatten = None
    smpl_lower_body_joints_multi_future_local_nonflat = None
    smpl_lower_body_joints_multi_future_local_flatten = None
    smpl_joints_lower_multi_future_local_flatten = None
    smpl_transl_z_multi_future_nonflat = None
    smpl_root_ori_b_multi_future = None
    smpl_root_ori_b_multi_future_flatten = None
    smpl_root_ori_refheading_multi_future = None
    smpl_root_ori_heading_multi_future = None
    smpl_elbow_wrist_pose_multi_future = None
    smpl_wrist_pose_multi_future = None
    joint_pos_multi_future_wrist = None
    joint_pos_multi_future_wrist_flatten = None
    joint_pos_multi_future_wrist_for_smpl = None
    soma_joints_multi_future_local_nonflat = None
    soma_root_ori_b_multi_future = None
    joint_pos_multi_future_wrist_for_soma = None
    object_pos_b = None
    object_ori_b = None
    object_ori_b_6d = None
    ref_root_pos_future_b = None
    ref_root_ori_future_b = None
    diff_body_pos_future_local = None
    diff_body_ori_future_local = None
    diff_body_lin_vel_future_local = None
    diff_body_ang_vel_future_local = None
    compliance = None
    command = None
    motion_anchor_pos_b = None
    motion_anchor_ori_b_mf = None
    base_lin_vel = None
    base_ang_vel = None
    joint_pos = None
    joint_vel = None
    actions = None
    target_object_pos = None
    hand_object_transform_6d = None
    finger_tips_force = None
    table_pos_b = None
    table_ori_b = None
    object_pos_delta_multi_future = None
    object_ori_delta_multi_future_6d = None
    command_multi_future = None


@configclass
class HeightMapCfg(ObsGroup):
    """Height map observation group for terrain-aware locomotion."""

    height_map = None


@configclass
class CameraRGBCfg(ObsGroup):
    """Camera RGB image observation group.

    This is a separate observation group for vision observations.
    It will be passed through the wrapper as 'camera_rgb' key in obs_dict,
    not concatenated with other policy observations.
    """

    camera_rgb = None


@configclass
class ResidualAction(ObsGroup):
    """Observation group for residual action feedback."""

    residual_action = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PolicyCfg = None
    critic: PrivilegedCfg = None
    disc: DiscriminatorCfg = None
    tokenizer: TokenizerCfg = None
    policy_atm: PolicyAtmCfg = None
    height_map: HeightMapCfg = None
    teacher: TeacherCfg = None
    camera_rgb: CameraRGBCfg = None
    residual_action: ResidualAction = None


def command_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get reference root z-height.

    Returns:
        torch.Tensor: Root z-height, shape (num_envs, 1).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_z


def command_z_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference root z-heights for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 1).

    Returns:
        torch.Tensor: Z-heights, shape (num_envs, num_future_frames) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_z_multi_future.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_z_multi_future


def command_multi_future(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference body positions in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, num_bodies * 3).

    Returns:
        torch.Tensor: Body positions, shape (num_envs, num_future_frames * num_bodies * 3)
            when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.command_multi_future.reshape(
            command.num_envs, command.num_future_frames, -1
        )
    else:
        return command.command_multi_future


def command_multi_future_lower_body(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Get reference lower-body positions for multiple future frames in body-local frame.

    Returns:
        torch.Tensor: Flattened lower-body positions,
            shape (num_envs, num_future_frames * lower_body_dim).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return command.command_multi_future_lower_body


def gravity_dir(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Compute gravity direction in the robot anchor (pelvis) frame.

    Transforms the world-frame down vector into the robot's local frame.
    Provides the policy with tilt/orientation information.

    Returns:
        torch.Tensor: Gravity direction vector, shape (num_envs, 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    gravity_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return gravity_dir


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body positions in robot anchor (pelvis) local frame.

    Transforms all tracked body positions from world frame into the robot's
    anchor frame using subtract_frame_transforms.

    Returns:
        torch.Tensor: Flattened body positions, shape (num_envs, num_bodies * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get robot body orientations in robot anchor (pelvis) local frame as 6D rotation.

    Returns:
        torch.Tensor: First two columns of each body's rotation matrix,
            shape (num_envs, num_bodies * 6).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(
    env: ManagerBasedEnv, command_name: str, mask_out_z=False
) -> torch.Tensor:
    """Get reference motion anchor position relative to robot anchor in robot-local frame.

    Computes the position of the reference motion's root relative to the robot's
    root, expressed in the robot's local coordinate frame.

    Args:
        command_name: Name of the tracking command term.
        mask_out_z: If True, return only xy components (discard z).

    Returns:
        torch.Tensor: Position offset, shape (num_envs, 3) or (num_envs, 2) if mask_out_z.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    if mask_out_z:
        pos = pos[:, :2]
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference anchor orientation relative to robot anchor as 6D rotation.

    Computes the orientation difference between the reference motion root and the
    robot root, expressed in the robot's local frame.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, repeat across future frames and return
            shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation representation, shape (num_envs, 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    ori = mat[..., :2].reshape(mat.shape[0], -1)
    if non_flatten:
        return ori.unsqueeze(1).repeat(1, command.num_future_frames, 1)
    else:
        return ori


def motion_anchor_ori_b_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=False
) -> torch.Tensor:
    """Get reference-vs-robot root orientation difference for multiple future frames in body-local frame.

    Uses the robot's full orientation (including pitch/roll) for normalization.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences,
            shape (num_envs, num_future_frames * 6) when flattened.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        return command.root_rot_dif_l_multi_future.reshape(
            env.num_envs, command.num_future_frames, -1
        )
    else:
        return command.root_rot_dif_l_multi_future.reshape(env.num_envs, -1)


def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point positions in reference motion anchor local frame.

    NOTE: "local" here means relative to the reference motion root, not the robot root.
    This gives the policy the reference pose structure independent of global position.

    Returns:
        torch.Tensor: Flattened 3-point positions, shape (num_envs, num_points * 3).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get VR 3-point body orientations in reference anchor local frame.

    Returns:
        torch.Tensor: Flattened 3-point quaternions, shape (num_envs, num_points * 4).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat), ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)


def get_command_obs(
    env: ManagerBasedEnv, command_name: str, obs_name: str
) -> torch.Tensor:
    """Get an arbitrary attribute from the tracking command by name.

    Generic accessor for command observations not covered by dedicated functions.

    Args:
        command_name: Name of the tracking command term.
        obs_name: Attribute name on the command object.

    Returns:
        torch.Tensor: The requested observation tensor.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    return getattr(command, obs_name)


def joint_pos_multi_future_select_joints_for_smpl(
    env: ManagerBasedEnv, command_name: str, joints_idx: list
) -> torch.Tensor:
    """Extract reference joint positions for selected joints across SMPL-aligned future frames.

    NOTE: This is a duplicate definition that shadows the previous one. Only this
    version is active at runtime.

    Args:
        command_name: Name of the tracking command term.
        joints_idx: Indices of joints to select.

    Returns:
        torch.Tensor: Selected joint positions,
            shape (num_envs, smpl_num_future_frames, len(joints_idx)).
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = command.joint_pos_multi_future_for_smpl
    joint_pos_reshaped = joint_pos.view(
        env.num_envs, command.smpl_num_future_frames, -1
    )
    joint_pos_selected = joint_pos_reshaped[..., joints_idx]
    return joint_pos_selected


def smpl_root_ori_b_mf(
    env: ManagerBasedEnv, command_name: str, non_flatten=True
) -> torch.Tensor:
    """Get SMPL root orientation difference in body-local frame for multiple future frames.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, 6).

    Returns:
        torch.Tensor: 6D rotation differences.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    if non_flatten:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(
            env.num_envs, command.smpl_num_future_frames, -1
        )
    else:
        rot = command.smpl_root_quat_w_dif_l_multi_future.reshape(env.num_envs, -1)
    return rot


def smpl_joints_multi_future_local(
    env: ManagerBasedEnv, command_name: str, non_flatten=False, joints_idx=None
) -> torch.Tensor:
    """Get SMPL joint positions relative to each frame's own root orientation.

    Unlike smpl_joints_multi_future which uses only the first frame's root,
    this version canonicalizes each future frame independently using its own
    root orientation. Optionally select a subset of joints.

    Args:
        command_name: Name of the tracking command term.
        non_flatten: If True, return shape (num_envs, smpl_num_future_frames, num_joints * 3).
        joints_idx: Optional list of joint indices to select.

    Returns:
        torch.Tensor: Per-frame root-relative SMPL joint positions.
    """
    command: commands.TrackingCommand = env.command_manager.get_term(command_name)
    ref_joints = command.smpl_joints_multi_future
    ref_root_quat = command.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
        1, 1, ref_joints.shape[-2], 1
    )
    ref_joints_root = quat_apply(quat_inv(ref_root_quat), ref_joints)
    if joints_idx is not None:
        ref_joints_root = ref_joints_root[..., joints_idx, :]
    if non_flatten:
        return ref_joints_root.reshape(env.num_envs, command.smpl_num_future_frames, -1)
    else:
        return ref_joints_root.view(env.num_envs, -1)
