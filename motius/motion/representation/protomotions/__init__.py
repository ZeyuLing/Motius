# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""G1 actor observation construction and PD action conversion, outside neural modules."""
from typing import Union,List,Dict
import torch
from torch import Tensor
from . import rotations

def select_step_indices(
    tensor: Tensor,
    steps: Union[int, List[int]],
    dim: int = 1
) -> Tensor:
    """Select steps from tensor by index.

    Supports both consecutive steps (int) and arbitrary step indices (list).
    Uses 1-indexed step numbers that are converted to 0-indexed tensor positions.

    Args:
        tensor: Input tensor with steps along dim.
        steps: If int N, selects first N steps (like arange(1, N+1)).
               If list, selects specific 1-indexed steps (e.g., [1, 3, 5] -> indices [0, 2, 4]).
        dim: Dimension containing steps.

    Returns:
        Tensor with selected steps.
    """
    if isinstance(steps, int):
        return tensor.narrow(dim, 0, steps)
    else:
        indices = torch.tensor([s - 1 for s in steps], device=tensor.device, dtype=torch.long)
        return tensor.index_select(dim, indices)



def root_projected_gravity(root_rot: Tensor, w_last: bool = True) -> torch.Tensor:
    """
    Computes the direction of gravity in the root's local frame.

    Given the root orientation quaternion(s) and assuming gravity points as [0, 0, -1] in world coordinates,
    this function transforms the gravity vector into the local (body) frame of the root by applying the
    inverse rotation. The resulting vector describes how gravity would be measured relative to the root's axes.

    Args:
        root_rot: Root orientation quaternion(s), shape [batch, 4].
        w_last: If True, quaternions use xyzw ordering.

    Returns:
        Tensor of shape [batch, 3]: Gravity vector(s) expressed in the local root frame.
        Returns None if root_rot is None.
    """
    GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 3)
    if root_rot is not None:
        if GRAVITY_VEC_W.device != root_rot.device:
            GRAVITY_VEC_W = GRAVITY_VEC_W.to(root_rot.device)
        return rotations.quat_rotate_inverse(
            root_rot, GRAVITY_VEC_W.repeat(root_rot.shape[0], 1), w_last=w_last
        )
    return None


def compute_humanoid_reduced_coords_observations(
    dof_pos: Tensor,
    dof_vel: Tensor,
    anchor_rot: Tensor,
    root_local_ang_vel: Tensor,
    w_last: bool,
    root_rot: Tensor = None,
    root_pos: Tensor = None,
    root_vel: Tensor = None,
    ground_height: Tensor = None,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> Tensor:
    """Compute reduced coordinates observations for humanoid.

    Output format (in order):
        - dof_pos [num_dofs]
        - dof_vel [num_dofs]
        - root_ang_vel [3]: root angular velocity in local frame
        - proj_gravity [3]: projected gravity vector
        - root_vel [3]: root linear velocity in local frame (if root_vel_obs=True)
        - root_height [1]: root height above ground (if root_height_obs=True)

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        anchor_rot: Anchor body orientation for gravity projection [num_envs, 4].
        root_local_ang_vel: Root angular velocity pre-normalized to local frame [num_envs, 3].
        w_last: Whether quaternion w component is last.
        root_rot: Root orientation quaternion [num_envs, 4]. Required if root_vel_obs=True.
        root_pos: Root position [num_envs, 3]. Required if root_height_obs=True.
        root_vel: Root linear velocity in world frame [num_envs, 3]. Required if root_vel_obs=True.
        ground_height: Ground height beneath root [num_envs] or [num_envs, 1].
        root_height_obs: Whether to include root height observation.
        root_vel_obs: Whether to include root velocity observation.

    Returns:
        Concatenated observation tensor [num_envs, obs_dim].
    """
    num_envs = dof_pos.shape[0]
    proj_gravity = root_projected_gravity(anchor_rot, w_last)

    obs = [
        dof_pos.view(num_envs, -1),
        dof_vel.view(num_envs, -1),
        root_local_ang_vel.view(num_envs, -1),
        proj_gravity.view(num_envs, -1),
    ]

    if root_vel_obs:
        if root_vel is None:
            raise ValueError("root_vel is required when root_vel_obs=True")
        if root_rot is None:
            raise ValueError("root_rot is required when root_vel_obs=True")
        normalized_root_vel = rotations.quat_rotate_inverse(root_rot, root_vel, w_last)
        obs.append(normalized_root_vel.view(num_envs, -1))

    if root_height_obs:
        if root_pos is None:
            raise ValueError("root_pos is required when root_height_obs=True")
        if ground_height is None:
            ground_height = torch.zeros(num_envs, 1, device=root_pos.device)
        elif ground_height.dim() == 1:
            ground_height = ground_height.unsqueeze(-1)
        root_h = root_pos[:, 2:3] - ground_height
        obs.append(root_h)

    obs = torch.cat(obs, dim=-1)
    return obs



def build_reduced_coords_target_poses(
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_rot: Tensor,
    mimic_ref_dof_vel: Tensor,
    mimic_ref_dof_pos: Tensor,
    w_last: bool = True,
    current_state_anchor_pos: Tensor = None,
    mimic_ref_anchor_pos: Tensor = None,
    mimic_ref_anchor_vel: Tensor = None,
    mimic_ref_anchor_ang_vel: Tensor = None,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_dof_vel: bool = True,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    future_steps: Union[int, List[int]] = None,
    current_ref_anchor_pos: Tensor = None,
    zero_xy_offset: bool = False,
):
    """Build target pose observations in reduced coordinates.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_anchor_rot: Reference body rotations [envs, future_steps, 4]
        mimic_ref_dof_vel: Reference DOF velocities [envs, future_steps, num_dofs]
        mimic_ref_dof_pos: Reference DOF positions [envs, future_steps, num_dofs]
        w_last: If True, quaternions are in XYZW format
        current_state_anchor_pos: Current anchor position [envs, 3]
        mimic_ref_anchor_pos: Reference body positions [envs, future_steps, 3]
        mimic_ref_anchor_vel: Reference anchor linear velocity [envs, future_steps, 3]
        mimic_ref_anchor_ang_vel: Reference anchor angular velocity [envs, future_steps, 3]
        include_xy_offset: If True, includes XY offset [2]
        include_height: If True, includes absolute height [1]
        include_dof_vel: If True, includes DOF velocities [num_dofs]
        include_anchor_vel: If True, includes anchor linear velocity in local frame [3]
        include_anchor_ang_vel: If True, includes anchor angular velocity in local frame [3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used with include_xy_offset to compute drift from current reference.
        zero_xy_offset: If True, emit zeros for the XY offset (for inference).

    Returns:
        Target pose observations [envs, features]
    """
    num_envs = current_state_anchor_rot.shape[0]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)
        if mimic_ref_anchor_pos is not None:
            mimic_ref_anchor_pos = select_step_indices(
                mimic_ref_anchor_pos, future_steps
            )
        if mimic_ref_anchor_vel is not None:
            mimic_ref_anchor_vel = select_step_indices(
                mimic_ref_anchor_vel, future_steps
            )
        if mimic_ref_anchor_ang_vel is not None:
            mimic_ref_anchor_ang_vel = select_step_indices(
                mimic_ref_anchor_ang_vel, future_steps
            )

    future_steps = mimic_ref_anchor_rot.shape[1]

    # Flatten: [envs, future_steps, dim] -> [envs*future_steps, dim]
    ref_state_anchor_rot = mimic_ref_anchor_rot.reshape(-1, 4)
    ref_state_dof_vel = mimic_ref_dof_vel.reshape(-1, mimic_ref_dof_vel.shape[-1])
    ref_state_dof_pos = mimic_ref_dof_pos.reshape(-1, mimic_ref_dof_pos.shape[-1])

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)

    current_state_anchor_rot_expanded = (
        current_state_anchor_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    heading_inv_rot_expanded = (
        heading_inv_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    # Target root rot relative to current root rot
    rel_target_anchor_rot = rotations.quat_mul(
        rotations.quat_conjugate(current_state_anchor_rot_expanded, w_last),
        ref_state_anchor_rot,
        w_last,
    )
    target_anchor_rot_obs = rotations.quat_to_tan_norm(rel_target_anchor_rot, w_last)

    # Build observation components
    obs_components = [target_anchor_rot_obs]  # [N, 6]

    if include_dof_vel:
        obs_components.append(ref_state_dof_vel)  # [N, num_dofs]

    obs_components.append(ref_state_dof_pos)  # [N, num_dofs]

    if include_xy_offset or include_height:
        # Compute position components
        # Extract root position: [envs, future_steps, 3] -> [envs*future_steps, 3]
        ref_state_anchor_pos = mimic_ref_anchor_pos.reshape(-1, 3)

        if include_xy_offset:
            if zero_xy_offset:
                # Inference: zero out XY offset ("you're tracking perfectly")
                xy_offset_local = torch.zeros(
                    num_envs * future_steps,
                    2,
                    device=current_state_anchor_rot.device,
                    dtype=current_state_anchor_rot.dtype,
                )
            else:
                # Training: drift from current reference frame (state_t - hat_state_t)
                drift_origin = (
                    current_ref_anchor_pos
                    if current_ref_anchor_pos is not None
                    else current_state_anchor_pos
                )
                drift_origin_expanded = (
                    drift_origin.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)
                )

                # XY drift in world frame: agent pos - reference pos
                xy_drift_world = (
                    current_state_anchor_pos.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)[:, :2]
                    - drift_origin_expanded[:, :2]
                )

                # Rotate to heading-aligned frame
                xy_drift_3d = torch.cat(
                    [xy_drift_world, torch.zeros_like(xy_drift_world[:, :1])], dim=-1
                )
                xy_drift_local_3d = rotations.quat_rotate(
                    heading_inv_rot_expanded, xy_drift_3d, w_last
                )
                xy_offset_local = xy_drift_local_3d[:, :2]  # [num_envs*future_steps, 2]
            obs_components.append(xy_offset_local)

        if include_height:
            # Absolute height (not offset from current)
            height = ref_state_anchor_pos[:, 2:3]  # [num_envs*future_steps, 1]
            obs_components.append(height)

    if include_anchor_vel:
        if mimic_ref_anchor_vel is None:
            raise ValueError(
                "mimic_ref_anchor_vel is required when include_anchor_vel=True"
            )
        ref_state_anchor_vel = mimic_ref_anchor_vel.reshape(-1, 3)
        # Transform to local frame (heading-aligned)
        local_anchor_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_vel, w_last
        )
        obs_components.append(local_anchor_vel)

    if include_anchor_ang_vel:
        if mimic_ref_anchor_ang_vel is None:
            raise ValueError(
                "mimic_ref_anchor_ang_vel is required when include_anchor_ang_vel=True"
            )
        ref_state_anchor_ang_vel = mimic_ref_anchor_ang_vel.reshape(-1, 3)
        # Transform to local frame
        local_anchor_ang_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_ang_vel, w_last
        )
        obs_components.append(local_anchor_ang_vel)

    # Concatenate all observations
    obs = torch.cat(obs_components, dim=-1)

    # Reshape to [num_envs, future_steps * features]
    return obs.view(num_envs, -1)



def compute_historical_actions_from_state(
    historical_actions: Tensor,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute flattened historical actions from state history.

    Args:
        historical_actions: Historical actions [envs, history_steps, action_dim].
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.

    Returns:
        Flattened historical actions [envs, history_steps * action_dim].

    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_actions is None:
        raise ValueError(
            "Historical actions tensor is None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )

    num_envs = historical_actions.shape[0]

    if history_steps is None:
        actions = historical_actions
    else:
        actions = select_step_indices(historical_actions, history_steps)

    actual_steps = actions.shape[1]
    action_dim = actions.shape[2]

    return actions.reshape(num_envs, actual_steps * action_dim)



def bm_pd_action(
    action: Tensor,
    pd_action_offset: Tensor,
    action_scale: Tensor,
    stiffness: Tensor,
    damping: Tensor,
) -> Dict[str, Tensor]:
    """BeyondMimic-style action scaling for implicit PD control.

    Scales raw policy output by ``0.25 * effort_limit / stiffness`` per joint,
    so ~O(1) policy outputs map to ~25% of the maximum achievable deflection.
    No tanh or clamp is applied — the policy learns its own output magnitude.

    Args:
        action: Raw action tensor from policy [num_envs, num_actions]
        pd_action_offset: Per-joint default positions [num_actions]
        action_scale: Per-joint scale ``0.25 * effort_limit / stiffness`` [num_actions]
        stiffness: Fixed per-joint stiffness gains [num_actions]
        damping: Fixed per-joint damping gains [num_actions]

    Returns:
        Dict with:
            - processed_action: Joint position targets [num_envs, num_actions]
            - stiffness_targets: Fixed stiffness values [num_envs, num_actions]
            - damping_targets: Fixed damping values [num_envs, num_actions]
    """
    batch_size = action.shape[0]
    processed_action = pd_action_offset + action_scale * action

    # Clone outputs for CUDA graphs compatibility
    return {
        "processed_action": processed_action.clone(),
        "stiffness_targets": stiffness.unsqueeze(0).expand(batch_size, -1).clone(),
        "damping_targets": damping.unsqueeze(0).expand(batch_size, -1).clone(),
    }
