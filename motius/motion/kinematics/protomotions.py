# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Batch-Optimized Forward Kinematics and Related Utilities for MJCF Models.

This module provides functions for performing batched forward kinematics (FK)
calculations on articulated rigid body systems following MuJoCo MJCF XML definition.
It leverages PyTorch for efficient computation on GPUs or CPUs.

**Key Functionalities:**

- Parsing kinematic structure (parent indices, local transforms, joint axes)
  from an MJCF file (`extract_kinematic_info`).
- Converting batched MuJoCo joint position vectors (`qpos`) into intermediate
  root positions and relative joint rotation matrices (`extract_transforms_from_qpos`).
  Handles bodies with multiple hinge degrees of freedom (DOFs), supporting
  both independent hinge angle and coupled exponential map representations for 3-DOF joints.
- Reconstructing `qpos` vectors from root positions and relative joint rotation
  matrices (`extract_qpos_from_transforms`), with options for decomposing
  3-DOF rotations ('euler_xyz', 'exp_map').
- Computing forward kinematics (world positions and rotations) from the
  intermediate transforms (`compute_forward_kinematics_from_transforms`).
- Calculating relative joint rotation matrices from global body rotations
  (`compute_joint_rot_mats_from_global_mats`).
- Estimating Cartesian linear and angular velocities from time series of
  poses using finite differences
  (`compute_cartesian_velocity`, `compute_angular_velocity`,
  `compute_kinematics_velocities`).
- High-level functions combining these steps to perform FK and velocity
  computation directly from `qpos` (`fk_batch_mjcf_with_velocities`) or
  intermediate transforms (`fk_from_transforms_with_velocities`), returning
  results packaged in a `RobotState` object.
- Test functions for verifying FK results against MuJoCo and checking the
  invertibility of the `qpos` <-> transform conversions.

**Assumptions:**

- The root body must have a free joint (either via `<freejoint/>` tag or
  `<joint type="free"/>`).
- Hinge joints must have their axes defined.
- Bodies are expected to have either 1 (hinge) or 3 DOFs associated with them
  for qpos reconstruction (`extract_qpos_from_transforms`).
- The MuJoCo `qpos` layout is assumed: [root_pos(3), root_quat_wxyz(4), joint_angles(...)].
"""
import torch
import numpy as np
from dm_control import mjcf
import logging
import sys
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import re
import mujoco
from motius.motion.representation.protomotions.rotations import quat_from_angle_axis
from motius.motion.representation.protomotions.rotations import quaternion_to_matrix
from motius.motion.representation.protomotions.rotations import matrix_to_quaternion
from motius.motion.representation.protomotions.rotations import quat_conjugate
from motius.motion.representation.protomotions.rotations import quat_mul_norm
from motius.motion.representation.protomotions.rotations import quat_angle_axis
from motius.motion.representation.protomotions.rotations import exp_map_to_quat
from motius.motion.representation.protomotions.rotations import quat_to_exp_map
from motius.motion.representation.protomotions.rotations import angle_from_matrix_axis
from motius.simulators.isaaclab.protomotions.physics.simulator_state import RobotState
from motius.simulators.isaaclab.protomotions.physics.simulator_state import (
    StateConversion,
)


@dataclass
class KinematicInfo:
    """Stores kinematic information extracted from an MJCF model.

    Contains structural data about the robot including body hierarchy, joint limits,
    local transformations, and degrees of freedom information. All fields are populated
    by parsing the MJCF file.
    """

    body_names: List[str]
    dof_names: List[str]
    parent_indices: List[int]
    local_pos: torch.Tensor
    local_rot_ref_mat: torch.Tensor
    hinge_axes_map: Dict[int, torch.Tensor]
    nq: int
    nv: int
    num_bodies: int
    num_dofs: int
    dof_limits_lower: torch.Tensor
    dof_limits_upper: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype = None):
        """Move all tensors to the specified device."""
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self.local_pos = self.local_pos.to(**kwargs)
        self.local_rot_ref_mat = self.local_rot_ref_mat.to(**kwargs)
        self.dof_limits_lower = self.dof_limits_lower.to(**kwargs)
        self.dof_limits_upper = self.dof_limits_upper.to(**kwargs)
        self.hinge_axes_map = {
            k: v.to(**kwargs) for k, v in self.hinge_axes_map.items()
        }
        return self


@dataclass
class ControlInfo:
    """Stores control information extracted from an MJCF model.

    Contains PD control parameters and joint limits for a single degree of freedom.
    All values are extracted from the MJCF file's actuator and joint definitions.
    """

    stiffness: Optional[float] = field(default=None)
    damping: Optional[float] = field(default=None)
    armature: Optional[float] = field(default=None)
    friction: Optional[float] = field(default=None)
    effort_limit: Optional[float] = field(default=None)
    velocity_limit: Optional[float] = field(default=None)


def compute_joint_loss_weights(
    kinematic_info: KinematicInfo, discount: float = 0.9, min_weight: float = 0.01
) -> torch.Tensor:
    """Compute per-joint loss weights based on kinematic chain importance.

    Joints near the root get higher weights because their angular errors propagate
    to all descendant bodies, causing large positional errors at extremities.
    Based on "Total Descendants Length" heuristic from:
    https://theorangeduck.com/page/joint-error-propagation

    Args:
        kinematic_info: KinematicInfo with parent_indices, local_pos, hinge_axes_map
        discount: Decay factor for descendant contributions (0.9 typical).
                  Lower values bias toward uniform weighting.
        min_weight: Minimum weight for leaf joints (end effectors).

    Returns:
        Tensor of shape [num_joints] with normalized weights summing to num_joints.
        Caller should expand based on their representation:
        - For 6D rotation: weights.repeat_interleave(6)
        - For raw DOF: weights.repeat_interleave(dofs_per_joint)

    Example:
        >>> weights = compute_joint_loss_weights(robot_config.kinematic_info)
        >>> weights_6d = weights.repeat_interleave(6)  # For 6D representation
        >>> weighted_loss = ((pred - target) ** 2 * weights_6d).mean()
    """
    parent_indices = kinematic_info.parent_indices
    local_pos = kinematic_info.local_pos
    hinge_axes_map = kinematic_info.hinge_axes_map
    num_bodies = kinematic_info.num_bodies
    bone_lengths = local_pos.norm(dim=-1).cpu()
    body_weights = torch.full((num_bodies,), min_weight)
    for i in range(num_bodies - 1, -1, -1):
        parent_idx = parent_indices[i]
        if parent_idx != -1:
            body_weights[parent_idx] += discount * (bone_lengths[i] + body_weights[i])
    joint_weights = []
    for body_idx in range(num_bodies):
        if body_idx in hinge_axes_map:
            joint_weights.append(body_weights[body_idx].item())
    weights = torch.tensor(joint_weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    return weights


def compute_body_density_weights(
    kinematic_info: KinematicInfo, discount: float = 0.9
) -> torch.Tensor:
    """Compute per-body weights based on kinematic chain density.

    Bodies surrounded by many nearby bodies (in kinematic chain distance) get
    lower weights. This prevents clustered bodies (e.g., finger chains) from
    over-representing their region in mean reward calculations.

    Algorithm:
    1. Compute pairwise chain distances (sum of bone lengths along tree path)
    2. For each body i: density_i = sum_{j != i}(discount^chain_distance_ij)
    3. weight_i = 1 / density_i
    4. Normalize so weights sum to num_bodies

    Args:
        kinematic_info: KinematicInfo with parent_indices and local_pos
        discount: Per-unit-length discount factor (0.9 typical).
                  Lower values make distant bodies contribute less to density.

    Returns:
        Tensor of shape [num_bodies] with normalized weights summing to num_bodies.
    """
    parent_indices = kinematic_info.parent_indices
    local_pos = kinematic_info.local_pos
    num_bodies = kinematic_info.num_bodies
    if num_bodies == 1:
        return torch.ones(1, dtype=torch.float32)
    bone_lengths = local_pos.norm(dim=-1).cpu()
    paths_to_root = []
    for i in range(num_bodies):
        path = []
        current = i
        cumulative_dist = 0.0
        while current != -1:
            path.append((current, cumulative_dist))
            parent = parent_indices[current]
            if parent != -1:
                cumulative_dist += bone_lengths[current].item()
            current = parent
        paths_to_root.append(path)
    ancestor_dists = []
    for path in paths_to_root:
        ancestor_dists.append({body_idx: dist for body_idx, dist in path})
    chain_distances = torch.zeros(num_bodies, num_bodies)
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            i_ancestors = ancestor_dists[i]
            j_ancestors = ancestor_dists[j]
            lca = -1
            lca_dist_i = 0.0
            for ancestor, dist_i in paths_to_root[i]:
                if ancestor in j_ancestors:
                    lca = ancestor
                    lca_dist_i = dist_i
                    break
            if lca != -1:
                lca_dist_j = j_ancestors[lca]
                chain_dist = lca_dist_i + lca_dist_j
            else:
                chain_dist = float("inf")
            chain_distances[i, j] = chain_dist
            chain_distances[j, i] = chain_dist
    discounted = torch.pow(discount, chain_distances)
    discounted.fill_diagonal_(0.0)
    densities = discounted.sum(dim=1)
    weights = 1.0 / densities
    weights = weights / weights.sum() * num_bodies
    return weights


def compute_region_uniform_weights(kinematic_info: KinematicInfo) -> torch.Tensor:
    """Compute per-body weights by discovering regions from kinematic tree.

    Automatically discovers anatomical regions by analyzing the kinematic tree
    structure, without relying on body naming conventions.

    Algorithm:
    1. Build children mapping from parent_indices
    2. Find all leaf bodies (no children) - these are end effectors
    3. For each leaf, trace path back to root - this defines a limb/region
    4. Bodies on paths to multiple leaves form the "core" region
    5. Each region gets equal total weight, shared among its bodies

    For a standard humanoid, this discovers ~5-6 regions:
    - Head (1 leaf)
    - Left hand, Right hand (2 leaves)
    - Left foot, Right foot (2 leaves)
    - Core (pelvis, spine - shared by multiple paths)

    Args:
        kinematic_info: KinematicInfo with parent_indices

    Returns:
        Tensor of shape [num_bodies] with normalized weights summing to num_bodies.
    """
    parent_indices = kinematic_info.parent_indices
    num_bodies = kinematic_info.num_bodies
    children = {i: [] for i in range(num_bodies)}
    for i, parent in enumerate(parent_indices):
        if parent != -1:
            children[parent].append(i)
    leaves = [i for i in range(num_bodies) if len(children[i]) == 0]
    leaf_paths = {}
    for leaf in leaves:
        path = set()
        current = leaf
        while current != -1:
            path.add(current)
            current = parent_indices[current]
        leaf_paths[leaf] = path
    body_to_region = {}
    for i in range(num_bodies):
        belonging_leaves = [leaf for leaf, path in leaf_paths.items() if i in path]
        if len(belonging_leaves) == 1:
            body_to_region[i] = belonging_leaves[0]
        else:
            body_to_region[i] = "core"
    region_counts = {}
    for region in set(body_to_region.values()):
        region_counts[region] = sum((1 for r in body_to_region.values() if r == region))
    num_regions = len(region_counts)
    weight_per_region = num_bodies / num_regions
    weights = torch.zeros(num_bodies)
    for i in range(num_bodies):
        region = body_to_region[i]
        count = region_counts[region]
        if count > 0:
            weights[i] = weight_per_region / count
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_bodies
    return weights


def build_body_ids_tensor(
    all_body_names: List[str], subset_body_names: List[str], device: torch.device
) -> torch.Tensor:
    """
    Build a tensor of body IDs based on the provided body names.

    Args:
        all_body_names (List[str]): List of all body names.
        subset_body_names (List[str]): List of subset body names.
        device (torch.device): Device to store the tensor on.

    Returns:
        torch.Tensor: Tensor containing indices corresponding to the body names.
    """
    if subset_body_names is None:
        return torch.tensor([], dtype=torch.long, device=device)
    body_ids: List[int] = []
    for body_name in subset_body_names:
        body_id = all_body_names.index(body_name)
        assert body_id != -1, f"Body part {body_name} not found in {all_body_names}"
        body_ids.append(body_id)
    body_ids = torch.tensor(body_ids, dtype=torch.long, device=device)
    return body_ids


def extract_kinematic_info(mjcf_path: str) -> KinematicInfo:
    """
    Extracts kinematic information needed for FK from an mjcf XML file.
    Handles multiple hinge joints per body.

    Args:
        mjcf_path (str): Path to the MJCF XML file.

    Returns:
        KinematicInfo: An object containing the kinematic information.
            See KinematicInfo dataclass for more details.

    Raises:
        ValueError: If any body is unnamed or the root body configuration is invalid.
        AssertionError: If internal consistency checks fail (e.g., joint axis missing).
    """
    mjcf_model = mjcf.from_path(mjcf_path)
    global_default_joint = getattr(mjcf_model.default, "joint", None)
    angle_unit = getattr(mjcf_model.compiler, "angle", None)
    angle_to_radians = 1.0 if angle_unit == "radian" else np.pi / 180.0
    bodies = []
    non_root_dof_names = []
    parent_indices_list = []
    local_pos_list = []
    local_quat_list = []
    dof_limits_lower_list = []
    dof_limits_upper_list = []
    hinge_axes_map_dict = {}
    body_name_to_idx = {}
    root_processed = False
    worldbody = mjcf_model.worldbody

    def _resolve_joint_attr(joint, body_default_joint, attr_name):
        value = getattr(joint, attr_name)
        if value is None and joint.dclass is not None:
            value = getattr(joint.dclass.joint, attr_name)
        if value is None and body_default_joint is not None:
            value = getattr(body_default_joint, attr_name)
        if value is None and global_default_joint is not None:
            value = getattr(global_default_joint, attr_name)
        return value

    def _traverse(mjcf_body, parent_idx, inherited_default=None):
        nonlocal root_processed
        body_name = mjcf_body.name
        if not body_name:
            raise ValueError(
                f"All bodies need a name for FK. Found unnamed body parented to {parent_idx}"
            )
        body_idx = len(bodies)
        body_name_to_idx[body_name] = body_idx
        bodies.append(body_name)
        parent_indices_list.append(parent_idx)
        pos_val = np.array(
            mjcf_body.pos if mjcf_body.pos is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        quat_mjcf = np.array(
            mjcf_body.quat if mjcf_body.quat is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        local_pos_list.append(pos_val)
        local_quat_list.append(quat_mjcf)
        body_default = (
            mjcf_body.childclass
            if mjcf_body.childclass is not None
            else inherited_default
        )
        body_default_joint = body_default.joint if body_default is not None else None
        if body_idx == 0:
            assert np.allclose(
                local_quat_list[0], [1.0, 0.0, 0.0, 0.0]
            ), "Root body must have a quat of [1.0, 0.0, 0.0, 0.0]"
            assert not root_processed, "Multiple root bodies"
            has_free_joint_element = any((j.type == "free" for j in mjcf_body.joint))
            has_freejoint_tag = mjcf_body.freejoint is not None
            assert (
                has_free_joint_element or has_freejoint_tag
            ), "Root body must have a free joint defined."
            root_processed = True
        else:
            current_hinge_axes = []
            joints = mjcf_body.joint
            if len(joints) > 0:
                assert (
                    len(joints) == 1 or len(joints) == 3
                ), f"Body {body_name} has {len(joints)} joints, expected 1 or 3"
                for joint in joints:
                    axis = _resolve_joint_attr(joint, body_default_joint, "axis")
                    assert (
                        axis is not None
                    ), f"Hinge joint '{joint.name}' for body '{body_name}' has no axis defined."
                    axis_val = np.array(axis, dtype=np.float32)
                    current_hinge_axes.append(axis_val)
                    non_root_dof_names.append(joint.name)
                    joint_range = _resolve_joint_attr(
                        joint, body_default_joint, "range"
                    )
                    if joint_range is not None:
                        dof_limits_lower_list.append(joint_range[0] * angle_to_radians)
                        dof_limits_upper_list.append(joint_range[1] * angle_to_radians)
                    else:
                        dof_limits_lower_list.append(-10000000000.0)
                        dof_limits_upper_list.append(10000000000.0)
                current_hinge_axes_np = np.array(current_hinge_axes, dtype=np.float32)
                hinge_axes_map_dict[body_idx] = torch.from_numpy(current_hinge_axes_np)
        child_bodies = mjcf_body.body
        for child in child_bodies:
            _traverse(child, body_idx, body_default)

    assert worldbody.body, "MJCF worldbody contains no bodies."
    assert len(worldbody.body) == 1, "Multiple root bodies"
    root_body = worldbody.body[0]
    _traverse(root_body, -1)
    assert root_processed, "Failed to identify/process root body."
    local_quat_numpy = np.array(local_quat_list, dtype=np.float32)
    local_rot_ref_mat_tensor = quaternion_to_matrix(
        torch.from_numpy(local_quat_numpy), w_last=False
    )
    local_pos_numpy = np.array(local_pos_list, dtype=np.float32)
    num_articulated_dofs = sum(
        (len(axes_tensor) for axes_tensor in hinge_axes_map_dict.values())
    )
    dof_limits_lower = torch.tensor(dof_limits_lower_list, dtype=torch.float32)
    dof_limits_upper = torch.tensor(dof_limits_upper_list, dtype=torch.float32)
    return KinematicInfo(
        body_names=bodies,
        dof_names=non_root_dof_names,
        parent_indices=parent_indices_list,
        local_pos=torch.from_numpy(local_pos_numpy),
        local_rot_ref_mat=local_rot_ref_mat_tensor,
        hinge_axes_map=hinge_axes_map_dict,
        nq=num_articulated_dofs + 7,
        nv=num_articulated_dofs + 6,
        num_bodies=len(bodies),
        num_dofs=len(non_root_dof_names),
        dof_limits_lower=dof_limits_lower,
        dof_limits_upper=dof_limits_upper,
    )


def extract_control_info(
    mjcf_path: str, override_control_info: Optional[Dict[str, ControlInfo]] = None
) -> Dict[str, ControlInfo]:
    """
    Extracts control information (stiffness, damping, etc.) from an MJCF XML file.

    This function traverses the MJCF model and extracts control parameters for all DOFs.
    It looks for control parameters in the following order of priority:
    1. Joint-level attributes (stiffness, damping, armature, frictionloss)
    2. Actuator-specific parameters if defined
    3. Default values if parameters are not specified
    4. Override values if specified in the override_control_info dictionary

    Args:
        mjcf_path (str): Path to the MJCF XML file.
        override_control_info (Optional[Dict[str, ControlInfo]]): Override control information for specific joints.
    Returns:
        ControlInfo: An object containing the control information for all DOFs.

    Raises:
        ValueError: If the MJCF file cannot be parsed or if required information is missing.
    """
    try:
        mjcf_model = mjcf.from_path(mjcf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse MJCF file {mjcf_path}: {e}")
    DEFAULT_STIFFNESS = None
    DEFAULT_DAMPING = None
    DEFAULT_ARMATURE = None
    DEFAULT_FRICTION = None
    DEFAULT_EFFORT_LIMIT = None
    control_info = {}

    def _optional_float(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.item()
        return float(value)

    def _extract_joint_control_params(joint_name, joint):
        """Extract control parameters from a joint element."""
        stiffness = getattr(joint, "stiffness", DEFAULT_STIFFNESS)
        damping = getattr(joint, "damping", DEFAULT_DAMPING)
        armature = getattr(joint, "armature", DEFAULT_ARMATURE)
        friction = getattr(joint, "frictionloss", DEFAULT_FRICTION)
        velocity_limit = None
        effort_limit = getattr(joint, "actuatorfrcrange", DEFAULT_EFFORT_LIMIT)
        if isinstance(stiffness, str):
            stiffness = float(stiffness)
        if isinstance(damping, str):
            damping = float(damping)
        if isinstance(armature, str):
            armature = float(armature)
        if isinstance(friction, str):
            friction = float(friction)
        if isinstance(effort_limit, str):
            try:
                _, max_val = map(float, effort_limit.split())
                effort_limit = max_val
            except ValueError:
                effort_limit = DEFAULT_EFFORT_LIMIT
        if isinstance(effort_limit, (list, tuple, np.ndarray)):
            values = np.asarray(effort_limit).reshape(-1)
            if values.size == 0:
                effort_limit = DEFAULT_EFFORT_LIMIT
            elif values.size == 1:
                effort_limit = values[0]
            else:
                effort_limit = values[1]
        if override_control_info is not None:
            for (
                joint_expr,
                override_joint_control_info,
            ) in override_control_info.items():
                if re.fullmatch(joint_expr, joint_name):
                    print(f"Overriding control info for {joint_name} with {joint_expr}")
                    if override_joint_control_info.stiffness is not None:
                        stiffness = override_joint_control_info.stiffness
                    if override_joint_control_info.damping is not None:
                        damping = override_joint_control_info.damping
                    if override_joint_control_info.armature is not None:
                        armature = override_joint_control_info.armature
                    if override_joint_control_info.friction is not None:
                        friction = override_joint_control_info.friction
                    if override_joint_control_info.effort_limit is not None:
                        effort_limit = override_joint_control_info.effort_limit
                    if override_joint_control_info.velocity_limit is not None:
                        velocity_limit = override_joint_control_info.velocity_limit
        stiffness = _optional_float(stiffness)
        damping = _optional_float(damping)
        armature = _optional_float(armature)
        friction = _optional_float(friction)
        effort_limit = _optional_float(effort_limit)
        dof_control_info = ControlInfo(
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            friction=friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        )
        return dof_control_info

    def _traverse_bodies_for_control(mjcf_body):
        """Traverse bodies to find joints and extract control parameters."""
        for joint in mjcf_body.joint:
            if joint.type != "free":
                joint_name = joint.name
                if joint_name:
                    dof_control_info = _extract_joint_control_params(joint_name, joint)
                    control_info[joint_name] = dof_control_info
        for child_body in mjcf_body.body:
            _traverse_bodies_for_control(child_body)

    if hasattr(mjcf_model, "worldbody") and mjcf_model.worldbody:
        for body in mjcf_model.worldbody.body:
            _traverse_bodies_for_control(body)
    if not control_info:
        raise ValueError(f"No DOFs found in MJCF file {mjcf_path}")
    return control_info


def extract_transforms_from_qpos_non_root_ignore_fixed_helper(
    hinge_axes_map: Dict[int, torch.Tensor],
    qpos_non_root: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> torch.Tensor:
    """
    Helper function to extract transforms from qpos, ignoring bodies with no DOFs.
    """
    device = qpos_non_root.device
    dtype = qpos_non_root.dtype
    B = qpos_non_root.shape[0]
    hinge_axes_map = {k: v.to(device).to(dtype) for k, v in hinge_axes_map.items()}
    joint_start = 0
    num_movable_bodies = len(hinge_axes_map.keys())
    assert num_movable_bodies > 0, "No movable bodies found"
    total_expected_dofs = sum((len(v) for v in hinge_axes_map.values()))
    actual_dofs = qpos_non_root.shape[1]
    if actual_dofs != total_expected_dofs:
        raise ValueError(
            f"DOF count mismatch: input has {actual_dofs} DOFs but robot kinematic model expects {total_expected_dofs} DOFs ({num_movable_bodies} movable bodies). This usually means the motion file was generated for a different robot than the one being used. Check that --motion-file matches the robot in your checkpoint/config."
        )
    joint_rot_mats = (
        torch.eye(3, device=device, dtype=dtype)
        .view(1, 1, 3, 3)
        .expand(B, num_movable_bodies, 3, 3)
        .clone()
    )
    for i, (body_idx, axes) in enumerate(hinge_axes_map.items()):
        assert body_idx != 0, "Hinge DOF mapped to root body index 0"
        num_body_dofs = len(axes)
        qpos_indices = torch.arange(
            joint_start, joint_start + num_body_dofs, device=device, dtype=torch.long
        )
        joint_start += num_body_dofs
        if num_body_dofs == 3 and qpos_is_exp_map_on_3dof_joints:
            exp_map_vec = qpos_non_root[:, qpos_indices]
            quat_k = exp_map_to_quat(exp_map_vec, w_last=False)
            joint_rot_mats[:, i, :, :] = quaternion_to_matrix(quat_k, w_last=False)
        else:
            all_hinge_angles = qpos_non_root[:, qpos_indices]
            all_hinge_quats_wxyz = quat_from_angle_axis(
                all_hinge_angles, axes.unsqueeze(0).expand(B, -1, -1), w_last=False
            )
            all_hinge_rot_mats = quaternion_to_matrix(
                all_hinge_quats_wxyz, w_last=False
            )
            for k in range(num_body_dofs):
                single_dof_rot_mat = all_hinge_rot_mats[:, k, :, :]
                joint_rot_mats[:, i, :, :] = torch.matmul(
                    joint_rot_mats[:, i, :, :].clone(), single_dof_rot_mat
                )
    return joint_rot_mats


def extract_transforms_from_qpos_non_root(
    kinematic_info: KinematicInfo,
    qpos_non_root: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> torch.Tensor:
    """
    Extracts COMPOUNDED relative joint rotation matrices from qpos.
    Handles multiple hinge DOFs per body, including 3-DOF joints represented
    either as independent hinges or coupled exponential maps.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure
                        information from `extract_kinematic_info`.
        qpos: Batch of joint positions (B, Nq-7). The layout must match the
              order defined by MuJoCo (hinge angles...).
        qpos_is_exp_map_on_3dof_joints: If True, assumes qpos values for 3-DOF joints
                                      represent a single 3D exponential map vector.
                                      If False, assumes they are three independent
                                      hinge angles around the body's joint axes.

    Returns:
        torch.Tensor:
            - joint_rot_mats (B, Nb, 3, 3): Relative rotation matrix for each body.
                Index 0 contains the root's world rotation matrix, which is identity in this case.
    """
    device = qpos_non_root.device
    dtype = qpos_non_root.dtype
    B = qpos_non_root.shape[0]
    Nb = kinematic_info.num_bodies
    hinge_axes_map = kinematic_info.hinge_axes_map
    assert (
        qpos_non_root.shape[1] == kinematic_info.nq - 7
    ), f"qpos_non_root has {qpos_non_root.shape[1]} DOFs, expected {kinematic_info.nq - 7}"
    joint_rot_mats = (
        torch.eye(3, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, Nb, 3, 3)
        .clone()
    )
    if not hinge_axes_map:
        return joint_rot_mats
    joint_rot_mats[:, list(hinge_axes_map.keys()), :, :] = (
        extract_transforms_from_qpos_non_root_ignore_fixed_helper(
            hinge_axes_map, qpos_non_root, qpos_is_exp_map_on_3dof_joints
        )
    )
    return joint_rot_mats


def extract_transforms_from_qpos(
    kinematic_info: KinematicInfo,
    qpos: torch.Tensor,
    qpos_is_exp_map_on_3dof_joints: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts root position and COMPOUNDED relative joint rotation matrices from qpos.
    Stores root rotation in the first index of joint_rot_mats.
    Handles multiple hinge DOFs per body, including 3-DOF joints represented
    either as independent hinges or coupled exponential maps.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure
                        information from `extract_kinematic_info`.
        qpos: Batch of joint positions (B, Nq). The layout must match the
              order defined by MuJoCo (root pos, root quat WXYZ, hinge angles...).
        qpos_is_exp_map_on_3dof_joints: If True, assumes qpos values for 3-DOF joints
                                      represent a single 3D exponential map vector.
                                      If False, assumes they are three independent
                                      hinge angles around the body's joint axes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - root_pos (B, 3): Root position extracted from qpos.
            - joint_rot_mats (B, Nb, 3, 3): Relative rotation matrix for each body.
                                          Index 0 contains the root's world rotation matrix.
                                          Other indices contain the combined hinge rotations
                                          relative to the parent frame (identity if no hinge).
    """
    device = qpos.device
    dtype = qpos.dtype
    root_pos = qpos[:, 0:3].to(device).to(dtype)
    root_quat_wxyz = qpos[:, 3:7].to(device).to(dtype)
    root_quat_wxyz_norm = root_quat_wxyz / torch.linalg.norm(
        root_quat_wxyz, dim=-1, keepdim=True
    )
    root_rot_mat = quaternion_to_matrix(root_quat_wxyz_norm, w_last=False)
    joint_rot_mats = extract_transforms_from_qpos_non_root(
        kinematic_info, qpos[:, 7:], qpos_is_exp_map_on_3dof_joints
    )
    joint_rot_mats[:, 0, :, :] = root_rot_mat
    return (root_pos, joint_rot_mats)


def extract_qpos_from_transforms(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    multi_dof_decomposition_method: Optional[str] = None,
) -> torch.Tensor:
    """
    Reconstructs the qpos tensor from root position and relative joint rotations.

    This is the inverse operation of `extract_transforms_from_qpos`.
    It assumes the MuJoCo qpos layout (root pos, root quat WXYZ, hinge angles...).
    Handles bodies with 1 or 3 hinge DOFs. For 3 DOFs, requires specifying
    `multi_dof_decomposition_method`.

    Args:
        kinematic_info: Dictionary containing pre-parsed kinematic structure.
        root_pos (B, 3): Root position.
        joint_rot_mats (B, Nb, 3, 3): Relative rotation matrices for each body.
                                      Index 0 contains the root's world rotation matrix.
                                      Other indices contain the compounded hinge rotations
                                      relative to the parent frame (or identity if no hinges).
        multi_dof_decomposition_method (str, optional): Method to decompose rotation
                                      for bodies with 3 DOFs. Options: 'euler_xyz', 'exp_map'.
                                      Required if any body has 3 DOFs. Defaults to None.

    Returns:
        torch.Tensor (B, Nq): Reconstructed qpos tensor.

    Raises:
        ValueError: If a non-root body has a number of hinge DOFs other than 1 or 3.
        ValueError: If a body has 3 DOFs and `multi_dof_decomposition_method` is
                    not specified or is invalid.
        AssertionError: If internal consistency checks fail (e.g., no hinge DOFs found).
    """
    device = root_pos.device
    dtype = root_pos.dtype
    B = root_pos.shape[0]
    nq = kinematic_info.nq
    qpos = torch.zeros(B, nq, device=device, dtype=dtype)
    qpos[:, 0:3] = root_pos
    root_rot_mat = joint_rot_mats[:, 0, :, :]
    root_quat_wxyz = matrix_to_quaternion(root_rot_mat, w_last=False)
    root_quat_wxyz_norm = root_quat_wxyz / torch.linalg.norm(
        root_quat_wxyz, dim=-1, keepdim=True
    )
    qpos[:, 3:7] = root_quat_wxyz_norm
    hinge_axes_map = kinematic_info.hinge_axes_map
    hinge_axes_map = {k: v.to(device).to(dtype) for k, v in hinge_axes_map.items()}
    num_hinge_dofs = nq - 7
    if num_hinge_dofs == 0:
        return qpos
    assert num_hinge_dofs > 0, "No hinge DOFs found"
    joint_start = 7
    for body_idx, axes in hinge_axes_map.items():
        num_body_dofs = len(axes)
        qpos_indices = torch.arange(
            joint_start, joint_start + num_body_dofs, device=device, dtype=torch.long
        )
        joint_start += num_body_dofs
        rot_mat_k = joint_rot_mats[:, body_idx, :, :]
        if num_body_dofs == 1:
            axis_k = axes[0]
            angle_k = angle_from_matrix_axis(rot_mat_k, axis_k)
            qpos[:, qpos_indices[0]] = angle_k
        elif num_body_dofs == 3:
            quat_k = matrix_to_quaternion(rot_mat_k, w_last=False)
            if multi_dof_decomposition_method == "euler_xyz":
                pitch = torch.asin(torch.clamp(rot_mat_k[:, 0, 2], -1.0, 1.0))
                roll = torch.atan2(-rot_mat_k[:, 1, 2], rot_mat_k[:, 2, 2])
                yaw = torch.atan2(-rot_mat_k[:, 0, 1], rot_mat_k[:, 0, 0])
                qpos[:, qpos_indices[0]] = roll
                qpos[:, qpos_indices[1]] = pitch
                qpos[:, qpos_indices[2]] = yaw
            elif multi_dof_decomposition_method == "exp_map":
                exp_map_vec = quat_to_exp_map(quat_k, w_last=False)
                qpos[:, qpos_indices[0]] = exp_map_vec[:, 0]
                qpos[:, qpos_indices[1]] = exp_map_vec[:, 1]
                qpos[:, qpos_indices[2]] = exp_map_vec[:, 2]
            else:
                body_name = kinematic_info.body_names[body_idx]
                raise ValueError(
                    f"Invalid 'multi_dof_decomposition_method' ('{multi_dof_decomposition_method}') provided for body '{body_name}' (index {body_idx}). Use 'euler_xyz' or 'exp_map'."
                )
        else:
            body_name = kinematic_info.body_names[body_idx]
            raise ValueError(
                f"Body '{body_name}' (index {body_idx}) has {num_body_dofs} hinge DOFs. Expected 1 or 3 DOFs for decomposition."
            )
    return qpos


def compute_forward_kinematics_from_transforms(
    kinematic_info: KinematicInfo, root_pos: torch.Tensor, joint_rot_mats: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes world poses given kinematic structure and transforms.
    Assumes joint_rot_mats contains root rotation at index 0.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        root_pos (B, 3): Root position.
        joint_rot_mats (B, Nb, 3, 3): Rotation matrices for each body.
                                      Index 0 is root's world rotation.
                                      Other indices are compounded relative joint rotations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - world_pos (B, Nb, 3): World positions.
            - world_rot_mat (B, Nb, 3, 3): World rotation matrices.
    """
    device = root_pos.device
    dtype = root_pos.dtype
    B = root_pos.shape[0]
    Nb = kinematic_info.num_bodies
    parent_indices = torch.tensor(
        kinematic_info.parent_indices, dtype=torch.long, device=device
    )
    local_pos = (
        kinematic_info.local_pos[None, ...].expand(B, -1, -1).to(device).to(dtype)
    )
    local_rot_ref_mat = (
        kinematic_info.local_rot_ref_mat[None, ...]
        .expand(B, -1, -1, -1)
        .to(device)
        .to(dtype)
    )
    world_pos = torch.zeros(B, Nb, 3, device=device, dtype=dtype)
    world_rot_mat = torch.zeros(B, Nb, 3, 3, device=device, dtype=dtype)
    for i in range(Nb):
        if parent_indices[i] == -1:
            world_pos[:, i, :] = root_pos
            world_rot_mat[:, i, :, :] = joint_rot_mats[:, 0, :, :]
        else:
            parent_idx = parent_indices[i]
            parent_pos_world = world_pos[:, parent_idx, :]
            parent_rot_mat_world = world_rot_mat[:, parent_idx, :, :]
            ref_rot_mat = local_rot_ref_mat[:, i, :, :]
            joint_rot_mat = joint_rot_mats[:, i, :, :]
            effective_local_rot = torch.matmul(ref_rot_mat, joint_rot_mat)
            world_rot_mat[:, i, :, :] = torch.matmul(
                parent_rot_mat_world, effective_local_rot
            )
            offset_in_world = torch.matmul(
                parent_rot_mat_world, local_pos[:, i, :, None]
            ).squeeze(-1)
            world_pos[:, i, :] = parent_pos_world + offset_in_world
    return (world_pos, world_rot_mat)


def compute_joint_rot_mats_from_global_mats(
    kinematic_info: KinematicInfo, global_rot_mats: torch.Tensor
) -> torch.Tensor:
    """
    Computes joint rotation matrices relative to the body's reference frame
    from global rotations.

    Calculates J_i = R_ref_i^(-1) * P_p(i)^(-1) * G_i
    where J_i is the joint rotation of body i relative to its reference frame,
    R_ref_i is the reference rotation of body i in the parent frame,
    P_p(i) is the global rotation of its parent p(i), and G_i is the global
    rotation of body i.

    The reference rotation R_ref_i represents the orientation offset specified
    in the MJCF for the body relative to its parent's frame *before* any joint
    rotation is applied. The computed `joint_rot_mats` represents the rotation
    *due to the joint(s)* connecting body `i` to its parent, expressed relative
    to the body's reference frame.

    For the root body (index 0), its reference rotation R_ref_0 is assumed to be
    identity, and its parent is the world frame (also identity rotation).
    Therefore, the "joint" rotation J_0 is simply its global rotation G_0.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        global_rot_mats (torch.Tensor): Global rotation matrices for each body
                                        (B, Nb, 3, 3).

    Returns:
        torch.Tensor (B, Nb, 3, 3): Joint rotation matrices for each body
                                    relative to its reference orientation frame.
                                    Index 0 contains the global rotation of the root.
                                    Other indices contain rotations purely due to joints.
    """
    device = global_rot_mats.device
    dtype = global_rot_mats.dtype
    B, Nb, _, _ = global_rot_mats.shape
    parent_indices = torch.tensor(
        kinematic_info.parent_indices, dtype=torch.long, device=device
    )
    local_rot_ref_mat = (
        kinematic_info.local_rot_ref_mat[None, ...]
        .expand(B, -1, -1, -1)
        .to(device)
        .to(dtype)
    )
    local_rot_ref_mat_inv = local_rot_ref_mat.transpose(-1, -2)
    joint_rot_mats = torch.zeros(B, Nb, 3, 3, device=device, dtype=dtype)
    for i in range(Nb):
        ref_rot_inv = local_rot_ref_mat_inv[:, i, :, :]
        child_global_rot = global_rot_mats[:, i, :, :]
        if parent_indices[i] == -1:
            assert torch.allclose(
                ref_rot_inv, torch.eye(3, device=device, dtype=dtype)
            ), "Root body must have a ref_rot_inv of identity"
            joint_rot_mats[:, i, :, :] = child_global_rot
        else:
            parent_idx = parent_indices[i]
            parent_global_rot = global_rot_mats[:, parent_idx, :, :]
            parent_global_rot_inv = parent_global_rot.transpose(-1, -2)
            effective_local_rot = torch.matmul(parent_global_rot_inv, child_global_rot)
            joint_rot = torch.matmul(ref_rot_inv, effective_local_rot)
            joint_rot_mats[:, i, :, :] = joint_rot
    return joint_rot_mats


def compute_cartesian_velocity(
    batched_robot_pos: torch.Tensor, fps: int, velocity_max_horizon: int = 1
) -> torch.Tensor:
    """
    Computes Cartesian velocity from position data over time.

    When velocity_max_horizon=1, uses simple forward difference (original behavior).
    When velocity_max_horizon>1, uses multi-horizon minimum to filter noise: computes
    velocity over horizons 1 to velocity_max_horizon and selects the one with minimum
    magnitude for each frame/body. This filters out spurious high velocities
    caused by noise/errors while preserving genuine fast motion.

    The intuition: noise spikes don't persist across multiple time horizons,
    but genuine motion does. A 2-3cm mocap error at 30fps creates ~1m/s velocity
    over 1 frame, but only ~0.3m/s over 3 frames.

    Args:
        batched_robot_pos (T, Nb, 3): Robot positions over time.
        fps (int): Frames per second.
        velocity_max_horizon (int): Maximum number of frames to look ahead for
                          numerical velocity computation (default: 1).
                          Use 3 for noise filtering, 1 for original behavior.

    Returns:
        torch.Tensor (T, Nb, 3): Cartesian velocities.
    """
    T = batched_robot_pos.shape[0]
    if T < 2:
        return torch.zeros_like(batched_robot_pos)
    velocities = []
    for horizon in range(1, velocity_max_horizon + 1):
        dt = horizon / fps
        vel = torch.zeros_like(batched_robot_pos)
        if T > horizon:
            vel[:-horizon] = (
                batched_robot_pos[horizon:] - batched_robot_pos[:-horizon]
            ) / dt
            vel[-horizon:] = vel[-horizon - 1].unsqueeze(0).expand(horizon, -1, -1)
        else:
            vel[:-1] = (batched_robot_pos[1:] - batched_robot_pos[:-1]) * fps
            vel[-1] = vel[-2]
        velocities.append(vel)
    if velocity_max_horizon == 1:
        return velocities[0]
    velocities_stacked = torch.stack(velocities, dim=0)
    magnitudes = torch.norm(velocities_stacked, dim=-1)
    min_indices = magnitudes.argmin(dim=0)
    min_indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, 3)
    velocities_for_gather = velocities_stacked.permute(1, 2, 3, 0)
    result = torch.gather(
        velocities_for_gather, dim=-1, index=min_indices_expanded.unsqueeze(-1)
    )
    result = result.squeeze(-1)
    return result


def compute_angular_velocity(
    batched_robot_rot_mats: torch.Tensor, fps: int, velocity_max_horizon: int = 1
) -> torch.Tensor:
    """
    Computes angular velocity from rotation matrices over time.

    When velocity_max_horizon=1, uses simple quaternion differentiation (original behavior).
    When velocity_max_horizon>1, uses multi-horizon minimum to filter noise: computes
    angular velocity over horizons 1 to velocity_max_horizon and selects the one with
    minimum magnitude for each frame/body.

    Args:
        batched_robot_rot_mats (T, Nb, 3, 3): Rotation matrices over time.
        fps (int): Frames per second.
        velocity_max_horizon (int): Maximum number of frames to look ahead for
                          numerical velocity computation (default: 1).
                          Use 3 for noise filtering, 1 for original behavior.

    Returns:
        torch.Tensor (T, Nb, 3): Angular velocities (axis * angle / dt).
    """
    T = batched_robot_rot_mats.shape[0]
    if T < 2:
        vel_shape = batched_robot_rot_mats.shape[:-2] + (3,)
        return torch.zeros(
            vel_shape,
            device=batched_robot_rot_mats.device,
            dtype=batched_robot_rot_mats.dtype,
        )
    device = batched_robot_rot_mats.device
    dtype = batched_robot_rot_mats.dtype
    batched_robot_quats = matrix_to_quaternion(batched_robot_rot_mats, w_last=True)
    angular_velocities = []
    for horizon in range(1, velocity_max_horizon + 1):
        dt = horizon / fps
        ang_vel = torch.zeros(
            batched_robot_rot_mats.shape[:-2] + (3,), device=device, dtype=dtype
        )
        if T > horizon:
            quat_t = batched_robot_quats[:-horizon]
            quat_t_plus_h = batched_robot_quats[horizon:]
            tail_len = horizon
        else:
            quat_t = batched_robot_quats[:-1]
            quat_t_plus_h = batched_robot_quats[1:]
            dt = 1 / fps
            tail_len = 1
        quat_t_inv = quat_conjugate(quat_t, w_last=True)
        diff_quat = quat_mul_norm(quat_t_plus_h, quat_t_inv, w_last=True)
        diff_angle, diff_axis = quat_angle_axis(diff_quat, w_last=True)
        ang_vel_valid = diff_axis * diff_angle.unsqueeze(-1) / dt
        ang_vel[:-tail_len] = ang_vel_valid
        ang_vel[-tail_len:] = ang_vel_valid[-1:].expand(tail_len, -1, -1)
        angular_velocities.append(ang_vel)
    if velocity_max_horizon == 1:
        return angular_velocities[0]
    angular_velocities_stacked = torch.stack(angular_velocities, dim=0)
    magnitudes = torch.norm(angular_velocities_stacked, dim=-1)
    min_indices = magnitudes.argmin(dim=0)
    min_indices_expanded = min_indices.unsqueeze(-1).expand(-1, -1, 3)
    ang_vel_for_gather = angular_velocities_stacked.permute(1, 2, 3, 0)
    result = torch.gather(
        ang_vel_for_gather, dim=-1, index=min_indices_expanded.unsqueeze(-1)
    )
    result = result.squeeze(-1)
    return result


def compute_kinematics_velocities(
    batched_robot_pos: torch.Tensor,
    batched_robot_rot_mats: torch.Tensor,
    fps: int,
    velocity_max_horizon: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute linear and angular velocities from poses over time.

    Args:
        batched_robot_pos (T, Nb, 3): Robot positions over time.
        batched_robot_rot_mats (T, Nb, 3, 3): Rotation matrices over time.
        fps (int): Frames per second.
        velocity_max_horizon (int): Maximum horizon for numerical velocity
                          computation (default: 3). Use 1 for simple finite
                          difference, 3 for noise filtering.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Linear and angular velocities.
    """
    lin_vel = compute_cartesian_velocity(batched_robot_pos, fps, velocity_max_horizon)
    ang_vel = compute_angular_velocity(
        batched_robot_rot_mats, fps, velocity_max_horizon
    )
    return (lin_vel, ang_vel)


def fk_batch_mjcf_with_velocities(
    kinematic_info: KinematicInfo,
    qpos: torch.Tensor,
    fps: Optional[int] = None,
    compute_velocities: bool = True,
    velocity_max_horizon: int = 3,
) -> RobotState:
    """
    Performs batched forward kinematics with velocities using PyTorch.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        qpos: Batch of joint positions (B, Nq), expects MuJoCo order.
              (root pos, root quat WXYZ, hinge angles...)
        fps: Frames per second.
        compute_velocities: Whether to compute velocities.
        velocity_max_horizon: Maximum horizon for numerical velocity computation
                    (default: 3). Use 1 for simple finite difference, 3 for
                    noise filtering.

    Returns:

        RobotState:
            rigid_body_pos (B, Nb, 3): World positions of each body.
            rigid_body_rot (B, Nb, 4): World quaternions of each body.
            fps (int): Frames per second.
            rigid_body_vel (B, Nb, 3): World velocities of each body.
            rigid_body_ang_vel (B, Nb, 3): World angular velocities of each body.

    """
    root_pos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)
    return fk_from_transforms_with_velocities(
        kinematic_info,
        root_pos,
        joint_rot_mats,
        fps,
        compute_velocities,
        velocity_max_horizon,
    )


def fk_from_transforms_with_velocities(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    fps: Optional[int] = None,
    compute_velocities: bool = True,
    velocity_max_horizon: int = 3,
) -> RobotState:
    """
    Performs forward kinematics with velocities from root position and joint rotations.

    Args:
        kinematic_info: Dictionary from extract_kinematic_info.
        root_pos (B, 3): Root positions.
        joint_rot_mats (B, Nb, 3, 3): Rotation matrices for each body.
        fps: Frames per second.
        compute_velocities: Whether to compute velocities.
        velocity_max_horizon: Maximum horizon for numerical velocity computation
                    (default: 3). Use 1 for simple finite difference, 3 for
                    noise filtering.

    Returns:

        RobotState:
            rigid_body_pos (B, Nb, 3): World positions of each body.
            rigid_body_rot (B, Nb, 4): World quaternions of each body.
            fps (int): Frames per second.
            rigid_body_vel (B, Nb, 3): World velocities of each body.
            rigid_body_ang_vel (B, Nb, 3): World angular velocities of each body.

    """
    world_pos, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, joint_rot_mats
    )
    world_quat = matrix_to_quaternion(world_rot_mat, w_last=True)
    result = RobotState(
        rigid_body_pos=world_pos,
        rigid_body_rot=world_quat,
        state_conversion=StateConversion.COMMON,
    )
    result.fps = fps
    if compute_velocities and root_pos.shape[0] > 1:
        assert fps is not None, "fps is required when compute_velocities is True"
        lin_vel, ang_vel = compute_kinematics_velocities(
            world_pos, world_rot_mat, fps, velocity_max_horizon
        )
        result.rigid_body_vel = lin_vel
        result.rigid_body_ang_vel = ang_vel
    return result
