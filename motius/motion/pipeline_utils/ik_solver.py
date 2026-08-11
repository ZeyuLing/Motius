"""IK Solver for SMPL-22 skeleton: root, analytic 2-bone, 1-bone, and gradient IK.

Solves inverse kinematics to satisfy world-space position constraints on any
joint of the SMPL-22 skeleton. Uses the most efficient strategy for each joint:

- **Root (Pelvis)**: Direct translation adjustment.
- **2-bone IK**: Analytic solution for end-effectors (ankles, feet, wrists).
- **1-bone IK**: Swing rotation of parent joint (hips, spine1, knees).
- **Gradient IK**: Optimization-based for complex chains (spine, head, collars, etc).

All solvers operate on local rotation matrices and are differentiable where needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS
from motius.motion.pipeline_utils.differentiable_fk import (
    differentiable_fk,
    rot6d_to_rotmat_row_major,
    rotmat_to_rot6d_row_major,
)

NUM_JOINTS = 22

# --- Joint Classification for IK Strategy ---

# 2-bone IK chains: (base, mid, end)
TWO_BONE_CHAINS: Dict[int, Tuple[int, int, int]] = {
    7:  (1, 4, 7),    # L_Ankle: L_Hip -> L_Knee -> L_Ankle
    8:  (2, 5, 8),    # R_Ankle: R_Hip -> R_Knee -> R_Ankle
    10: (4, 7, 10),   # L_Foot: L_Knee -> L_Ankle -> L_Foot
    11: (5, 8, 11),   # R_Foot: R_Knee -> R_Ankle -> R_Foot
    20: (16, 18, 20), # L_Wrist: L_Shoulder -> L_Elbow -> L_Wrist
    21: (17, 19, 21), # R_Wrist: R_Shoulder -> R_Elbow -> R_Wrist
}

# 1-bone IK targets: target -> parent whose rotation is adjusted
ONE_BONE_TARGETS: Dict[int, int] = {
    1: 0,   # L_Hip -> adjust Pelvis
    2: 0,   # R_Hip -> adjust Pelvis
    3: 0,   # Spine1 -> adjust Pelvis
    4: 1,   # L_Knee -> adjust L_Hip
    5: 2,   # R_Knee -> adjust R_Hip
}

# Gradient IK: target -> list of ancestor joints to optimize
GRADIENT_IK_ANCESTORS: Dict[int, List[int]] = {
    6:  [0, 3],          # Spine2 -> Pelvis, Spine1
    9:  [3, 6],          # Spine3 -> Spine1, Spine2
    12: [6, 9],          # Neck -> Spine2, Spine3
    15: [9, 12],         # Head -> Spine3, Neck
    13: [6, 9],          # L_Collar -> Spine2, Spine3
    14: [6, 9],          # R_Collar -> Spine2, Spine3
    16: [9, 13],         # L_Shoulder -> Spine3, L_Collar
    17: [9, 14],         # R_Shoulder -> Spine3, R_Collar
    18: [13, 16],        # L_Elbow -> L_Collar, L_Shoulder
    19: [14, 17],        # R_Elbow -> R_Collar, R_Shoulder
}


def get_ik_strategy(joint_idx: int) -> str:
    """Determine which IK strategy to use for a given joint.

    Returns:
        One of 'root', 'two_bone', 'one_bone', 'gradient'.
    """
    if joint_idx == 0:
        return 'root'
    elif joint_idx in TWO_BONE_CHAINS:
        return 'two_bone'
    elif joint_idx in ONE_BONE_TARGETS:
        return 'one_bone'
    else:
        return 'gradient'


# ========================================================================
# Root IK
# ========================================================================

def solve_root_ik(
    translation: Tensor,
    bone_offsets: Tensor,
    target_pos: Tensor,
) -> Tensor:
    """Adjust translation so the root joint (Pelvis) reaches target position.

    Args:
        translation: Current root translation, shape ``(3,)``.
        bone_offsets: ``(22, 3)`` bone offsets.
        target_pos: Target world position for Pelvis, shape ``(3,)``.

    Returns:
        new_translation: Adjusted translation, shape ``(3,)``.
    """
    # Root world position = translation + bone_offsets[0]
    # To reach target_pos: new_translation = target_pos - bone_offsets[0]
    return target_pos - bone_offsets[0]


# ========================================================================
# 2-bone Analytic IK
# ========================================================================

def _rotation_between_vectors(v_from: Tensor, v_to: Tensor) -> Tensor:
    """Compute rotation matrix that rotates v_from to align with v_to.

    Args:
        v_from: Source direction, shape ``(3,)``.
        v_to: Target direction, shape ``(3,)``.

    Returns:
        Rotation matrix, shape ``(3, 3)``.
    """
    v_from = F.normalize(v_from, dim=-1)
    v_to = F.normalize(v_to, dim=-1)

    cross = torch.cross(v_from, v_to, dim=-1)
    dot = (v_from * v_to).sum(-1)
    sin_angle = cross.norm()
    cos_angle = dot

    if sin_angle < 1e-7:
        if cos_angle > 0:
            return torch.eye(3, device=v_from.device, dtype=v_from.dtype)
        else:
            # 180 degree rotation: find orthogonal axis
            ortho = torch.zeros(3, device=v_from.device, dtype=v_from.dtype)
            min_idx = v_from.abs().argmin()
            ortho[min_idx] = 1.0
            axis = F.normalize(torch.cross(v_from, ortho, dim=-1), dim=-1)
            # Rodrigues: R = -I + 2 * axis * axis^T
            return -torch.eye(3, device=v_from.device, dtype=v_from.dtype) + 2 * axis.unsqueeze(-1) @ axis.unsqueeze(0)

    axis = F.normalize(cross, dim=-1)
    # Rodrigues' rotation formula
    K = torch.zeros(3, 3, device=v_from.device, dtype=v_from.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    R = torch.eye(3, device=v_from.device, dtype=v_from.dtype) + sin_angle * K + (1 - cos_angle) * (K @ K)
    return R


def solve_two_bone_ik(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
    target_joint: int,
    target_pos: Tensor,
    num_iters: int = 20,
) -> Tensor:
    """Solve 2-bone IK using iterative CCD (Cyclic Coordinate Descent).

    Iteratively adjusts base and mid joint rotations to bring the end-effector
    to the target position.

    Args:
        local_rotmat: Local rotation matrices, shape ``(22, 3, 3)``.
        translation: Root translation, shape ``(3,)``.
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        target_joint: Target joint index (must be in TWO_BONE_CHAINS).
        target_pos: Target world position, shape ``(3,)``.
        num_iters: Number of CCD iterations.

    Returns:
        new_local_rotmat: Modified local rotations, shape ``(22, 3, 3)``.
    """
    base, mid, end = TWO_BONE_CHAINS[target_joint]
    device = local_rotmat.device
    dtype = local_rotmat.dtype

    new_local_rotmat = local_rotmat.clone()

    # Clamp target to reachable range from base
    world_pos_init, _ = differentiable_fk(new_local_rotmat, translation, bone_offsets)
    base_pos = world_pos_init[base]
    len_upper = bone_offsets[mid].norm()
    len_lower = bone_offsets[end].norm()
    max_reach = len_upper + len_lower
    to_target = target_pos - base_pos
    dist = to_target.norm()
    if dist > max_reach * 0.999:
        # Clamp to max reach
        target_pos = base_pos + F.normalize(to_target, dim=-1) * max_reach * 0.99

    # CCD: iterate between joints in the chain
    # Order: mid first (closer to end-effector), then base
    chain_joints = [mid, base]

    for iteration in range(num_iters):
        world_pos, world_rot = differentiable_fk(new_local_rotmat, translation, bone_offsets)

        current_error = (world_pos[end] - target_pos).norm().item()
        if current_error < 1e-5:
            break

        for joint in chain_joints:
            # Fresh FK for each joint adjustment
            world_pos, world_rot = differentiable_fk(new_local_rotmat, translation, bone_offsets)

            joint_pos = world_pos[joint]
            end_pos = world_pos[end]
            joint_world_rot = world_rot[joint]

            # Direction from this joint to current end-effector
            current_dir = F.normalize(end_pos - joint_pos, dim=-1)
            # Direction from this joint to target
            desired_dir = F.normalize(target_pos - joint_pos, dim=-1)

            # Rotation correction in world space
            R_corr = _rotation_between_vectors(current_dir, desired_dir)

            # Apply correction: new_world_rot = R_corr @ old_world_rot
            new_world_rot = R_corr @ joint_world_rot

            # Convert back to local
            parent = SMPL22_PARENTS[joint]
            if parent < 0:
                parent_world_rot = torch.eye(3, device=device, dtype=dtype)
            else:
                parent_world_rot = world_rot[parent]

            new_local_rotmat[joint] = parent_world_rot.T @ new_world_rot

    return new_local_rotmat


# ========================================================================
# 1-bone IK
# ========================================================================

def solve_one_bone_ik(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
    target_joint: int,
    target_pos: Tensor,
) -> Tensor:
    """Solve 1-bone IK by rotating the parent joint.

    Uses minimum swing rotation to align the child bone direction with the
    target direction, preserving existing twist component.

    Args:
        local_rotmat: Local rotation matrices, shape ``(22, 3, 3)``.
        translation: Root translation, shape ``(3,)``.
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        target_joint: Target joint index (must be in ONE_BONE_TARGETS).
        target_pos: Target world position, shape ``(3,)``.

    Returns:
        new_local_rotmat: Modified local rotations, shape ``(22, 3, 3)``.
    """
    parent_joint = ONE_BONE_TARGETS[target_joint]
    device = local_rotmat.device
    dtype = local_rotmat.dtype

    new_local_rotmat = local_rotmat.clone()

    # Current FK
    world_pos, world_rot = differentiable_fk(local_rotmat, translation, bone_offsets)

    parent_world_pos = world_pos[parent_joint]
    parent_world_rot = world_rot[parent_joint]

    # Current direction from parent to target_joint (in world space)
    current_dir = F.normalize(world_pos[target_joint] - parent_world_pos, dim=-1)
    desired_dir = F.normalize(target_pos - parent_world_pos, dim=-1)

    # Compute swing rotation in world space
    R_swing = _rotation_between_vectors(current_dir, desired_dir)

    # Apply to parent's world rotation
    new_parent_world_rot = R_swing @ parent_world_rot

    # Convert back to local
    grandparent = SMPL22_PARENTS[parent_joint]
    if grandparent < 0:
        grandparent_world_rot = torch.eye(3, device=device, dtype=dtype)
    else:
        grandparent_world_rot = world_rot[grandparent]

    new_local_rotmat[parent_joint] = grandparent_world_rot.T @ new_parent_world_rot

    return new_local_rotmat


# ========================================================================
# Gradient IK
# ========================================================================

def solve_gradient_ik(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
    target_joint: int,
    target_pos: Tensor,
    lr: float = 0.05,
    num_steps: int = 100,
    reg_weight: float = 0.001,
    tol: float = 1e-4,  # 0.1mm in meters
) -> Tensor:
    """Solve IK via gradient optimization on ancestor joints.

    Optimizes the rotations of 1-3 ancestor joints to minimize position error
    at the target joint, with regularization to stay close to original pose.

    Args:
        local_rotmat: Local rotation matrices, shape ``(22, 3, 3)``.
        translation: Root translation, shape ``(3,)``.
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        target_joint: Target joint index.
        target_pos: Target world position, shape ``(3,)``.
        lr: Learning rate for Adam.
        num_steps: Maximum optimization steps.
        reg_weight: Regularization weight for rotation deviation.
        tol: Convergence tolerance (meters).

    Returns:
        new_local_rotmat: Modified local rotations, shape ``(22, 3, 3)``.
    """
    device = local_rotmat.device
    dtype = local_rotmat.dtype

    # Get ancestors to optimize
    ancestors = GRADIENT_IK_ANCESTORS.get(target_joint)
    if ancestors is None:
        # If not in the predefined map, use direct parent
        ancestors = [SMPL22_PARENTS[target_joint]]

    # Convert ancestors' rotations to axis-angle for optimization
    from motius.models.motioncanvas.network.geometry import (
        matrix_to_axis_angle,
        axis_angle_to_matrix,
    )

    original_rotmat = local_rotmat.clone().detach()

    # Parameterize as axis-angle delta
    deltas = []
    for anc in ancestors:
        delta = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
        deltas.append(delta)

    optimizer = torch.optim.Adam(deltas, lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Build modified local_rotmat
        mod_rotmat = original_rotmat.clone()
        for i, anc in enumerate(ancestors):
            # Apply delta rotation: R_new = R_delta @ R_original
            R_delta = axis_angle_to_matrix(deltas[i])
            mod_rotmat[anc] = R_delta @ original_rotmat[anc]

        # FK
        world_pos, _ = differentiable_fk(mod_rotmat, translation.detach(), bone_offsets.detach())

        # Position error
        pos_error = (world_pos[target_joint] - target_pos.detach()).pow(2).sum()

        # Regularization: keep rotations close to original
        reg_loss = sum(d.pow(2).sum() for d in deltas) * reg_weight

        loss = pos_error + reg_loss
        loss.backward()
        optimizer.step()

        # Check convergence
        with torch.no_grad():
            err = (world_pos[target_joint] - target_pos).norm().item()
            if err < tol:
                break

    # Build final rotmat
    with torch.no_grad():
        final_rotmat = original_rotmat.clone()
        for i, anc in enumerate(ancestors):
            R_delta = axis_angle_to_matrix(deltas[i].detach())
            final_rotmat[anc] = R_delta @ original_rotmat[anc]

    return final_rotmat


# ========================================================================
# Unified IK Dispatcher
# ========================================================================

def solve_single_constraint(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
    target_joint: int,
    target_pos: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Solve IK for a single position constraint, auto-selecting strategy.

    Args:
        local_rotmat: Local rotation matrices, shape ``(22, 3, 3)``.
        translation: Root translation, shape ``(3,)``.
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        target_joint: Target joint index (0-21).
        target_pos: Target world position, shape ``(3,)``.

    Returns:
        new_local_rotmat: Modified local rotations, shape ``(22, 3, 3)``.
        new_translation: Modified translation, shape ``(3,)``.
    """
    strategy = get_ik_strategy(target_joint)
    new_translation = translation.clone()

    if strategy == 'root':
        new_translation = solve_root_ik(translation, bone_offsets, target_pos)
        return local_rotmat.clone(), new_translation

    elif strategy == 'two_bone':
        new_rotmat = solve_two_bone_ik(
            local_rotmat, translation, bone_offsets, target_joint, target_pos
        )
        return new_rotmat, new_translation

    elif strategy == 'one_bone':
        new_rotmat = solve_one_bone_ik(
            local_rotmat, translation, bone_offsets, target_joint, target_pos
        )
        return new_rotmat, new_translation

    else:  # gradient
        new_rotmat = solve_gradient_ik(
            local_rotmat, translation, bone_offsets, target_joint, target_pos
        )
        return new_rotmat, new_translation
