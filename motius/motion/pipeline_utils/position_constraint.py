"""Position constraint solver for HyMotion M2M motion generation.

Provides a unified interface for applying world-space position constraints
to generated motion. Orchestrates FK and IK to ensure constraints are met.

Usage::

    from motius.motion.pipeline_utils.position_constraint import (
        PositionConstraint,
        PositionConstraintSolver,
    )

    constraints = [
        PositionConstraint(frame=30, joint=20, target_xyz=torch.tensor([1.0, 1.5, 0.3])),
    ]
    solver = PositionConstraintSolver(bone_offsets, rotation_space='local')
    motion_fixed, max_error = solver.solve(motion_denorm, constraints)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS
from motius.motion.pipeline_utils.differentiable_fk import (
    differentiable_fk,
    fk_to_motion135,
    motion135_to_fk,
)
from motius.motion.pipeline_utils.ik_solver import (
    GRADIENT_IK_ANCESTORS,
    ONE_BONE_TARGETS,
    TWO_BONE_CHAINS,
    get_ik_strategy,
    solve_gradient_ik,
    solve_one_bone_ik,
    solve_root_ik,
    solve_single_constraint,
    solve_two_bone_ik,
)


@dataclass
class PositionConstraint:
    """A single position constraint specifying a target world-space position
    for a joint at a specific frame.

    Attributes:
        frame: Frame index.
        joint: Joint index (0-21 for SMPL-22).
        target_xyz: Target world-space position, shape ``(3,)``.
    """
    frame: int
    joint: int
    target_xyz: Tensor  # (3,)


class PositionConstraintSolver:
    """Solves position constraints on denormalized 135-dim motion.

    Orchestrates IK solving with proper ordering to minimize cascading
    conflicts. Constraints on different kinematic chains are independent
    and won't conflict.

    Args:
        bone_offsets: Bone offsets, shape ``(22, 3)``.
        rotation_space: ``'local'`` or ``'global'``.
        hard_projection_tol: If any constraint error exceeds this after IK,
            a second pass with tighter parameters is applied (meters).
        hard_projection_lr: Learning rate for the hard projection pass.
        hard_projection_steps: Number of gradient IK steps for hard projection.
    """

    def __init__(
        self,
        bone_offsets: Tensor,
        rotation_space: str = 'local',
        hard_projection_tol: float = 5e-4,  # 0.5mm
        hard_projection_lr: float = 0.005,
        hard_projection_steps: int = 200,
    ):
        self.bone_offsets = bone_offsets
        self.rotation_space = rotation_space
        self.hard_projection_tol = hard_projection_tol
        self.hard_projection_lr = hard_projection_lr
        self.hard_projection_steps = hard_projection_steps

    def solve(
        self,
        motion_denorm: Tensor,
        constraints: List[PositionConstraint],
    ) -> Tuple[Tensor, float]:
        """Apply position constraints to denormalized motion.

        Solving order (upstream first to avoid cascading conflicts):
        1. Root constraints (translation adjustment)
        2. 1-bone constraints (parent rotation adjustment)
        3. 2-bone constraints (chain rotation adjustment)
        4. Gradient constraints (ancestor optimization)
        5. Hard projection verification pass

        Args:
            motion_denorm: Denormalized motion, shape ``(T, 135)`` or ``(B, T, 135)``.
            constraints: List of position constraints.

        Returns:
            motion_fixed: Corrected motion, same shape as input.
            max_error: Maximum position error (meters) across all constraints.
        """
        if len(constraints) == 0:
            return motion_denorm.clone(), 0.0

        has_batch = motion_denorm.ndim == 3
        if has_batch:
            # Process batch dimension (constraints apply to all batches)
            # For now, assume single batch or handle first batch
            assert motion_denorm.shape[0] == 1, (
                "Batch position constraints only supported for B=1"
            )
            motion_2d = motion_denorm[0]
            result, err = self._solve_single(motion_2d, constraints)
            return result.unsqueeze(0), err
        else:
            return self._solve_single(motion_denorm, constraints)

    def _solve_single(
        self,
        motion_denorm: Tensor,
        constraints: List[PositionConstraint],
    ) -> Tuple[Tensor, float]:
        """Solve constraints for a single motion sequence (T, 135)."""
        device = motion_denorm.device
        dtype = motion_denorm.dtype
        bone_offsets = self.bone_offsets.to(device=device, dtype=dtype)

        motion = motion_denorm.clone()

        # Group constraints by priority
        root_cs: List[PositionConstraint] = []
        one_bone_cs: List[PositionConstraint] = []
        two_bone_cs: List[PositionConstraint] = []
        gradient_cs: List[PositionConstraint] = []

        for c in constraints:
            strategy = get_ik_strategy(c.joint)
            if strategy == 'root':
                root_cs.append(c)
            elif strategy == 'one_bone':
                one_bone_cs.append(c)
            elif strategy == 'two_bone':
                two_bone_cs.append(c)
            else:
                gradient_cs.append(c)

        # Sort within each group: closer to root first
        def _chain_depth(joint: int) -> int:
            depth = 0
            j = joint
            while SMPL22_PARENTS[j] >= 0:
                j = SMPL22_PARENTS[j]
                depth += 1
            return depth

        one_bone_cs.sort(key=lambda c: _chain_depth(c.joint))
        two_bone_cs.sort(key=lambda c: _chain_depth(c.joint))
        gradient_cs.sort(key=lambda c: _chain_depth(c.joint))

        # Apply constraints in priority order
        for cs_group in [root_cs, one_bone_cs, two_bone_cs, gradient_cs]:
            for c in cs_group:
                motion = self._apply_single_constraint(
                    motion, c, bone_offsets
                )

        # Hard projection pass: verify all constraints and re-solve if needed
        max_error = self._compute_max_error(motion, constraints, bone_offsets)
        if max_error > self.hard_projection_tol:
            motion = self._hard_projection_pass(motion, constraints, bone_offsets)
            max_error = self._compute_max_error(motion, constraints, bone_offsets)

        return motion, max_error

    def _apply_single_constraint(
        self,
        motion: Tensor,
        constraint: PositionConstraint,
        bone_offsets: Tensor,
    ) -> Tensor:
        """Apply a single constraint to a motion frame."""
        frame = constraint.frame
        target_pos = constraint.target_xyz.to(device=motion.device, dtype=motion.dtype)

        # Parse frame
        frame_motion = motion[frame]  # (135,)
        world_pos, world_rot, translation, local_rotmat = motion135_to_fk(
            frame_motion, bone_offsets, rotation_space=self.rotation_space
        )

        # Solve IK
        new_rotmat, new_translation = solve_single_constraint(
            local_rotmat, translation, bone_offsets,
            constraint.joint, target_pos,
        )

        # Re-encode to 135-dim
        new_frame = fk_to_motion135(
            new_rotmat, new_translation,
            rotation_space=self.rotation_space,
        )

        motion = motion.clone()
        motion[frame] = new_frame
        return motion

    def _compute_max_error(
        self,
        motion: Tensor,
        constraints: List[PositionConstraint],
        bone_offsets: Tensor,
    ) -> float:
        """Compute maximum position error across all constraints."""
        if len(constraints) == 0:
            return 0.0

        max_err = 0.0
        for c in constraints:
            frame_motion = motion[c.frame]
            world_pos, _, _, _ = motion135_to_fk(
                frame_motion, bone_offsets, rotation_space=self.rotation_space
            )
            target = c.target_xyz.to(device=motion.device, dtype=motion.dtype)
            err = (world_pos[c.joint] - target).norm().item()
            max_err = max(max_err, err)
        return max_err

    def _hard_projection_pass(
        self,
        motion: Tensor,
        constraints: List[PositionConstraint],
        bone_offsets: Tensor,
    ) -> Tensor:
        """Re-solve constraints that exceed tolerance with tighter parameters."""
        for c in constraints:
            frame_motion = motion[c.frame]
            world_pos, _, _, _ = motion135_to_fk(
                frame_motion, bone_offsets, rotation_space=self.rotation_space
            )
            target = c.target_xyz.to(device=motion.device, dtype=motion.dtype)
            err = (world_pos[c.joint] - target).norm().item()

            if err > self.hard_projection_tol:
                # Use gradient IK with tighter params regardless of joint type
                _, _, translation, local_rotmat = motion135_to_fk(
                    frame_motion, bone_offsets, rotation_space=self.rotation_space
                )

                if c.joint == 0:
                    # Root: just re-apply (should be exact)
                    new_translation = solve_root_ik(translation, bone_offsets, target)
                    new_frame = fk_to_motion135(
                        local_rotmat, new_translation,
                        rotation_space=self.rotation_space,
                    )
                else:
                    new_rotmat = solve_gradient_ik(
                        local_rotmat, translation, bone_offsets,
                        c.joint, target,
                        lr=self.hard_projection_lr,
                        num_steps=self.hard_projection_steps,
                        reg_weight=0.0001,
                        tol=1e-4,
                    )
                    new_frame = fk_to_motion135(
                        new_rotmat, translation,
                        rotation_space=self.rotation_space,
                    )

                motion = motion.clone()
                motion[c.frame] = new_frame

        return motion


def get_affected_dims(constraints: List[PositionConstraint]) -> List[int]:
    """Get the set of motion dimensions affected by IK for given constraints.

    Used for selective replacement in the ODE loop — only replace dims that
    IK actually modified.

    Args:
        constraints: List of position constraints.

    Returns:
        Sorted list of affected dimension indices in the 135-dim motion.
    """
    affected = set()

    for c in constraints:
        strategy = get_ik_strategy(c.joint)

        if strategy == 'root':
            # Translation dims
            affected.update([0, 1, 2])

        elif strategy == 'two_bone':
            chain = TWO_BONE_CHAINS[c.joint]
            for j in chain[:2]:  # base and mid joints get modified
                start = 3 + j * 6
                affected.update(range(start, start + 6))

        elif strategy == 'one_bone':
            parent = ONE_BONE_TARGETS[c.joint]
            start = 3 + parent * 6
            affected.update(range(start, start + 6))

        else:  # gradient
            ancestors = GRADIENT_IK_ANCESTORS.get(c.joint, [SMPL22_PARENTS[c.joint]])
            for anc in ancestors:
                start = 3 + anc * 6
                affected.update(range(start, start + 6))

    return sorted(affected)
