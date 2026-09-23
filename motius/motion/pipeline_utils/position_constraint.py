"""Batched position projection for motion135 and motion198.

Constraints sharing a frame are solved together. Frames sharing a cue layout
are batched; no per-cue full-motion copies or backward/Adam loops are needed.
Extra representation channels are preserved verbatim, not used as FK evidence.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import List, Tuple

import torch
from torch import Tensor

from motius.motion.skeleton.fk import motion135_to_fk, fk_to_motion135, differentiable_fk
from motius.motion.pipeline_utils.position_ik import (
    TWO_BONE_CHAINS, ONE_BONE_TARGETS, ancestors, solve_two_bone_ik,
    solve_one_bone_ik, solve_joint_positions,
)


@dataclass
class PositionConstraint:
    frame: int
    joint: int
    target_xyz: Tensor
    axes: Tuple[bool, bool, bool] = (True, True, True)


class PositionConstraintSolver:
    """Position IK with final FK verification against original cues.

    ``fixed_mask`` in solve() locks input translation coordinates and whole
    rotation-6D blocks. Partial rotation blocks are rejected, not silently
    relaxed. Global-space rotation locks conservatively freeze their local
    ancestors too. This can reduce reachability but never breaks supplied
    global orientations. Translation is fixed unless a root-position cue is
    present. All coordinates/returned errors are in meters.

    Legacy hard_projection_* argument names remain accepted. The old Adam
    learning rate is unused; steps bounds DLS, and tol is the actual positional
    stopping tolerance. This solver is NOT the paper's temporal SQP solver.
    """

    def __init__(self, bone_offsets: Tensor, rotation_space: str = 'local',
                 hard_projection_tol: float = 1e-5, hard_projection_lr: float = .005,
                 hard_projection_steps: int = 100):
        if rotation_space not in ('local', 'global'):
            raise ValueError('rotation_space must be local or global')
        if hard_projection_tol <= 0 or hard_projection_steps < 1:
            raise ValueError('IK tolerance and iteration budget must be positive')
        self.bone_offsets = bone_offsets
        self.rotation_space = rotation_space
        self.hard_projection_tol = hard_projection_tol
        self.hard_projection_lr = hard_projection_lr
        self.hard_projection_steps = hard_projection_steps

    @torch.no_grad()
    def solve(self, motion_denorm: Tensor, constraints: List[PositionConstraint],
              *, fixed_mask: Tensor = None) -> Tuple[Tensor, float]:
        if motion_denorm.ndim not in (2, 3) or motion_denorm.shape[-1] < 135:
            raise ValueError('Expected (T,D) or (B,T,D), D >= 135')
        if not motion_denorm.is_floating_point():
            raise ValueError('Motion must be floating-point')
        if not constraints:
            return motion_denorm.clone(), 0.0
        batched = motion_denorm.ndim == 3
        original = motion_denorm if batched else motion_denorm[None]
        batch_size, length, dim = original.shape
        if batch_size == 0 or length == 0:
            raise ValueError('Cannot constrain an empty motion')
        dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
        offsets = self.bone_offsets.to(device=original.device, dtype=dtype)
        if offsets.shape != (22, 3) or not bool(torch.isfinite(offsets).all()):
            raise ValueError('Expected finite bone offsets (22,3)')
        if fixed_mask is None:
            locks = torch.zeros_like(original, dtype=torch.bool)
        else:
            locks = torch.as_tensor(fixed_mask, device=original.device, dtype=torch.bool)
            try:
                locks = torch.broadcast_to(locks, original.shape)
            except RuntimeError as exc:
                raise ValueError('fixed_mask must broadcast to the input motion') from exc
        by_frame = {}
        for c in constraints:
            if not isinstance(c.frame, Integral) or not 0 <= c.frame < length:
                raise ValueError(f'Constraint frame {c.frame} outside [0,{length})')
            if not isinstance(c.joint, Integral) or not 0 <= c.joint < 22:
                raise ValueError(f'Invalid SMPL-22 joint {c.joint}')
            if len(c.axes) != 3 or not any(c.axes) or any(v not in (True, False) for v in c.axes):
                raise ValueError('axes must contain three booleans, with at least one true')
            target = torch.as_tensor(c.target_xyz, device=original.device, dtype=dtype)
            if target.shape != (3,) or not bool(torch.isfinite(target).all()):
                raise ValueError('target_xyz must be finite and have shape (3,)')
            by_frame.setdefault(c.frame, []).append((c.joint, tuple(c.axes), target))
        groups = {}
        for frame, cues in by_frame.items():
            cues.sort(key=lambda item: (item[0], item[1]))
            layout = tuple((joint, axes) for joint, axes, _ in cues)
            groups.setdefault(layout, []).append(frame)
        result = original.clone()
        for layout, frames in groups.items():
            # Sort frames so input cue ordering cannot alter arithmetic order.
            frames.sort()
            joints = [entry[0] for entry in layout]
            count, k = batch_size * len(frames), len(layout)
            frame_input = original[:, frames, :135].reshape(count, 135)
            if not bool(torch.isfinite(frame_input).all()):
                raise ValueError('Constrained motion frames must be finite')
            q0_pos, _, root, q = motion135_to_fk(frame_input.to(dtype), offsets, self.rotation_space)
            frame_locks = locks[:, frames, :135].reshape(count, 135)
            rotation_locks = frame_locks[:, 3:].reshape(count, 22, 6)
            if bool((rotation_locks.any(-1) != rotation_locks.all(-1)).any()):
                raise ValueError('A rotation cue must lock its entire six-dimensional block')
            fixed_rot = rotation_locks.all(-1).clone()
            if self.rotation_space == 'global':
                global_locks = fixed_rot.clone()
                for j in range(22):
                    for parent in ancestors(j):
                        fixed_rot[:, parent] |= global_locks[:, j]
            axes = torch.tensor([entry[1] for entry in layout], device=original.device, dtype=dtype)
            axes = axes[None].expand(count, k, 3)
            targets = torch.stack([torch.stack([entry[2] for entry in by_frame[f]]) for f in frames])
            targets = targets[None].expand(batch_size, len(frames), k, 3).reshape(count, k, 3)
            original_error = ((targets - q0_pos[:, joints]) * axes).norm(dim=-1).amax(-1)
            if bool((original_error <= self.hard_projection_tol).all()):
                continue
            root_axes = torch.zeros(3, device=original.device, dtype=torch.bool)
            for j, axis in layout:
                if j == 0:
                    root_axes |= torch.tensor(axis, device=original.device)
            fixed_root = frame_locks[:, :3] | ~root_axes[None]
            start_q, start_root = q.clone(), root.clone()
            # Analytic seeds: cheap on wrists/ankles and avoid straight-limb
            # Jacobian singularities. Never overwrite a fixed rotation.
            for index, (joint, axis) in enumerate(layout):
                if joint == 0:
                    enabled = axes[:, index].bool() & ~fixed_root
                    root = torch.where(enabled, targets[:, index] - offsets[0], root)
                elif joint in TWO_BONE_CHAINS:
                    base, mid, _ = TWO_BONE_CHAINS[joint]
                    current = differentiable_fk(q, root, offsets)[0][:, joint]
                    seed_target = torch.where(axes[:, index].bool(), targets[:, index], current)
                    candidate = solve_two_bone_ik(q, root, offsets, joint, seed_target)
                    allowed = ~fixed_rot[:, base] & ~fixed_rot[:, mid]
                    q = torch.where(allowed[:, None, None, None], candidate, q)
                elif joint in ONE_BONE_TARGETS:
                    current = differentiable_fk(q, root, offsets)[0][:, joint]
                    seed_target = torch.where(axes[:, index].bool(), targets[:, index], current)
                    candidate = solve_one_bone_ik(q, root, offsets, joint, seed_target)
                    allowed = ~fixed_rot[:, ONE_BONE_TARGETS[joint]]
                    q = torch.where(allowed[:, None, None, None], candidate, q)
            q, root = solve_joint_positions(q, root, offsets, joints, targets, axes,
                                            fixed_rot, fixed_root, self.hard_projection_steps,
                                            self.hard_projection_tol)
            error = ((targets - differentiable_fk(q, root, offsets)[0][:, joints]) * axes).norm(dim=-1).amax(-1)
            improved = torch.isfinite(error) & (error < original_error)
            q = torch.where(improved[:, None, None, None], q, start_q)
            root = torch.where(improved[:, None], root, start_root)
            encoded = fk_to_motion135(q, root, self.rotation_space).to(original.dtype)
            if self.rotation_space == 'local':
                changed_rotations = (q != start_q).any(-1).any(-1)
                changed = torch.cat((root != start_root,
                                     changed_rotations[..., None].expand(count, 22, 6).reshape(count, 132)), -1)
                encoded = torch.where(changed, encoded, frame_input)
            encoded = torch.where(frame_locks, frame_input, encoded)
            encoded = torch.where(improved[:, None], encoded, frame_input)
            # Measure AFTER rotation6D encoding, dtype conversion and lock copy.
            decoded = motion135_to_fk(encoded.to(dtype), offsets, self.rotation_space)[0]
            returned_error = ((targets - decoded[:, joints]) * axes).norm(dim=-1).amax(-1)
            accept = torch.isfinite(returned_error) & (returned_error < original_error)
            encoded = torch.where(accept[:, None], encoded, frame_input)
            result[:, frames, :135] = encoded.reshape(batch_size, len(frames), 135)
        max_error = self._compute_max_error(result, constraints, offsets)
        return (result if batched else result[0]), max_error

    def _solve_single(self, motion_denorm, constraints, bone_offsets=None):
        """Compatibility path; constraint frame indices remain sequence-relative."""
        if bone_offsets is None:
            return self.solve(motion_denorm, constraints)
        solver = PositionConstraintSolver(bone_offsets, self.rotation_space,
                                         self.hard_projection_tol, self.hard_projection_lr,
                                         self.hard_projection_steps)
        return solver.solve(motion_denorm, constraints)

    def _compute_max_error(self, motion, constraints, bone_offsets):
        if not constraints:
            return 0.0
        batched = motion if motion.ndim == 3 else motion[None]
        frames = sorted({c.frame for c in constraints})
        dtype = torch.float64 if batched.dtype == torch.float64 else torch.float32
        positions = motion135_to_fk(batched[:, frames, :135].to(dtype), bone_offsets.to(dtype), self.rotation_space)[0]
        lookup = {frame: index for index, frame in enumerate(frames)}
        residuals = []
        for c in constraints:
            target = torch.as_tensor(c.target_xyz, device=positions.device, dtype=dtype)
            axes = torch.tensor(c.axes, device=positions.device, dtype=dtype)
            residuals.append(((positions[:, lookup[c.frame], c.joint] - target) * axes).norm(dim=-1))
        errors = torch.stack(residuals, -1)
        return float(errors.max()) if bool(torch.isfinite(errors).all()) else float('inf')


def get_affected_dims(constraints):
    """Conservative rotation/translation support, including fallback ancestors."""
    affected = set()
    for c in constraints:
        if c.joint == 0:
            affected.update(axis for axis, enabled in enumerate(c.axes) if enabled)
        for joint in ancestors(c.joint):
            affected.update(range(3 + 6 * joint, 3 + 6 * (joint + 1)))
    return sorted(affected)
