"""Deterministic regression cases; no checkpoints, SMPL assets or CUDA needed."""

import math

import pytest
import torch

from motius.motion.pipeline_utils.ik_solver import solve_two_bone_ik
from motius.motion.pipeline_utils.position_constraint import (
    PositionConstraint,
    PositionConstraintSolver,
)
from motius.motion.skeleton.fk import differentiable_fk, fk_to_motion135, motion135_to_fk


def skeleton(dtype=torch.float64):
    offsets = torch.zeros(22, 3, dtype=dtype)
    offsets[1] = torch.tensor([.1, -.1, 0.])
    offsets[2] = torch.tensor([-.1, -.1, 0.])
    offsets[[3, 6, 9, 12, 15], 1] = .1
    offsets[[4, 5, 7, 8], 1] = -.4
    offsets[[10, 11], 2] = .15
    offsets[[13, 16], 0] = .12
    offsets[[14, 17], 0] = -.12
    offsets[[18, 20], 0] = .3
    offsets[[19, 21], 0] = -.3
    return offsets


def pose():
    return torch.eye(3, dtype=torch.float64).repeat(22, 1, 1), torch.zeros(3, dtype=torch.float64)


def turn_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def test_two_bone_does_not_shrink_reachable_extended_target():
    offsets = skeleton()
    rotations, translation = pose()
    target = differentiable_fk(rotations, translation, offsets)[0][20].clone()
    rotations[18] = turn_z(.7)
    solved = solve_two_bone_ik(rotations, translation, offsets, 20, target)
    actual = differentiable_fk(solved, translation, offsets)[0][20]
    assert torch.linalg.vector_norm(actual - target) < 1e-6


def test_nonzero_frame_and_198_channels_preserved():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation).repeat(6, 1)
    extra = torch.randn(6, 63, dtype=motion.dtype)
    motion = torch.cat((motion, extra), dim=-1)
    target = differentiable_fk(rotations, translation, offsets)[0][20] + torch.tensor([-.15, .1, 0])
    original = motion.clone()
    fixed, error = PositionConstraintSolver(offsets).solve(motion, [PositionConstraint(4, 20, target)])
    assert fixed.shape == (6, 198)
    assert error < 1e-4
    assert torch.equal(fixed[:, 135:], extra)
    assert torch.equal(fixed[[0, 1, 2, 3, 5]], original[[0, 1, 2, 3, 5]])
    assert torch.equal(motion, original)


def test_unreachable_target_is_not_reported_as_success():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation)[None]
    target = torch.tensor([100., 100., 100.], dtype=motion.dtype)
    _, error = PositionConstraintSolver(offsets).solve(motion, [PositionConstraint(0, 20, target)])
    assert math.isfinite(error) and error > 1


def test_axis_mask_checks_only_controlled_axes():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation)[None]
    target = differentiable_fk(rotations, translation, offsets)[0][20].clone()
    target[1] = 99  # unobserved Y must not be pulled towards this value
    fixed, error = PositionConstraintSolver(offsets).solve(
        motion, [PositionConstraint(0, 20, target, axes=(True, False, True))]
    )
    assert error < 1e-6
    assert torch.equal(fixed, motion)


def test_partial_axis_cue_can_bend_an_initially_straight_limb():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation)[None]
    target = differentiable_fk(rotations, translation, offsets)[0][20].clone()
    target[0] -= .15
    target[1] = 99
    fixed, error = PositionConstraintSolver(offsets).solve(
        motion, [PositionConstraint(0, 20, target, axes=(True, False, False))]
    )
    assert error < 1e-4
    assert torch.isfinite(fixed).all()


def test_shared_chain_constraints_solved_together_and_order_independent():
    offsets = skeleton()
    rotations, translation = pose()
    reference = rotations.clone()
    reference[16] = turn_z(.35)
    reference[18] = turn_z(.6)
    targets = differentiable_fk(reference, translation, offsets)[0]
    motion = fk_to_motion135(rotations, translation)[None]
    constraints = [PositionConstraint(0, j, targets[j]) for j in (18, 20)]
    solver = PositionConstraintSolver(offsets)
    first, e1 = solver.solve(motion, constraints)
    second, e2 = solver.solve(motion, constraints[::-1])
    assert max(e1, e2) < 1e-4
    torch.testing.assert_close(first, second)


def test_fixed_rotations_and_translation_are_not_relaxed():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation)[None]
    target = differentiable_fk(rotations, translation, offsets)[0][20] + torch.tensor([0., .2, 0.])
    fixed_mask = torch.ones_like(motion, dtype=torch.bool)
    fixed, error = PositionConstraintSolver(offsets).solve(
        motion, [PositionConstraint(0, 20, target)], fixed_mask=fixed_mask
    )
    assert torch.equal(fixed, motion)
    assert error > .1


@pytest.mark.parametrize('rotation_space', ['local', 'global'])
def test_batch_and_final_fk_residual(rotation_space):
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation, rotation_space).repeat(2, 3, 1)
    target = differentiable_fk(rotations, translation, offsets)[0][20] + torch.tensor([-.1, .15, .05])
    fixed, error = PositionConstraintSolver(offsets, rotation_space).solve(
        motion, [PositionConstraint(2, 20, target)]
    )
    actual = motion135_to_fk(fixed[:, 2, :135], offsets, rotation_space)[0][:, 20]
    measured = torch.linalg.vector_norm(actual - target, dim=-1).max().item()
    assert measured < 1e-4
    assert error == pytest.approx(measured, abs=1e-10)


@pytest.mark.parametrize('frame,joint', [(-1, 20), (3, 20), (0, 22)])
def test_invalid_constraint_indices_are_rejected(frame, joint):
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation).repeat(3, 1)
    with pytest.raises(ValueError):
        PositionConstraintSolver(offsets).solve(motion, [PositionConstraint(frame, joint, torch.zeros(3))])


def test_nonfinite_target_rejected_not_hidden_by_max():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation)[None]
    with pytest.raises(ValueError):
        PositionConstraintSolver(offsets).solve(
            motion, [PositionConstraint(0, 20, torch.tensor([float('nan'), 0., 0.]))]
        )


def test_root_axes_and_locked_global_rotation():
    offsets = skeleton()
    rotations, translation = pose()
    motion = fk_to_motion135(rotations, translation, 'global')[None]
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mask[:, 3 + 16 * 6:3 + 17 * 6] = True
    target = torch.tensor([.1, .2, .3], dtype=motion.dtype)
    fixed, error = PositionConstraintSolver(offsets, 'global').solve(
        motion, [PositionConstraint(0, 0, target, axes=(True, False, True))], fixed_mask=mask
    )
    assert error < 1e-6
    assert fixed[0, 1] == motion[0, 1]
    assert torch.equal(fixed[mask], motion[mask])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_batched_analytic_rotation_integrity(dtype):
    offsets = skeleton(dtype)
    rotations = torch.eye(3, dtype=dtype).repeat(3, 22, 1, 1)
    translation = torch.zeros(3, 3, dtype=dtype)
    base = differentiable_fk(rotations, translation, offsets)[0][:, 16]
    targets = base + torch.tensor([[.6, 0., 0.], [-.45, .1, .05], [0., 0., 0.]], dtype=dtype)
    solved = solve_two_bone_ik(rotations, translation, offsets, 20, targets)
    actual = differentiable_fk(solved, translation, offsets)[0][:, 20]
    assert (actual - targets).norm(dim=-1).max() < 1e-5
    torch.testing.assert_close(solved @ solved.transpose(-1, -2), torch.eye(3, dtype=dtype).expand_as(solved), atol=2e-6, rtol=2e-6)
    assert torch.all(torch.linalg.det(solved) > .99999)


class _ProjectionBundle(torch.nn.Module):
    """Tiny deterministic x1 predictor to test the actual ODE integration path."""
    def __init__(self, motion, offsets):
        super().__init__()
        self.motion_transformer = torch.nn.Linear(1, 1).to(motion.dtype)
        self.register_buffer('target', motion)
        self.register_buffer('offsets', offsets)
        self.register_buffer('null_vtxt_feat', torch.zeros(1, 1, 1, dtype=motion.dtype))
        self.register_buffer('null_ctxt_input', torch.zeros(1, 1, 1, dtype=motion.dtype))
        self.uncondition_mode = True
        self.pred_type = 'x1'
        self.rotation_space = 'local'
        self._noise_scheduler_cfg = {}

    def prepare_condition_context(self, src_motion, **kwargs):
        return src_motion

    def predict_flow(self, **kwargs):
        return self.target

    def get_bone_offsets(self):
        return self.offsets

    def denormalize_motion(self, x):
        return x

    def normalize_motion(self, x):
        return x

    def decode_motion_from_latent(self, x):
        return {'motion_198': x}


@pytest.mark.parametrize('mode', ['all', 'skip_last', 'flow_interp'])
def test_pipeline_nonzero_frame_198_and_fixed_cues(mode):
    from motius.pipelines.motioncanvas import MotionCanvasPipeline

    offsets = skeleton(torch.float32)
    rotations, translation = pose()
    motion = fk_to_motion135(rotations.float(), translation.float()).repeat(1, 6, 1)
    motion = torch.cat((motion, torch.zeros(1, 6, 63)), dim=-1)
    mask = torch.zeros_like(motion)
    mask[:, :, 3 + 16 * 6:3 + 17 * 6] = 1
    mask[:, :, 3 + 18 * 6:3 + 19 * 6] = 1
    target = motion135_to_fk(motion[0, 4, :135], offsets)[0][20] + torch.tensor([-.15, .1, 0.])
    pipeline = MotionCanvasPipeline(_ProjectionBundle(motion, offsets), num_steps=1,
                                    replacement_guidance=mode, official_t2m_frames=6)
    result = pipeline._inference({'src_motion': motion, 'src_mask': mask, 'clean_motion': motion,
                                   'position_constraints': [PositionConstraint(4, 20, target)]})
    assert result['latent'].shape == motion.shape
    assert torch.equal(result['latent'][mask == 0], motion[mask == 0])
    assert result['position_constraint_max_error_m'] < 1e-4


def test_pipeline_rejects_padding_cue():
    from motius.pipelines.motioncanvas import MotionCanvasPipeline

    offsets = skeleton(torch.float32)
    rotations, translation = pose()
    motion = fk_to_motion135(rotations.float(), translation.float()).repeat(1, 6, 1)
    motion = torch.cat((motion, torch.zeros(1, 6, 63)), dim=-1)
    pipeline = MotionCanvasPipeline(_ProjectionBundle(motion, offsets), num_steps=1, official_t2m_frames=6)
    with pytest.raises(ValueError, match='valid sequence'):
        pipeline._inference({'src_motion': motion, 'src_length': [3],
                             'position_constraints': [PositionConstraint(4, 20, torch.zeros(3))]})
