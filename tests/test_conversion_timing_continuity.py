"""Independent clocks and singular poses that average MPJPE cannot detect."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motius.motion.representation.motion272 import _canonical_272_offsets
from motius.motion.retarget._hml263_smpl_impl import resample_linear
from motius.motion.retarget.position_ik import (
    _preserve_twist_continuity,
    estimate_local_rotations,
)
from motius.motion.skeleton.names import SMPL22_PARENTS


@pytest.mark.parametrize("src,dst", [(20, 30), (30, 20), (20, 60), (60, 20)])
def test_resample_preserves_physical_clock(src, dst):
    source = np.arange(40, dtype=np.float32)[:, None] / src
    result = resample_linear(source, src, dst)
    expected = np.minimum(np.arange(round(40 * dst / src)) / dst, 39 / src)
    np.testing.assert_allclose(result[:, 0], expected, atol=2e-7)


@pytest.mark.parametrize(
    "src,dst", [(0, 30), (-1, 30), (20, 0), (np.nan, 30), (20, np.inf)]
)
def test_resample_rejects_invalid_rates_even_for_single_frame(src, dst):
    with pytest.raises(ValueError, match="fps"):
        resample_linear(np.zeros((1, 3)), src, dst)


def test_single_frame_respects_output_duration():
    assert resample_linear(np.ones((1, 3)), 20, 60).shape == (3, 3)
    assert resample_linear(np.empty((0, 3)), 20, 60).shape == (0, 3)
    assert resample_linear(np.ones((2, 3)), 60, 20).shape == (1, 3)


def test_single_bone_antipode_has_no_twist_flip():
    parents = np.asarray(SMPL22_PARENTS)
    rest = _canonical_272_offsets().copy()
    for j in range(1, 22):
        rest[j] += rest[parents[j]]
    joints = np.broadcast_to(rest, (5, 22, 3)).copy()
    bone = rest[18] - rest[16]
    axis = np.cross(bone, [0, 0, 1.0])
    axis /= np.linalg.norm(axis)
    for t, angle in enumerate(np.deg2rad([178, 179, 180, 181, 182])):
        joints[t, 18] = joints[t, 16] + Rotation.from_rotvec(axis * angle).apply(bone)
    rotations = estimate_local_rotations(joints, rest, parents)
    steps = Rotation.from_matrix(
        rotations[1:, 16] @ rotations[:-1, 16].transpose(0, 2, 1)
    ).magnitude()
    assert np.rad2deg(steps).max() < 1.01


def test_twist_projection_does_not_mutate_caller_bone_lengths():
    bone = np.array([0.0, 0.0, 0.4])
    saved = bone.copy()
    _preserve_twist_continuity(np.eye(3), np.eye(3), bone)
    np.testing.assert_array_equal(bone, saved)


def test_grounding_moves_mesh_once_and_preserves_root_displacements():
    from pathlib import Path

    import torch

    from motius.motion.retarget.hml263_smpl import retarget_hml263_clip
    from tools.smpl_audit_rig import GallerySMPL

    model = GallerySMPL(
        Path(__file__).resolve().parents[1]
        / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model"
    )
    rest = model.rest[:22].numpy()
    rig = (model, rest, model.parents[:22].numpy())
    positions = np.broadcast_to(rest, (4, 22, 3)).copy()
    positions[:, :, 0] += np.arange(4)[:, None] * 0.03
    raw = retarget_hml263_clip(
        positions, smpl_rest=rig, device="cpu", source_fps=30, target_fps=30
    )
    grounded = retarget_hml263_clip(
        positions,
        smpl_rest=rig,
        device="cpu",
        source_fps=30,
        target_fps=30,
        floor_align=True,
    )
    np.testing.assert_allclose(
        np.diff(raw["transl"], axis=0), np.diff(grounded["transl"], axis=0), atol=1e-6
    )
    body = np.zeros((4, 69), np.float32)
    body[:, :63] = grounded["body_pose"]
    with torch.no_grad():
        vertices = model(
            betas=torch.zeros(4, 10),
            body_pose=torch.from_numpy(body),
            global_orient=torch.from_numpy(grounded["global_orient"]),
            transl=torch.from_numpy(grounded["transl"]),
        ).vertices
    assert abs(float(vertices[..., 1].min())) < 1e-6
    restored = retarget_hml263_clip(
        positions,
        smpl_rest=rig,
        device="cpu",
        source_fps=30,
        target_fps=30,
        floor_align=True,
        source_motion135_transl=raw["transl"],
    )
    np.testing.assert_array_equal(restored["transl"], raw["transl"])


def test_soma_straight_spine_does_not_use_collinear_parent_reference():
    from motius.motion.retarget.smpl_soma import SOMAToSMPLIKConfig

    parents = np.asarray(SMPL22_PARENTS)
    rest = _canonical_272_offsets().astype(np.float64).copy()
    for j in range(1, 22):
        rest[j] += rest[parents[j]]
    joints = np.broadcast_to(rest, (9, 22, 3)).copy()
    joints[:, 6, 0] += np.linspace(-0.001, 0.001, 9)
    rotations = estimate_local_rotations(
        joints, rest, parents, orientation_mode=SOMAToSMPLIKConfig().orientation_mode
    )
    delta = rotations[1:, 3] @ rotations[:-1, 3].transpose(0, 2, 1)
    assert np.rad2deg(Rotation.from_matrix(delta).magnitude()).max() < 1


def test_skeleton_only_adapter_does_not_claim_mesh_grounding():
    from types import SimpleNamespace

    import torch

    from motius.motion.retarget._hml263_smpl_impl import smpl_mesh_floor_offset

    def model(**kwargs):
        return SimpleNamespace(vertices=None, joints=torch.zeros(2, 22, 3))

    delta = smpl_mesh_floor_offset(
        model,
        np.zeros((2, 3), np.float32),
        np.zeros((2, 63), np.float32),
        np.zeros((2, 3), np.float32),
        torch.device("cpu"),
    )
    assert delta is None
