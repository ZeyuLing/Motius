"""Contract tests for the native ARDY integration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from motius.models.ardy import ARDYBundle
from motius.motion.representation import get_spec, split_ardy_features


def test_ardy_specs_match_explicit_feature_widths():
    core = get_spec("ARDY-Core-330")
    g1 = get_spec("ARDY-G1-414")
    assert core.dim == 330
    assert core.layout[-1][2] == 330
    assert g1.dim == 414
    assert g1.layout[-1][2] == 414


@pytest.mark.parametrize("name,dim", [("ardy_core330", 330), ("ardy_g1_414", 414)])
def test_split_ardy_features_covers_every_channel(name, dim):
    value = np.arange(2 * dim, dtype=np.float32).reshape(2, dim)
    fields = split_ardy_features(value, name)
    assert sum(item.shape[-1] for item in fields.values()) == dim
    assert np.array_equal(fields["root_position"], value[:, :3])
    assert fields["foot_contacts"].shape[-1] == 4


def test_ardy_bundle_alias_without_loading_weights():
    bundle = ARDYBundle.from_pretrained("core8", load_model=False)
    assert bundle.model_name == "nvidia/ARDY-Core-RP-20FPS-Horizon8"
    assert bundle.SUPPORTED_TASKS["streaming_text_to_motion"]


@pytest.mark.parametrize(
    "skeleton_name,fps,expected_dim",
    [("CoreSkeleton27", 20, 330), ("G1Skeleton34", 25, 414)],
)
def test_ardy_explicit_representation_roundtrip(skeleton_name, fps, expected_dim):
    from motius.models.ardy.network import skeleton as skeleton_module
    from motius.models.ardy.network.motion_rep import ArdyMotionRep

    skeleton = getattr(skeleton_module, skeleton_name)()
    rep = ArdyMotionRep(skeleton, fps)
    frames = 5
    rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(frames, skeleton.nbjoints, 1, 1)
    root = torch.zeros(frames, 3)
    root[:, 2] = torch.linspace(0, 0.4, frames)
    features = rep(rotations, root, to_normalize=False)
    decoded = rep.inverse(features, is_normalized=False)

    assert features.shape == (frames, expected_dim)
    assert torch.allclose(decoded["root_positions"], root, atol=1e-5)
    assert torch.isfinite(decoded["posed_joints"]).all()


def test_g1_qpos_export_accepts_repository_scipy_version():
    from motius.models.ardy.network.exports.mujoco import MujocoQposConverter
    from motius.models.ardy.network.skeleton import G1Skeleton34

    skeleton = G1Skeleton34()
    rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 3, skeleton.nbjoints, 1, 1)
    roots = torch.zeros(1, 3, 3)
    qpos = MujocoQposConverter(skeleton).to_qpos(rotations, roots)
    assert qpos.shape == (1, 3, 36)
    assert torch.isfinite(qpos).all()


def test_ardy_constraint_loader_honors_dtype():
    from motius.models.ardy.network.constraints import load_constraints_lst
    from motius.models.ardy.network.skeleton import CoreSkeleton27

    constraints = load_constraints_lst(
        [
            {
                "type": "root2d",
                "frame_indices": [0, 4],
                "root_2d": [[0.0, 0.0], [1.0, 0.5]],
            }
        ],
        CoreSkeleton27(),
        dtype=torch.float64,
    )
    assert constraints[0].frame_indices.dtype == torch.int64
    assert constraints[0].root_2d.dtype == torch.float64


def test_ardy_core27_to_smpl22_joint_bridge_is_explicit():
    from motius.motion import ardy_core27_to_smpl22_joints

    joints = np.arange(2 * 27 * 3, dtype=np.float32).reshape(2, 27, 3)
    smpl = ardy_core27_to_smpl22_joints(joints)
    assert smpl.shape == (2, 22, 3)
    np.testing.assert_array_equal(smpl[:, 0], joints[:, 0])    # Pelvis <- Hips
    np.testing.assert_array_equal(smpl[:, 1], joints[:, 23])   # L_Hip <- LeftUpLeg
    np.testing.assert_array_equal(smpl[:, 2], joints[:, 19])   # R_Hip <- RightUpLeg
    np.testing.assert_array_equal(smpl[:, 20], joints[:, 16])  # L_Wrist <- LeftHand
    np.testing.assert_array_equal(smpl[:, 21], joints[:, 10])  # R_Wrist <- RightHand

    recentered = ardy_core27_to_smpl22_joints(joints, recenter_root=True)
    np.testing.assert_allclose(recentered[:, 0, [0, 2]], 0.0)
    np.testing.assert_array_equal(recentered[:, 0, 1], smpl[:, 0, 1])
