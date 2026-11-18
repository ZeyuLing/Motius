"""Regress torso folding hidden by small position residuals."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from motius.motion.retarget.anatomical_fit import fit_anatomical_pose
from motius.motion.retarget.position_ik import estimate_local_rotations
from tools.smpl_audit_rig import GallerySMPL
from tools.validate_hml263_motion272 import forward_kinematics


def test_noisy_short_spine_bones_do_not_force_opposing_large_bends():
    rig = GallerySMPL(
        Path(__file__).resolve().parents[1]
        / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model"
    )
    rest = rig.rest[:22].numpy()
    parents = rig.parents[:22].numpy()
    target = np.broadcast_to(rest, (5, 22, 3)).copy()
    target[:, 6, 2] -= 0.06
    target[:, 9, 2] -= 0.10
    target[:, 15, 2] -= 0.13
    target[:, 0, 0] += np.arange(5) * 0.02
    initial = estimate_local_rotations(target, rest, parents)
    # Public inference callers commonly disable gradient recording.
    with torch.no_grad():
        result = fit_anatomical_pose(target, rest, parents, initial)
    angle = lambda r: Rotation.from_matrix(
        r[:, [3, 6, 9, 12]].reshape(-1, 3, 3)
    ).magnitude()
    assert np.rad2deg(angle(initial)).max() > 90
    assert np.rad2deg(angle(result)).max() < 45
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    fitted, _ = forward_kinematics(result, target[:, 0], offsets, parents)
    previous, _ = forward_kinematics(initial, target[:, 0], offsets, parents)
    assert (
        np.linalg.norm(fitted - target, axis=-1).mean()
        < np.linalg.norm(previous - target, axis=-1).mean()
    )
    np.testing.assert_allclose(fitted[:, 0], target[:, 0], atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(result), 1, atol=1e-6)


def test_fit_can_be_disabled_and_rejects_invalid_configuration():
    initial = np.broadcast_to(np.eye(3), (1, 22, 3, 3)).copy()
    for kwargs in (
        {"iterations": -1},
        {"prior_weight": float("nan")},
        {"prior_weight": -1},
    ):
        with pytest.raises(ValueError):
            fit_anatomical_pose(
                np.zeros((1, 22, 3)), np.zeros((22, 3)), np.zeros(22), initial, **kwargs
            )
    np.testing.assert_array_equal(
        fit_anatomical_pose(np.zeros((1, 22, 3)), None, None, initial, iterations=0),
        initial,
    )
