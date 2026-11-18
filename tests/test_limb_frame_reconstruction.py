"""Known SMPL poses expose twist errors invisible to joint-position checks."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motius.motion.retarget.position_ik import estimate_local_rotations
from tools.smpl_audit_rig import GallerySMPL
from tools.validate_hml263_motion272 import forward_kinematics


def test_bent_limbs_recover_hinge_and_upper_limb_twist():
    model = GallerySMPL(
        Path(__file__).resolve().parents[1]
        / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model"
    )
    rest = model.rest[:22].numpy()
    parents = model.parents[:22].numpy()
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    reference = np.tile(np.eye(3), (30, 22, 1, 1))
    reference[:, 16] = Rotation.from_euler(
        "xz", np.stack([np.linspace(0, 70, 30), np.full(30, -50)], -1), degrees=True
    ).as_matrix()
    reference[:, 17] = Rotation.from_euler(
        "xz", np.stack([np.linspace(0, -70, 30), np.full(30, 50)], -1), degrees=True
    ).as_matrix()
    reference[:, 18] = Rotation.from_euler("y", -80, degrees=True).as_matrix()
    reference[:, 19] = Rotation.from_euler("y", 80, degrees=True).as_matrix()
    reference[:, 1] = Rotation.from_euler("yx", [30, -30], degrees=True).as_matrix()
    reference[:, 4] = Rotation.from_euler("x", 70, degrees=True).as_matrix()
    joints, _ = forward_kinematics(
        reference, np.broadcast_to(rest[0], (30, 3)), offsets, parents
    )
    result = estimate_local_rotations(joints, rest, parents)
    recovered, _ = forward_kinematics(result, joints[:, 0], offsets, parents)
    np.testing.assert_allclose(recovered, joints, atol=1e-6)
    selected = [1, 4, 16, 17, 18, 19]
    errors = Rotation.from_matrix(
        (result[:, selected] @ reference[:, selected].transpose(0, 1, 3, 2)).reshape(
            -1, 3, 3
        )
    ).magnitude()
    # Previous solver matched joints to 4e-8 m while missing rotations by 35 deg.
    assert np.rad2deg(errors).max() < 0.01


def test_torso_twist_is_not_concentrated_at_the_chest():
    model = GallerySMPL(
        Path(__file__).resolve().parents[1]
        / "docs/leaderboards/hf_space_t2m_humanml3d/cases/smpl_model"
    )
    rest = model.rest[:22].numpy()
    parents = model.parents[:22].numpy()
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    reference = np.tile(np.eye(3), (20, 22, 1, 1))
    for joint in [3, 6, 9]:
        reference[:, joint] = Rotation.from_euler(
            "y", np.linspace(0, 30, 20)[:, None], degrees=True
        ).as_matrix()
    joints, _ = forward_kinematics(
        reference, np.broadcast_to(rest[0], (20, 3)), offsets, parents
    )
    result = estimate_local_rotations(joints, rest, parents)
    errors = Rotation.from_matrix(
        (result[:, [3, 6, 9]] @ reference[:, [3, 6, 9]].transpose(0, 1, 3, 2)).reshape(
            -1, 3, 3
        )
    ).magnitude()
    # Old solve put ~82 degrees on the chest for three authored 30-degree turns.
    assert np.rad2deg(errors).max() < 0.1
