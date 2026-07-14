import numpy as np

from motius.motion.representation import motion135_to_motion272
from motius.motion.representation.babel135 import (
    babel135_to_joints,
    babel135_to_motion135,
    babel_rows6d_to_matrix,
    encode_babel135,
    infer_smpl22_offsets,
    matrix_to_babel_rows6d,
)
from motius.motion.representation.rotation import axis_angle_to_matrix, matrix_to_axis_angle
from motius.motion.skeleton.fk import forward_kinematics
from motius.motion.skeleton.names import SMPL22_PARENTS


def _synthetic_sequence(frames=12):
    rng = np.random.default_rng(7)
    poses = rng.normal(scale=0.08, size=(frames, 22, 3))
    poses[:, 0, 2] += 0.35
    trans = np.zeros((frames, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(1.2, 1.8, frames)
    trans[:, 1] = np.linspace(-0.4, 0.1, frames)
    trans[:, 2] = 0.92 + np.sin(np.linspace(0, np.pi, frames)) * 0.02

    offsets = np.zeros((22, 3), dtype=np.float64)
    for joint, parent in enumerate(SMPL22_PARENTS):
        if parent >= 0:
            offsets[joint] = np.asarray(
                [0.015 * ((joint % 3) - 1), 0.01 * (joint % 2), 0.045 + joint * 0.001]
            )
    local = axis_angle_to_matrix(poses.reshape(-1, 3)).reshape(frames, 22, 3, 3)

    import torch

    joints, _ = forward_kinematics(
        torch.from_numpy(local),
        torch.from_numpy(trans),
        torch.from_numpy(offsets),
    )
    return poses, trans, offsets, joints.numpy()


def test_babel_rotation_layout_roundtrip():
    rng = np.random.default_rng(4)
    matrix = axis_angle_to_matrix(rng.normal(scale=0.2, size=(9, 3)))
    encoded = matrix_to_babel_rows6d(matrix)
    np.testing.assert_allclose(babel_rows6d_to_matrix(encoded), matrix, atol=1e-6)


def test_babel135_roundtrip_to_canonical_y_up_joints():
    poses, trans, offsets_z, joints_z = _synthetic_sequence()
    encoded = encode_babel135(poses, trans)
    offsets_y = infer_smpl22_offsets(poses, trans, joints_z, target_up_axis="y")
    predicted = babel135_to_joints(encoded, bone_offsets=offsets_y)

    root_matrix = axis_angle_to_matrix(poses[0, 0])
    root_axis_angle = np.asarray(matrix_to_axis_angle(root_matrix))
    canonicalizer = axis_angle_to_matrix(
        np.asarray([0.0, 0.0, root_axis_angle[2] + np.pi / 2])
    )
    origin = np.asarray([trans[0, 0], trans[0, 1], 0.0])
    canonical_z = (joints_z - origin) @ canonicalizer
    z_to_y = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    expected = canonical_z @ z_to_y.T
    np.testing.assert_allclose(predicted, expected, atol=2e-5, rtol=0)

    motion135 = babel135_to_motion135(encoded)
    assert motion135.shape == (len(poses), 135)
    assert not np.allclose(encoded[:, :3], motion135[:, :3])


def test_babel135_normalization_is_exact():
    poses, trans, _, _ = _synthetic_sequence(frames=5)
    mean = np.linspace(-0.2, 0.2, 135)
    std = np.linspace(0.5, 1.5, 135)
    raw = encode_babel135(poses, trans)
    normalized = encode_babel135(poses, trans, mean=mean, std=std)
    motion_raw = babel135_to_motion135(raw)
    motion_normalized = babel135_to_motion135(normalized, mean=mean, std=std)
    np.testing.assert_allclose(motion_normalized, motion_raw, atol=1e-6, rtol=0)


def test_motion135_to_motion272_is_exported() -> None:
    assert callable(motion135_to_motion272)
