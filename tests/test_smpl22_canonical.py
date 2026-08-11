import numpy as np

from motius.motion import canonicalize_smpl22_joints


def _body(frames=12):
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, :, 1] = np.linspace(0.1, 1.1, 22)[None]
    joints[:, 1, 0] = 0.2
    joints[:, 2, 0] = -0.2
    joints[:, 16, 0] = 0.35
    joints[:, 17, 0] = -0.35
    joints[:, [7, 8, 10, 11], 1] = 0.0
    joints[:, :, 2] += np.linspace(4.0, 5.0, frames)[:, None]
    return joints


def test_canonicalize_smpl22_is_rigid_and_faces_positive_z():
    joints = _body()
    yaw = 0.8
    rotation = np.asarray(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    )
    transformed = joints @ rotation.T + np.asarray([3.0, 2.0, -7.0])
    expected = canonicalize_smpl22_joints(joints)
    actual = canonicalize_smpl22_joints(transformed)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert np.isclose(actual[:, [7, 8, 10, 11], 1].min(), 0.0)
    first_left = (actual[0, 1] - actual[0, 2]) + (actual[0, 16] - actual[0, 17])
    first_left[1] = 0.0
    first_left /= np.linalg.norm(first_left)
    np.testing.assert_allclose(first_left, [1.0, 0.0, 0.0], atol=2e-6)
    first_forward = np.cross(first_left, np.asarray([0.0, 1.0, 0.0]))
    root_travel = actual[-1, 0] - actual[0, 0]
    assert np.dot(first_forward, root_travel) > 0.0


def test_canonicalize_smpl22_preserves_flat_layout_and_jerk():
    joints = _body()
    original_jerk = np.diff(joints, n=3, axis=0)
    canonical = canonicalize_smpl22_joints(joints.reshape(len(joints), 66))
    assert canonical.shape == (len(joints), 66)
    canonical_jerk = np.diff(canonical.reshape(len(joints), 22, 3), n=3, axis=0)
    np.testing.assert_allclose(
        np.linalg.norm(canonical_jerk, axis=-1),
        np.linalg.norm(original_jerk, axis=-1),
        atol=2e-6,
        rtol=0,
    )
