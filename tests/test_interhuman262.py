import numpy as np

from motius.motion.representation import get_spec
from motius.motion.representation.interhuman262 import (
    interhuman262_to_joint_velocities,
    interhuman262_to_joints,
    interhuman262_to_local_rotmat,
    joints_pair_to_interhuman262,
    joints_to_interhuman262,
)


def _pair(frames=12):
    base = np.zeros((22, 3), dtype=np.float32)
    base[:, 1] = np.linspace(0.0, 1.6, 22)
    base[1] = [-0.15, 0.9, 0.0]
    base[2] = [0.15, 0.9, 0.0]
    base[7] = [-0.1, 0.06, 0.0]
    base[8] = [0.1, 0.06, 0.0]
    base[10] = [-0.1, 0.02, 0.1]
    base[11] = [0.1, 0.02, 0.1]
    time = np.linspace(0.0, 0.4, frames, dtype=np.float32)
    first = np.stack([base + [0.0, 0.0, value] for value in time])
    second = np.stack([base + [1.25, 0.0, 0.5 - value] for value in time])
    joints_y_up = np.stack([first, second], axis=1)
    joints_raw = joints_y_up[..., [0, 2, 1]].copy()
    joints_raw[..., 1] *= -1
    rotations = np.zeros((frames, 2, 21, 6), dtype=np.float32)
    rotations[..., 0] = 1.0
    rotations[..., 3] = 1.0
    return joints_raw, rotations


def test_interhuman_spec_and_channel_decode():
    assert get_spec("InterHuman-262").dim == 262
    motion = np.arange(3 * 262, dtype=np.float32).reshape(3, 262)
    np.testing.assert_array_equal(interhuman262_to_joints(motion).reshape(3, 66), motion[:, :66])
    np.testing.assert_array_equal(
        interhuman262_to_joint_velocities(motion).reshape(3, 66), motion[:, 66:132]
    )


def test_single_encoder_has_official_layout_and_length():
    joints, rotations = _pair()
    encoded = joints_to_interhuman262(joints[:, 0], rotations[:, 0])
    assert encoded.shape == (len(joints) - 1, 262)
    decoded = interhuman262_to_joints(encoded)
    velocities = interhuman262_to_joint_velocities(encoded)
    np.testing.assert_allclose(velocities[:-1], decoded[1:] - decoded[:-1], atol=1e-6)
    assert np.isfinite(encoded).all()


def test_pair_encoder_preserves_relative_placement():
    joints, rotations = _pair()
    encoded = joints_pair_to_interhuman262(joints, rotations)
    assert encoded.shape == (len(joints) - 1, 2, 262)
    positions = interhuman262_to_joints(encoded)
    relative_root = positions[:, 1, 0] - positions[:, 0, 0]
    assert np.linalg.norm(relative_root[:, [0, 2]], axis=-1).mean() > 0.8
    assert not np.allclose(positions[:, 0], positions[:, 1])


def test_interhuman_rotation_channels_use_official_row_interleaved_layout():
    quarter_turn_z = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    motion = np.zeros((2, 262), dtype=np.float32)
    row_interleaved = quarter_turn_z[:, :2].reshape(6)
    motion[:, 132:258] = np.tile(row_interleaved, 21)

    decoded = interhuman262_to_local_rotmat(motion, include_root=False)
    expected = np.broadcast_to(quarter_turn_z, decoded.shape)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


def test_motion135_interhuman_encoder_is_public():
    from motius.motion import motion135_to_interhuman262

    assert callable(motion135_to_interhuman262)
