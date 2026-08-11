from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motius.trainers.beyondmimic.motion_contract import load_motion_npz


BODY_NAMES = ("pelvis", "left_foot", "right_foot")
JOINT_NAMES = ("hip", "knee")


def _write_motion(
    path: Path,
    *,
    body_count: int,
    body_names: tuple[str, ...] | None = None,
    joint_names: tuple[str, ...] = JOINT_NAMES,
) -> None:
    frames = 4
    payload = {
        "fps": np.float32(50.0),
        "joint_pos": np.zeros((frames, 2), dtype=np.float32),
        "joint_vel": np.zeros((frames, 2), dtype=np.float32),
        "body_pos_w": np.zeros((frames, body_count, 3), dtype=np.float32),
        "body_quat_w": np.zeros((frames, body_count, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros(
            (frames, body_count, 3), dtype=np.float32
        ),
        "body_ang_vel_w": np.zeros(
            (frames, body_count, 3), dtype=np.float32
        ),
        "joint_names": np.asarray(joint_names),
    }
    payload["body_quat_w"][..., 0] = 1.0
    if body_names is not None:
        payload["body_names"] = np.asarray(body_names)
    np.savez(path, **payload)


def test_named_compact_motion_uses_names_instead_of_robot_indexes(tmp_path):
    motion = tmp_path / "compact.npz"
    _write_motion(
        motion,
        body_count=3,
        body_names=("right_foot", "pelvis", "left_foot"),
    )

    _, indexes = load_motion_npz(
        motion,
        robot_body_indexes=(0, 5, 9),
        target_body_names=BODY_NAMES,
        target_joint_names=JOINT_NAMES,
    )

    assert indexes == (1, 2, 0)


def test_named_motion_reorders_joints_to_robot_order(tmp_path):
    motion = tmp_path / "reordered_joints.npz"
    _write_motion(
        motion,
        body_count=3,
        body_names=BODY_NAMES,
        joint_names=("knee", "hip"),
    )
    with np.load(motion) as source:
        payload = {key: source[key] for key in source.files}
    payload["joint_pos"][:, 0] = 1.0
    payload["joint_pos"][:, 1] = 2.0
    np.savez(motion, **payload)

    arrays, _ = load_motion_npz(
        motion,
        robot_body_indexes=(0, 5, 9),
        target_body_names=BODY_NAMES,
        target_joint_names=JOINT_NAMES,
    )

    np.testing.assert_array_equal(arrays["joint_pos"][0], (2.0, 1.0))


def test_official_full_body_motion_uses_robot_indexes(tmp_path):
    motion = tmp_path / "full.npz"
    _write_motion(motion, body_count=10)

    _, indexes = load_motion_npz(
        motion,
        robot_body_indexes=(0, 5, 9),
        target_body_names=BODY_NAMES,
        target_joint_names=JOINT_NAMES,
    )

    assert indexes == (0, 5, 9)


def test_motion_contract_rejects_incompatible_compact_layout(tmp_path):
    motion = tmp_path / "bad.npz"
    _write_motion(
        motion,
        body_count=2,
        body_names=("pelvis", "left_foot"),
    )

    with pytest.raises(ValueError, match="missing tracked bodies"):
        load_motion_npz(
            motion,
            robot_body_indexes=(0, 5, 9),
            target_body_names=BODY_NAMES,
            target_joint_names=JOINT_NAMES,
        )


def test_motion_contract_rejects_nonfinite_values(tmp_path):
    motion = tmp_path / "nonfinite.npz"
    _write_motion(motion, body_count=10)
    with np.load(motion) as source:
        payload = {key: source[key] for key in source.files}
    payload["joint_pos"][0, 0] = np.nan
    np.savez(motion, **payload)

    with pytest.raises(ValueError, match="non-finite"):
        load_motion_npz(
            motion,
            robot_body_indexes=(0, 5, 9),
            target_body_names=BODY_NAMES,
            target_joint_names=JOINT_NAMES,
        )
