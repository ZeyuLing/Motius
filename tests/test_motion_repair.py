"""Parity and contract tests for source-only D3 MotionRepair."""

from __future__ import annotations

import inspect

from mmengine.config import Config
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motius.motion.repair import (
    DenseD3RepairConfig,
    MotionRepairResult,
    repair_motion135,
)
from motius.motion.representation.rotation import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from motius.motion.skeleton.names import SMPL22_PARENTS
from motius.pipelines.motionrepair import MotionRepairPipeline
from motius.registry import PIPELINES


def _synthetic_case(length: int = 64) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 1.0, length, dtype=np.float64)
    joint = np.arange(22, dtype=np.float64)[None, :]
    rotvec = np.zeros((length, 22, 3), dtype=np.float64)
    rotvec[..., 0] = 0.08 * np.sin(2.0 * np.pi * time[:, None] + joint / 9.0)
    rotvec[..., 1] = 0.06 * np.cos(3.0 * np.pi * time[:, None] + joint / 7.0)
    rotvec[..., 2] = 0.05 * np.sin(5.0 * np.pi * time[:, None] + joint / 5.0)
    rotation = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    rotation = rotation.reshape(length, 22, 3, 3)
    translation = np.stack(
        [
            0.3 * time,
            0.95 + 0.015 * np.sin(4.0 * np.pi * time),
            0.1 * np.cos(2.0 * np.pi * time),
        ],
        axis=-1,
    )
    motion = np.empty((length, 135), dtype=np.float32)
    motion[:, :3] = translation
    motion[:, 3:] = matrix_to_rotation_6d(
        rotation,
        convention="row",
    ).reshape(length, -1)

    offsets = np.zeros((22, 3), dtype=np.float32)
    for index in range(1, 22):
        offsets[index] = (
            0.015 * (-1.0 if index % 2 else 1.0),
            -0.08 if index in (4, 5, 7, 8, 10, 11) else 0.055,
            0.01 * ((index % 3) - 1),
        )
    return motion, offsets


def _reference_repair(
    motion: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent SciPy-xyzw transcription of the frozen paper policy."""

    length = len(motion)
    source_rotation = np.asarray(
        rotation_6d_to_matrix(
            motion[:, 3:].reshape(length, 22, 6),
            convention="row",
        ),
        dtype=np.float64,
    )
    source_translation = np.asarray(motion[:, :3], dtype=np.float64)
    global_rotation = np.empty_like(source_rotation)
    global_rotation[:, 0] = source_rotation[:, 0]
    for index in range(1, 22):
        global_rotation[:, index] = (
            global_rotation[:, SMPL22_PARENTS[index]] @ source_rotation[:, index]
        )

    quaternion = Rotation.from_matrix(global_rotation.reshape(-1, 3, 3))
    quaternion = quaternion.as_quat().reshape(length, 22, 4)
    for frame in range(1, length):
        flip = np.sum(quaternion[frame - 1] * quaternion[frame], axis=-1) < 0.0
        quaternion[frame, flip] *= -1.0
    identity = np.eye(length, dtype=np.float64)
    third = np.diff(identity, n=3, axis=0)
    system = identity + 3.0 * (third.T @ third)
    filtered = np.linalg.solve(system, quaternion.reshape(length, -1))
    filtered = filtered.reshape(quaternion.shape)
    filtered /= np.linalg.norm(filtered, axis=-1, keepdims=True)
    smooth_global = Rotation.from_quat(filtered.reshape(-1, 4)).as_matrix()
    smooth_global = smooth_global.reshape(length, 22, 3, 3)
    repaired_rotation = np.empty_like(smooth_global)
    repaired_rotation[:, 0] = smooth_global[:, 0]
    for index in range(1, 22):
        parent = SMPL22_PARENTS[index]
        repaired_rotation[:, index] = (
            np.swapaxes(smooth_global[:, parent], -1, -2)
            @ smooth_global[:, index]
        )

    repaired_translation = np.linalg.solve(system, source_translation)
    global_rotation[:, 0] = repaired_rotation[:, 0]
    positions = np.empty((length, 22, 3), dtype=np.float64)
    positions[:, 0] = repaired_translation + offsets[0]
    for index in range(1, 22):
        parent = SMPL22_PARENTS[index]
        global_rotation[:, index] = (
            global_rotation[:, parent] @ repaired_rotation[:, index]
        )
        positions[:, index] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], offsets[index]
        )
    vertical = positions[:, (10, 11), 1]
    baseline = float(np.mean(vertical[:10]))
    depth = max(
        0.0,
        float(np.quantile((baseline - 0.049 - vertical[10:]).reshape(-1), 0.98)),
    )
    shift = np.zeros(length, dtype=np.float64)
    shift[:10] = -depth
    stop = min(length, 40)
    x = np.arange(1, stop - 10 + 1, dtype=np.float64) / 30.0
    smoother = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
    shift[10:stop] = -depth * (1.0 - smoother)
    repaired_translation[:, 1] += shift

    rotation_support = np.any(
        np.abs(repaired_rotation - source_rotation) > 1.0e-12,
        axis=(-2, -1),
    )
    translation_support = np.any(
        np.abs(repaired_translation - source_translation) > 1.0e-12,
        axis=-1,
    )
    repaired = motion.copy()
    repaired[:, :3] = repaired_translation.astype(np.float32)
    repaired[:, 3:] = matrix_to_rotation_6d(
        repaired_rotation,
        convention="row",
    ).reshape(length, -1).astype(np.float32)
    return repaired, rotation_support, translation_support, shift


def test_default_policy_matches_frozen_reference() -> None:
    motion, offsets = _synthetic_case()
    expected, expected_rotation, expected_translation, expected_shift = (
        _reference_repair(motion, offsets)
    )

    result = repair_motion135(motion, offsets)

    assert result.policy == {
        "lambda": 3.0,
        "mode": "both",
        "ground": "prefix_q98",
    }
    assert result.motion135.dtype == motion.dtype
    assert np.allclose(result.motion135, expected, rtol=0.0, atol=2.0e-6)
    assert np.array_equal(result.rotation_support, expected_rotation)
    assert np.array_equal(result.translation_support, expected_translation)
    assert np.allclose(result.ground_shift, expected_shift, rtol=0.0, atol=1.0e-12)


def test_pipeline_registry_and_direct_api_are_identical() -> None:
    motion, offsets = _synthetic_case(length=45)
    cfg = Config.fromfile("configs/motionrepair/d3_source_only.py")
    pipeline = PIPELINES.build(cfg.pipeline)

    direct = repair_motion135(motion, offsets)
    actual = pipeline.infer_motion_repair(motion, offsets)

    assert isinstance(pipeline, MotionRepairPipeline)
    assert isinstance(actual, MotionRepairResult)
    assert np.array_equal(actual.motion135, direct.motion135)
    assert np.array_equal(actual.rotation_support, direct.rotation_support)
    assert np.array_equal(actual.translation_support, direct.translation_support)
    assert np.array_equal(actual.ground_shift, direct.ground_shift)


def test_short_sequence_is_supported_without_ground_tail() -> None:
    motion, offsets = _synthetic_case(length=8)
    result = repair_motion135(motion, offsets)

    assert result.motion135.shape == (8, 135)
    assert np.array_equal(result.ground_shift, np.zeros(8, dtype=np.float64))
    assert np.isfinite(result.motion135).all()


def test_values_outside_reported_support_are_source_exact() -> None:
    motion, offsets = _synthetic_case(length=3)
    result = repair_motion135(motion, offsets)
    repaired_rotation = result.motion135[:, 3:].reshape(3, 22, 6)
    source_rotation = motion[:, 3:].reshape(3, 22, 6)

    assert np.array_equal(
        repaired_rotation[~result.rotation_support],
        source_rotation[~result.rotation_support],
    )
    assert np.array_equal(
        result.motion135[~result.translation_support, :3],
        motion[~result.translation_support, :3],
    )


def test_public_api_has_no_clean_target_or_label_input() -> None:
    parameters = inspect.signature(repair_motion135).parameters
    assert tuple(parameters) == ("motion135", "bone_offsets", "config")
    assert "clean" not in parameters
    assert "label" not in parameters


@pytest.mark.parametrize(
    ("motion_shape", "offset_shape", "error"),
    [
        ((2, 134), (22, 3), ValueError),
        ((0, 135), (22, 3), ValueError),
        ((2, 135), (21, 3), ValueError),
    ],
)
def test_invalid_shapes_are_rejected(motion_shape, offset_shape, error) -> None:
    with pytest.raises(error):
        repair_motion135(
            np.zeros(motion_shape, dtype=np.float32),
            np.zeros(offset_shape, dtype=np.float32),
        )


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="regularization"):
        DenseD3RepairConfig(regularization=-1.0)
    with pytest.raises(ValueError, match="ground_quantile"):
        DenseD3RepairConfig(ground_quantile=1.1)
    with pytest.raises(TypeError, match="prefix_frames"):
        DenseD3RepairConfig(prefix_frames=10.5)
