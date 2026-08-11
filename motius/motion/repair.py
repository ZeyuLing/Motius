"""Training-free, source-only repair for SMPL-22 ``motion135`` sequences.

The default policy is the frozen rule used by the MotionRepair formal
evaluation: third-difference (D3) Whittaker smoothing of global rotations and
root translation, followed by a smooth prefix-ground correction.  Inference
uses only the corrupted motion and fixed skeleton offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Dict

import numpy as np

from .representation.rotation import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
from .skeleton.names import SMPL22_PARENTS


_JOINT_COUNT = 22
_MOTION135_DIM = 135
_QUATERNION_NORM_EPS = 1.0e-12
_GROUND_FOOT_JOINTS = (10, 11)


@dataclass(frozen=True)
class DenseD3RepairConfig:
    """Configuration for the frozen dense D3 source-only repair policy."""

    regularization: float = 3.0
    ground_quantile: float = 0.98
    prefix_frames: int = 10
    transition_frames: int = 30
    penetration_margin_m: float = 0.049
    support_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        scalar_fields = {
            "regularization": self.regularization,
            "ground_quantile": self.ground_quantile,
            "penetration_margin_m": self.penetration_margin_m,
            "support_tolerance": self.support_tolerance,
        }
        for name, value in scalar_fields.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {value!r}.")
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}.")
        integer_fields = {
            "prefix_frames": self.prefix_frames,
            "transition_frames": self.transition_frames,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}.")
        if float(self.regularization) < 0.0:
            raise ValueError("regularization must be non-negative.")
        if not 0.0 <= float(self.ground_quantile) <= 1.0:
            raise ValueError("ground_quantile must be in [0, 1].")
        if int(self.prefix_frames) < 1:
            raise ValueError("prefix_frames must be at least one.")
        if int(self.transition_frames) < 1:
            raise ValueError("transition_frames must be at least one.")
        if float(self.penetration_margin_m) < 0.0:
            raise ValueError("penetration_margin_m must be non-negative.")
        if float(self.support_tolerance) < 0.0:
            raise ValueError("support_tolerance must be non-negative.")

    @property
    def policy(self) -> Dict[str, object]:
        """Return the compact policy identity used in evaluation artifacts."""

        quantile = int(round(100.0 * self.ground_quantile))
        return {
            "lambda": float(self.regularization),
            "mode": "both",
            "ground": f"prefix_q{quantile:02d}",
        }


@dataclass(frozen=True)
class MotionRepairResult:
    """Repaired motion plus the exact dense support used by evaluation."""

    motion135: np.ndarray
    rotation_support: np.ndarray
    translation_support: np.ndarray
    ground_shift: np.ndarray
    config: DenseD3RepairConfig

    @property
    def policy(self) -> Dict[str, object]:
        return self.config.policy


@lru_cache(maxsize=None)
def _whittaker_system(length: int, regularization: float) -> np.ndarray:
    identity = np.eye(int(length), dtype=np.float64)
    third_difference = np.diff(identity, n=3, axis=0)
    system = identity + float(regularization) * (
        third_difference.T @ third_difference
    )
    system.setflags(write=False)
    return system


def _smooth_d3(value: np.ndarray, regularization: float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    flattened = source.reshape(len(source), -1)
    output = np.linalg.solve(
        _whittaker_system(len(source), float(regularization)),
        flattened,
    )
    return output.reshape(source.shape)


def _global_rotations(local_rotation: np.ndarray) -> np.ndarray:
    local = np.asarray(local_rotation, dtype=np.float64)
    global_rotation = np.empty_like(local)
    global_rotation[:, 0] = local[:, 0]
    for joint in range(1, _JOINT_COUNT):
        global_rotation[:, joint] = (
            global_rotation[:, SMPL22_PARENTS[joint]] @ local[:, joint]
        )
    return global_rotation


def _local_rotations(global_rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(global_rotation, dtype=np.float64)
    local_rotation = np.empty_like(value)
    local_rotation[:, 0] = value[:, 0]
    for joint in range(1, _JOINT_COUNT):
        parent = SMPL22_PARENTS[joint]
        local_rotation[:, joint] = (
            np.swapaxes(value[:, parent], -1, -2) @ value[:, joint]
        )
    return local_rotation


def _continuous_quaternions(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(matrix_to_quaternion(rotation), dtype=np.float64)
    output = quaternion.copy()
    for frame in range(1, len(output)):
        flip = np.sum(output[frame - 1] * output[frame], axis=-1) < 0.0
        output[frame, flip] *= -1.0
    return output


def _smooth_global_rotations(
    global_rotation: np.ndarray,
    regularization: float,
) -> np.ndarray:
    quaternion = _continuous_quaternions(global_rotation)
    filtered = _smooth_d3(quaternion, regularization)
    norm = np.linalg.norm(filtered, axis=-1, keepdims=True)
    if np.any(norm <= _QUATERNION_NORM_EPS) or not np.isfinite(norm).all():
        raise ValueError("Global D3 quaternion smoothing became singular.")
    return np.asarray(quaternion_to_matrix(filtered / norm), dtype=np.float64)


def _forward_kinematics(
    local_rotation: np.ndarray,
    translation: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64)
    bone_offsets = np.asarray(offsets, dtype=np.float64)
    global_rotation = np.empty_like(local)
    positions = np.empty((len(local), _JOINT_COUNT, 3), dtype=np.float64)
    global_rotation[:, 0] = local[:, 0]
    positions[:, 0] = trans + bone_offsets[0]
    for joint in range(1, _JOINT_COUNT):
        parent = SMPL22_PARENTS[joint]
        global_rotation[:, joint] = (
            global_rotation[:, parent] @ local[:, joint]
        )
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], bone_offsets[joint]
        )
    return positions


def _prefix_ground_shift(
    positions: np.ndarray,
    config: DenseD3RepairConfig,
) -> np.ndarray:
    vertical = np.asarray(positions, dtype=np.float64)[
        :, _GROUND_FOOT_JOINTS, 1
    ]
    prefix_stop = min(len(vertical), int(config.prefix_frames))
    baseline = float(np.mean(vertical[:prefix_stop]))
    tail = (
        baseline
        - float(config.penetration_margin_m)
        - vertical[prefix_stop:]
    ).reshape(-1)
    depth = (
        max(0.0, float(np.quantile(tail, float(config.ground_quantile))))
        if len(tail)
        else 0.0
    )
    shift = np.zeros(len(vertical), dtype=np.float64)
    shift[:prefix_stop] = -depth
    stop = min(
        len(shift),
        prefix_stop + int(config.transition_frames),
    )
    if stop > prefix_stop:
        x = np.arange(1, stop - prefix_stop + 1, dtype=np.float64) / float(
            config.transition_frames
        )
        x = np.clip(x, 0.0, 1.0)
        smoother = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
        shift[prefix_stop:stop] = -depth * (1.0 - smoother)
    return shift


def _validate_inputs(
    motion135: np.ndarray,
    bone_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    motion = np.asarray(motion135)
    if motion.ndim != 2 or motion.shape[1] != _MOTION135_DIM:
        raise ValueError(
            f"motion135 must have shape [T, {_MOTION135_DIM}], got {motion.shape}."
        )
    if len(motion) < 1:
        raise ValueError("motion135 must contain at least one frame.")
    if not np.issubdtype(motion.dtype, np.floating):
        raise TypeError(f"motion135 must use a floating dtype, got {motion.dtype}.")
    if not np.isfinite(motion).all():
        raise ValueError("motion135 contains non-finite values.")
    offsets = np.asarray(bone_offsets)
    if offsets.shape != (_JOINT_COUNT, 3):
        raise ValueError(
            f"bone_offsets must have shape [{_JOINT_COUNT}, 3], got {offsets.shape}."
        )
    if not np.issubdtype(offsets.dtype, np.number) or not np.isfinite(offsets).all():
        raise ValueError("bone_offsets must be finite numeric values.")
    return motion, np.asarray(offsets, dtype=np.float64)


def repair_motion135(
    motion135: np.ndarray,
    bone_offsets: np.ndarray,
    config: DenseD3RepairConfig | None = None,
) -> MotionRepairResult:
    """Repair a Y-up, metre-scale SMPL-22 ``motion135`` sequence.

    Args:
        motion135: ``[T, 135]`` array containing root XYZ translation followed
            by 22 local rotations in Motius' row-convention 6D layout.
        bone_offsets: Fixed SMPL-22 offsets with shape ``[22, 3]`` in metres.
        config: Optional policy configuration.  Defaults reproduce the frozen
            formal-evaluation policy ``lambda=3, mode=both, ground=prefix_q98``.

    Returns:
        A :class:`MotionRepairResult`. ``rotation_support`` has shape
        ``[T, 22]`` and ``translation_support`` has shape ``[T]``.
    """

    resolved_config = config or DenseD3RepairConfig()
    if not isinstance(resolved_config, DenseD3RepairConfig):
        raise TypeError("config must be a DenseD3RepairConfig instance or None.")
    source_motion, offsets = _validate_inputs(motion135, bone_offsets)
    length = len(source_motion)
    source_rotation = np.asarray(
        rotation_6d_to_matrix(
            source_motion[:, 3:].reshape(length, _JOINT_COUNT, 6),
            convention="row",
        ),
        dtype=np.float64,
    )
    source_translation = np.asarray(source_motion[:, :3], dtype=np.float64)

    repaired_rotation = _local_rotations(
        _smooth_global_rotations(
            _global_rotations(source_rotation),
            resolved_config.regularization,
        )
    )
    repaired_translation = _smooth_d3(
        source_translation,
        resolved_config.regularization,
    )
    repaired_positions = _forward_kinematics(
        repaired_rotation,
        repaired_translation,
        offsets,
    )
    ground_shift = _prefix_ground_shift(repaired_positions, resolved_config)
    repaired_translation[:, 1] += ground_shift

    tolerance = float(resolved_config.support_tolerance)
    rotation_support = np.any(
        np.abs(repaired_rotation - source_rotation) > tolerance,
        axis=(-2, -1),
    )
    translation_support = np.any(
        np.abs(repaired_translation - source_translation) > tolerance,
        axis=-1,
    )
    repaired_motion = source_motion.copy()
    repaired_motion[:, :3] = repaired_translation.astype(
        source_motion.dtype,
        copy=False,
    )
    repaired_motion[:, 3:] = np.asarray(
        matrix_to_rotation_6d(repaired_rotation, convention="row"),
        dtype=source_motion.dtype,
    ).reshape(length, -1)
    repaired_rotation_6d = repaired_motion[:, 3:].reshape(
        length,
        _JOINT_COUNT,
        6,
    )
    source_rotation_6d = source_motion[:, 3:].reshape(
        length,
        _JOINT_COUNT,
        6,
    )
    repaired_rotation_6d[~rotation_support] = source_rotation_6d[
        ~rotation_support
    ]
    repaired_motion[~translation_support, :3] = source_motion[
        ~translation_support,
        :3,
    ]
    if not np.isfinite(repaired_motion).all():
        raise ValueError("Motion repair produced non-finite values.")

    return MotionRepairResult(
        motion135=repaired_motion,
        rotation_support=rotation_support,
        translation_support=translation_support,
        ground_shift=ground_shift,
        config=resolved_config,
    )


__all__ = [
    "DenseD3RepairConfig",
    "MotionRepairResult",
    "repair_motion135",
]
