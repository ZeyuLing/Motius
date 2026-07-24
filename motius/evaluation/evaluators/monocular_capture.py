"""Protocol-locked monocular motion-capture metrics.

The global metrics intentionally follow the public GVHMR/WHAM implementation:
100-frame chunks, first-two-frame alignment for W-MPJPE, full-chunk alignment
for WA-MPJPE, and the published RTE/jitter/foot-sliding definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from motius.motion.representation.monocular_joints import select_common_hmr15


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, points, 3).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _align_pcl(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    fixed_scale: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment matching GVHMR's ``align_pcl`` convention."""

    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 3)
    if target.shape != prediction.shape or target.shape[0] < 3:
        raise ValueError("Alignment inputs must share at least three 3D points.")
    target_mean = target.mean(axis=0)
    prediction_mean = prediction.mean(axis=0)
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean
    covariance = target_centered.T @ prediction_centered / target.shape[0]
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh.T) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    if fixed_scale:
        scale = 1.0
    else:
        variance = np.square(prediction_centered).sum() / target.shape[0]
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("Cannot align a zero-variance prediction.")
        scale = float(np.dot(singular_values, sign) / variance)
    translation = target_mean - scale * (rotation @ prediction_mean)
    return scale, rotation, translation


def _apply_alignment(
    points: np.ndarray,
    alignment: tuple[float, np.ndarray, np.ndarray],
) -> np.ndarray:
    scale, rotation, translation = alignment
    return scale * np.einsum("ij,...j->...i", rotation, points) + translation


def _joint_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=-1).mean(axis=-1)


def _pelvis(
    joints: np.ndarray,
    pelvis_indices: Sequence[int],
) -> np.ndarray:
    indices = np.asarray(tuple(pelvis_indices), dtype=np.int64)
    if not len(indices) or indices.min() < 0 or indices.max() >= joints.shape[1]:
        raise ValueError("pelvis_indices are outside the joint array.")
    return joints[:, indices].mean(axis=1, keepdims=True)


def _masked(
    array: Optional[np.ndarray],
    valid: np.ndarray,
) -> Optional[np.ndarray]:
    return None if array is None else array[valid]


@dataclass(frozen=True)
class MonocularMetricResult:
    """Per-sequence vectors plus scalar means ready for aggregation."""

    per_frame: dict[str, np.ndarray]
    means: dict[str, float]
    protocol: dict[str, object]


def evaluate_camera_coordinates(
    prediction_joints: np.ndarray,
    target_joints: np.ndarray,
    *,
    prediction_vertices: Optional[np.ndarray] = None,
    target_vertices: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    pelvis_indices: Sequence[int] = (1, 2),
    fps: float = 30.0,
) -> MonocularMetricResult:
    """Compute PA-MPJPE, MPJPE, optional PVE, and acceleration error."""

    prediction_joints = _points(prediction_joints, "prediction_joints")
    target_joints = _points(target_joints, "target_joints")
    if prediction_joints.shape != target_joints.shape:
        raise ValueError("Camera-coordinate joint arrays must share a shape.")
    frames = len(prediction_joints)
    valid_mask = (
        np.ones(frames, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid_mask.shape != (frames,):
        raise ValueError("valid must align to the frame axis.")
    prediction_joints = prediction_joints[valid_mask]
    target_joints = target_joints[valid_mask]
    if len(prediction_joints) == 0:
        raise ValueError("No valid frames remain for camera evaluation.")

    prediction_pelvis = _pelvis(prediction_joints, pelvis_indices)
    target_pelvis = _pelvis(target_joints, pelvis_indices)
    prediction_aligned = prediction_joints - prediction_pelvis
    target_aligned = target_joints - target_pelvis
    per_frame: dict[str, np.ndarray] = {
        "mpjpe_mm": _joint_error(prediction_aligned, target_aligned) * 1000.0
    }
    pa_prediction = np.stack(
        [
            _apply_alignment(prediction, _align_pcl(target, prediction))
            for prediction, target in zip(prediction_aligned, target_aligned)
        ],
        axis=0,
    )
    per_frame["pa_mpjpe_mm"] = (
        _joint_error(pa_prediction, target_aligned) * 1000.0
    )

    if (prediction_vertices is None) != (target_vertices is None):
        raise ValueError("PVE requires both prediction and target vertices.")
    if prediction_vertices is not None:
        prediction_vertices = _points(prediction_vertices, "prediction_vertices")
        target_vertices = _points(target_vertices, "target_vertices")
        if prediction_vertices.shape != target_vertices.shape:
            raise ValueError("Camera-coordinate vertex arrays must share a shape.")
        prediction_vertices = prediction_vertices[valid_mask] - prediction_pelvis
        target_vertices = target_vertices[valid_mask] - target_pelvis
        per_frame["pve_mm"] = (
            _joint_error(prediction_vertices, target_vertices) * 1000.0
        )

    if len(prediction_aligned) >= 3:
        prediction_acceleration = (
            prediction_aligned[:-2]
            - 2.0 * prediction_aligned[1:-1]
            + prediction_aligned[2:]
        )
        target_acceleration = (
            target_aligned[:-2]
            - 2.0 * target_aligned[1:-1]
            + target_aligned[2:]
        )
        per_frame["accel_mps2"] = (
            _joint_error(prediction_acceleration, target_acceleration) * fps**2
        )
    means = {
        name: float(values.mean())
        for name, values in per_frame.items()
        if len(values)
    }
    return MonocularMetricResult(
        per_frame=per_frame,
        means=means,
        protocol={
            "revision": "motius_camera_v1_gvhmr_parity",
            "pelvis_indices": list(pelvis_indices),
            "fps": float(fps),
            "units": {
                "pa_mpjpe_mm": "millimeter",
                "mpjpe_mm": "millimeter",
                "pve_mm": "millimeter",
                "accel_mps2": "meter_per_second_squared",
            },
        },
    )


def evaluate_global_coordinates(
    prediction_joints: np.ndarray,
    target_joints: np.ndarray,
    *,
    prediction_vertices: Optional[np.ndarray] = None,
    target_vertices: Optional[np.ndarray] = None,
    valid: Optional[np.ndarray] = None,
    fps: float = 30.0,
    chunk_frames: int = 100,
) -> MonocularMetricResult:
    """Compute official GVHMR/WHAM-style world-coordinate metrics."""

    prediction_joints = _points(prediction_joints, "prediction_joints")
    target_joints = _points(target_joints, "target_joints")
    if prediction_joints.shape != target_joints.shape:
        raise ValueError("World-coordinate joint arrays must share a shape.")
    frames = len(prediction_joints)
    valid_mask = (
        np.ones(frames, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if valid_mask.shape != (frames,):
        raise ValueError("valid must align to the frame axis.")
    prediction_joints = prediction_joints[valid_mask]
    target_joints = target_joints[valid_mask]
    if len(prediction_joints) < 2:
        raise ValueError("Global evaluation requires at least two valid frames.")

    w_mpjpe, wa_mpjpe = [], []
    for start in range(0, len(prediction_joints), chunk_frames):
        prediction = prediction_joints[start : start + chunk_frames]
        target = target_joints[start : start + chunk_frames]
        first_count = min(2, len(prediction))
        first_alignment = _align_pcl(
            target[:first_count],
            prediction[:first_count],
        )
        global_alignment = _align_pcl(target, prediction)
        w_mpjpe.append(
            _joint_error(_apply_alignment(prediction, first_alignment), target)
        )
        wa_mpjpe.append(
            _joint_error(_apply_alignment(prediction, global_alignment), target)
        )
    per_frame: dict[str, np.ndarray] = {
        "w_mpjpe_mm": np.concatenate(w_mpjpe) * 1000.0,
        "wa_mpjpe_mm": np.concatenate(wa_mpjpe) * 1000.0,
    }

    root_target = target_joints[:, 0]
    root_prediction = prediction_joints[:, 0]
    rigid_alignment = _align_pcl(
        root_target,
        root_prediction,
        fixed_scale=True,
    )
    aligned_root = _apply_alignment(root_prediction, rigid_alignment)
    displacement = np.linalg.norm(np.diff(root_target, axis=0), axis=-1).sum()
    if displacement > np.finfo(np.float64).eps:
        per_frame["rte_percent"] = (
            np.linalg.norm(root_target - aligned_root, axis=-1)
            / displacement
            * 100.0
        )

    if len(prediction_joints) >= 4:
        third_difference = (
            prediction_joints[3:]
            - 3.0 * prediction_joints[2:-1]
            + 3.0 * prediction_joints[1:-2]
            - prediction_joints[:-3]
        )
        per_frame["jitter_mps3"] = (
            np.linalg.norm(third_difference * fps**3, axis=-1).mean(axis=-1)
            / 10.0
        )

    if (prediction_vertices is None) != (target_vertices is None):
        raise ValueError(
            "Foot sliding requires both prediction and target vertices."
        )
    if prediction_vertices is not None:
        prediction_vertices = _points(prediction_vertices, "prediction_vertices")
        target_vertices = _points(target_vertices, "target_vertices")
        if prediction_vertices.shape != target_vertices.shape:
            raise ValueError("World-coordinate vertex arrays must share a shape.")
        prediction_vertices = prediction_vertices[valid_mask]
        target_vertices = target_vertices[valid_mask]
        if prediction_vertices.shape[1] != 6890:
            raise ValueError("GVHMR foot sliding requires 6,890 SMPL vertices.")
        foot_indices = np.asarray([3216, 3387, 6617, 6787])
        target_displacement = np.linalg.norm(
            np.diff(target_vertices[:, foot_indices], axis=0),
            axis=-1,
        )
        contact = target_displacement < 1e-2
        prediction_displacement = np.linalg.norm(
            np.diff(prediction_vertices[:, foot_indices], axis=0),
            axis=-1,
        )
        per_frame["foot_sliding_mm"] = prediction_displacement[contact] * 1000.0

    means = {
        name: float(values.mean())
        for name, values in per_frame.items()
        if len(values)
    }
    return MonocularMetricResult(
        per_frame=per_frame,
        means=means,
        protocol={
            "revision": "motius_global_v1_gvhmr_wham_parity",
            "chunk_frames": int(chunk_frames),
            "fps": float(fps),
            "w_alignment": "similarity_first_two_frames_per_chunk",
            "wa_alignment": "similarity_all_frames_per_chunk",
            "rte_alignment": "rigid_full_root_trajectory",
            "jitter_scale": "gvhmr_divide_by_10",
        },
    )


def evaluate_common_joint_coordinates(
    prediction_joints: np.ndarray,
    prediction_joint_names: Sequence[str],
    prediction_body_model: str,
    target_joints: np.ndarray,
    target_joint_names: Sequence[str],
    target_body_model: str,
    *,
    space: str,
    valid: Optional[np.ndarray] = None,
    fps: float = 30.0,
) -> MonocularMetricResult:
    """Evaluate unlike body models only on the audited named 15-joint subset."""

    prediction_common = select_common_hmr15(
        prediction_joints,
        prediction_joint_names,
        body_model=prediction_body_model,
    )
    target_common = select_common_hmr15(
        target_joints,
        target_joint_names,
        body_model=target_body_model,
    )
    if space == "camera":
        result = evaluate_camera_coordinates(
            prediction_common,
            target_common,
            valid=valid,
            pelvis_indices=(0,),
            fps=fps,
        )
    elif space == "world":
        result = evaluate_global_coordinates(
            prediction_common,
            target_common,
            valid=valid,
            fps=fps,
        )
    else:
        raise ValueError("space must be camera or world.")
    protocol = dict(result.protocol)
    protocol.update(
        {
            "joint_protocol": "common_hmr15_named_v1",
            "prediction_body_model": prediction_body_model,
            "target_body_model": target_body_model,
            "mesh_metrics_available": False,
        }
    )
    return MonocularMetricResult(
        per_frame=result.per_frame,
        means=result.means,
        protocol=protocol,
    )


__all__ = [
    "MonocularMetricResult",
    "evaluate_camera_coordinates",
    "evaluate_common_joint_coordinates",
    "evaluate_global_coordinates",
]
