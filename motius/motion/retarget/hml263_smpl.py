"""HumanML3D-263 to SMPL ``motion_135`` retargeting API.

This module exposes the public library wrapper for the repository's validated
HML263 -> SMPL chain. The conversion recovers HumanML3D joint positions, solves
the articulated SMPL chain from those positions, and refines against all
recovered 22 joints. HumanML3D local rotations are never used to construct SMPL poses.

The implementation shares one in-package IK/FK backend with the public API so
that evaluation and standalone conversion use identical math.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from ._hml263_smpl_impl import (
    N_JOINTS,
    fit_length_linear,
    load_smpl_rest as _load_smpl_rest,
    matrix_to_rot6d_rowmajor,
    recover_from_ric,
    refine_smpl_fit,
    resample_linear,
    smpl_forward_22,
    smpl_motion_integrity_metrics,
    smpl_mesh_floor_offset,
    validate_smpl_motion_integrity,
)

from .position_ik import estimate_local_rotations


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_smpl_rest(
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    gender: str = "neutral",
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Load a zero-shape SMPL model, rest joints, and parent chain.

    Args:
        model_dir: SMPL model directory. When omitted, reads
            ``MOTIUS_SMPL_MODEL_DIR``.
        device: Torch device for the SMPL layer.
        gender: SMPL body-model gender to load.
    """

    resolved = model_dir or os.environ.get("MOTIUS_SMPL_MODEL_DIR")
    if not resolved:
        raise FileNotFoundError(
            "SMPL model assets are required; pass model_dir=... or set MOTIUS_SMPL_MODEL_DIR"
        )
    return _load_smpl_rest(Path(resolved), _as_device(device), gender=gender)


@torch.inference_mode(False)
def retarget_hml263_clip(
    feats: np.ndarray,
    *,
    smpl_rest: tuple[Any, np.ndarray, np.ndarray] | None = None,
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    gender: str = "neutral",
    source_fps: float = 20.0,
    target_fps: float = 30.0,
    batch_size: int = 256,
    floor_align: bool = False,
    refine_iters: int = 0,
    refine_lr: float = 2e-2,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
    temporal_twist_stabilization: bool = True,
    anatomical_fit_iters: int = 300,
    anatomical_prior_weight: float = 1e-3,
    pose_keep_weight: float = 1e-4,
    pose_l2_weight: float = 0.0,
    angle_prior_weight: float = 0.0,
    target_len: int | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    source_motion135_transl: np.ndarray | None = None,
    root_translation_mode: str = "auto",
    rot6d_convention: str = "row",
    lock_global_orient: bool | None = None,
    lock_body_joint_ids: Iterable[int] | None = None,
    compute_mesh_metrics: bool = False,
    mesh_metric_samples: int = 16,
) -> dict[str, Any]:
    """Retarget one un-normalized HML263 clip to SMPL ``motion_135``.

    The returned ``motion_135`` is ROW-major rot6d:
    ``[root_translation(3), 22 * rot6d_row(132)]`` at ``target_fps``.

    All inputs are decoded to positions before the shared position IK solve.
    Shared position IK initializes a regularized whole-chain fit. Robust
    landmark residuals downweight outliers, and temporal penalties use
    rotation matrices to keep the fit continuous across axis-angle wraps.
    Both fitting stages preserve the root trajectory before optional grounding.
    Set anatomical_fit_iters=0 for the unregularized diagnostic initialization.
    """

    if rot6d_convention != "row":
        raise ValueError("hml263_smpl public API only emits ROW-major motion_135")
    if orientation_mode not in {"bone", "parent_frame"}:
        raise ValueError(f"unsupported orientation_mode: {orientation_mode}")
    if root_translation_mode not in {"auto", "canonical", "source_transl"}:
        raise ValueError(f"unsupported root_translation_mode: {root_translation_mode}")
    device_t = _as_device(device)
    model, rest_joints, parents = smpl_rest or load_smpl_rest(
        model_dir, device_t, gender=gender
    )

    arr = np.asarray(feats, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (N_JOINTS, 3):
        target = resample_linear(arr, source_fps, target_fps)
    else:
        if arr.ndim != 2 or arr.shape[-1] != 263:
            raise ValueError(f"expected (T,263) or (T,{N_JOINTS},3), got {arr.shape}")
        if mean is not None and std is not None:
            arr = arr * std + mean
        target = resample_linear(recover_from_ric(arr, N_JOINTS), source_fps, target_fps)
    target = fit_length_linear(target, target_len)
    if floor_align:
        target = target.copy()
        target[..., 1] -= target[..., 1].min()

    local_r = estimate_local_rotations(
        target,
        rest_joints,
        parents,
        orientation_mode=orientation_mode,
        parent_ref_weight=parent_ref_weight,
        temporal_twist_stabilization=temporal_twist_stabilization,
    )
    from .anatomical_fit import fit_anatomical_pose

    local_r = fit_anatomical_pose(
        target, rest_joints, parents, local_r,
        iterations=anatomical_fit_iters,
        prior_weight=anatomical_prior_weight,
        device=device_t,
    )
    if lock_global_orient is None:
        lock_global_orient = False
    aa = R.from_matrix(local_r.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(len(target), N_JOINTS, 3)
    global_orient = aa[:, 0]
    body_pose = aa[:, 1:].reshape(len(target), 63)

    joints_no_trans = smpl_forward_22(model, global_orient, body_pose, None, batch_size, device_t)
    transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
    global_orient, body_pose, transl, fitted = refine_smpl_fit(
        model,
        target,
        global_orient,
        body_pose,
        transl,
        refine_iters,
        refine_lr,
        pose_l2_weight,
        pose_keep_weight,
        angle_prior_weight,
        device_t,
        lock_global_orient=bool(lock_global_orient),
        lock_body_joint_ids=lock_body_joint_ids,
    )
    # An explicitly restored source translation takes precedence over grounding.
    restore_source_transl = (
        root_translation_mode == "source_transl"
        or (root_translation_mode == "auto" and source_motion135_transl is not None)
    )
    mesh_floor_delta = 0.0
    mesh_floor_aligned = False
    if floor_align and not restore_source_transl:
        offset = smpl_mesh_floor_offset(
            model, global_orient, body_pose, transl, device_t,
        )
        if offset is not None:
            mesh_floor_delta = offset
            mesh_floor_aligned = True
        transl[:, 1] += mesh_floor_delta
        fitted[..., 1] += mesh_floor_delta
    canonical_transl = transl.copy()
    canonical_fitted = fitted.copy()
    canonical_mpjpe_mm = np.linalg.norm(canonical_fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0
    root_translation_restored = False
    if restore_source_transl:
        if source_motion135_transl is None:
            raise ValueError("source_motion135_transl is required when root_translation_mode='source_transl'")
        transl = fit_length_linear(np.asarray(source_motion135_transl, dtype=np.float32), len(target)).astype(np.float32)
        fitted = smpl_forward_22(model, global_orient, body_pose, transl, batch_size, device_t)
        root_translation_restored = True
    local_r = R.from_rotvec(
        np.concatenate([global_orient[:, None, :], body_pose.reshape(len(target), 21, 3)], axis=1)
        .reshape(-1, 3)
    ).as_matrix().reshape(len(target), N_JOINTS, 3, 3).astype(np.float32)
    output_target_mpjpe_mm = np.linalg.norm(fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0

    motion_135 = np.concatenate(
        [transl, matrix_to_rot6d_rowmajor(local_r).reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)
    result = {
        "motion_135": motion_135,
        "transl": transl.astype(np.float32),
        "canonical_transl": canonical_transl.astype(np.float32),
        "global_orient": global_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "target_joints": target.astype(np.float32),
        "fitted_joints": fitted.astype(np.float32),
        "canonical_fitted_joints": canonical_fitted.astype(np.float32),
        "fit_mpjpe_mm": canonical_mpjpe_mm,
        "output_vs_canonical_target_mpjpe_mm": output_target_mpjpe_mm,
        "source_fps": np.array(source_fps, dtype=np.float32),
        "target_fps": np.array(target_fps, dtype=np.float32),
        "refine_iters": np.array(refine_iters, dtype=np.int32),
        "anatomical_fit_iters": np.array(anatomical_fit_iters, dtype=np.int32),
        "anatomical_prior_weight": np.array(anatomical_prior_weight, dtype=np.float32),
        "mesh_floor_delta_y": np.array(mesh_floor_delta, dtype=np.float32),
        "mesh_floor_aligned": np.array(mesh_floor_aligned),
        "rotation_init": np.array("position_ik"),
        "root_translation_restore_mode": np.array("source_transl" if root_translation_restored else "canonical"),
        "root_translation_restored": np.array(root_translation_restored),
        "global_orient_locked": np.array(bool(lock_global_orient)),
        "temporal_twist_stabilization": np.array(
            bool(temporal_twist_stabilization)
        ),
    }
    if compute_mesh_metrics:
        result["mesh_integrity"] = smpl_motion_integrity_metrics(
            model,
            global_orient,
            body_pose,
            device_t,
            sample_count=mesh_metric_samples,
        )
    return result


def hml263_to_motion135(feats: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Convert one HML263 clip to ROW-major SMPL ``motion_135``."""

    return retarget_hml263_clip(feats, **kwargs)["motion_135"]


__all__ = [
    "load_smpl_rest",
    "retarget_hml263_clip",
    "hml263_to_motion135",
    "smpl_motion_integrity_metrics",
    "validate_smpl_motion_integrity",
]
