"""HumanML3D-263 to SMPL ``motion_135`` retargeting API.

This module exposes the public library wrapper for the repository's validated
HML263 -> SMPL chain.  When full HML263 features are available, the conversion
maps HumanML3D's canonical-skeleton rotation block onto the SMPL rest skeleton
as pose initialization, then refines against the recovered 22 joints.
``position_ik`` is reserved for raw joint-only input or explicit diagnostics.

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
    estimate_local_rotations,
    fit_length_linear,
    hml263_rotations_to_smpl_init,
    load_smpl_rest as _load_smpl_rest,
    matrix_to_rot6d_rowmajor,
    merge_hml263_end_effectors,
    recover_hml263_local_rotations,
    recover_from_ric,
    refine_smpl_fit,
    fit_length_rotations,
    resample_rotations,
    resample_linear,
    smpl_forward_22,
)

def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_smpl_rest(
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Load the neutral SMPL model, rest joints, and parent chain.

    Args:
        model_dir: SMPL model directory. When omitted, reads
            ``MOTIUS_SMPL_MODEL_DIR``.
        device: Torch device for the SMPL layer.
    """

    resolved = model_dir or os.environ.get("MOTIUS_SMPL_MODEL_DIR")
    if not resolved:
        raise FileNotFoundError(
            "SMPL model assets are required; pass model_dir=... or set MOTIUS_SMPL_MODEL_DIR"
        )
    return _load_smpl_rest(Path(resolved), _as_device(device))


def retarget_hml263_clip(
    feats: np.ndarray,
    *,
    smpl_rest: tuple[Any, np.ndarray, np.ndarray] | None = None,
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    source_fps: float = 20.0,
    target_fps: float = 30.0,
    batch_size: int = 256,
    floor_align: bool = False,
    refine_iters: int = 0,
    refine_lr: float = 2e-2,
    rotation_init: str = "auto",
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
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
) -> dict[str, np.ndarray]:
    """Retarget one un-normalized HML263 clip to SMPL ``motion_135``.

    The returned ``motion_135`` is ROW-major rot6d:
    ``[root_translation(3), 22 * rot6d_row(132)]`` at ``target_fps``.

    Args:
        rotation_init: ``"auto"`` (default) maps the HumanML3D rotation block
            onto the SMPL rest skeleton for HML263 feature input and falls back
            to ``"position_ik"`` only when the input is already raw
            ``(T, 22, 3)`` joints.
    """

    if rot6d_convention != "row":
        raise ValueError("hml263_smpl public API only emits ROW-major motion_135")
    if orientation_mode not in {"bone", "parent_frame"}:
        raise ValueError(f"unsupported orientation_mode: {orientation_mode}")
    if root_translation_mode not in {"auto", "canonical", "source_transl"}:
        raise ValueError(f"unsupported root_translation_mode: {root_translation_mode}")
    if rotation_init not in {"auto", "position_ik", "hml263", "hml263_end_effectors", "hml263_init"}:
        raise ValueError(f"unsupported rotation_init: {rotation_init}")

    device_t = _as_device(device)
    model, rest_joints, parents = smpl_rest or load_smpl_rest(model_dir, device_t)

    arr = np.asarray(feats, dtype=np.float32)
    hml_local_r = None
    hml_feature_input = False
    use_hml_rot = rotation_init in {"auto", "hml263", "hml263_end_effectors", "hml263_init"}
    if arr.ndim == 3 and arr.shape[1:] == (N_JOINTS, 3):
        target = resample_linear(arr, source_fps, target_fps)
        if rotation_init == "auto":
            rotation_init = "position_ik"
    else:
        if arr.ndim != 2 or arr.shape[-1] != 263:
            raise ValueError(f"expected (T,263) or (T,{N_JOINTS},3), got {arr.shape}")
        hml_feature_input = True
        if rotation_init == "auto":
            rotation_init = "hml263_init"
        if mean is not None and std is not None:
            arr = arr * std + mean
        if use_hml_rot:
            hml_local_r = recover_hml263_local_rotations(arr, N_JOINTS)
            hml_local_r = resample_rotations(hml_local_r, source_fps, target_fps)
        target = recover_from_ric(arr, N_JOINTS)
        target = resample_linear(target, source_fps, target_fps)
    target = fit_length_linear(target, target_len)
    hml_local_r = fit_length_rotations(hml_local_r, target_len) if hml_local_r is not None else None
    if floor_align:
        target = target.copy()
        target[..., 1] -= target[..., 1].min()

    if rotation_init in {"hml263", "hml263_end_effectors", "hml263_init"} and hml_local_r is None:
        raise ValueError(f"rotation_init={rotation_init!r} requires HML263 feature input")
    position_local_r = estimate_local_rotations(
        target,
        rest_joints,
        parents,
        orientation_mode=orientation_mode,
        parent_ref_weight=parent_ref_weight,
    )
    if rotation_init == "hml263":
        local_r = hml_local_r
        rotation_init_used = "hml263"
    elif rotation_init == "hml263_end_effectors":
        local_r = merge_hml263_end_effectors(position_local_r, hml_local_r, rest_joints, parents)
        rotation_init_used = "hml263_end_effectors"
    elif rotation_init == "hml263_init":
        local_r = hml263_rotations_to_smpl_init(hml_local_r, position_local_r, rest_joints, parents)
        rotation_init_used = "hml263_init"
    else:
        local_r = position_local_r
        rotation_init_used = "position_ik"
    if lock_global_orient is None:
        lock_global_orient = bool(hml_feature_input and rotation_init_used != "position_ik")
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
    canonical_transl = transl.copy()
    canonical_fitted = fitted.copy()
    canonical_mpjpe_mm = np.linalg.norm(canonical_fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0
    root_translation_restored = False
    restore_source_transl = (
        root_translation_mode == "source_transl"
        or (root_translation_mode == "auto" and source_motion135_transl is not None)
    )
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
    return {
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
        "rotation_init": np.array(rotation_init_used),
        "root_translation_restore_mode": np.array("source_transl" if root_translation_restored else "canonical"),
        "root_translation_restored": np.array(root_translation_restored),
        "global_orient_locked": np.array(bool(lock_global_orient)),
    }


def hml263_to_motion135(feats: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Convert one HML263 clip to ROW-major SMPL ``motion_135``."""

    return retarget_hml263_clip(feats, **kwargs)["motion_135"]


__all__ = ["load_smpl_rest", "retarget_hml263_clip", "hml263_to_motion135"]
