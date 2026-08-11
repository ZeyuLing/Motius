"""DART / ViMoGen 276-dim motion representation.

This module lifts the DART-style representation used by ViMoGen into the public
motion representation library. It is a pure conversion layer: no model bundle,
text encoder, renderer, or external repository checkout is required.

Layout per frame (dim=276, 22 joints):

``[body_pose_rot6d(126), joints(66), joints_vel(66), root_rot6d(6),
root_rot_vel6d(6), transl(3), transl_vel(3)]``.

The representation stores velocity channels, so a DART276 sequence of length
``L`` represents an original SMPL/joint sequence of length ``L + 1`` when
decoded with ``equal_length=True``.

DART's 6D rotation vector is produced by ``R[..., :, :2].reshape(6)``. In this
repository's rotation API that is ``convention="row"`` (row-interleaved first
two columns), not the standard column-concatenated Zhou-6D vector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor

from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)

N_JOINTS = 22
DART276_DIM = 276

# Official ViMoGen / MBench conversion from DART canonical coordinates to the
# viewer/evaluator coordinate used by the released scripts:
# _FRONT_ROTATION @ _BASE_CONVERSION.
MBENCH_COORD_CONVERSION = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


def _as_tensor(x, *, dtype=torch.float32, device=None) -> Tensor:
    if torch.is_tensor(x):
        t = x
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t
    return torch.as_tensor(x, dtype=dtype, device=device)


def _same_type(reference, x: Tensor):
    if isinstance(reference, np.ndarray):
        return x.detach().cpu().numpy()
    return x


def validate_dart276(motion: Tensor, *, name: str = "motion") -> Tensor:
    """Validate a ``(..., 276)`` DART tensor and return it."""

    if motion.shape[-1] != DART276_DIM:
        raise ValueError(f"{name} must have last dim {DART276_DIM}, got {tuple(motion.shape)}")
    if motion.shape[-2] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    return motion


def _matrix_to_dart_rot6d(matrix: Tensor) -> Tensor:
    """Official DART/ViMoGen 6D layout: ``R[..., :, :2].reshape(..., 6)``."""

    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def _dart_rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Official DART/ViMoGen 6D decoder, including its normalization behavior."""

    r = rot6d.reshape(-1, 3, 2)
    a1 = r[:, :, 0]
    a2 = r[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    out = torch.stack((b1, b2, b3), dim=-1)
    return out.reshape(*rot6d.shape[:-1], 3, 3)


def _dart_rot6d_to_axis_angle(rot6d: Tensor) -> Tensor:
    aa = matrix_to_axis_angle(_dart_rot6d_to_matrix(rot6d))
    aa[torch.isnan(aa)] = 0.0
    return aa


def split_dart276(motion) -> dict[str, Tensor]:
    """Split DART276 channels into named tensors.

    Args:
        motion: ``(..., T, 276)`` tensor/array.

    Returns:
        Dictionary with tensors in torch format. Rotation channels remain in
        DART/row-interleaved 6D form.
    """

    m = validate_dart276(_as_tensor(motion), name="dart276")
    return {
        "body_pose_rot6d": m[..., 0:126].reshape(*m.shape[:-1], 21, 6),
        "joints": m[..., 126:192].reshape(*m.shape[:-1], N_JOINTS, 3),
        "joints_vel": m[..., 192:258].reshape(*m.shape[:-1], N_JOINTS, 3),
        "global_orient_rot6d": m[..., 258:264],
        "global_orient_vel_rot6d": m[..., 264:270],
        "transl": m[..., 270:273],
        "transl_vel": m[..., 273:276],
    }


def dart276_to_smpl_params(
    motion,
    *,
    recover_from_velocity: bool = False,
    equal_length: bool = False,
) -> Tuple[dict[str, Tensor], Tensor]:
    """Decode DART276 to SMPL body parameters and stored/recovered joints.

    Args:
        motion: ``(T,276)`` tensor/array.
        recover_from_velocity: If true, reconstruct root orientation,
            translation, and joints by integrating velocity channels.
        equal_length: If true, decode to ``T+1`` frames by integrating the final
            velocity and repeating the final body pose. This mirrors official
            ViMoGen rendering/SMPL export.

    Returns:
        ``(smpl_params, joints)`` where ``smpl_params`` contains
        ``global_orient`` ``(L,3)``, ``body_pose`` ``(L,63)``, and ``transl``
        ``(L,3)``. ``joints`` has shape ``(L,22,3)``.
    """

    m = validate_dart276(_as_tensor(motion), name="dart276")
    if m.ndim != 2:
        raise ValueError(f"dart276_to_smpl_params expects (T,276), got {tuple(m.shape)}")
    parts = split_dart276(m)
    seq_len = m.shape[0]

    body_pose = _dart_rot6d_to_axis_angle(parts["body_pose_rot6d"].reshape(-1, 6)).reshape(seq_len, -1)
    joints = parts["joints"]
    global_orient = _dart_rot6d_to_axis_angle(parts["global_orient_rot6d"])
    transl = parts["transl"]

    if recover_from_velocity or equal_length:
        seq_end = seq_len + 1 if equal_length else seq_len
        r_first = _dart_rot6d_to_matrix(parts["global_orient_rot6d"][0:1])
        r_vel = _dart_rot6d_to_matrix(parts["global_orient_vel_rot6d"])
        rec_rots = [r_first]
        for i in range(1, seq_end):
            rec_rots.append(r_vel[i - 1 : i] @ rec_rots[i - 1])
        global_orient = matrix_to_axis_angle(torch.cat(rec_rots, dim=0))

        rec_trans = [transl[0:1]]
        rec_joints = [joints[0:1]]
        for i in range(1, seq_end):
            rec_trans.append(rec_trans[i - 1] + parts["transl_vel"][i - 1 : i])
            rec_joints.append(rec_joints[i - 1] + parts["joints_vel"][i - 1 : i])
        transl = torch.cat(rec_trans, dim=0)
        joints = torch.cat(rec_joints, dim=0)
        if equal_length:
            body_pose = torch.cat([body_pose, body_pose[-1:]], dim=0)

    return {"global_orient": global_orient, "body_pose": body_pose, "transl": transl}, joints


def get_dart_canonical_transform(joints) -> Tensor:
    """Compute ViMoGen/DART first-frame canonical rotation.

    The transform uses the first frame's left/right hips and shoulders to define
    a horizontal body-facing frame. DART is Z-up; the first-frame pelvis becomes
    the translation origin during :func:`canonicalize_smpl_for_dart`.
    """

    j = _as_tensor(joints)
    if j.ndim != 3 or j.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"joints must have shape (T,22,3), got {tuple(j.shape)}")
    j0 = j[0]
    right_hip, left_hip = 2, 1
    right_shoulder, left_shoulder = 17, 16
    x_axis = j0[right_hip] - j0[left_hip]
    if torch.linalg.norm(x_axis) < 1e-6:
        x_axis = j0[right_shoulder] - j0[left_shoulder]
    x_axis = x_axis.clone()
    x_axis[2] = 0.0
    x_axis = x_axis / torch.linalg.norm(x_axis).clamp_min(1e-8)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=j.dtype, device=j.device)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-8)
    return torch.stack([x_axis, y_axis, z_axis], dim=1).T


def _default_smplx_root(device=None, dtype=torch.float32) -> Tensor:
    candidates = [
        Path(__file__).resolve().parents[1] / "assets" / "smplx_root.pt",
        Path(__file__).resolve().parents[2]
        / "models"
        / "motion"
        / "vimogen"
        / "network"
        / "assets"
        / "smplx_root.pt",
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True).to(device=device, dtype=dtype)
    return torch.zeros(3, device=device, dtype=dtype)


def apply_dart_rotation_to_smpl(
    smpl_params: Mapping[str, Tensor],
    rotation: Tensor,
    *,
    smplx_root: Tensor | None = None,
) -> dict[str, Tensor]:
    """Apply a DART/global coordinate rotation to SMPL root orient/transl."""

    go = _as_tensor(smpl_params["global_orient"])
    tr = _as_tensor(smpl_params["transl"], device=go.device, dtype=go.dtype)
    r = _as_tensor(rotation, device=go.device, dtype=go.dtype)
    root = (
        _as_tensor(smplx_root, device=go.device, dtype=go.dtype)
        if smplx_root is not None
        else _default_smplx_root(device=go.device, dtype=go.dtype)
    )

    go_mat = axis_angle_to_matrix(go.reshape(-1, 3))
    out = dict(smpl_params)
    out["global_orient"] = matrix_to_axis_angle(r[None] @ go_mat).reshape_as(go)
    out["transl"] = (r[None] @ (tr + root).unsqueeze(-1)).squeeze(-1) - root
    return out


def canonicalize_smpl_for_dart(
    smpl_params: Mapping[str, Tensor],
    joints,
    *,
    set_floor: bool = False,
    smplx_root: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
    """Canonicalize SMPL params/joints into DART's Z-up body-facing frame."""

    j = _as_tensor(joints)
    r_inv = get_dart_canonical_transform(j)
    aligned = apply_dart_rotation_to_smpl(smpl_params, r_inv, smplx_root=smplx_root)
    joints_base = (r_inv[None, None] @ j.unsqueeze(-1)).squeeze(-1)
    delta_transl = -joints_base[0, 0:1]
    if set_floor:
        delta_transl = delta_transl.clone()
        delta_transl[0, 2] = -torch.min(joints_base[..., 2])
    joints_canonical = joints_base + delta_transl[None]
    aligned["transl"] = aligned["transl"] + delta_transl
    return aligned, joints_canonical, r_inv, delta_transl


def smpl_params_and_joints_to_dart276(
    smpl_params: Mapping[str, Tensor],
    joints,
    *,
    canonicalize: bool = False,
    set_floor: bool = False,
    smplx_root: Tensor | None = None,
) -> Tensor:
    """Encode SMPL params plus matching 22-joint positions into DART276.

    ``joints`` must correspond to the same body model/shape as ``smpl_params``.
    This explicit requirement is important: DART stores both SMPL rotations and
    joint positions, so SMPL pose/trans alone is insufficient for a lossless
    encode unless a body model FK pass has already supplied the joints.
    """

    go = _as_tensor(smpl_params["global_orient"]).reshape(-1, 3)
    bp = _as_tensor(smpl_params["body_pose"], device=go.device, dtype=go.dtype)
    tr = _as_tensor(smpl_params["transl"], device=go.device, dtype=go.dtype).reshape(-1, 3)
    if bp.ndim == 3:
        bp = bp.reshape(bp.shape[0], -1)
    j = _as_tensor(joints, device=go.device, dtype=go.dtype)
    if j.ndim != 3 or j.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"joints must have shape (T,22,3), got {tuple(j.shape)}")
    seq_len = tr.shape[0]
    if go.shape[0] != seq_len or bp.shape[0] != seq_len or j.shape[0] != seq_len:
        raise ValueError("global_orient, body_pose, transl, and joints must share T")
    if seq_len < 2:
        raise ValueError("DART276 encoding needs at least two SMPL/joint frames")

    params: MutableMapping[str, Tensor] = {
        "global_orient": go,
        "body_pose": bp,
        "transl": tr,
    }
    if canonicalize:
        params, j, _, _ = canonicalize_smpl_for_dart(
            params,
            j,
            set_floor=set_floor,
            smplx_root=smplx_root,
        )
        go = params["global_orient"].reshape(seq_len, 3)
        bp = params["body_pose"].reshape(seq_len, -1)
        tr = params["transl"].reshape(seq_len, 3)

    root_rot = axis_angle_to_matrix(go)
    root_vel = root_rot[1:] @ root_rot[:-1].transpose(-1, -2)
    body_rot6d = _matrix_to_dart_rot6d(axis_angle_to_matrix(bp.reshape(-1, 3))).reshape(seq_len, -1)
    joints_flat = j.reshape(seq_len, -1)
    motion = torch.cat(
        [
            body_rot6d[:-1],
            joints_flat[:-1],
            joints_flat[1:] - joints_flat[:-1],
            _matrix_to_dart_rot6d(root_rot)[:-1],
            _matrix_to_dart_rot6d(root_vel),
            tr[:-1],
            tr[1:] - tr[:-1],
        ],
        dim=-1,
    )
    validate_dart276(motion, name="encoded_dart276")
    return motion


def dart276_to_joints(
    motion,
    *,
    recover_from_velocity: bool = True,
    equal_length: bool = False,
    coord: str = "dart",
):
    """Decode DART276 to 22 joint positions.

    ``coord="dart"`` returns the native DART canonical Z-up coordinates.
    ``coord="mbench"`` applies the official ViMoGen/MBench coordinate transform.
    """

    _, joints = dart276_to_smpl_params(
        motion,
        recover_from_velocity=recover_from_velocity,
        equal_length=equal_length,
    )
    if coord == "mbench":
        conv = MBENCH_COORD_CONVERSION.to(device=joints.device, dtype=joints.dtype)
        joints = torch.einsum("ij,tvj->tvi", conv, joints)
    elif coord != "dart":
        raise ValueError(f"coord must be 'dart' or 'mbench', got {coord!r}")
    return _same_type(motion, joints)


def dart276_to_motion135(
    motion,
    *,
    recover_from_velocity: bool = True,
    equal_length: bool = True,
    coord_conversion: str = "mbench",
    translation_source: str = "floor_aligned_smpl_transl",
    rotation_convention: str = "row",
):
    """Decode DART276 to SMPL ``motion_135``.

    Args:
        coord_conversion: ``"mbench"`` applies the official DART->viewer/eval
            coordinate conversion to root translation and root orientation;
            ``"none"`` keeps native DART canonical coordinates.
        translation_source: ``"floor_aligned_smpl_transl"`` keeps the decoded
            SMPL translation but shifts the vertical axis so the decoded joint
            floor is at zero, matching repository ``motion_135`` evaluator
            conventions. ``"floor_aligned_joints_pelvis"`` uses the pelvis
            joint with the same floor alignment. ``"joints_pelvis"`` and
            ``"smpl_transl"`` preserve raw decoded coordinates and are mainly
            useful for compatibility diagnostics.
        rotation_convention: ``"row"`` for repository-canonical ``motion_135``;
            ``"column"`` for legacy MotionCLIP evaluator inputs.
    """

    smpl, joints = dart276_to_smpl_params(
        motion,
        recover_from_velocity=recover_from_velocity,
        equal_length=equal_length,
    )
    allowed_translation_sources = {
        "floor_aligned_smpl_transl",
        "floor_aligned_joints_pelvis",
        "joints_pelvis",
        "smpl_transl",
    }
    if translation_source not in allowed_translation_sources:
        raise ValueError(f"unsupported translation_source={translation_source!r}")
    transl = smpl["transl"]
    global_orient = smpl["global_orient"].reshape(-1, 3)
    body_pose = smpl["body_pose"].reshape(len(transl), 21, 3)
    if coord_conversion == "mbench":
        conv = MBENCH_COORD_CONVERSION.to(device=transl.device, dtype=transl.dtype)
        transl = torch.einsum("ij,tj->ti", conv, transl)
        joints = torch.einsum("ij,tkj->tki", conv, joints)
        global_orient = matrix_to_axis_angle(conv[None] @ axis_angle_to_matrix(global_orient))
    elif coord_conversion != "none":
        raise ValueError(f"unsupported coord_conversion={coord_conversion!r}")
    if translation_source in {"joints_pelvis", "floor_aligned_joints_pelvis"}:
        transl = joints[:, 0, :]
    if translation_source.startswith("floor_aligned_"):
        vertical_idx = 1 if coord_conversion == "mbench" else 2
        floor = joints[:, :, vertical_idx].amin(dim=1)
        transl = transl.clone()
        transl[:, vertical_idx] = transl[:, vertical_idx] - floor

    root6 = matrix_to_rotation_6d(
        axis_angle_to_matrix(global_orient),
        convention=rotation_convention,
    ).reshape(len(transl), 6)
    body6 = matrix_to_rotation_6d(
        axis_angle_to_matrix(body_pose),
        convention=rotation_convention,
    ).reshape(len(transl), 126)
    out = torch.cat([transl, root6, body6], dim=-1).float()
    return _same_type(motion, out)


__all__ = [
    "N_JOINTS",
    "DART276_DIM",
    "MBENCH_COORD_CONVERSION",
    "validate_dart276",
    "split_dart276",
    "dart276_to_smpl_params",
    "get_dart_canonical_transform",
    "apply_dart_rotation_to_smpl",
    "canonicalize_smpl_for_dart",
    "smpl_params_and_joints_to_dart276",
    "dart276_to_joints",
    "dart276_to_motion135",
]
