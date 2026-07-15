"""MotionStreamer / GoToZero ``humanml3d_272`` representation (encode/decode).

Faithful re-implementation of the official MotionStreamer forward script
``representation_272.py`` (https://github.com/Li-xingXiao/272-dim-Motion-Representation).
See :class:`motius.motion.representation.specs.MS272` for the channel layout.

272 layout (per frame, ``njoint=22``)::

    [0:2]      root local xz velocity, heading-removed
    [2:8]      heading angular velocity, 6D rotation (frame 0 = identity)
    [8:74]     22 joint positions, heading-removed, per-frame xz origin (66)
    [74:140]   22 joint velocities, heading-removed (66)
    [140:272]  22 joint LOCAL rotations, 6D ROW-major (first two rows) (132)

The rotation block is ROW-major (``matrix[:2, :]``), matching the official
decoder and :func:`motius.motion.representation.rotation` ``convention="row"``.
MS272 contains enough channels to recover a root translation and local
rotations, but they are not stored as a contiguous ``motion_135`` prefix:
translation is reconstructed from ``[0:2]`` + ``[2:8]`` + root height in
``[8:74]``, while rotations live in ``[140:272]``.

Official raw-SMPL input uses :func:`smpl85_to_272`: SMPL-85 axis-angle
parameters are face-Z aligned, FK'ed with the SMPL-X body model used by
MotionStreamer's ``infer_get_joints.py``, then encoded with
:func:`encode_smpl_to_272`. On the bundled upstream examples
(``000000``/``M000000``), this path matches official ``Representation_272`` to
``max_abs < 5e-7`` over all 272 channels.

Important: :func:`motion135_to_272` is a different bridge for generated
``motion_135`` features. It FK's 22 rotations on the **GT-272 canonical
skeleton** (``bone_offsets_canon272.npy``), NOT the SMPL-H rest pose and NOT the
raw SMPL-X body model. Use it only when the input is already ``motion_135``.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import numpy as np

_NJOINT = 22
_SMPL85_DIM = 85

# Canonical parent-relative bone offsets (22,3) of the GT humanml3d_272 body.
_ASSET_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "bone_offsets_canon272.npy"),
)


def _canonical_272_offsets() -> np.ndarray:
    for p in _ASSET_CANDIDATES:
        if os.path.isfile(p):
            return np.load(p).astype(np.float64)
    raise FileNotFoundError(
        "bone_offsets_canon272.npy not found. Tried: " + ", ".join(_ASSET_CANDIDATES)
    )


def _rot_yaw(yaw: np.ndarray) -> np.ndarray:
    """Pure yaw (+Y) rotation matrices ``(...,3,3)`` (matches representation_272.rot_yaw)."""
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    z = np.zeros_like(yaw)
    o = np.ones_like(yaw)
    return np.stack(
        [
            np.stack([cs, z, sn], axis=-1),
            np.stack([z, o, z], axis=-1),
            np.stack([-sn, z, cs], axis=-1),
        ],
        axis=-2,
    )


def _matrix_to_rotation_6d_rows(mat: np.ndarray) -> np.ndarray:
    """ROW-major 6D = first two ROWS of the matrix, flattened."""
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def _default_smpl_model_dir() -> str:
    env = os.environ.get("MOTIUS_SMPL_MODEL_DIR")
    if env:
        return env
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(repo_root, "checkpoints", "smpl_models")


def _validate_smpl85(smpl_85: np.ndarray, *, name: str = "smpl_85") -> np.ndarray:
    arr = np.asarray(smpl_85)
    if arr.ndim != 2 or arr.shape[1] != _SMPL85_DIM:
        raise ValueError(f"{name} must have shape (T, 85), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one frame")
    return arr


def pack_smpl85(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pack SMPL-style arrays into MotionStreamer's SMPL-85 layout.

    Layout: ``[global_orient3, body_pose69, transl3, betas10]``. The official
    MotionStreamer-272 path consumes only the first 21 body joints
    (``body_pose[:, :63]``) when using the default SMPL-X FK backend; dims
    ``66:72`` are preserved for callers that pass 23 SMPL body joints or choose
    ``model_type="smpl"`` for diagnostics.
    """
    go = np.asarray(global_orient, dtype=np.float32).reshape(-1, 3)
    T = go.shape[0]
    bp = np.asarray(body_pose, dtype=np.float32)
    if bp.ndim == 3:
        bp = bp.reshape(T, -1)
    if bp.ndim != 2 or bp.shape[0] != T or bp.shape[1] < 63:
        raise ValueError(
            "body_pose must have shape (T, 63+), (T, 21+, 3), got "
            f"{np.asarray(body_pose).shape} for T={T}"
        )
    tr = np.asarray(transl, dtype=np.float32).reshape(T, 3)
    if betas is None:
        be = np.zeros((T, 10), dtype=np.float32)
    else:
        be_arr = np.asarray(betas, dtype=np.float32)
        if be_arr.ndim == 1:
            if be_arr.shape[0] != 10:
                raise ValueError(f"betas must have 10 dims, got {be_arr.shape}")
            be = np.broadcast_to(be_arr[None], (T, 10)).copy()
        elif be_arr.ndim == 2 and be_arr.shape == (T, 10):
            be = be_arr
        else:
            raise ValueError(f"betas must have shape (10,) or (T,10), got {be_arr.shape}")

    out = np.zeros((T, _SMPL85_DIM), dtype=np.float32)
    out[:, :3] = go
    out[:, 3 : 3 + min(69, bp.shape[1])] = bp[:, :69]
    out[:, 72:75] = tr
    out[:, 75:85] = be
    return out


def face_z_transform_smpl85(smpl_85: np.ndarray) -> np.ndarray:
    """Apply MotionStreamer's first-frame face-Z canonicalization to SMPL-85.

    This mirrors ``272-dim-Motion-Representation/face_z_transform.py``: the
    first root orientation's facing direction is rotated to +Z, and the same
    heading inverse is applied to every root orientation and translation.
    """
    arr = _validate_smpl85(smpl_85).astype(np.float64, copy=False)
    from motius.motion.representation.rotation import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
    )

    T = arr.shape[0]
    pose = arr[:, :72].reshape(T, 24, 3).copy()
    trans = arr[:, 72:75].copy()
    betas = arr[:, 75:85].copy()

    root_rot = np.asarray(axis_angle_to_matrix(pose[:, 0]), dtype=np.float64)
    first_forward = root_rot[0] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    heading = np.arctan2(first_forward[0], first_forward[2])
    heading_inv = _rot_yaw(np.asarray(-heading, dtype=np.float64))

    root_rot = np.einsum("ij,tjk->tik", heading_inv, root_rot)
    pose[:, 0] = np.asarray(matrix_to_axis_angle(root_rot), dtype=np.float64)
    trans = np.einsum("ij,tj->ti", heading_inv, trans)

    out = np.concatenate([pose.reshape(T, 72), trans, betas], axis=-1)
    return out.astype(np.float32)


def smpl85_to_local_rotmat(smpl_85: np.ndarray) -> np.ndarray:
    """Return the 22 local rotation matrices encoded by an SMPL-85 clip."""
    arr = _validate_smpl85(smpl_85)
    from motius.motion.representation.rotation import axis_angle_to_matrix

    aa = np.asarray(arr[:, :66], dtype=np.float64).reshape(arr.shape[0], _NJOINT, 3)
    rot = axis_angle_to_matrix(aa.reshape(-1, 3))
    return np.asarray(rot, dtype=np.float64).reshape(arr.shape[0], _NJOINT, 3, 3)


def fk_smpl85_joints(
    smpl_85: np.ndarray,
    *,
    smpl_model_dir: Optional[str] = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    device: str = "cpu",
    batch_size: int = 256,
    model: Optional[Any] = None,
    return_model: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
    """FK a face-Z SMPL-85 clip to MotionStreamer 22 joints.

    The default ``model_type="smplx"`` is the official MotionStreamer GT path
    (``infer_get_joints.py`` uses SMPL-X ``BodyModel`` and stores ``Jtr``).
    ``model_type="smpl"`` is kept only for diagnostics or legacy PRISM scripts.
    """
    arr = _validate_smpl85(smpl_85).astype(np.float32, copy=False)
    if model_type not in {"smplx", "smpl"}:
        raise ValueError(f"model_type must be 'smplx' or 'smpl', got {model_type!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    import torch
    import smplx

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    bs = int(getattr(model, "batch_size", batch_size) if model is not None else batch_size)
    if bs < 1:
        bs = min(max(arr.shape[0], 1), 256)

    if model is None:
        model_root = smpl_model_dir or _default_smpl_model_dir()
        if not os.path.exists(model_root):
            raise FileNotFoundError(
                f"SMPL model dir not found: {model_root}. Set MOTIUS_SMPL_MODEL_DIR "
                "or pass smpl_model_dir=..."
            )
        model = smplx.create(
            model_root,
            model_type=model_type,
            gender=gender,
            ext="npz" if model_type == "smplx" else "pkl",
            num_betas=10,
            batch_size=bs,
            use_pca=False,
            flat_hand_mean=True,
        ).to(dev)
        model.eval()
    else:
        model = model.to(dev)

    chunks = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], bs):
            clip = arr[start : start + bs]
            n = clip.shape[0]
            padded = np.zeros((bs, _SMPL85_DIM), dtype=np.float32)
            padded[:n] = clip
            pose = torch.from_numpy(padded[:, :72]).float().to(dev)
            trans = torch.from_numpy(padded[:, 72:75]).float().to(dev)
            betas = torch.from_numpy(padded[:, 75:85]).float().to(dev)
            if model_type == "smplx":
                out = model(
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:66],
                    betas=betas,
                    transl=trans,
                )
            else:
                out = model(
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:72],
                    betas=betas,
                    transl=trans,
                )
            chunks.append(out.joints[:n, :_NJOINT, :].detach().cpu().numpy())

    joints = np.concatenate(chunks, axis=0).astype(np.float32)
    if return_model:
        return joints, model
    return joints


def encode_smpl_to_272(joints_world: np.ndarray, local_rotmat: np.ndarray) -> np.ndarray:
    """Encode one clip to the 272 representation.

    Args:
        joints_world: ``(T, 22, 3)`` world joint positions (Y-up, metres).
        local_rotmat: ``(T, 22, 3, 3)`` SMPL local rotation matrices
            (joint 0 = global root orientation; joints 1..21 parent-relative).

    Returns:
        ``(T, 272)`` representation (float64).
    """
    pos = np.array(joints_world, dtype=np.float64)
    rot = np.array(local_rotmat, dtype=np.float64)
    nfrm, njoint = pos.shape[0], pos.shape[1]
    assert njoint == _NJOINT, f"expected 22 joints, got {njoint}"
    root_idx = 0

    # put on floor + root xz origin for the FIRST frame
    ori = pos[0, root_idx].copy()
    ori[1] = np.min(pos[:, :, 1])
    pos = pos - ori

    velocities_root = pos[1:, root_idx, :] - pos[:-1, root_idx, :]

    # per-frame xz origin (all joints relative to that frame's root)
    pos[:, :, 0] -= pos[:, root_idx : root_idx + 1, 0]
    pos[:, :, 2] -= pos[:, root_idx : root_idx + 1, 2]

    # heading from root rotation matrix
    R0 = rot[:, root_idx]
    global_heading = -np.arctan2(R0[:, 0, 2], R0[:, 2, 2])
    global_heading_rot = _rot_yaw(global_heading)  # (T,3,3)
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = _rot_yaw(global_heading_diff)  # (T-1,3,3)

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1), pos[..., None]
    ).squeeze(-1)  # (T,22,3)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]
    ).squeeze(-1)[..., [0, 2]]  # (T-1,2)

    rot = rot.copy()
    rot[:, root_idx] = np.matmul(global_heading_rot, rot[:, root_idx])

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame), dtype=np.float64)
    final_x[0, 2] = 1.0  # frame-0 heading 6D = identity rows [1,0,0,0,1,0]
    final_x[0, 6] = 1.0
    final_x[1:, 2:8] = _matrix_to_rotation_6d_rows(global_heading_diff_rot)
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8 : 8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint : 8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)
    final_x[:, 8 + 6 * njoint : 8 + 12 * njoint] = _matrix_to_rotation_6d_rows(rot).reshape(nfrm, -1)
    return final_x


def smpl85_to_272(
    smpl_85: np.ndarray,
    *,
    smpl_model_dir: Optional[str] = None,
    model_type: str = "smplx",
    gender: str = "neutral",
    device: str = "cpu",
    batch_size: int = 256,
    apply_face_z: bool = True,
    model: Optional[Any] = None,
    return_intermediates: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Convert MotionStreamer-layout SMPL-85 parameters to MS272.

    This is the canonical public API for raw SMPL/SMPL-X parameters. With the
    default ``model_type="smplx"`` and ``apply_face_z=True`` it follows the
    official MotionStreamer GT generation path:

    ``smpl_85 -> face_z_transform -> SMPL-X FK -> encode_smpl_to_272``.
    """
    smpl_85_fz = (
        face_z_transform_smpl85(smpl_85)
        if apply_face_z
        else _validate_smpl85(smpl_85).astype(np.float32, copy=False)
    )
    joints = fk_smpl85_joints(
        smpl_85_fz,
        smpl_model_dir=smpl_model_dir,
        model_type=model_type,
        gender=gender,
        device=device,
        batch_size=batch_size,
        model=model,
    )
    rot = smpl85_to_local_rotmat(smpl_85_fz)
    m272 = encode_smpl_to_272(joints, rot)
    if return_intermediates:
        return m272, {
            "smpl_85_face_z": smpl_85_fz,
            "joints": np.asarray(joints),
            "local_rotmat": rot,
        }
    return m272


def smpl_params_to_272(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Pack raw SMPL arrays and convert them to MS272.

    Args:
        global_orient: ``(T,3)`` root axis-angle.
        body_pose: ``(T,63+)`` or ``(T,21+,3)`` body axis-angle. The official
            SMPL-X path consumes the first 21 body joints.
        transl: ``(T,3)`` translation in metres, Y-up.
        betas: optional ``(10,)`` or ``(T,10)`` shape parameters; defaults to 0.
        **kwargs: forwarded to :func:`smpl85_to_272`.
    """
    smpl_85 = pack_smpl85(global_orient, body_pose, transl, betas)
    return smpl85_to_272(smpl_85, **kwargs)


def motion135_to_272(
    motion_135: np.ndarray,
    *,
    rotation_space: str = "local",
    bone_offsets: Optional[np.ndarray] = None,
    skeleton: str = "canon272",
    smplh_model: Optional[str] = None,
) -> np.ndarray:
    """Convert a 135-dim motion (trans3 + 22x6D rot6d, 30 fps) to MS272.

    Runs SMPL-22 forward kinematics then encodes via :func:`encode_smpl_to_272`.

    Args:
        motion_135: ``(T, >=135)`` motion (only the first 135 dims are used).
        rotation_space: ``"local"`` or ``"global"`` (model rot6d convention).
        bone_offsets: optional ``(22,3)`` rest offsets override (takes precedence).
        skeleton: ``"canon272"`` uses the bundled evaluator skeleton. For a
            different skeleton, pass ``bone_offsets`` explicitly.
        smplh_model: retained for API compatibility; not used by the public
            core converter.

    Returns:
        ``(T, 272)`` representation (float64).
    """
    import torch

    from motius.motion.skeleton.fk import motion135_to_fk

    arr = np.asarray(motion_135, dtype=np.float32)
    m135 = torch.from_numpy(arr[:, :135]).float()
    if bone_offsets is not None:
        bo = torch.as_tensor(bone_offsets).float()
    elif skeleton == "canon272":
        bo = torch.from_numpy(_canonical_272_offsets()).float()
    else:
        raise ValueError(
            f"unknown skeleton={skeleton!r}; use skeleton='canon272' or pass bone_offsets"
        )

    world_pos, _world_rot, _trans, local_rotmat = motion135_to_fk(
        m135, bo, rotation_space=rotation_space
    )
    joints = world_pos.detach().cpu().numpy().astype(np.float64)
    rot = local_rotmat.detach().cpu().numpy().astype(np.float64)
    return encode_smpl_to_272(joints, rot)


# --------------------------------------------------------------------------- #
# Native decode helpers
# --------------------------------------------------------------------------- #
def _rotation_6d_to_matrix_rows(d6: np.ndarray) -> np.ndarray:
    """Decode the row-major 6D rotations used by MS272."""

    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _accumulate_rotations(relative: np.ndarray) -> np.ndarray:
    out = [relative[0]]
    for rotation in relative[1:]:
        out.append(rotation @ out[-1])
    return np.asarray(out)


def recover_local_rotations_and_root(m272: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover local rotations and integrated world root translation."""

    motion = np.asarray(m272, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"m272 must have shape (T,272), got {motion.shape}")
    frames = len(motion)
    rotations = _rotation_6d_to_matrix_rows(
        motion[:, 140:272].reshape(frames, _NJOINT, 6)
    )
    heading = _accumulate_rotations(_rotation_6d_to_matrix_rows(motion[:, 2:8]))
    inverse_heading = np.transpose(heading, (0, 2, 1))
    rotations[:, 0] = inverse_heading @ rotations[:, 0]

    velocity = np.zeros((frames, 3), dtype=np.float64)
    velocity[:, [0, 2]] = motion[:, :2]
    if frames > 1:
        velocity[1:] = (inverse_heading[:-1] @ velocity[1:, :, None]).squeeze(-1)
    root = np.cumsum(velocity, axis=0)
    root[:, 1] = motion[:, 8:74].reshape(frames, _NJOINT, 3)[:, 0, 1]
    return rotations, root


def recover_272_stored_positions(m272: np.ndarray) -> np.ndarray:
    """Decode the native MS272 position block into world-space joints."""

    motion = np.asarray(m272, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"m272 must have shape (T,272), got {motion.shape}")
    frames = len(motion)
    positions = motion[:, 8:74].reshape(frames, _NJOINT, 3)
    heading = _accumulate_rotations(_rotation_6d_to_matrix_rows(motion[:, 2:8]))
    inverse_heading = np.transpose(heading, (0, 2, 1))
    positions = (
        np.repeat(inverse_heading[:, None], _NJOINT, axis=1) @ positions[..., None]
    ).squeeze(-1)

    velocity = np.zeros((frames, 3), dtype=np.float64)
    velocity[:, [0, 2]] = motion[:, :2]
    if frames > 1:
        velocity[1:] = (inverse_heading[:-1] @ velocity[1:, :, None]).squeeze(-1)
    root = np.cumsum(velocity, axis=0)
    positions[..., 0] += root[:, None, 0]
    positions[..., 2] += root[:, None, 2]
    return positions


def motion272_to_joints(
    m272: np.ndarray,
    *,
    bone_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode MS272 joints, optionally on an explicit canonical skeleton.

    Without ``bone_offsets`` this returns the representation's stored joint
    positions. With offsets it reconstructs local rotations and runs FK. The
    recovered translation is the pelvis position, so ``bone_offsets[0]`` must
    be zero in the latter mode.
    """

    if bone_offsets is None:
        return recover_272_stored_positions(m272)
    import torch

    from motius.motion.skeleton.fk import motion135_to_fk

    offsets = np.asarray(bone_offsets, dtype=np.float32)
    if offsets.shape != (_NJOINT, 3):
        raise ValueError(f"bone_offsets must have shape (22,3), got {offsets.shape}")
    if not np.allclose(offsets[0], 0.0, atol=1e-7):
        raise ValueError("motion272 pelvis-origin FK requires bone_offsets[0] == 0")
    motion135 = torch.from_numpy(motion272_to_motion135(m272))
    joints, _, _, _ = motion135_to_fk(motion135, torch.from_numpy(offsets))
    return joints.detach().cpu().numpy().astype(np.float32)


def motion272_to_motion135(m272: np.ndarray) -> np.ndarray:
    """Recover root translation and local rotations as canonical motion135.

    MS272 does not preserve subject shape, so this recovers the transform
    channels only. The MS272 first-two-rows rotation block is decoded first and
    then repacked into motion135's row-interleaved first-two-column layout.
    """

    from .rotation import matrix_to_rotation_6d

    rotations, root = recover_local_rotations_and_root(m272)
    rot6d = matrix_to_rotation_6d(rotations, convention="row")
    return np.concatenate([root, rot6d.reshape(len(root), 132)], axis=-1).astype(np.float32)


def reencode_272_via_stored_positions(m272: np.ndarray) -> np.ndarray:
    """GT272 -> decode (stored positions + local rotations) -> re-encode.

    Isolates the encoding math from any FK/body-model mismatch.
    """
    rot, _root = recover_local_rotations_and_root(m272)
    joints = recover_272_stored_positions(m272)
    return encode_smpl_to_272(joints, rot)


__all__ = [
    "encode_smpl_to_272",
    "face_z_transform_smpl85",
    "fk_smpl85_joints",
    "pack_smpl85",
    "motion135_to_272",
    "motion272_to_joints",
    "motion272_to_motion135",
    "recover_local_rotations_and_root",
    "recover_272_stored_positions",
    "smpl85_to_272",
    "smpl85_to_local_rotmat",
    "smpl_params_to_272",
    "reencode_272_via_stored_positions",
]
