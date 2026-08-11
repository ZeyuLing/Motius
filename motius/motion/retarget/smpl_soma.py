"""Reusable SMPL <-> KIMODO/SOMA retargeting utilities.

The defaults in this module mirror the validated retargeting setup used by the
KIMODO HumanML3D evaluation scripts:

* SMPL ``motion_135`` -> SOMA30 uses direct rotation transfer with a
  shoulder-only rest-direction correction.
* KIMODO/SOMA 22-joint positions -> SMPL ``motion_135`` uses a SMPL IK fit with
  SOMA77 leaf/head orientation guides when they are available.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


N_SMPL_JOINTS = 22
SMPL22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
SMPL22_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)
SOMA30_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
    "RightFoot", "RightToeBase",
]
SOMA30_PARENT_NAMES = [
    None, "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "Head", "Head", "Chest", "LeftShoulder", "LeftArm", "LeftForeArm",
    "LeftHand", "LeftHand", "Chest", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHand", "Hips", "LeftLeg", "LeftShin",
    "LeftFoot", "Hips", "RightLeg", "RightShin", "RightFoot",
]
SMPL_TO_SOMA_NAME = {
    "pelvis": "Hips",
    "left_hip": "LeftLeg",
    "right_hip": "RightLeg",
    "spine1": "Spine1",
    "left_knee": "LeftShin",
    "right_knee": "RightShin",
    "spine2": "Spine2",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Chest",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "neck": "Neck1",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}
_SOMA30_IDX = {name: idx for idx, name in enumerate(SOMA30_NAMES)}
_SMPL22_IDX = {name: idx for idx, name in enumerate(SMPL22_NAMES)}
SMPL22_TO_SOMA30 = np.array(
    [_SOMA30_IDX[SMPL_TO_SOMA_NAME[name]] for name in SMPL22_NAMES],
    dtype=np.int64,
)
SOMA30_PARENTS = np.array(
    [-1 if parent is None else _SOMA30_IDX[parent] for parent in SOMA30_PARENT_NAMES],
    dtype=np.int64,
)
SOMA77_IDX = {
    "Neck2": 5,
    "Head": 6,
    "HeadEnd": 7,
    "Jaw": 8,
    "LeftEye": 9,
    "RightEye": 10,
    "LeftHand": 14,
    "LeftHandThumbEnd": 18,
    "LeftHandMiddleEnd": 28,
    "RightHand": 42,
    "RightHandThumbEnd": 46,
    "RightHandMiddleEnd": 56,
    "LeftToeBase": 70,
    "LeftToeEnd": 71,
    "RightToeBase": 75,
    "RightToeEnd": 76,
}
SOMA30_IN_SOMA77 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 28, 39, 40, 41, 42, 46, 56, 67, 68, 69, 70, 72, 73, 74, 75],
    dtype=np.int64,
)
FOOT_HEIGHT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)


@dataclass(frozen=True)
class SMPLToSOMAConfig:
    """Config for SMPL ``motion_135`` -> SOMA30 retargeting."""

    assets_root: str | Path | None = None
    shoulder_offset_alpha: float = 0.75
    smpl_height_mode: str = "source_root"


@dataclass(frozen=True)
class SOMAToSMPLIKConfig:
    """Config for KIMODO/SOMA skeleton -> SMPL ``motion_135`` IK retargeting."""

    model_dir: str | Path | None = None
    device: str | torch.device | None = None
    batch_size: int = 512
    floor_align: bool = True
    foot_height_align: bool = True
    refine_iters: int = 5
    refine_lr: float = 2e-2
    orientation_mode: str = "parent_frame"
    parent_ref_weight: float = 0.25
    pose_l2_weight: float = 0.0
    angle_prior_weight: float = 0.0
    smooth_weight: float = 0.01
    joint_accel_weight: float = 0.001
    joint_fit_weight_preset: str = "relaxed_upper"
    soma_orientation_guides: bool = True
    head_guide_weight: float = 0.15
    leaf_guide_weight: float = 0.35
    smpl_height_mode: str = "source_root"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_assets_root(assets_root: str | Path | None) -> Path:
    if assets_root is None:
        env = os.environ.get("KIMODO_SKELETON_ASSETS")
        if not env:
            raise FileNotFoundError(
                "KIMODO skeleton assets are required; pass assets_root=... or set "
                "KIMODO_SKELETON_ASSETS"
            )
        assets_root = env
    root = Path(assets_root)
    if not root.is_absolute():
        root = _repo_root() / root
    if not root.exists():
        raise FileNotFoundError(f"KIMODO skeleton assets not found: {root}")
    return root


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _patch_numpy_chumpy_aliases() -> None:
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
        "int_": np.int64,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": object,
        "unicode_": str,
        "str_": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _import_smplx():
    _patch_numpy_chumpy_aliases()
    try:
        import smplx  # type: ignore
        return smplx
    except ModuleNotFoundError as exc:
        raise ImportError("SMPL/SOMA retargeting requires the optional 'smplx' package") from exc


@lru_cache(maxsize=None)
def _neutral_joints(assets_root: str, skeleton_name: str) -> torch.Tensor:
    path = Path(assets_root) / skeleton_name / "joints.p"
    try:
        joints = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        joints = torch.load(path, map_location="cpu")
    return joints.squeeze().float()


def rot6d_to_rotmat_row_major(rot6d: torch.Tensor) -> torch.Tensor:
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d_rowmajor(rotmat: np.ndarray) -> np.ndarray:
    return np.asarray(rotmat[..., :, :2], dtype=np.float32).reshape(*rotmat.shape[:-2], 6)


def _smpl22_bone_offsets(assets_root: Path) -> np.ndarray:
    neutral = _as_numpy(_neutral_joints(str(assets_root), "smplx22")).astype(np.float32)
    offsets = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
    for j, p in enumerate(SMPL22_PARENTS.tolist()):
        if p >= 0:
            offsets[j] = neutral[j] - neutral[p]
    return offsets


def _soma30_offsets(assets_root: Path) -> torch.Tensor:
    neutral = _neutral_joints(str(assets_root), "somaskel30")
    offsets = torch.zeros((len(SOMA30_PARENTS), 3), dtype=torch.float32)
    for j, p in enumerate(SOMA30_PARENTS.tolist()):
        if p >= 0:
            offsets[j] = neutral[j] - neutral[p]
    return offsets


def differentiable_fk(
    local_rotmat: torch.Tensor,
    translation: torch.Tensor,
    bone_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_rot: list[torch.Tensor] = [None] * N_SMPL_JOINTS  # type: ignore[list-item]
    world_pos: list[torch.Tensor] = [None] * N_SMPL_JOINTS  # type: ignore[list-item]
    for j, p in enumerate(SMPL22_PARENTS.tolist()):
        if p < 0:
            world_rot[j] = local_rotmat[..., j, :, :]
            world_pos[j] = translation + bone_offsets[j]
        else:
            world_rot[j] = world_rot[p] @ local_rotmat[..., j, :, :]
            offset = (world_rot[p] @ bone_offsets[j].unsqueeze(-1)).squeeze(-1)
            world_pos[j] = world_pos[p] + offset
    return torch.stack(world_pos, dim=-2), torch.stack(world_rot, dim=-3)


def _slerp_rot_matrices(r1: torch.Tensor, r2: torch.Tensor, t: float) -> torch.Tensor:
    r_delta = torch.einsum("...ij,...ik->...jk", r1, r2)
    tr = r_delta[..., 0, 0] + r_delta[..., 1, 1] + r_delta[..., 2, 2]
    cos_angle = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    small = angle.abs() < 1e-6
    sin_angle = torch.sin(angle).clamp(min=1e-8)
    axis = torch.stack(
        [
            r_delta[..., 2, 1] - r_delta[..., 1, 2],
            r_delta[..., 0, 2] - r_delta[..., 2, 0],
            r_delta[..., 1, 0] - r_delta[..., 0, 1],
        ],
        dim=-1,
    ) / (2.0 * sin_angle.unsqueeze(-1))
    axis = F.normalize(axis, dim=-1)
    scaled = angle * t
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    k = k.reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=r1.device, dtype=r1.dtype).expand_as(k)
    out = eye + torch.sin(scaled)[..., None, None] * k
    out = out + (1.0 - torch.cos(scaled))[..., None, None] * (k @ k)
    out = torch.where(small[..., None, None], eye, out)
    return r1 @ out


def _safe_normalize_np(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n[..., 0] > eps


def _rotation_between_np(src: np.ndarray, dst: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / max(float(np.linalg.norm(src)), eps)
    dst = dst / max(float(np.linalg.norm(dst)), eps)
    cross = np.cross(src, dst)
    sin = float(np.linalg.norm(cross))
    cos = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if sin < eps:
        if cos > 0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < eps:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis = axis / max(float(np.linalg.norm(axis)), eps)
        return R.from_rotvec(np.pi * axis).as_matrix()
    return R.from_rotvec(np.arctan2(sin, cos) * cross / sin).as_matrix()


def _scaled_rotation_np(rot: np.ndarray, alpha: float) -> np.ndarray:
    return R.from_rotvec(R.from_matrix(rot).as_rotvec() * float(alpha)).as_matrix()


def _neutral_bone_direction_offsets(assets_root: Path) -> dict[int, np.ndarray]:
    smpl_neutral = _as_numpy(_neutral_joints(str(assets_root), "smplx22")).astype(np.float64)
    soma_neutral = _as_numpy(_neutral_joints(str(assets_root), "somaskel30")).astype(np.float64)
    offsets: dict[int, np.ndarray] = {}
    for smpl_idx, soma_idx in enumerate(SMPL22_TO_SOMA30.tolist()):
        smpl_parent = int(SMPL22_PARENTS[smpl_idx])
        soma_parent = int(SOMA30_PARENTS[soma_idx])
        if smpl_parent < 0 or soma_parent < 0:
            offsets[soma_idx] = np.eye(3, dtype=np.float64)
            continue
        offsets[soma_idx] = _rotation_between_np(
            smpl_neutral[smpl_idx] - smpl_neutral[smpl_parent],
            soma_neutral[soma_idx] - soma_neutral[soma_parent],
        )
    return offsets


def _global_to_local_np(global_rots: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.zeros_like(global_rots, dtype=np.float32)
    for j, p in enumerate(parents.tolist()):
        if p < 0:
            local[:, j] = global_rots[:, j]
        else:
            local[:, j] = np.einsum("tki,tkl->til", global_rots[:, p], global_rots[:, j])
    return local


def _complete_soma30_global_rots(soma_global_rots: torch.Tensor) -> torch.Tensor:
    soma_global_rots[:, 5] = _slerp_rot_matrices(soma_global_rots[:, 4], soma_global_rots[:, 6], 0.5)
    soma_global_rots[:, 7] = soma_global_rots[:, 6]
    soma_global_rots[:, 8] = soma_global_rots[:, 6]
    soma_global_rots[:, 9] = soma_global_rots[:, 6]
    soma_global_rots[:, 14] = soma_global_rots[:, 13]
    soma_global_rots[:, 15] = soma_global_rots[:, 13]
    soma_global_rots[:, 20] = soma_global_rots[:, 19]
    soma_global_rots[:, 21] = soma_global_rots[:, 19]
    return soma_global_rots


def _soma30_fk_from_local(
    local_rots: np.ndarray,
    root_pos: torch.Tensor,
    soma_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_t = torch.from_numpy(np.asarray(local_rots, dtype=np.float32)).to(root_pos.device)
    global_rots: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    for j, p in enumerate(SOMA30_PARENTS.tolist()):
        if p < 0:
            global_rots.append(local_t[:, j])
            positions.append(root_pos)
            continue
        global_rots.append(global_rots[p] @ local_t[:, j])
        offset = soma_offsets[j].to(root_pos.device, root_pos.dtype)
        positions.append(positions[p] + (global_rots[p] @ offset[:, None]).squeeze(-1))
    return torch.stack(global_rots, dim=1), torch.stack(positions, dim=1)


def _smpl22_to_soma30_retarget_shoulder_offset(
    motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    soma_offsets: torch.Tensor,
    assets_root: Path,
    shoulder_offset_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32))
    t = motion.shape[0]
    translation = motion[:, :3]
    local_rotmat = rot6d_to_rotmat_row_major(motion[:, 3:135].reshape(t, N_SMPL_JOINTS, 6))
    smpl_pos, smpl_global_rots = differentiable_fk(local_rotmat, translation, offsets)

    eye = torch.eye(3, dtype=local_rotmat.dtype, device=local_rotmat.device)
    soma_global_rots = eye[None, None].expand(t, len(SOMA30_NAMES), 3, 3).clone()
    for smpl_idx, soma_idx in enumerate(SMPL22_TO_SOMA30.tolist()):
        soma_global_rots[:, soma_idx] = smpl_global_rots[:, smpl_idx]

    rest_offsets = _neutral_bone_direction_offsets(assets_root)
    for name in ("LeftShoulder", "RightShoulder"):
        soma_idx = _SOMA30_IDX[name]
        offset = _scaled_rotation_np(rest_offsets[soma_idx], shoulder_offset_alpha).astype(np.float32)
        soma_global_rots[:, soma_idx] = soma_global_rots[:, soma_idx] @ torch.from_numpy(offset).to(soma_global_rots)

    soma_global_rots = _complete_soma30_global_rots(soma_global_rots)
    soma_local = _global_to_local_np(_as_numpy(soma_global_rots), SOMA30_PARENTS)
    soma_root_pos = translation.clone()
    soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    smpl_feet = [_SMPL22_IDX[x] for x in ("left_foot", "right_foot", "left_ankle", "right_ankle")]
    soma_feet = [_SOMA30_IDX[x] for x in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")]
    y_delta = soma_joints[:, soma_feet, 1].min(dim=1).values - smpl_pos[:, smpl_feet, 1].min(dim=1).values
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    root_delta_xz = translation[:, [0, 2]] - soma_joints[:, 0, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)
    return soma_global_rots, soma_joints


def _soma30_to_smpl22_motion_rotation(
    soma_global_rots: torch.Tensor | np.ndarray,
    source_motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    height_mode: str = "source_root",
    assets_root: str | Path | None = None,
    shoulder_offset_alpha: float = 0.75,
) -> dict[str, np.ndarray]:
    if not isinstance(soma_global_rots, torch.Tensor):
        soma_global_rots = torch.from_numpy(np.asarray(soma_global_rots, dtype=np.float32))
    source = torch.from_numpy(np.asarray(source_motion_135, dtype=np.float32))
    t = source.shape[0]
    smpl_global = torch.eye(3, dtype=soma_global_rots.dtype, device=soma_global_rots.device)
    smpl_global = smpl_global[None, None].expand(t, N_SMPL_JOINTS, 3, 3).clone()
    for smpl_idx, soma_idx in enumerate(SMPL22_TO_SOMA30.tolist()):
        smpl_global[:, smpl_idx] = soma_global_rots[:, soma_idx]

    if shoulder_offset_alpha != 0.0:
        rest_offsets = _neutral_bone_direction_offsets(_resolve_assets_root(assets_root))
        for name in ("left_collar", "right_collar"):
            smpl_idx = _SMPL22_IDX[name]
            soma_idx = int(SMPL22_TO_SOMA30[smpl_idx])
            offset = _scaled_rotation_np(rest_offsets[soma_idx], shoulder_offset_alpha).astype(np.float32)
            inv_offset = torch.from_numpy(offset.T).to(smpl_global)
            smpl_global[:, smpl_idx] = smpl_global[:, smpl_idx] @ inv_offset

    local = torch.empty_like(smpl_global)
    for j, p in enumerate(SMPL22_PARENTS.tolist()):
        if p < 0:
            local[:, j] = smpl_global[:, j]
        else:
            local[:, j] = torch.einsum("tki,tkl->til", smpl_global[:, p], smpl_global[:, j])

    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32)).to(local.device)
    source_local = rot6d_to_rotmat_row_major(source[:, 3:135].reshape(t, N_SMPL_JOINTS, 6)).to(local.device)
    source_pos, _ = differentiable_fk(source_local, source[:, :3].to(local.device), offsets)
    transl = source[:, :3].to(local.device).clone()
    fitted, _ = differentiable_fk(local, transl, offsets)

    if height_mode == "foot_floor":
        foot_indices = [_SMPL22_IDX[x] for x in ("left_ankle", "right_ankle", "left_foot", "right_foot")]
        y_delta = fitted[:, foot_indices, 1].min(dim=1).values - source_pos[:, foot_indices, 1].min(dim=1).values
        if torch.max(torch.abs(y_delta)) > 1e-5:
            transl[:, 1] -= y_delta
            fitted, _ = differentiable_fk(local, transl, offsets)
    elif height_mode != "source_root":
        raise ValueError(f"unknown height_mode: {height_mode}")

    rot6d = torch.from_numpy(matrix_to_rot6d_rowmajor(_as_numpy(local)).reshape(t, N_SMPL_JOINTS * 6)).to(transl.device)
    motion_135 = torch.cat([transl, rot6d], dim=-1).detach().cpu().numpy().astype(np.float32)
    aa = R.from_matrix(_as_numpy(local).reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(t, N_SMPL_JOINTS, 3)
    return {
        "motion_135": motion_135,
        "transl": transl.detach().cpu().numpy().astype(np.float32),
        "global_orient": aa[:, 0].astype(np.float32),
        "body_pose": aa[:, 1:].reshape(t, 63).astype(np.float32),
        "fitted_joints": fitted.detach().cpu().numpy().astype(np.float32),
    }


class SMPLSOMARetargeter:
    """Bidirectional retargeter for SMPL ``motion_135`` and SOMA30 rotations."""

    def __init__(self, config: SMPLToSOMAConfig | None = None, **overrides: Any) -> None:
        values = vars(config or SMPLToSOMAConfig()).copy()
        values.update({k: v for k, v in overrides.items() if v is not None})
        self.config = SMPLToSOMAConfig(**values)
        self.assets_root = _resolve_assets_root(self.config.assets_root)
        self.smpl_bone_offsets = _smpl22_bone_offsets(self.assets_root)
        self.soma_offsets = _soma30_offsets(self.assets_root)

    def smpl_to_soma(self, motion_135: np.ndarray) -> dict[str, np.ndarray]:
        soma_rots, soma_joints = _smpl22_to_soma30_retarget_shoulder_offset(
            motion_135,
            self.smpl_bone_offsets,
            self.soma_offsets,
            self.assets_root,
            self.config.shoulder_offset_alpha,
        )
        soma_local = _global_to_local_np(_as_numpy(soma_rots), SOMA30_PARENTS)
        return {
            "soma30_joints": _as_numpy(soma_joints).astype(np.float32),
            "soma30_global_rots": _as_numpy(soma_rots).astype(np.float32),
            "soma30_local_rots": soma_local.astype(np.float32),
        }

    def soma_to_smpl_from_rotations(
        self,
        soma30_global_rots: np.ndarray | torch.Tensor,
        source_motion_135: np.ndarray,
        height_mode: str | None = None,
    ) -> dict[str, np.ndarray]:
        return _soma30_to_smpl22_motion_rotation(
            soma30_global_rots,
            source_motion_135,
            self.smpl_bone_offsets,
            height_mode=height_mode or self.config.smpl_height_mode,
            assets_root=self.assets_root,
            shoulder_offset_alpha=self.config.shoulder_offset_alpha,
        )

    def roundtrip_smpl(self, motion_135: np.ndarray) -> dict[str, np.ndarray]:
        soma = self.smpl_to_soma(motion_135)
        smpl = self.soma_to_smpl_from_rotations(soma["soma30_global_rots"], motion_135)
        return {**soma, **smpl, "source_motion_135": np.asarray(motion_135, dtype=np.float32)}


def smpl_motion135_to_soma30(motion_135: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
    """One-call SMPL ``motion_135`` -> SOMA30 joints/rotations."""

    return SMPLSOMARetargeter(**kwargs).smpl_to_soma(motion_135)


def smpl_soma30_roundtrip(motion_135: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
    """One-call SMPL -> SOMA30 -> SMPL round trip."""

    return SMPLSOMARetargeter(**kwargs).roundtrip_smpl(motion_135)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def _estimate_local_rotations(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
) -> np.ndarray:
    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_SMPL_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_SMPL_JOINTS)]
    for j in range(1, N_SMPL_JOINTS):
        p = int(parents[j])
        if 0 <= p < N_SMPL_JOINTS:
            children[p].append(j)

    offsets = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_SMPL_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_SMPL_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_SMPL_JOINTS, 1, 1))
    for t, joints in enumerate(target_joints):
        for j in range(N_SMPL_JOINTS):
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            child_ids = children[j]
            rest_vecs = [offsets[c] for c in child_ids]
            target_vecs = [joints[c] - joints[j] for c in child_ids]
            weights = [1.0] * len(rest_vecs)
            if orientation_mode == "parent_frame" and parent >= 0:
                rest_vecs.append(rest_joints[parent] - rest_joints[j])
                target_vecs.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            if not rest_vecs:
                rot_local = np.eye(3)
            else:
                rest_unit, rest_valid = _safe_normalize(np.stack(rest_vecs, axis=0))
                target_unit, target_valid = _safe_normalize(np.stack(target_vecs, axis=0))
                valid = rest_valid & target_valid
                if not np.any(valid):
                    rot_local = np.eye(3)
                else:
                    dst_local = (parent_global.T @ target_unit[valid].T).T
                    try:
                        rot_local = R.align_vectors(
                            dst_local,
                            rest_unit[valid],
                            weights=np.asarray(weights, dtype=np.float64)[valid],
                        )[0].as_matrix()
                    except Exception:
                        rot_local = np.eye(3)
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _append_guide(
    rest_vecs: list[np.ndarray],
    target_vecs: list[np.ndarray],
    weights: list[float],
    rest_vec: np.ndarray,
    target_vec: np.ndarray,
    weight: float,
) -> None:
    if weight <= 0 or np.linalg.norm(target_vec) < 1e-6:
        return
    scale = max(float(np.linalg.norm(rest_vec)), 1e-3)
    rest_vecs.append(np.asarray(rest_vec, dtype=np.float64))
    target_vecs.append(np.asarray(target_vec, dtype=np.float64) / np.linalg.norm(target_vec) * scale)
    weights.append(float(weight))


def _add_soma77_orientation_guides(
    joint_index: int,
    soma: np.ndarray,
    rest_vecs: list[np.ndarray],
    target_vecs: list[np.ndarray],
    weights: list[float],
    head_weight: float,
    leaf_weight: float,
) -> None:
    if joint_index == 15:
        head = soma[SOMA77_IDX["Head"]]
        eye_mid = 0.5 * (soma[SOMA77_IDX["LeftEye"]] + soma[SOMA77_IDX["RightEye"]])
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.08, 0.0]), soma[SOMA77_IDX["HeadEnd"]] - head, head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.08, 0.0, 0.0]), soma[SOMA77_IDX["LeftEye"]] - soma[SOMA77_IDX["RightEye"]], head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.0, 0.08]), eye_mid - head, 0.5 * head_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, -0.05, 0.06]), soma[SOMA77_IDX["Jaw"]] - head, 0.35 * head_weight)
    elif joint_index == 20:
        hand = soma[SOMA77_IDX["LeftHand"]]
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.10, 0.0, 0.0]), soma[SOMA77_IDX["LeftHandMiddleEnd"]] - hand, leaf_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.06, -0.04, 0.04]), soma[SOMA77_IDX["LeftHandThumbEnd"]] - hand, 0.5 * leaf_weight)
    elif joint_index == 21:
        hand = soma[SOMA77_IDX["RightHand"]]
        _append_guide(rest_vecs, target_vecs, weights, np.array([-0.10, 0.0, 0.0]), soma[SOMA77_IDX["RightHandMiddleEnd"]] - hand, leaf_weight)
        _append_guide(rest_vecs, target_vecs, weights, np.array([-0.06, -0.04, 0.04]), soma[SOMA77_IDX["RightHandThumbEnd"]] - hand, 0.5 * leaf_weight)
    elif joint_index == 10:
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.0, 0.12]), soma[SOMA77_IDX["LeftToeEnd"]] - soma[SOMA77_IDX["LeftToeBase"]], leaf_weight)
    elif joint_index == 11:
        _append_guide(rest_vecs, target_vecs, weights, np.array([0.0, 0.0, 0.12]), soma[SOMA77_IDX["RightToeEnd"]] - soma[SOMA77_IDX["RightToeBase"]], leaf_weight)


def _estimate_local_rotations_with_soma77_guides(
    target_joints: np.ndarray,
    soma77: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str,
    parent_ref_weight: float,
    head_guide_weight: float,
    leaf_guide_weight: float,
) -> np.ndarray:
    target_joints = np.asarray(target_joints, dtype=np.float64)
    soma77 = np.asarray(soma77, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_SMPL_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_SMPL_JOINTS)]
    for j in range(1, N_SMPL_JOINTS):
        parent = int(parents[j])
        if 0 <= parent < N_SMPL_JOINTS:
            children[parent].append(j)

    offsets = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_SMPL_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_SMPL_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_SMPL_JOINTS, 1, 1))
    for t, joints in enumerate(target_joints):
        soma = soma77[min(t, len(soma77) - 1)]
        for j in range(N_SMPL_JOINTS):
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs = [offsets[c] for c in children[j]]
            target_vecs = [joints[c] - joints[j] for c in children[j]]
            weights = [1.0] * len(rest_vecs)
            if orientation_mode == "parent_frame" and parent >= 0:
                rest_vecs.append(rest_joints[parent] - rest_joints[j])
                target_vecs.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            _add_soma77_orientation_guides(j, soma, rest_vecs, target_vecs, weights, head_guide_weight, leaf_guide_weight)
            if not rest_vecs:
                rot_local = np.eye(3)
            else:
                rest_unit, rest_valid = _safe_normalize(np.stack(rest_vecs, axis=0))
                target_unit, target_valid = _safe_normalize(np.stack(target_vecs, axis=0))
                valid = rest_valid & target_valid
                if not np.any(valid):
                    rot_local = np.eye(3)
                else:
                    dst_local = (parent_global.T @ target_unit[valid].T).T
                    try:
                        rot_local = R.align_vectors(
                            dst_local,
                            rest_unit[valid],
                            weights=np.asarray(weights, dtype=np.float64)[valid],
                        )[0].as_matrix()
                    except Exception:
                        rot_local = np.eye(3)
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _make_joint_fit_weights(preset: str) -> np.ndarray | None:
    if preset == "uniform":
        return None
    weights = np.ones(N_SMPL_JOINTS, dtype=np.float32)
    if preset == "relaxed_torso":
        for j in [6, 9, 12, 15]:
            weights[j] = 0.15
        for j in [3, 13, 14]:
            weights[j] = 0.4
        return weights
    if preset == "relaxed_upper":
        for j in [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            weights[j] = 0.08
        for j in [10, 11]:
            weights[j] = 0.25
        return weights
    raise ValueError(f"unknown joint fit weight preset: {preset}")


def _resolve_smpl_model_dir(model_dir: str | Path | None) -> Path:
    resolved = model_dir or os.environ.get("MOTIUS_SMPL_MODEL_DIR")
    if not resolved:
        raise FileNotFoundError(
            "SMPL model assets are required; pass model_dir=... or set MOTIUS_SMPL_MODEL_DIR"
        )
    path = Path(resolved)
    if not path.is_absolute():
        path = _repo_root() / path
    candidates = []
    if path.name == "body_models":
        candidates.append(path.with_name("body_models_nochumpy"))
    candidates.append(path)
    for candidate in candidates:
        if (candidate / "smpl/SMPL_NEUTRAL.pkl").exists():
            return candidate
    return path


def _load_smpl_rest(model_dir: str | Path, device: torch.device):
    smplx = _import_smplx()
    model_dir = _resolve_smpl_model_dir(model_dir)
    model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
    rest = out.joints[0, :N_SMPL_JOINTS].detach().cpu().numpy().astype(np.float32)
    parents = model.parents.detach().cpu().numpy().astype(np.int64)
    return model, rest, parents


def _smpl_forward_22(
    model,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(global_orient), batch_size):
        end = min(start + batch_size, len(global_orient))
        b = end - start
        body_23 = np.zeros((b, 69), dtype=np.float32)
        body_23[:, :63] = body_pose_21[start:end]
        tr = np.zeros((b, 3), dtype=np.float32) if transl is None else transl[start:end]
        with torch.no_grad():
            out = model(
                betas=torch.zeros(b, 10, device=device),
                body_pose=torch.from_numpy(body_23).to(device),
                global_orient=torch.from_numpy(global_orient[start:end]).to(device),
                transl=torch.from_numpy(tr).to(device),
            )
        chunks.append(out.joints[:, :N_SMPL_JOINTS].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _smpl_forward_22_and_vertex_floor(
    model,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    joint_chunks = []
    floor_chunks = []
    for start in range(0, len(global_orient), batch_size):
        end = min(start + batch_size, len(global_orient))
        b = end - start
        body_23 = np.zeros((b, 69), dtype=np.float32)
        body_23[:, :63] = body_pose_21[start:end]
        with torch.no_grad():
            out = model(
                betas=torch.zeros(b, 10, device=device),
                body_pose=torch.from_numpy(body_23).to(device),
                global_orient=torch.from_numpy(global_orient[start:end]).to(device),
                transl=torch.from_numpy(transl[start:end]).to(device),
            )
        joints = out.joints[:, :N_SMPL_JOINTS]
        joint_chunks.append(joints.detach().cpu().numpy().astype(np.float32))
        if hasattr(out, "vertices"):
            floor = out.vertices[..., 1].amin(dim=1)
        else:
            floor = joints[:, FOOT_HEIGHT_JOINTS, 1].amin(dim=1)
        floor_chunks.append(floor.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(joint_chunks, axis=0), np.concatenate(floor_chunks, axis=0)


def _refine_smpl_fit(
    model,
    target_joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray,
    config: SOMAToSMPLIKConfig,
    device: torch.device,
    joint_fit_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.refine_iters <= 0:
        fitted = _smpl_forward_22(model, global_orient, body_pose_21, transl, config.batch_size, device)
        return global_orient, body_pose_21, transl, fitted

    target = torch.from_numpy(target_joints.astype(np.float32)).to(device)
    if joint_fit_weights is None:
        fit_weights = torch.ones((1, N_SMPL_JOINTS), dtype=torch.float32, device=device)
    else:
        fit_weights = torch.from_numpy(joint_fit_weights.reshape(1, N_SMPL_JOINTS).astype(np.float32)).to(device)
    fit_weights = fit_weights / fit_weights.mean().clamp_min(1e-6)
    n = len(target_joints)
    g = torch.tensor(global_orient, dtype=torch.float32, device=device, requires_grad=True)
    b21 = torch.tensor(body_pose_21, dtype=torch.float32, device=device, requires_grad=True)
    tr = torch.tensor(transl, dtype=torch.float32, device=device, requires_grad=True)
    b21_init = b21.detach().clone()
    opt = torch.optim.Adam([g, b21, tr], lr=config.refine_lr)

    for _ in range(config.refine_iters):
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        joints = out.joints[:, :N_SMPL_JOINTS]
        data_loss = (((joints - target) ** 2).sum(dim=-1) * fit_weights).mean()
        pose_keep = ((b21 - b21_init) ** 2).mean()
        pose_prior = (body_23 ** 2).mean()
        if config.angle_prior_weight > 0:
            idx = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=device)
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=device)
            angle_prior = torch.exp(body_23[:, idx] * signs).pow(2).mean()
        else:
            angle_prior = torch.tensor(0.0, device=device)
        if n >= 3:
            tr_acc = tr[2:] - 2 * tr[1:-1] + tr[:-2]
            pose_acc = b21[2:] - 2 * b21[1:-1] + b21[:-2]
            smooth = (tr_acc ** 2).mean() + 1e-2 * (pose_acc ** 2).mean()
            if config.joint_accel_weight > 0:
                joints_acc = joints[2:] - 2 * joints[1:-1] + joints[:-2]
                target_acc = target[2:] - 2 * target[1:-1] + target[:-2]
                joint_accel = ((joints_acc - target_acc) ** 2).sum(dim=-1).mean()
            else:
                joint_accel = torch.tensor(0.0, device=device)
        else:
            smooth = torch.tensor(0.0, device=device)
            joint_accel = torch.tensor(0.0, device=device)
        loss = (
            data_loss
            + 1e-4 * pose_keep
            + config.pose_l2_weight * pose_prior
            + config.angle_prior_weight * angle_prior
            + config.smooth_weight * smooth
            + config.joint_accel_weight * joint_accel
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        fitted = out.joints[:, :N_SMPL_JOINTS].detach().cpu().numpy().astype(np.float32)
    return (
        g.detach().cpu().numpy().astype(np.float32),
        b21.detach().cpu().numpy().astype(np.float32),
        tr.detach().cpu().numpy().astype(np.float32),
        fitted,
    )


class KIMODOSOMAToSMPLRetargeter:
    """Retarget KIMODO/SOMA skeleton outputs to SMPL ``motion_135``."""

    def __init__(self, config: SOMAToSMPLIKConfig | None = None, **overrides: Any) -> None:
        values = vars(config or SOMAToSMPLIKConfig()).copy()
        values.update({k: v for k, v in overrides.items() if v is not None})
        self.config = SOMAToSMPLIKConfig(**values)
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.rest_joints, self.parents = _load_smpl_rest(self.config.model_dir, self.device)
        self.joint_fit_weights = _make_joint_fit_weights(self.config.joint_fit_weight_preset)
        self.assets_root = _resolve_assets_root(None)
        self.smpl_bone_offsets = _smpl22_bone_offsets(self.assets_root)

    def _align_rotation_result_floor(
        self,
        result: dict[str, np.ndarray],
        target_joints: np.ndarray | None,
        soma77: np.ndarray | None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        target_floor_y = 0.0
        floor_source = "zero"
        if soma77 is not None and soma77.ndim == 3 and soma77.shape[1] > 75:
            target_floor_y = float(soma77[:, [69, 70, 74, 75], 1].min())
            floor_source = "soma77_feet"
        elif target_joints is not None:
            target_floor_y = float(target_joints[:, FOOT_HEIGHT_JOINTS, 1].min())
            floor_source = "smpl22_target_feet"

        target_for_fit = target_joints
        if target_joints is not None and self.config.floor_align:
            target_for_fit = target_joints.copy()
            target_for_fit[..., 1] -= target_floor_y

        if not self.config.foot_height_align:
            return result, target_for_fit

        transl = np.asarray(result["transl"], dtype=np.float32).copy()
        global_orient = np.asarray(result["global_orient"], dtype=np.float32)
        body_pose = np.asarray(result["body_pose"], dtype=np.float32)
        body_joints, vertex_floor_y = _smpl_forward_22_and_vertex_floor(
            self.model,
            global_orient,
            body_pose,
            transl,
            self.config.batch_size,
            self.device,
        )
        desired_floor_y = 0.0 if self.config.floor_align else target_floor_y
        smpl_floor_before = float(vertex_floor_y.min())
        floor_delta = float(desired_floor_y - smpl_floor_before)
        if abs(floor_delta) > 1e-6:
            transl[:, 1] += floor_delta
            result = {**result}
            result["transl"] = transl.astype(np.float32)
            motion_135 = np.asarray(result["motion_135"], dtype=np.float32).copy()
            motion_135[:, 1] += floor_delta
            result["motion_135"] = motion_135
            body_joints = body_joints.copy()
            body_joints[..., 1] += floor_delta
            vertex_floor_y = vertex_floor_y + floor_delta
        else:
            result = {**result}
        result["fitted_joints"] = body_joints.astype(np.float32)
        result["height_align_delta_y"] = np.array(floor_delta, dtype=np.float32)
        result["smpl_vertex_floor_y_before"] = np.array(smpl_floor_before, dtype=np.float32)
        result["smpl_vertex_floor_y_after"] = np.array(float(vertex_floor_y.min()), dtype=np.float32)
        result["target_floor_y"] = np.array(target_floor_y, dtype=np.float32)
        result["height_align_source"] = np.array(floor_source, dtype=object)
        return result, target_for_fit

    def retarget_rotations(
        self,
        global_rot_mats: np.ndarray,
        positions22: np.ndarray | None = None,
        root_positions: np.ndarray | None = None,
        soma77: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Retarget KIMODO/SOMA global rotations to SMPL ``motion_135``.

        ``global_rot_mats`` may be SOMA-77 or SOMA-30. This path is preferred
        for KIMODO mesh visualization because position-only IK cannot recover
        twist and upper-body orientation reliably.
        """
        rots = np.asarray(global_rot_mats, dtype=np.float32)
        if rots.ndim != 4 or rots.shape[-2:] != (3, 3):
            raise ValueError(f"expected global_rot_mats with shape (T,J,3,3), got {rots.shape}")
        if rots.shape[1] == 77:
            soma30_rots = rots[:, SOMA30_IN_SOMA77]
        elif rots.shape[1] == len(SOMA30_NAMES):
            soma30_rots = rots
        else:
            raise ValueError(
                "expected global_rot_mats joint dimension 30 or 77, "
                f"got {rots.shape[1]}"
            )

        target = None if positions22 is None else np.asarray(positions22, dtype=np.float32)
        if root_positions is not None:
            transl = np.asarray(root_positions, dtype=np.float32)
        elif target is not None:
            transl = target[:, 0]
        else:
            transl = np.zeros((rots.shape[0], 3), dtype=np.float32)
        if transl.shape != (rots.shape[0], 3):
            raise ValueError(f"expected root positions with shape ({rots.shape[0]},3), got {transl.shape}")

        source = np.zeros((rots.shape[0], 135), dtype=np.float32)
        source[:, :3] = transl
        source[:, 3:] = matrix_to_rot6d_rowmajor(
            np.broadcast_to(np.eye(3, dtype=np.float32), (rots.shape[0], N_SMPL_JOINTS, 3, 3))
        ).reshape(rots.shape[0], N_SMPL_JOINTS * 6)
        result = _soma30_to_smpl22_motion_rotation(
            soma30_rots,
            source,
            self.smpl_bone_offsets,
            height_mode=self.config.smpl_height_mode,
            assets_root=self.assets_root,
        )
        if target is not None:
            if target.shape != (rots.shape[0], N_SMPL_JOINTS, 3):
                raise ValueError(
                    f"expected positions22 with shape ({rots.shape[0]},{N_SMPL_JOINTS},3), "
                    f"got {target.shape}"
                )
        soma77_arr = None if soma77 is None else np.asarray(soma77, dtype=np.float32)
        result, target_for_fit = self._align_rotation_result_floor(result, target, soma77_arr)
        if target_for_fit is not None:
            result["target_joints"] = target_for_fit.astype(np.float32)
            result["fit_mpjpe_mm"] = (
                np.linalg.norm(result["fitted_joints"] - target_for_fit, axis=-1).mean(axis=1) * 1000.0
            ).astype(np.float32)
        result["retarget_method"] = np.array("soma_global_rotation_transfer", dtype=object)
        return result

    def retarget_positions(
        self,
        positions22: np.ndarray,
        soma77: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        target = np.asarray(positions22, dtype=np.float32).copy()
        if target.ndim != 3 or target.shape[1:] != (N_SMPL_JOINTS, 3):
            raise ValueError(f"expected positions22 with shape (T,{N_SMPL_JOINTS},3), got {target.shape}")
        soma77_arr = None if soma77 is None else np.asarray(soma77, dtype=np.float32).copy()
        if self.config.floor_align:
            floor = float(target[..., 1].min())
            target[..., 1] -= floor
            if soma77_arr is not None:
                soma77_arr[..., 1] -= floor

        if self.config.soma_orientation_guides and soma77_arr is not None:
            local_r = _estimate_local_rotations_with_soma77_guides(
                target,
                soma77_arr,
                self.rest_joints,
                self.parents,
                self.config.orientation_mode,
                self.config.parent_ref_weight,
                self.config.head_guide_weight,
                self.config.leaf_guide_weight,
            )
        else:
            local_r = _estimate_local_rotations(
                target,
                self.rest_joints,
                self.parents,
                self.config.orientation_mode,
                self.config.parent_ref_weight,
            )

        aa = R.from_matrix(local_r.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
        aa = aa.reshape(len(target), N_SMPL_JOINTS, 3)
        global_orient = aa[:, 0]
        body_pose = aa[:, 1:].reshape(len(target), 63)
        joints_no_trans = _smpl_forward_22(
            self.model,
            global_orient,
            body_pose,
            None,
            self.config.batch_size,
            self.device,
        )
        transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
        global_orient, body_pose, transl, fitted = _refine_smpl_fit(
            self.model,
            target,
            global_orient,
            body_pose,
            transl,
            self.config,
            self.device,
            self.joint_fit_weights,
        )
        if self.config.foot_height_align:
            target_floor_y = target[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
            fitted_floor_y = fitted[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
            y_delta = (target_floor_y - fitted_floor_y).astype(np.float32)
            transl = transl.copy()
            fitted = fitted.copy()
            transl[:, 1] += y_delta
            fitted[..., 1] += y_delta[:, None]

        local_r = R.from_rotvec(
            np.concatenate([global_orient[:, None, :], body_pose.reshape(len(target), 21, 3)], axis=1).reshape(-1, 3)
        ).as_matrix().reshape(len(target), N_SMPL_JOINTS, 3, 3).astype(np.float32)
        motion_135 = np.concatenate(
            [transl, matrix_to_rot6d_rowmajor(local_r).reshape(len(target), N_SMPL_JOINTS * 6)],
            axis=-1,
        ).astype(np.float32)
        mpjpe_mm = (np.linalg.norm(fitted - target, axis=-1).mean(axis=1) * 1000.0).astype(np.float32)
        return {
            "motion_135": motion_135,
            "transl": transl.astype(np.float32),
            "global_orient": global_orient.astype(np.float32),
            "body_pose": body_pose.astype(np.float32),
            "target_joints": target.astype(np.float32),
            "fitted_joints": fitted.astype(np.float32),
            "fit_mpjpe_mm": mpjpe_mm,
            "retarget_method": np.array("soma_position_ik_fallback", dtype=object),
        }

    def retarget_file(self, path: str | Path) -> dict[str, np.ndarray]:
        path = Path(path)
        if path.suffix == ".npy":
            return self.retarget_positions(np.load(path).astype(np.float32))
        with np.load(path, allow_pickle=True) as data:
            if "positions" not in data.files:
                raise KeyError(f"{path} has no 'positions' key")
            positions = np.asarray(data["positions"], dtype=np.float32)
            soma77 = np.asarray(data["posed_joints"], dtype=np.float32) if "posed_joints" in data.files else None
            if "global_rot_mats" in data.files:
                root_positions = (
                    np.asarray(data["root_positions"], dtype=np.float32)
                    if "root_positions" in data.files
                    else positions[:, 0]
                )
                return self.retarget_rotations(
                    np.asarray(data["global_rot_mats"], dtype=np.float32),
                    positions22=positions,
                    root_positions=root_positions,
                    soma77=soma77,
                )
        return self.retarget_positions(positions, soma77)

    @staticmethod
    def save_npz(path: str | Path, result: dict[str, np.ndarray], **metadata: Any) -> None:
        arrays: dict[str, Any] = {k: v for k, v in result.items()}
        arrays.update(metadata)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def kimodo_soma_to_smpl_motion135(
    positions22: np.ndarray,
    soma77: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """One-call KIMODO/SOMA 22-joint positions -> SMPL ``motion_135``."""

    return KIMODOSOMAToSMPLRetargeter(**kwargs).retarget_positions(positions22, soma77)


__all__ = [
    "N_SMPL_JOINTS",
    "SMPL22_NAMES",
    "SMPL22_PARENTS",
    "SOMA30_NAMES",
    "SOMA30_PARENTS",
    "SMPL22_TO_SOMA30",
    "SOMA77_IDX",
    "SOMA30_IN_SOMA77",
    "SMPLToSOMAConfig",
    "SOMAToSMPLIKConfig",
    "SMPLSOMARetargeter",
    "KIMODOSOMAToSMPLRetargeter",
    "smpl_motion135_to_soma30",
    "smpl_soma30_roundtrip",
    "kimodo_soma_to_smpl_motion135",
    "rot6d_to_rotmat_row_major",
    "matrix_to_rot6d_rowmajor",
]
