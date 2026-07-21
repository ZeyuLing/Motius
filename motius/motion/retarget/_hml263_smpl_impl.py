"""IK/FK backend for HumanML3D-263 to SMPL-style motion135 retargeting.

This script keeps the HumanML3D-to-SMPL conversion auditable in one place:

    HML3D-263 -> 22 joints -> hierarchical IK on SMPL rest skeleton
              -> global_orient/body_pose/transl + motion_135

The conversion is not mathematically exact: HumanML3D-263 does not uniquely
determine SMPL pose twist, shape, or mesh details.  The default ``auto`` path
maps the HML263 rotation block from the canonical HumanML skeleton to the SMPL
rest skeleton for full feature input, and falls back to position IK only when
the input is already raw joints.  The saved fit MPJPE is a diagnostic for how
well the SMPL skeleton tracks the recovered 22 joints, but it does not by itself
validate terminal mesh orientation.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp


def _patch_numpy_chumpy_aliases() -> None:
    """Keep legacy SMPL/chumpy pickles loadable under newer NumPy releases."""
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
    # NumPy 2.x pickles reference ``numpy._core`` while NumPy 1.x exposes the
    # same implementation as ``numpy.core``. SMPL assets exist in both forms.
    import numpy.core.multiarray as numpy_multiarray

    private_core = types.ModuleType("numpy._core")
    private_core.__path__ = []
    private_multiarray = types.ModuleType("numpy._core.multiarray")
    private_multiarray._reconstruct = numpy_multiarray._reconstruct
    private_core.multiarray = private_multiarray
    sys.modules.setdefault("numpy._core", private_core)
    sys.modules.setdefault("numpy._core.multiarray", private_multiarray)


_patch_numpy_chumpy_aliases()


def _ensure_chumpy_unpickle_support() -> None:
    """Provide the tiny part of chumpy needed by legacy SMPL pickles."""
    try:
        import chumpy.ch  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    class Ch:
        @property
        def r(self) -> np.ndarray:
            return np.asarray(getattr(self, "x"))

        @property
        def shape(self) -> tuple[int, ...]:
            return self.r.shape

        def __array__(self, dtype=None):
            return np.asarray(self.r, dtype=dtype)

        def __getitem__(self, item):
            return self.r[item]

        def __len__(self) -> int:
            return len(self.r)

    ch_mod = types.ModuleType("chumpy.ch")
    ch_mod.Ch = Ch
    root_mod = types.ModuleType("chumpy")
    root_mod.ch = ch_mod
    sys.modules.setdefault("chumpy", root_mod)
    sys.modules.setdefault("chumpy.ch", ch_mod)


_ensure_chumpy_unpickle_support()

import smplx  # noqa: E402


# HumanML3D 22-joint skeleton order follows the first 22 SMPL joints.
N_JOINTS = 22
HML263_END_EFFECTOR_JOINTS = np.asarray([10, 11, 15, 20, 21], dtype=np.int64)
HML263_RAW_OFFSETS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


def _qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _recover_root_rot_pos(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)
    r_rot_quat = np.zeros(data.shape[:-1] + (4,), dtype=data.dtype)
    r_rot_quat[..., 0] = np.cos(r_rot_ang)
    r_rot_quat[..., 2] = np.sin(r_rot_ang)
    r_pos = np.zeros(data.shape[:-1] + (3,), dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = N_JOINTS) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (-1, 3))
    q = _qinv(r_rot_quat)[..., None, :]
    q = np.broadcast_to(q, positions.shape[:-1] + (4,))
    positions = _qrot(q, positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return np.concatenate([r_pos[..., None, :], positions], axis=-2)


def cont6d_to_matrix_hml(cont6d: np.ndarray) -> np.ndarray:
    """HumanML/MoMask column-major cont6d -> rotation matrix."""
    cont6d = np.asarray(cont6d, dtype=np.float32)
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=-1, keepdims=True), 1e-8)
    z = np.cross(x, y_raw)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    xyzw = quat[..., [1, 2, 3, 0]]
    return R.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)


def recover_hml263_local_rotations(feats: np.ndarray, joints_num: int = N_JOINTS) -> np.ndarray:
    """Recover HumanML canonical-skeleton local rotations from a 263D clip.

    These are not original SMPL rotations. They are the local rotations produced
    by HumanML3D/MoMask inverse kinematics on the canonical 22-joint skeleton,
    and are useful as a twist/orientation prior for SMPL fitting.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[-1] != 263:
        raise ValueError(f"expected HML263 features (T,263), got {feats.shape}")
    root_quat, _ = _recover_root_rot_pos(feats)
    root_mat = quat_wxyz_to_matrix(root_quat)
    start = 4 + (joints_num - 1) * 3
    end = start + (joints_num - 1) * 6
    body = feats[:, start:end].reshape(len(feats), joints_num - 1, 6)
    body_mat = cont6d_to_matrix_hml(body)
    return np.concatenate([root_mat[:, None], body_mat], axis=1).astype(np.float32)


def _slerp_length(rot: np.ndarray, target_len: int) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32)
    if target_len <= 0 or len(rot) == target_len:
        return rot
    if len(rot) < 2:
        return np.repeat(rot[:1], target_len, axis=0).astype(np.float32)
    src_times = np.arange(len(rot), dtype=np.float64)
    dst_times = np.linspace(0.0, len(rot) - 1, int(target_len), dtype=np.float64)
    out = np.empty((int(target_len), rot.shape[1], 3, 3), dtype=np.float32)
    for j in range(rot.shape[1]):
        out[:, j] = Slerp(src_times, R.from_matrix(rot[:, j]))(dst_times).as_matrix().astype(np.float32)
    return out


def resample_rotations(rot: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if abs(src_fps - dst_fps) < 1e-6 or len(rot) < 2:
        return np.asarray(rot, dtype=np.float32)
    return _slerp_length(rot, max(2, int(round(len(rot) * dst_fps / src_fps))))


def fit_length_rotations(rot: np.ndarray, target_len: int | None) -> np.ndarray:
    if target_len is None:
        return np.asarray(rot, dtype=np.float32)
    return _slerp_length(rot, int(target_len))


def resample_linear(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-6 or len(x) < 2:
        return x
    new_t = max(2, int(round(len(x) * dst_fps / src_fps)))
    grid = np.linspace(0.0, len(x) - 1, new_t)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    shape = (new_t,) + (1,) * (x.ndim - 1)
    return x[lo] * (1.0 - w.reshape(shape)) + x[hi] * w.reshape(shape)


def fit_length_linear(x: np.ndarray, target_len: int | None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if target_len is None:
        return x
    target_len = int(target_len)
    if target_len <= 0 or len(x) == target_len:
        return x
    if len(x) < 2:
        if len(x) == 0:
            return x
        return np.repeat(x[:1], target_len, axis=0).astype(np.float32)
    grid = np.linspace(0.0, len(x) - 1, target_len)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    shape = (target_len,) + (1,) * (x.ndim - 1)
    return (x[lo] * (1.0 - w.reshape(shape)) + x[hi] * w.reshape(shape)).astype(np.float32)


def _load_canonical_meta(meta_dir: Path | None, sid: str) -> dict[str, np.ndarray] | None:
    if meta_dir is None:
        return None
    path = meta_dir / f"{sid}.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _restore_root_translation(
    transl: np.ndarray,
    meta: dict[str, np.ndarray] | None,
    mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if mode == "none" or meta is None:
        return transl.astype(np.float32), {"mode": "none", "applied": False}
    if mode not in {"auto", "source_transl"}:
        raise ValueError(f"unsupported root translation restore mode: {mode}")
    if "source_motion135_transl" not in meta:
        return transl.astype(np.float32), {
            "mode": mode,
            "applied": False,
            "reason": "source_motion135_transl_missing",
        }
    source = np.asarray(meta["source_motion135_transl"], dtype=np.float32)
    restored = fit_length_linear(source, len(transl)).astype(np.float32)
    return restored, {
        "mode": mode,
        "applied": True,
        "source_frames": int(len(source)),
        "output_frames": int(len(restored)),
    }


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def _preserve_twist_continuity(
    current: np.ndarray,
    previous: np.ndarray,
    bone_axis: np.ndarray,
) -> np.ndarray:
    """Choose the one-bone IK twist closest to the preceding frame.

    Aligning one rest bone to one observed bone determines swing but leaves
    rotation around the rest bone unconstrained. Right-multiplying by a twist
    around that rest axis preserves the observed child position. Projecting the
    current-to-previous relative quaternion onto that axis selects the smoothest
    member of this equivalent family without changing the positional fit.
    """

    axis = np.asarray(bone_axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        return current
    axis /= norm
    relative = np.asarray(current, dtype=np.float64).T @ np.asarray(
        previous, dtype=np.float64
    )
    quat = R.from_matrix(relative).as_quat()
    projected = axis * float(np.dot(quat[:3], axis))
    twist = np.concatenate([projected, quat[3:4]])
    twist_norm = float(np.linalg.norm(twist))
    if twist_norm <= 1e-8:
        return current
    twist /= twist_norm
    return np.asarray(current, dtype=np.float64) @ R.from_quat(twist).as_matrix()


def estimate_local_rotations(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
    temporal_twist_stabilization: bool = True,
) -> np.ndarray:
    """Estimate local rotations by aligning SMPL rest bones to target bones."""
    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_JOINTS)]
    for j in range(1, N_JOINTS):
        p = int(parents[j])
        if 0 <= p < N_JOINTS:
            children[p].append(j)

    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))

    for t, joints in enumerate(target_joints):
        for j in range(N_JOINTS):
            child_ids = children[j]
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs_list = [offsets[c] for c in child_ids]
            target_vecs_list = [joints[c] - joints[j] for c in child_ids]
            weights = [1.0] * len(rest_vecs_list)
            if orientation_mode == "parent_frame" and parent >= 0:
                # Position-only IK leaves twist around a single bone undefined.
                # A weak joint-to-parent reference chooses a stable local frame
                # without letting the virtual axis dominate child-bone fitting.
                rest_vecs_list.append(rest_joints[parent] - rest_joints[j])
                target_vecs_list.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            if not rest_vecs_list:
                local[t, j] = np.eye(3)
                global_r[t, j] = parent_global @ local[t, j]
                continue

            rest_vecs = np.stack(rest_vecs_list, axis=0)
            target_vecs = np.stack(target_vecs_list, axis=0)
            rest_unit, rest_valid = _safe_normalize(rest_vecs)
            target_unit, target_valid = _safe_normalize(target_vecs)
            valid = rest_valid & target_valid
            if not np.any(valid):
                rot_local = np.eye(3)
            else:
                src = rest_unit[valid]
                dst_world = target_unit[valid]
                dst_local = (parent_global.T @ dst_world.T).T
                valid_weights = np.asarray(weights, dtype=np.float64)[valid]
                try:
                    rot_local = R.align_vectors(dst_local, src, weights=valid_weights)[0].as_matrix()
                except Exception:
                    rot_local = np.eye(3)
            if (
                temporal_twist_stabilization
                and t > 0
                and orientation_mode == "bone"
                and len(child_ids) == 1
            ):
                rot_local = _preserve_twist_continuity(
                    rot_local,
                    local[t - 1, j],
                    offsets[child_ids[0]],
                )
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _global_rotations_from_local(local_r: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local_r = np.asarray(local_r, dtype=np.float32)
    parents = np.asarray(parents[: local_r.shape[1]], dtype=np.int64)
    out = np.zeros_like(local_r)
    for j in range(local_r.shape[1]):
        parent = int(parents[j])
        if parent < 0:
            out[:, j] = local_r[:, j]
        else:
            out[:, j] = out[:, parent] @ local_r[:, j]
    return out


def hml263_rotations_to_smpl_init(
    hml_local_r: np.ndarray,
    position_local_r: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Convert HumanML IK rotations into an SMPL-local initialization.

    HumanML3D stores local rotations on the canonical T2M skeleton.  In that
    skeleton, the rotation attached to joint ``j`` is used together with the
    incoming offset ``parent(j) -> j``.  SMPL local pose instead defines the
    skinning/outgoing frame at joint ``j``.  Directly copying the HumanML local
    matrices into SMPL therefore corrupts terminal mesh orientation.

    This initializer keeps the useful twist/orientation signal from HML263 by
    first converting HML local matrices to global joint frames, aligning each
    SMPL incoming rest bone to the corresponding HumanML raw incoming axis, and
    then converting those desired global frames back to SMPL local rotations.
    The root frame is copied directly from the HumanML root quaternion.  Unlike
    body twist, HumanML root heading is an explicit trajectory channel and should
    survive HML263 <-> SMPL conversion without being re-estimated from positions.
    """

    if hml_local_r.shape != position_local_r.shape:
        raise ValueError(f"rotation shapes differ: {hml_local_r.shape} vs {position_local_r.shape}")
    parents22 = np.asarray(parents[: N_JOINTS], dtype=np.int64)
    hml_global = _global_rotations_from_local(hml_local_r, parents22)
    pos_global = _global_rotations_from_local(position_local_r, parents22)
    rest_joints = np.asarray(rest_joints[: N_JOINTS], dtype=np.float32)

    desired_global = np.empty_like(hml_global)
    desired_global[:, 0] = hml_global[:, 0]
    for j in range(1, N_JOINTS):
        parent = int(parents22[j])
        smpl_incoming = rest_joints[j] - rest_joints[parent]
        hml_incoming = HML263_RAW_OFFSETS[j]
        correction = _align_vector_matrix(smpl_incoming, hml_incoming)
        desired_global[:, j] = hml_global[:, j] @ correction

    out = np.empty_like(desired_global)
    out[:, 0] = desired_global[:, 0]
    for j in range(1, N_JOINTS):
        parent = int(parents22[j])
        out[:, j] = np.einsum(
            "tij,tjk->tik",
            desired_global[:, parent].transpose(0, 2, 1),
            desired_global[:, j],
        )
    return out.astype(np.float32)


def _align_vector_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / max(np.linalg.norm(src), 1e-8)
    dst = dst / max(np.linalg.norm(dst), 1e-8)
    try:
        return R.align_vectors(dst[None], src[None])[0].as_matrix().astype(np.float32)
    except Exception:
        return np.eye(3, dtype=np.float32)


def merge_hml263_end_effectors(
    position_local_r: np.ndarray,
    hml_local_r: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Diagnostic injection of HML263 terminal orientation.

    HumanML3D/MoMask ``cont6d_params[j]`` rotates the incoming bone
    ``parent(j) -> j`` on the canonical skeleton. SMPL local pose ``pose[j]``
    instead defines joint ``j``'s own skinning/child frame. Copying HML local
    rotations directly into SMPL local rotations therefore deforms terminal
    meshes. This function is kept only for explicit diagnostics; it must pass a
    mesh-integrity check before any result using it is trusted.
    """

    if position_local_r.shape != hml_local_r.shape:
        raise ValueError(f"rotation shapes differ: {position_local_r.shape} vs {hml_local_r.shape}")
    out = position_local_r.copy()
    parents22 = np.asarray(parents[: N_JOINTS], dtype=np.int64)
    smpl_global = _global_rotations_from_local(position_local_r, parents22)
    hml_global = _global_rotations_from_local(hml_local_r, parents22)
    rest_joints = np.asarray(rest_joints[: N_JOINTS], dtype=np.float32)
    for j in HML263_END_EFFECTOR_JOINTS:
        parent = int(parents22[int(j)])
        if parent < 0:
            continue
        smpl_incoming = rest_joints[int(j)] - rest_joints[parent]
        hml_incoming = HML263_RAW_OFFSETS[int(j)]
        correction = _align_vector_matrix(smpl_incoming, hml_incoming)
        desired_global = hml_global[:, int(j)] @ correction
        out[:, int(j)] = np.einsum("tij,tjk->tik", smpl_global[:, parent].transpose(0, 2, 1), desired_global)
    return out


def matrix_to_rot6d_rowmajor(rotmat: np.ndarray) -> np.ndarray:
    return np.asarray(rotmat[..., :, :2], dtype=np.float32).reshape(*rotmat.shape[:-2], 6)


def matrix_to_rot6d(rotmat: np.ndarray, convention: str = "row") -> np.ndarray:
    """Convert rotation matrices to 6D with explicit row/column convention."""

    if convention == "row":
        return matrix_to_rot6d_rowmajor(rotmat)
    if convention == "column":
        return np.asarray(rotmat[..., :2, :], dtype=np.float32).reshape(*rotmat.shape[:-2], 6)
    raise ValueError(f"unsupported rot6d convention: {convention}")


def _resolve_smplx_model_root(model_dir: Path) -> Path:
    if (model_dir / "SMPL_NEUTRAL.pkl").is_file():
        return model_dir.parent
    return model_dir


def load_smpl_rest(
    model_dir: Path,
    device: torch.device,
    gender: str = "neutral",
):
    if model_dir.name == "body_models":
        nochumpy = model_dir.with_name("body_models_nochumpy")
        if (nochumpy / "smpl" / "SMPL_NEUTRAL.pkl").exists():
            model_dir = nochumpy
    model = smplx.create(
        str(_resolve_smplx_model_root(model_dir)),
        model_type="smpl",
        gender=gender,
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
    rest = out.joints[0, :N_JOINTS].detach().cpu().numpy().astype(np.float32)
    parents = model.parents.detach().cpu().numpy().astype(np.int64)
    return model, rest, parents


def smpl_forward_22(
    model,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = len(global_orient)
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
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
        chunks.append(out.joints[:, :N_JOINTS].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def smpl_motion_integrity_metrics(
    model,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    device: torch.device,
    sample_count: int = 16,
) -> dict[str, float]:
    """Measure temporal rotation jumps and sampled SMPL surface distortion."""

    global_orient = np.asarray(global_orient, dtype=np.float32).reshape(-1, 3)
    body_pose_21 = np.asarray(body_pose_21, dtype=np.float32).reshape(-1, 63)
    if len(global_orient) != len(body_pose_21) or len(global_orient) < 1:
        raise ValueError("global_orient and body_pose must have the same non-zero length")

    axis_angle = np.concatenate(
        [global_orient[:, None], body_pose_21.reshape(-1, 21, 3)], axis=1
    )
    local = R.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix().reshape(
        len(axis_angle), N_JOINTS, 3, 3
    )
    if len(local) > 1:
        delta = local[1:] @ local[:-1].transpose(0, 1, 3, 2)
        angle = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(delta, axis1=-2, axis2=-1) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        rotation_p99 = float(np.percentile(angle, 99))
        rotation_max = float(angle.max())
    else:
        rotation_p99 = rotation_max = 0.0

    count = min(max(1, int(sample_count)), len(global_orient))
    indices = np.linspace(0, len(global_orient) - 1, count).round().astype(np.int64)
    body_23 = np.zeros((count, 69), dtype=np.float32)
    body_23[:, :63] = body_pose_21[indices]
    with torch.no_grad():
        posed = model(
            betas=torch.zeros(count, 10, device=device),
            body_pose=torch.from_numpy(body_23).to(device),
            global_orient=torch.from_numpy(global_orient[indices]).to(device),
            transl=torch.zeros(count, 3, device=device),
        ).vertices.detach().cpu().numpy()
        rest = model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        ).vertices[0].detach().cpu().numpy()
    faces = np.asarray(model.faces, dtype=np.int64)
    edges = np.unique(
        np.sort(
            np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
            ),
            axis=1,
        ),
        axis=0,
    )
    rest_lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=-1)
    posed_lengths = np.linalg.norm(
        posed[:, edges[:, 0]] - posed[:, edges[:, 1]], axis=-1
    )
    ratios = posed_lengths / np.maximum(rest_lengths[None], 1e-8)
    return {
        "rotation_jump_deg_p99": rotation_p99,
        "rotation_jump_deg_max": rotation_max,
        "mesh_edge_ratio_p01": float(np.percentile(ratios, 1)),
        "mesh_edge_ratio_p99": float(np.percentile(ratios, 99)),
        "mesh_edge_ratio_max": float(ratios.max()),
        "mesh_sample_count": float(count),
    }


def validate_smpl_motion_integrity(
    metrics: dict[str, float],
    *,
    max_rotation_jump_p99_deg: float = 90.0,
    max_mesh_edge_ratio_p99: float = 1.8,
    min_mesh_edge_ratio_p01: float = 0.2,
) -> None:
    """Reject low-MPJPE poses whose local rotations deform the SMPL mesh."""

    required = {
        "rotation_jump_deg_p99",
        "mesh_edge_ratio_p01",
        "mesh_edge_ratio_p99",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"missing SMPL integrity metrics: {sorted(missing)}")
    values = np.asarray([metrics[key] for key in required], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("SMPL integrity metrics contain non-finite values")
    if metrics["rotation_jump_deg_p99"] > max_rotation_jump_p99_deg:
        raise ValueError(
            "local-rotation jump p99 "
            f"{metrics['rotation_jump_deg_p99']:.2f} deg exceeds "
            f"{max_rotation_jump_p99_deg:.2f} deg"
        )
    if metrics["mesh_edge_ratio_p99"] > max_mesh_edge_ratio_p99:
        raise ValueError(
            "SMPL edge-stretch p99 "
            f"{metrics['mesh_edge_ratio_p99']:.3f} exceeds "
            f"{max_mesh_edge_ratio_p99:.3f}"
        )
    if metrics["mesh_edge_ratio_p01"] < min_mesh_edge_ratio_p01:
        raise ValueError(
            "SMPL edge-collapse p01 "
            f"{metrics['mesh_edge_ratio_p01']:.3f} is below "
            f"{min_mesh_edge_ratio_p01:.3f}"
        )


def refine_smpl_fit(
    model,
    target_joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray,
    iters: int,
    lr: float,
    pose_l2_weight: float,
    pose_keep_weight: float,
    angle_prior_weight: float,
    device: torch.device,
    smooth_weight: float = 1e-3,
    lock_global_orient: bool = False,
    lock_body_joint_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine IK initialization by optimizing SMPL pose/transl against joints."""
    if iters <= 0:
        fitted = smpl_forward_22(model, global_orient, body_pose_21, transl, 512, device)
        return global_orient, body_pose_21, transl, fitted

    target = torch.from_numpy(target_joints.astype(np.float32)).to(device)
    n = len(target_joints)
    g = torch.tensor(global_orient, dtype=torch.float32, device=device, requires_grad=not lock_global_orient)
    b21 = torch.tensor(body_pose_21, dtype=torch.float32, device=device, requires_grad=True)
    tr = torch.tensor(transl, dtype=torch.float32, device=device, requires_grad=True)
    b21_init = b21.detach().clone()
    params = [b21, tr] if lock_global_orient else [g, b21, tr]
    opt = torch.optim.Adam(params, lr=lr)
    lock_slices: list[slice] = []
    for joint_id in lock_body_joint_ids or []:
        joint_id = int(joint_id)
        if not (1 <= joint_id <= 21):
            raise ValueError(f"SMPL body joint id must be in [1,21], got {joint_id}")
        start = (joint_id - 1) * 3
        lock_slices.append(slice(start, start + 3))

    for _ in range(iters):
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        joints = out.joints[:, :N_JOINTS]
        data_loss = ((joints - target) ** 2).sum(dim=-1).mean()
        pose_keep = ((b21 - b21_init) ** 2).mean()
        pose_prior = (body_23 ** 2).mean()
        if angle_prior_weight > 0:
            # SMPLify angle prior indices in the 69-dim body-pose vector:
            # left/right knees and elbows are discouraged from bending backward.
            idx = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=device)
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=device)
            angle_prior = torch.exp(body_23[:, idx] * signs).pow(2).mean()
        else:
            angle_prior = torch.tensor(0.0, device=device)
        if n >= 3:
            tr_acc = tr[2:] - 2 * tr[1:-1] + tr[:-2]
            pose_acc = b21[2:] - 2 * b21[1:-1] + b21[:-2]
            smooth = (tr_acc ** 2).mean() + 1e-2 * (pose_acc ** 2).mean()
        else:
            smooth = torch.tensor(0.0, device=device)
        loss = (
            data_loss
            + pose_keep_weight * pose_keep
            + pose_l2_weight * pose_prior
            + angle_prior_weight * angle_prior
            + smooth_weight * smooth
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if lock_slices:
            with torch.no_grad():
                for sl in lock_slices:
                    b21[:, sl] = b21_init[:, sl]

    with torch.no_grad():
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        fitted = out.joints[:, :N_JOINTS].detach().cpu().numpy().astype(np.float32)
    return (
        g.detach().cpu().numpy().astype(np.float32),
        b21.detach().cpu().numpy().astype(np.float32),
        tr.detach().cpu().numpy().astype(np.float32),
        fitted,
    )
