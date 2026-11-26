"""IK/FK backend for HumanML3D-263 to SMPL-style motion135 retargeting.

This script keeps the HumanML3D-to-SMPL conversion auditable in one place:

    HML3D-263 -> 22 joints -> hierarchical IK on SMPL rest skeleton
              -> global_orient/body_pose/transl + motion_135

The conversion is not mathematically exact: HumanML3D-263 does not uniquely
determine SMPL pose twist, shape, or mesh details. The shared position IK solver handles both features and raw joints.
There are no HumanML rotation initializers or alternative conversion paths.
The saved fit MPJPE is a diagnostic for how
well the SMPL skeleton tracks the recovered 22 joints, but it does not by itself
validate terminal mesh orientation.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


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
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes  # noqa: E402


# HumanML3D 22-joint skeleton order follows the first 22 SMPL joints.
N_JOINTS = 22
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


def resample_linear(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Sample at physical output timestamps, holding the final source sample.

    Clip duration uses the frame-count convention ``T / fps``. Unlike explicit
    ``fit_length_linear`` time warping, FPS conversion must not stretch the
    interval between the first and last observations to fill the output grid.
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite([src_fps, dst_fps]).all() or src_fps <= 0 or dst_fps <= 0:
        raise ValueError("src_fps and dst_fps must be finite and positive")
    if abs(src_fps - dst_fps) < 1e-6 or len(x) == 0:
        return x
    new_t = max(1, int(round(len(x) * dst_fps / src_fps)))
    grid = np.minimum(np.arange(new_t, dtype=np.float64) * src_fps / dst_fps, len(x) - 1)
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


def _resolve_smplh_npz(model_dir: Path, gender: str) -> Path | None:
    if model_dir.is_file() and model_dir.suffix.lower() == ".npz":
        return model_dir
    candidates = (
        model_dir / gender / "model.npz",
        model_dir / "smplh" / gender / "model.npz",
        model_dir / f"SMPLH_{gender.upper()}.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


class _SMPLHSkeletonAdapter(torch.nn.Module):
    """Expose a lightweight SMPL-compatible API for HY-Motion SMPL-H NPZs."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        with np.load(model_path) as data:
            joint_regressor = np.asarray(data["J_regressor"], dtype=np.float32)[
                :N_JOINTS
            ]
            vertex_template = np.asarray(data["v_template"], dtype=np.float32)
            shape_dirs = np.asarray(data["shapedirs"], dtype=np.float32)
            kinematic_tree = np.asarray(data["kintree_table"], dtype=np.int64)
        joint_template = joint_regressor @ vertex_template
        joint_shape_dirs = np.einsum(
            "jv,vdb->jdb", joint_regressor, shape_dirs, optimize=True
        )
        parents = kinematic_tree[0] if kinematic_tree.ndim == 2 else kinematic_tree
        parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
        parents[0] = -1
        self.register_buffer("joint_template", torch.from_numpy(joint_template))
        self.register_buffer("joint_shape_dirs", torch.from_numpy(joint_shape_dirs))
        self.register_buffer("parents", torch.from_numpy(parents))
        self.faces = None

    def forward(
        self,
        *,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        batch_size = global_orient.shape[0]
        pose = torch.cat((global_orient, body_pose[:, :63]), dim=-1)
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(
            batch_size, N_JOINTS, 3, 3
        )
        shape_count = min(betas.shape[-1], self.joint_shape_dirs.shape[-1])
        shaped_joints = self.joint_template.unsqueeze(0) + blend_shapes(
            betas[:, :shape_count], self.joint_shape_dirs[..., :shape_count]
        )
        joints, _ = batch_rigid_transform(
            rotations, shaped_joints, self.parents, dtype=rotations.dtype
        )
        return SimpleNamespace(joints=joints + transl[:, None, :], vertices=None)


def load_smpl_rest(
    model_dir: Path,
    device: torch.device,
    gender: str = "neutral",
):
    smplh_npz = _resolve_smplh_npz(model_dir, gender)
    if smplh_npz is not None:
        model = _SMPLHSkeletonAdapter(smplh_npz).to(device)
        model.eval()
        with torch.no_grad():
            out = model(
                betas=torch.zeros(1, 10, device=device),
                body_pose=torch.zeros(1, 69, device=device),
                global_orient=torch.zeros(1, 3, device=device),
                transl=torch.zeros(1, 3, device=device),
            )
        rest = out.joints[0].detach().cpu().numpy().astype(np.float32)
        parents = model.parents.detach().cpu().numpy().astype(np.int64)
        return model, rest, parents
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
    max_mesh_edge_ratio_p99: float = 2.0,
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


def smpl_mesh_floor_offset(model, global_orient, body_pose_21, transl, device, batch_size=32):
    """One clip-wide offset placing the lowest skinned vertex on Y=0.

    Joint centers are above the soles. Aligning joint minima leaves the actual
    mesh underground; per-frame offsets instead introduce artificial root bob.
    Return None for skeleton-only adapters, which cannot certify mesh contact.
    """
    minimum = float("inf")
    with torch.no_grad():
        for start in range(0, len(transl), batch_size):
            end = min(start + batch_size, len(transl))
            body = torch.zeros(end - start, 69, device=device)
            body[:, :63] = torch.as_tensor(body_pose_21[start:end], device=device)
            output = model(
                betas=torch.zeros(end - start, 10, device=device),
                body_pose=body,
                global_orient=torch.as_tensor(global_orient[start:end], device=device),
                transl=torch.as_tensor(transl[start:end], device=device),
            )
            if getattr(output, "vertices", None) is None:
                return None
            minimum = min(minimum, float(output.vertices[..., 1].min()))
    if not np.isfinite(minimum):
        raise ValueError("cannot align an empty or non-finite SMPL mesh")
    return -minimum


@torch.inference_mode(False)
@torch.enable_grad()
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
    smooth_weight: float = 1e-2,
    lock_global_orient: bool = False,
    lock_body_joint_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine SMPL pose against joints while preserving the pelvis trajectory."""
    if iters <= 0:
        fitted = smpl_forward_22(model, global_orient, body_pose_21, transl, 512, device)
        return global_orient, body_pose_21, transl, fitted

    from motius.motion.representation.rotation import axis_angle_to_matrix

    from .anatomical_fit import robust_landmark_loss, rotation_temporal_loss

    target = torch.from_numpy(target_joints.astype(np.float32)).to(device)
    n = len(target_joints)
    g = torch.tensor(global_orient, dtype=torch.float32, device=device, requires_grad=not lock_global_orient)
    b21 = torch.tensor(body_pose_21, dtype=torch.float32, device=device, requires_grad=True)
    tr = torch.tensor(transl, dtype=torch.float32, device=device)
    b21_init = b21.detach().clone()
    initial_rotations = axis_angle_to_matrix(b21_init.reshape(n, 21, 3))
    params = [b21] if lock_global_orient else [g, b21]
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
        data_loss = robust_landmark_loss(
            (joints[:, 1:] - joints[:, :1]) - (target[:, 1:] - target[:, :1])
        )
        local = axis_angle_to_matrix(torch.cat([g, b21], dim=1).reshape(n, 22, 3))
        pose_keep = (local[:, 1:] - initial_rotations).square().sum((-1, -2)).mean() / 6
        pose_prior = (body_23 ** 2).mean()
        if angle_prior_weight > 0:
            # SMPLify angle prior indices in the 69-dim body-pose vector:
            # left/right knees and elbows are discouraged from bending backward.
            idx = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=device)
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=device)
            angle_prior = torch.exp(body_23[:, idx] * signs).pow(2).mean()
        else:
            angle_prior = torch.tensor(0.0, device=device)
        smooth = rotation_temporal_loss(local)
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
        joints = out.joints[:, :N_JOINTS]
        root_delta = target[:, 0] - joints[:, 0]
        tr = tr + root_delta
        fitted = (joints + root_delta[:, None]).cpu().numpy().astype(np.float32)
    return (
        g.detach().cpu().numpy().astype(np.float32),
        b21.detach().cpu().numpy().astype(np.float32),
        tr.detach().cpu().numpy().astype(np.float32),
        fitted,
    )
