"""Shape-aware SMPL-family skeleton loading and 22-joint forward kinematics.

Only the body-model arrays needed for joint locations are loaded. Mesh skinning
is deliberately outside this module: the public conversion API needs the first
22 body joints, their shape-dependent rest locations, and the kinematic tree.
Licensed SMPL model files are supplied by the caller and are never distributed
with Motius.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from motius.motion.representation.rotation import axis_angle_to_matrix
from motius.motion.skeleton.names import SMPL22_PARENTS


_NUM_BODY_JOINTS = 22


def _dense_array(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def _load_model_data(path: Path) -> Mapping[str, object]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix.lower() in {".pkl", ".pickle"}:
        import pickle

        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        if not isinstance(data, Mapping):
            raise TypeError(f"expected a mapping in {path}, got {type(data).__name__}")
        return data
    raise ValueError(f"unsupported SMPL model file: {path}; expected .npz or .pkl")


def resolve_smpl_model_path(
    model_path: str | Path,
    *,
    model_type: str = "smplh",
    gender: str = "neutral",
) -> Path:
    """Resolve a direct model file or a common SMPL-family directory layout."""

    path = Path(model_path).expanduser()
    model_type = model_type.lower()
    gender = gender.lower()
    if model_type not in {"smpl", "smplh", "smplx"}:
        raise ValueError(f"model_type must be smpl/smplh/smplx, got {model_type!r}")
    if gender not in {"neutral", "male", "female"}:
        raise ValueError(f"gender must be neutral/male/female, got {gender!r}")
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"SMPL model path does not exist: {path}")

    upper_type = model_type.upper()
    upper_gender = gender.upper()
    candidates = [
        path / model_type / gender / "model.npz",
        path / gender / "model.npz",
        path / model_type / f"{upper_type}_{upper_gender}.npz",
        path / model_type / f"{upper_type}_{upper_gender}.pkl",
        path / f"{upper_type}_{upper_gender}.npz",
        path / f"{upper_type}_{upper_gender}.pkl",
        path / f"{upper_gender}.npz",
        path / f"{upper_gender}.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"could not resolve a {model_type}/{gender} model under {path}; tried:\n  {attempted}"
    )


@dataclass(frozen=True)
class SMPLSkeletonModel:
    """The shape-dependent subset of an SMPL-family model needed for FK."""

    joint_template: np.ndarray
    joint_shape_dirs: np.ndarray
    parents: tuple[int, ...]
    path: Path

    @property
    def num_betas(self) -> int:
        return int(self.joint_shape_dirs.shape[-1])

    def rest_joints(self, betas: np.ndarray) -> np.ndarray:
        betas = np.asarray(betas, dtype=np.float64)
        if betas.shape[-1] > self.num_betas:
            betas = betas[..., : self.num_betas]
        elif betas.shape[-1] < self.num_betas:
            pad = [(0, 0)] * betas.ndim
            pad[-1] = (0, self.num_betas - betas.shape[-1])
            betas = np.pad(betas, pad)
        shaped = np.einsum("...b,jcb->...jc", betas, self.joint_shape_dirs)
        return self.joint_template + shaped


@lru_cache(maxsize=12)
def _load_smpl_skeleton_model_cached(path_str: str) -> SMPLSkeletonModel:
    path = Path(path_str)
    data = _load_model_data(path)
    required = {"v_template", "shapedirs", "J_regressor", "kintree_table"}
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"SMPL model {path} is missing arrays: {missing}")

    vertices = _dense_array(data["v_template"]).astype(np.float64)
    shapedirs = _dense_array(data["shapedirs"]).astype(np.float64)
    regressor = _dense_array(data["J_regressor"]).astype(np.float64)
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != vertices.shape:
        raise ValueError(
            f"invalid shapedirs {shapedirs.shape} for v_template {vertices.shape} in {path}"
        )
    if regressor.shape[-1] != vertices.shape[0]:
        raise ValueError(
            f"invalid J_regressor {regressor.shape} for v_template {vertices.shape} in {path}"
        )

    joint_template = (regressor @ vertices)[:_NUM_BODY_JOINTS]
    joint_shape_dirs = np.einsum(
        "jv,vcb->jcb", regressor[:_NUM_BODY_JOINTS], shapedirs
    )
    tree = _dense_array(data["kintree_table"])
    parents = np.asarray(tree[0] if tree.ndim == 2 else tree, dtype=np.int64)
    parents = parents[:_NUM_BODY_JOINTS]
    parents[0] = -1
    expected = np.asarray(SMPL22_PARENTS, dtype=np.int64)
    if not np.array_equal(parents, expected):
        raise ValueError(
            f"first 22 joints in {path} do not use the SMPL-22 hierarchy: {parents.tolist()}"
        )
    return SMPLSkeletonModel(
        joint_template=joint_template,
        joint_shape_dirs=joint_shape_dirs,
        parents=tuple(int(parent) for parent in parents),
        path=path,
    )


def load_smpl_skeleton_model(
    model_path: str | Path,
    *,
    model_type: str = "smplh",
    gender: str = "neutral",
) -> SMPLSkeletonModel:
    """Load and cache the shape-aware first 22 joints of an SMPL-family model."""

    resolved = resolve_smpl_model_path(
        model_path, model_type=model_type, gender=gender
    ).resolve()
    return _load_smpl_skeleton_model_cached(str(resolved))


def _time_major(value, *, frames: int, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = np.broadcast_to(array, (frames, array.shape[0]))
    else:
        array = array.reshape(array.shape[0], -1)
    if array.shape != (frames, width):
        raise ValueError(f"{name} must have shape ({frames},{width}) or ({width},), got {array.shape}")
    return np.asarray(array, dtype=np.float64)


def smpl_to_joints(
    global_orient,
    body_pose,
    transl,
    *,
    betas=None,
    gender: str = "neutral",
    model_type: str = "smplh",
    model_path: str | Path,
) -> np.ndarray:
    """Materialize shape-aware SMPL-22 joints from SMPL-family parameters.

    Args:
        global_orient: ``(T,3)`` root axis-angle rotations.
        body_pose: ``(T,>=63)`` or ``(T,>=21,3)`` local body rotations. Only
            the 21 joints shared by SMPL and SMPL-H are used.
        transl: ``(T,3)`` root translations.
        betas: ``(B,)`` sequence shape or ``(T,B)`` per-frame shapes. Defaults
            to zero shape for the loaded model.
        gender/model_type/model_path: identify the licensed body-model file.

    Returns:
        ``(T,22,3)`` joints in the input SMPL coordinate system.
    """

    root = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    frames = len(root)
    pose = np.asarray(body_pose, dtype=np.float64).reshape(frames, -1, 3)
    if pose.shape[1] < 21:
        raise ValueError(f"body_pose needs at least 21 joints, got {pose.shape}")
    pose = pose[:, :21]
    translation = _time_major(transl, frames=frames, width=3, name="transl")

    model = load_smpl_skeleton_model(
        model_path, model_type=model_type, gender=gender
    )
    if betas is None:
        shape = np.zeros((frames, model.num_betas), dtype=np.float64)
    else:
        raw_betas = np.asarray(betas, dtype=np.float64)
        if raw_betas.ndim == 1:
            shape = np.broadcast_to(raw_betas, (frames, raw_betas.shape[0]))
        else:
            shape = raw_betas.reshape(raw_betas.shape[0], -1)
            if shape.shape[0] == 1:
                shape = np.broadcast_to(shape, (frames, shape.shape[1]))
            elif shape.shape[0] != frames:
                raise ValueError(
                    f"betas must have one row or {frames} rows, got {shape.shape}"
                )

    rest = model.rest_joints(shape)
    offsets = np.empty_like(rest)
    offsets[:, 0] = rest[:, 0]
    for joint, parent in enumerate(model.parents[1:], start=1):
        offsets[:, joint] = rest[:, joint] - rest[:, parent]

    local_axis_angle = np.concatenate([root[:, None], pose], axis=1)
    local_rotations = axis_angle_to_matrix(
        local_axis_angle.reshape(-1, 3)
    ).reshape(frames, _NUM_BODY_JOINTS, 3, 3)
    world_rotations = np.empty_like(local_rotations)
    joints = np.empty((frames, _NUM_BODY_JOINTS, 3), dtype=np.float64)
    for joint, parent in enumerate(model.parents):
        if parent < 0:
            world_rotations[:, joint] = local_rotations[:, joint]
            joints[:, joint] = translation + offsets[:, joint]
        else:
            world_rotations[:, joint] = (
                world_rotations[:, parent] @ local_rotations[:, joint]
            )
            joints[:, joint] = joints[:, parent] + (
                world_rotations[:, parent] @ offsets[:, joint, :, None]
            ).squeeze(-1)
    return joints.astype(np.float32)


__all__ = [
    "SMPLSkeletonModel",
    "resolve_smpl_model_path",
    "load_smpl_skeleton_model",
    "smpl_to_joints",
]
