"""Parse the fixed GEM-SMPL demo output into the monocular capture contract."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from motius.models.gem_smpl.runtime import CHECKPOINT_SHA256, SOURCE_REVISION
from motius.motion.representation.monocular_capture import (
    GRAVITY_WORLD_Y_UP,
    MonocularCaptureResult,
    MonocularTrack,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES


_PARAMETER_GROUPS = ("body_params_incam", "body_params_global")
_REQUIRED_PARAMETERS = ("body_pose", "global_orient", "transl")


def _numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must be a numeric array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _inflate_npz(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in values:
        value = values[key]
        prefix, separator, suffix = key.partition(".")
        if separator and prefix in _PARAMETER_GROUPS:
            result.setdefault(prefix, {})[suffix] = value
        else:
            result[key] = value
    return result


def load_gem_smpl_payload(path: str | Path) -> Mapping[str, Any]:
    """Load trusted official PT/PKL output or the adapter's numeric NPZ export."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as values:
            return _inflate_npz({key: values[key] for key in values.files})
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Reading official GEM-SMPL .pt output requires torch; "
                "use the isolated NPZ exporter when torch is unavailable."
            ) from exc
        try:
            value = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            value = torch.load(source, map_location="cpu")
    elif suffix in {".pkl", ".pickle"}:
        # Pickle is accepted only for trusted, user-supplied official artifacts.
        with source.open("rb") as handle:
            value = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported GEM-SMPL output extension: {suffix}")
    if not isinstance(value, Mapping):
        raise TypeError("GEM-SMPL output must contain a mapping.")
    return value


def _parameter_group(
    payload: Mapping[str, Any],
    name: str,
) -> dict[str, np.ndarray] | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    values = {key: _numpy(value, name=f"{name}.{key}") for key, value in raw.items()}
    missing = [key for key in _REQUIRED_PARAMETERS if key not in values]
    if missing:
        raise KeyError(f"{name} is missing official fields: {missing}.")
    return values


def _poses(parameters: Mapping[str, np.ndarray], frames: int) -> np.ndarray:
    orient = parameters["global_orient"].reshape(frames, 1, 3)
    body = parameters["body_pose"].reshape(frames, -1, 3)
    if body.shape[1] != 21:
        raise ValueError(
            "Fixed GEM-SMPL revision must export 21 body joints "
            f"(63 values), got shape {parameters['body_pose'].shape}."
        )
    return np.concatenate([orient, body], axis=1).astype(np.float32, copy=False)


def _native_parameters(
    groups: Mapping[str, Mapping[str, np.ndarray] | None],
) -> dict[str, np.ndarray]:
    return {
        f"{group_name}.{parameter_name}": value
        for group_name, parameters in groups.items()
        if parameters is not None
        for parameter_name, value in parameters.items()
    }


def parse_gem_smpl_output(
    payload: Mapping[str, Any],
    *,
    original_fps: float,
    output_fps: float = 30.0,
    track_id: str = "person_0",
) -> MonocularCaptureResult:
    """Parse the official ``smpl_params.pt`` field layout without inference."""

    camera = _parameter_group(payload, "body_params_incam")
    world = _parameter_group(payload, "body_params_global")
    reference = camera or world
    if reference is None:
        raise KeyError(
            "GEM-SMPL output requires body_params_incam or body_params_global."
        )
    frames = int(reference["body_pose"].shape[0])
    if frames < 1:
        raise ValueError("GEM-SMPL output is empty.")
    for name, parameters in (("body_params_incam", camera), ("body_params_global", world)):
        if parameters is not None and parameters["body_pose"].shape[0] != frames:
            raise ValueError(f"{name} frame count is not aligned.")

    joints_camera = (
        _numpy(payload["joints_camera"], name="joints_camera")
        if "joints_camera" in payload
        else None
    )
    joints_world = (
        _numpy(payload["joints_world"], name="joints_world")
        if "joints_world" in payload
        else None
    )
    for name, joints in (("joints_camera", joints_camera), ("joints_world", joints_world)):
        if joints is not None and joints.shape != (frames, 24, 3):
            raise ValueError(f"{name} must have shape ({frames}, 24, 3), got {joints.shape}.")

    betas = reference.get("betas")
    intrinsics = (
        _numpy(payload["K_fullimg"], name="K_fullimg")
        if "K_fullimg" in payload
        else None
    )
    groups = {"body_params_incam": camera, "body_params_global": world}
    track = MonocularTrack(
        track_id=track_id,
        frame_ids=np.arange(frames, dtype=np.int64),
        valid=np.ones(frames, dtype=bool),
        body_model="smpl",
        joint_names=SMPL24_NAMES if joints_camera is not None or joints_world is not None else (),
        poses_axis_angle=_poses(reference, frames),
        shape_parameters=betas,
        joints_camera=joints_camera,
        joints_world=joints_world,
        root_translation_camera=None if camera is None else camera["transl"],
        root_translation_world=None if world is None else world["transl"],
        native_parameters=_native_parameters(groups),
        availability={
            "camera_parameters": "native" if camera is not None else "unavailable",
            "world_parameters": "native" if world is not None else "unavailable",
            "camera_joints": (
                "official_smpl24_forward" if joints_camera is not None else "not_exported"
            ),
            "world_joints": (
                "official_smpl24_forward" if joints_world is not None else "not_exported"
            ),
            "vertices": "not_exported",
            "pve": "unavailable_without_exported_vertices",
            "confidence": "not_predicted_by_official_demo",
        },
        metadata={
            "official_output": "smpl_params.pt",
            "native_parameter_groups": tuple(
                name for name, value in groups.items() if value is not None
            ),
            "pose_joint_count": 22,
        },
    )
    return MonocularCaptureResult(
        source_model="NVlabs/GEM-SMPL",
        source_revision=SOURCE_REVISION,
        checkpoint_sha256=CHECKPOINT_SHA256,
        original_fps=float(original_fps),
        output_fps=float(output_fps),
        tracks=(track,),
        world_coordinate_system=GRAVITY_WORLD_Y_UP if world is not None else None,
        camera_intrinsics=intrinsics,
        metadata={
            "upstream_name": "GEM-SMPL",
            "former_paper_name": "GENMO",
            "official_serialization": "PyTorch smpl_params.pt",
        },
    )


def parse_gem_smpl_file(
    path: str | Path,
    *,
    original_fps: float,
    output_fps: float = 30.0,
    track_id: str = "person_0",
) -> MonocularCaptureResult:
    return parse_gem_smpl_output(
        load_gem_smpl_payload(path),
        original_fps=original_fps,
        output_fps=output_fps,
        track_id=track_id,
    )


__all__ = [
    "load_gem_smpl_payload",
    "parse_gem_smpl_file",
    "parse_gem_smpl_output",
]
