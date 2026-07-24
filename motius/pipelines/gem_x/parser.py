"""Parse GEM-X while preserving its native SOMA-77 representation."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from motius.models.gem_x.runtime import CHECKPOINT_SHA256, SOURCE_REVISION
from motius.motion.representation.monocular_capture import (
    GRAVITY_WORLD_Y_UP,
    MonocularCaptureResult,
    MonocularTrack,
)
from motius.motion.representation.monocular_joints import (
    COMMON_HMR15_FROM_SOMA77,
    SOMA77_NAMES,
)


_PARAMETER_GROUPS = ("body_params_incam", "body_params_global")
_REQUIRED_PARAMETERS = (
    "body_pose",
    "global_orient",
    "transl",
    "identity_coeffs",
    "scale_params",
)


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


def load_gem_x_payload(path: str | Path) -> Mapping[str, Any]:
    """Load trusted official PT/PKL output or the numeric NPZ export."""

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
                "Reading official GEM-X .pt output requires torch; "
                "use the isolated NPZ exporter when torch is unavailable."
            ) from exc
        try:
            value = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            value = torch.load(source, map_location="cpu")
    elif suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            value = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported GEM-X output extension: {suffix}")
    if not isinstance(value, Mapping):
        raise TypeError("GEM-X output must contain a mapping.")
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
        raise KeyError(f"{name} is missing official SOMA fields: {missing}.")
    return values


def _poses(parameters: Mapping[str, np.ndarray], frames: int) -> np.ndarray:
    orient = parameters["global_orient"].reshape(frames, 1, 3)
    body = parameters["body_pose"].reshape(frames, -1, 3)
    if body.shape[1] != 76:
        raise ValueError(
            "Fixed GEM-X revision must export 76 non-root SOMA joints "
            f"(228 values), got shape {parameters['body_pose'].shape}."
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


def parse_gem_x_output(
    payload: Mapping[str, Any],
    *,
    original_fps: float,
    output_fps: float = 30.0,
    track_id: str = "person_0",
) -> MonocularCaptureResult:
    """Parse ``hpe_results.pt`` without converting SOMA parameters to SMPL."""

    camera = _parameter_group(payload, "body_params_incam")
    world = _parameter_group(payload, "body_params_global")
    reference = camera or world
    if reference is None:
        raise KeyError("GEM-X output requires body_params_incam or body_params_global.")
    frames = int(reference["body_pose"].shape[0])
    if frames < 1:
        raise ValueError("GEM-X output is empty.")
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
        if joints is not None and joints.shape != (frames, 77, 3):
            raise ValueError(f"{name} must have shape ({frames}, 77, 3), got {joints.shape}.")

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
        body_model="soma77",
        joint_names=SOMA77_NAMES if joints_camera is not None or joints_world is not None else (),
        poses_axis_angle=_poses(reference, frames),
        # SOMA identity/scale coefficients are not SMPL betas.
        shape_parameters=None,
        joints_camera=joints_camera,
        joints_world=joints_world,
        root_translation_camera=None if camera is None else camera["transl"],
        root_translation_world=None if world is None else world["transl"],
        native_parameters=_native_parameters(groups),
        availability={
            "camera_parameters": "native_soma77" if camera is not None else "unavailable",
            "world_parameters": "native_soma77" if world is not None else "unavailable",
            "camera_joints": (
                "official_soma77_forward" if joints_camera is not None else "not_exported"
            ),
            "world_joints": (
                "official_soma77_forward" if joints_world is not None else "not_exported"
            ),
            "vertices": "not_exported",
            "pve": "not_comparable_to_smpl_topology",
            "confidence": "not_predicted_by_official_demo",
        },
        metadata={
            "official_output": "hpe_results.pt",
            "native_parameter_groups": tuple(
                name for name, value in groups.items() if value is not None
            ),
            "native_body_model": "SOMA-77",
            "joint_only_common_mapping": dict(COMMON_HMR15_FROM_SOMA77),
        },
    )
    return MonocularCaptureResult(
        source_model="NVlabs/GEM-X",
        source_revision=SOURCE_REVISION,
        checkpoint_sha256=CHECKPOINT_SHA256,
        original_fps=float(original_fps),
        output_fps=float(output_fps),
        tracks=(track,),
        world_coordinate_system=GRAVITY_WORLD_Y_UP if world is not None else None,
        camera_intrinsics=intrinsics,
        metadata={
            "upstream_name": "GEM-X",
            "native_representation": "SOMA-77",
            "official_serialization": "PyTorch hpe_results.pt",
            "comparison_scope": "named common joints only; no cross-topology PVE",
        },
    )


def parse_gem_x_file(
    path: str | Path,
    *,
    original_fps: float,
    output_fps: float = 30.0,
    track_id: str = "person_0",
) -> MonocularCaptureResult:
    return parse_gem_x_output(
        load_gem_x_payload(path),
        original_fps=original_fps,
        output_fps=output_fps,
        track_id=track_id,
    )


__all__ = ["load_gem_x_payload", "parse_gem_x_file", "parse_gem_x_output"]
