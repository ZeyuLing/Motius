"""Motius monocular-capture adapter for official GVHMR demo outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from motius.models.gvhmr.bundle import OFFICIAL_RUNTIME_REVISION
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
    MonocularCaptureResult,
    MonocularTrack,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


_PARAMETER_KEYS = ("global_orient", "body_pose", "betas", "transl")
_OUTPUT_FPS = 30.0


def _numpy(value: Any, *, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(
            f"{name} must have shape (frames,{','.join(map(str, shape_tail))}), "
            f"got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return array.astype(np.float32, copy=False)


def _optional_points(
    payload: Mapping[str, Any],
    name: str,
    *,
    frames: int,
) -> Optional[np.ndarray]:
    value = payload.get(name)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[0] != frames
        or array.shape[-1] != 3
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must have finite shape (frames,points,3).")
    return array.astype(np.float32, copy=False)


def _text_scalar(payload: Mapping[str, Any], name: str) -> Optional[str]:
    value = payload.get(name)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string.")
    return str(array.item())


def load_gvhmr_output(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load an official ``hmr4d_results.pt`` or safe converted ``.npz``."""

    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Official GVHMR output must be a mapping.")
    return dict(payload)


def _parameter_groups(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "smpl_params_global" in payload or "smpl_params_incam" in payload:
        world = payload.get("smpl_params_global")
        camera = payload.get("smpl_params_incam")
        if not isinstance(world, Mapping) or not isinstance(camera, Mapping):
            raise ValueError(
                "smpl_params_global and smpl_params_incam must both be mappings."
            )
        return dict(world), dict(camera)

    world = {
        name: payload.get(f"smpl_params_global_{name}")
        for name in _PARAMETER_KEYS
    }
    camera = {
        name: payload.get(f"smpl_params_incam_{name}")
        for name in _PARAMETER_KEYS
    }
    return world, camera


def parse_gvhmr_output(
    source: str | Path | Mapping[str, Any],
    *,
    checkpoint_sha256: Optional[str] = None,
    original_fps: float,
    valid: Optional[np.ndarray] = None,
    frame_ids: Optional[np.ndarray] = None,
    inference_metadata: Optional[Mapping[str, object]] = None,
) -> MonocularCaptureResult:
    """Convert documented GVHMR demo fields to the shared capture contract."""

    payload = load_gvhmr_output(source)
    embedded_revision = _text_scalar(payload, "runtime_revision")
    if (
        embedded_revision is not None
        and embedded_revision != OFFICIAL_RUNTIME_REVISION
    ):
        raise ValueError(
            "Converted GVHMR output revision does not match the pinned runtime."
        )
    embedded_sha256 = _text_scalar(payload, "checkpoint_sha256")
    if checkpoint_sha256 is None:
        checkpoint_sha256 = embedded_sha256
    elif embedded_sha256 is not None and embedded_sha256 != checkpoint_sha256:
        raise ValueError("Provided checkpoint SHA256 disagrees with converted output.")
    if checkpoint_sha256 is None:
        raise ValueError(
            "checkpoint_sha256 is required for raw official GVHMR outputs."
        )

    world_raw, camera_raw = _parameter_groups(payload)
    missing = [
        f"{group_name}.{name}"
        for group_name, group in (
            ("smpl_params_global", world_raw),
            ("smpl_params_incam", camera_raw),
        )
        for name in _PARAMETER_KEYS
        if group.get(name) is None
    ]
    if missing:
        raise ValueError(f"GVHMR output is missing required fields: {missing}.")

    world = {
        "global_orient": _numpy(
            world_raw["global_orient"],
            name="smpl_params_global.global_orient",
            shape_tail=(3,),
        ),
        "body_pose": _numpy(
            world_raw["body_pose"],
            name="smpl_params_global.body_pose",
            shape_tail=(63,),
        ),
        "betas": _numpy(
            world_raw["betas"],
            name="smpl_params_global.betas",
            shape_tail=(10,),
        ),
        "transl": _numpy(
            world_raw["transl"],
            name="smpl_params_global.transl",
            shape_tail=(3,),
        ),
    }
    frames = len(world["transl"])
    camera = {
        "global_orient": _numpy(
            camera_raw["global_orient"],
            name="smpl_params_incam.global_orient",
            shape_tail=(3,),
        ),
        "body_pose": _numpy(
            camera_raw["body_pose"],
            name="smpl_params_incam.body_pose",
            shape_tail=(63,),
        ),
        "betas": _numpy(
            camera_raw["betas"],
            name="smpl_params_incam.betas",
            shape_tail=(10,),
        ),
        "transl": _numpy(
            camera_raw["transl"],
            name="smpl_params_incam.transl",
            shape_tail=(3,),
        ),
    }
    for group_name, group in (("world", world), ("camera", camera)):
        for name, array in group.items():
            if len(array) != frames:
                raise ValueError(
                    f"{group_name}.{name} has {len(array)} frames, expected {frames}."
                )
    if not np.array_equal(world["body_pose"], camera["body_pose"]):
        raise ValueError(
            "Official GVHMR postprocessed camera/world body_pose fields must agree."
        )
    if not np.array_equal(world["betas"], camera["betas"]):
        raise ValueError("Official GVHMR camera/world betas fields must agree.")

    intrinsics_value = payload.get("K_fullimg")
    if intrinsics_value is None:
        raise ValueError("GVHMR output is missing K_fullimg.")
    intrinsics = _numpy(
        intrinsics_value,
        name="K_fullimg",
        shape_tail=(3, 3),
    )
    if len(intrinsics) != frames:
        raise ValueError("K_fullimg must align with the SMPL parameter frames.")

    payload_valid = payload.get("valid")
    valid_source = "caller"
    if valid is None and payload_valid is not None:
        valid = np.asarray(payload_valid)
        valid_source = "converted_output"
    elif valid is None:
        valid = np.ones(frames, dtype=bool)
        valid_source = "official_dense_output_after_bbox_interpolation"
    else:
        valid = np.asarray(valid)
    if valid.shape != (frames,) or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean vector aligned to GVHMR frames.")

    if frame_ids is None:
        payload_frame_ids = payload.get("frame_ids")
        frame_ids = (
            np.arange(frames, dtype=np.int64)
            if payload_frame_ids is None
            else np.asarray(payload_frame_ids)
        )
    else:
        frame_ids = np.asarray(frame_ids)
    if (
        frame_ids.shape != (frames,)
        or not np.issubdtype(frame_ids.dtype, np.integer)
        or (frames > 1 and np.any(np.diff(frame_ids) <= 0))
    ):
        raise ValueError("frame_ids must be strictly increasing integers.")
    frame_ids = frame_ids.astype(np.int64, copy=False)

    joints_camera = _optional_points(payload, "joints_camera", frames=frames)
    joints_world = _optional_points(payload, "joints_world", frames=frames)
    vertices_camera = _optional_points(payload, "vertices_camera", frames=frames)
    vertices_world = _optional_points(payload, "vertices_world", frames=frames)
    joint_counts = {
        array.shape[1]
        for array in (joints_camera, joints_world)
        if array is not None
    }
    if joint_counts and joint_counts != {len(SMPL24_NAMES)}:
        raise ValueError("Converted GVHMR joints must use the official SMPL24 order.")

    world_pose = np.concatenate(
        (
            world["global_orient"][:, None],
            world["body_pose"].reshape(frames, 21, 3),
        ),
        axis=1,
    )
    converted = any(
        value is not None
        for value in (
            joints_camera,
            joints_world,
            vertices_camera,
            vertices_world,
        )
    )
    availability = {
        "poses_axis_angle": "native_world_global_orient_plus_body_pose",
        "shape_parameters": "native_per_frame_betas",
        "root_translation_camera": "native",
        "root_translation_world": "native",
        "joints_camera": (
            "official_smplx_to_smpl_materialization"
            if joints_camera is not None
            else "requires_official_body_model_materialization"
        ),
        "joints_world": (
            "official_smplx_to_smpl_materialization"
            if joints_world is not None
            else "requires_official_body_model_materialization"
        ),
        "vertices_camera": (
            "official_smplx_to_smpl_materialization"
            if vertices_camera is not None
            else "requires_official_body_model_materialization"
        ),
        "vertices_world": (
            "official_smplx_to_smpl_materialization"
            if vertices_world is not None
            else "requires_official_body_model_materialization"
        ),
    }
    track = MonocularTrack(
        track_id="person_0",
        frame_ids=frame_ids,
        valid=valid,
        body_model="smpl",
        joint_names=SMPL24_NAMES if joint_counts else (),
        poses_axis_angle=world_pose,
        shape_parameters=world["betas"],
        joints_camera=joints_camera,
        joints_world=joints_world,
        vertices_camera=vertices_camera,
        vertices_world=vertices_world,
        root_translation_camera=camera["transl"],
        root_translation_world=world["transl"],
        native_parameters={
            "global_orient_camera": camera["global_orient"],
            "global_orient_world": world["global_orient"],
            "body_pose": world["body_pose"],
            "betas": world["betas"],
            "transl_camera": camera["transl"],
            "transl_world": world["transl"],
        },
        availability=availability,
        metadata={
            "valid_mask_source": valid_source,
            "valid_mask_semantics": (
                "finite output frame; original YOLO detection validity is not "
                "retained by hmr4d_results.pt"
            ),
            "native_body_model": "SMPL-X body parameters",
            "contract_body_model": "SMPL after official sparse SMPL-X-to-SMPL mapping",
            "poses_axis_angle_coordinate_system": "gravity_world_y_up",
            "materialized_geometry": converted,
        },
    )
    metadata = {
        "official_repository": "zju3dv/GVHMR",
        "official_demo_result": "hmr4d_results.pt",
        "output_clock": "official_demo_rewrites_video_to_30_fps",
        "camera_to_world": "not_emitted_by_official_demo",
        "single_person_tracking": (
            "official Tracker.get_one_track selects top-area track and "
            "interpolates missing boxes"
        ),
    }
    if inference_metadata:
        metadata.update(dict(inference_metadata))
    return MonocularCaptureResult(
        source_model="GVHMR",
        source_revision=OFFICIAL_RUNTIME_REVISION,
        checkpoint_sha256=checkpoint_sha256,
        original_fps=float(original_fps),
        output_fps=_OUTPUT_FPS,
        tracks=(track,),
        camera_coordinate_system=CAMERA_OPENCV,
        world_coordinate_system=GRAVITY_WORLD_Y_UP,
        camera_intrinsics=intrinsics,
        frame_timestamps=frame_ids.astype(np.float64) / _OUTPUT_FPS,
        metadata=metadata,
    )


def _video_fps(path: str | Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "Could not determine the input video FPS; pass original_fps explicitly."
        )
    return fps


@PIPELINES.register_module()
class GVHMRPipeline(BasePipeline):
    """Run and parse the pinned official single-person GVHMR demo."""

    BUNDLE_CLS = "motius.models.gvhmr.GVHMRBundle"

    def infer_video(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        original_fps: Optional[float] = None,
        static_camera: bool = False,
        use_dpvo: bool = False,
        focal_length_mm: Optional[int] = None,
        verbose: bool = False,
        materialize_geometry: bool = True,
        bbox_xyxy: Optional[np.ndarray] = None,
    ) -> MonocularCaptureResult:
        fps = _video_fps(video) if original_fps is None else float(original_fps)
        raw_result = self.bundle.run_official_demo(
            video,
            output_root,
            static_camera=static_camera,
            use_dpvo=use_dpvo,
            focal_length_mm=focal_length_mm,
            verbose=verbose,
            bbox_xyxy=bbox_xyxy,
        )
        source = (
            self.bundle.convert_official_result(raw_result)
            if materialize_geometry
            else raw_result
        )
        result = parse_gvhmr_output(
            source,
            checkpoint_sha256=self.bundle.checkpoint_sha256,
            original_fps=fps,
            inference_metadata={
                "static_camera": bool(static_camera),
                "visual_odometry": "DPVO" if use_dpvo else "SimpleVO",
                "focal_length_mm": (
                    None if focal_length_mm is None else int(focal_length_mm)
                ),
                "geometry_materialized": bool(materialize_geometry),
                "tracking_bbox_source": (
                    "caller" if bbox_xyxy is not None else "official_yolov8x"
                ),
            },
        )
        if bbox_xyxy is None:
            return result
        boxes = np.asarray(bbox_xyxy, dtype=np.float32)
        if boxes.shape != (result.tracks[0].num_frames, 4):
            raise ValueError(
                "bbox_xyxy must align with the official 30 FPS output frames."
            )
        track = replace(
            result.tracks[0],
            native_parameters={
                **dict(result.tracks[0].native_parameters),
                "tracking_bboxes": boxes,
            },
            metadata={
                **dict(result.tracks[0].metadata),
                "tracking_bbox_source": "caller",
            },
        )
        return replace(
            result,
            tracks=(track,),
            metadata={
                **dict(result.metadata),
                "single_person_tracking": "caller-provided dense bbox track",
            },
        )

    def parse_output(
        self,
        source: str | Path | Mapping[str, Any],
        *,
        original_fps: float,
        checkpoint_sha256: Optional[str] = None,
        **kwargs,
    ) -> MonocularCaptureResult:
        digest = checkpoint_sha256 or self.bundle.checkpoint_sha256
        return parse_gvhmr_output(
            source,
            checkpoint_sha256=digest,
            original_fps=original_fps,
            **kwargs,
        )

    def __call__(self, video: str | Path, output_root: str | Path, **kwargs):
        return self.infer_video(video, output_root, **kwargs)


__all__ = [
    "GVHMRPipeline",
    "load_gvhmr_output",
    "parse_gvhmr_output",
]
