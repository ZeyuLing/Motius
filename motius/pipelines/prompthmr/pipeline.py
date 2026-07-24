"""Official PromptHMR-Video command runner and results.pkl converter."""

from __future__ import annotations

import os
import pickle
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from motius.models.prompthmr.bundle import (
    PROMPTHMR_REVISION,
    PromptHMRBundle,
)
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    CoordinateSystem,
    MonocularCaptureResult,
    MonocularTrack,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


PROMPTHMR_WORLD_COORDINATE = CoordinateSystem(
    name="prompthmr_gravity_world",
    up_axis="+Y",
    forward_axis="-Z",
    handedness="right",
    units="meter",
    origin="sequence_local_fitted_floor",
)

DEFAULT_TRACKER_PROVENANCE = {
    "name": "SAM 2 video predictor",
    "variant": "sam2_hiera_tiny",
    "initial_detector": "Detectron2 Keypoint R-CNN X-101-32x8d-FPN",
    "segmentation_filter": "torchvision DeepLabV3-ResNet50",
    "upstream_config": "pipeline/config.yaml: tracker=sam2",
}
DEFAULT_DETECTOR_PROVENANCE = {
    "name": "ViTPose-H",
    "keypoint_layout": "COCO-25 converted to OpenPose",
    "checkpoint": "data/pretrain/vitpose-h-coco_25.pth",
    "upstream_config": "pipeline/config.yaml: kp2d_detector=vitpose",
}
DEFAULT_CAMERA_PROVENANCE = {
    "motion": "Masked DROID-SLAM",
    "metric_scale": "Metric3D ViT-Large",
    "gravity": "SPEC camera calibration",
    "world_postprocess": "PromptHMR floor fitting and optional reprojection/contact optimization",
}


@dataclass(frozen=True)
class PromptHMROfficialCommand:
    """A reproducible invocation of the pinned official video demo."""

    argv: Tuple[str, ...]
    cwd: Path
    output_path: Path


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"Official PromptHMR field {name!r} must be a mapping.")
    return value


def _load_official_results(
    source: Union[str, Path, Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Load only trusted official results.pkl files.

    Both pickle and joblib formats can execute code while loading. Callers must
    never pass an untrusted file.
    """

    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"PromptHMR results file does not exist: {path}")
    try:
        import joblib
    except ImportError:
        with path.open("rb") as stream:
            value = pickle.load(stream)
    else:
        value = joblib.load(path)
    return _mapping(value, "results")


def _frames(value: Any, track_name: str) -> np.ndarray:
    frames = _numpy(value)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"{track_name}.frames must be a one-dimensional integer array.")
    frames = frames.astype(np.int64, copy=False)
    if len(frames) and (frames[0] < 0 or np.any(np.diff(frames) <= 0)):
        raise ValueError(f"{track_name}.frames must be non-negative and increasing.")
    return frames


def _track_array(
    value: Any,
    *,
    frames: np.ndarray,
    sequence_frames: int,
    name: str,
) -> np.ndarray:
    array = _numpy(value)
    if array.ndim < 1 or array.shape[0] != len(frames):
        raise ValueError(
            f"{name} must have {len(frames)} rows, got shape {array.shape}."
        )
    if len(frames) and frames[-1] >= sequence_frames:
        raise ValueError(
            f"{name} references frame {frames[-1]}, but the sequence has "
            f"{sequence_frames} frames."
        )
    dense = np.zeros((sequence_frames,) + array.shape[1:], dtype=array.dtype)
    dense[frames] = array
    return dense


def _pose_array(
    value: Any,
    *,
    frames: np.ndarray,
    sequence_frames: int,
    name: str,
) -> np.ndarray:
    pose = _numpy(value)
    if pose.ndim == 2:
        if pose.shape[1] % 3:
            raise ValueError(f"{name} width must be divisible by three.")
        pose = pose.reshape(len(frames), -1, 3)
    elif pose.ndim < 3 or pose.shape[-1] != 3:
        raise ValueError(f"{name} must be flattened axis-angle or end in 3.")
    return _track_array(
        pose,
        frames=frames,
        sequence_frames=sequence_frames,
        name=name,
    )


def _sequence_length(results: Mapping[str, Any], people: Mapping) -> int:
    candidates = []
    for camera_key, rotation_key in (
        ("camera", "pred_cam_R"),
        ("camera_world", "Rwc"),
    ):
        camera = results.get(camera_key)
        if isinstance(camera, Mapping) and rotation_key in camera:
            rotation = _numpy(camera[rotation_key])
            if rotation.ndim >= 1:
                candidates.append(int(rotation.shape[0]))
    for key, person in people.items():
        person = _mapping(person, f"people[{key!r}]")
        frames = _frames(person.get("frames", []), f"people[{key!r}]")
        if len(frames):
            candidates.append(int(frames[-1]) + 1)
    return max(candidates, default=0)


def _camera_intrinsics(results: Mapping[str, Any]) -> Optional[np.ndarray]:
    camera = results.get("camera")
    if not isinstance(camera, Mapping):
        return None
    if "img_focal" not in camera or "img_center" not in camera:
        return None
    focal = _numpy(camera["img_focal"]).reshape(-1)
    center = _numpy(camera["img_center"]).reshape(-1)
    if not len(focal) or len(center) < 2:
        return None
    fx = float(focal[0])
    fy = float(focal[1] if len(focal) > 1 else focal[0])
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = float(center[0])
    intrinsics[1, 2] = float(center[1])
    return intrinsics


def _camera_to_world(
    results: Mapping[str, Any], sequence_frames: int
) -> Optional[np.ndarray]:
    camera = results.get("camera_world")
    if not isinstance(camera, Mapping) or not {"Rwc", "Twc"} <= set(camera):
        return None
    rotation = _numpy(camera["Rwc"])
    translation = _numpy(camera["Twc"])
    if rotation.shape != (sequence_frames, 3, 3):
        raise ValueError(
            "camera_world.Rwc must have shape "
            f"({sequence_frames}, 3, 3), got {rotation.shape}."
        )
    if translation.shape != (sequence_frames, 3):
        raise ValueError(
            "camera_world.Twc must have shape "
            f"({sequence_frames}, 3), got {translation.shape}."
        )
    transforms = np.broadcast_to(
        np.eye(4, dtype=np.result_type(rotation, translation)),
        (sequence_frames, 4, 4),
    ).copy()
    transforms[:, :3, :3] = rotation
    transforms[:, :3, 3] = translation
    return transforms


def _dense_native(
    source: Mapping,
    keys: Sequence[str],
    prefix: str,
    frames: np.ndarray,
    sequence_frames: int,
) -> Dict[str, np.ndarray]:
    output = {}
    for key in keys:
        if key not in source:
            continue
        output[f"{prefix}{key}"] = _track_array(
            source[key],
            frames=frames,
            sequence_frames=sequence_frames,
            name=f"{prefix}{key}",
        )
    return output


def parse_prompthmr_results(
    source: Union[str, Path, Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    original_fps: float,
    output_fps: Optional[float] = None,
    prompt_types: Sequence[str] = ("box", "keypoint", "mask"),
    tracker_provenance: Optional[Mapping[str, Any]] = None,
    detector_provenance: Optional[Mapping[str, Any]] = None,
    camera_provenance: Optional[Mapping[str, Any]] = None,
    checkpoint_sha256s: Optional[Mapping[str, str]] = None,
) -> MonocularCaptureResult:
    """Convert the exact official ``results.pkl`` schema to the shared contract."""

    results = _load_official_results(source)
    people = _mapping(results.get("people", {}), "people")
    sequence_frames = _sequence_length(results, people)
    tracks = []
    has_world_tracks = False

    for key, raw_person in people.items():
        track_name = f"people[{key!r}]"
        person = _mapping(raw_person, track_name)
        frames = _frames(person.get("frames", []), track_name)
        camera_smplx = _mapping(person.get("smplx_cam"), f"{track_name}.smplx_cam")
        for required in ("pose", "shape", "trans"):
            if required not in camera_smplx:
                raise ValueError(
                    f"Official PromptHMR output is missing {track_name}."
                    f"smplx_cam.{required}."
                )

        valid = np.zeros(sequence_frames, dtype=np.bool_)
        valid[frames] = True
        poses = _pose_array(
            camera_smplx["pose"],
            frames=frames,
            sequence_frames=sequence_frames,
            name=f"{track_name}.smplx_cam.pose",
        )
        shape = _track_array(
            camera_smplx["shape"],
            frames=frames,
            sequence_frames=sequence_frames,
            name=f"{track_name}.smplx_cam.shape",
        )
        translation_camera = _track_array(
            camera_smplx["trans"],
            frames=frames,
            sequence_frames=sequence_frames,
            name=f"{track_name}.smplx_cam.trans",
        )

        native = _dense_native(
            camera_smplx,
            ("pose", "shape", "trans", "rotmat", "contact", "static_conf_logits"),
            "smplx_camera_",
            frames,
            sequence_frames,
        )
        native.update(
            _dense_native(
                person,
                ("bboxes", "detected"),
                "tracking_",
                frames,
                sequence_frames,
            )
        )

        translation_world = None
        world_smplx = person.get("smplx_world")
        if world_smplx is not None:
            world_smplx = _mapping(world_smplx, f"{track_name}.smplx_world")
            for required in ("pose", "shape", "trans"):
                if required not in world_smplx:
                    raise ValueError(
                        f"Official PromptHMR output is missing {track_name}."
                        f"smplx_world.{required}."
                    )
            translation_world = _track_array(
                world_smplx["trans"],
                frames=frames,
                sequence_frames=sequence_frames,
                name=f"{track_name}.smplx_world.trans",
            )
            native.update(
                _dense_native(
                    world_smplx,
                    ("pose", "shape", "trans"),
                    "smplx_world_",
                    frames,
                    sequence_frames,
                )
            )
            has_world_tracks = True

        availability = {
            "poses_axis_angle": (
                "official smplx_cam.pose; camera-space root orientation with "
                "SMPL-X local body parameters"
            ),
            "shape_parameters": "official smplx_cam.shape",
            "root_translation_camera": "official smplx_cam.trans, meters",
            "root_translation_world": (
                "official smplx_world.trans after gravity/floor alignment and "
                "optional official post-optimization"
                if translation_world is not None
                else "not present in this official output; not synthesized"
            ),
            "joints_camera": (
                "not saved by official video results.pkl; requires licensed "
                "SMPL-X body-model replay"
            ),
            "joints_world": (
                "not saved by official video results.pkl; requires licensed "
                "SMPL-X body-model replay"
            ),
            "vertices_camera": "not saved by official video results.pkl",
            "vertices_world": "deleted by the official world conversion before save",
            "confidence": (
                "no calibrated per-frame person confidence; contact logits are "
                "retained only as native parameters"
            ),
        }
        track_id = str(person.get("track_id", key))
        tracks.append(
            MonocularTrack(
                track_id=track_id,
                frame_ids=np.arange(sequence_frames, dtype=np.int64),
                valid=valid,
                body_model="SMPL-X neutral",
                poses_axis_angle=poses,
                shape_parameters=shape,
                root_translation_camera=translation_camera,
                root_translation_world=translation_world,
                native_parameters=native,
                availability=availability,
                metadata={
                    "official_track_key": str(key),
                    "official_frame_ids": frames.tolist(),
                    "invalid_rows": (
                        "zero placeholders; valid is authoritative for sparse tracks"
                    ),
                    "standard_pose_space": "camera",
                },
            )
        )

    camera_to_world = _camera_to_world(results, sequence_frames)
    if camera_to_world is not None and not tracks:
        camera_to_world = None
    world_coordinate = (
        PROMPTHMR_WORLD_COORDINATE
        if has_world_tracks or camera_to_world is not None
        else None
    )
    component_hashes = dict(checkpoint_sha256s or {})
    if "video_head" in component_hashes:
        if component_hashes["video_head"].lower() != checkpoint_sha256.lower():
            raise ValueError(
                "checkpoint_sha256 must match checkpoint_sha256s['video_head']."
            )
    else:
        component_hashes["video_head"] = checkpoint_sha256

    result_flags = {
        key: bool(results[key])
        for key in (
            "has_tracks",
            "has_hps_cam",
            "has_hps_world",
            "has_slam",
            "has_2d_kpts",
            "has_post_opt",
        )
        if key in results
    }
    effective_output_fps = float(
        original_fps if output_fps is None else output_fps
    )
    return MonocularCaptureResult(
        source_model="PromptHMR-Video",
        source_revision=PROMPTHMR_REVISION,
        checkpoint_sha256=checkpoint_sha256.lower(),
        original_fps=float(original_fps),
        output_fps=effective_output_fps,
        tracks=tuple(tracks),
        camera_coordinate_system=CAMERA_OPENCV,
        world_coordinate_system=world_coordinate,
        camera_intrinsics=_camera_intrinsics(results),
        camera_to_world=camera_to_world,
        frame_timestamps=(
            np.arange(sequence_frames, dtype=np.float64) / effective_output_fps
            if tracks
            else None
        ),
        metadata={
            "integration": "official scripts/demo_video.py results.pkl",
            "prompt_types": list(prompt_types),
            "tracker": dict(tracker_provenance or DEFAULT_TRACKER_PROVENANCE),
            "detector": dict(detector_provenance or DEFAULT_DETECTOR_PROVENANCE),
            "camera_estimation": dict(
                camera_provenance or DEFAULT_CAMERA_PROVENANCE
            ),
            "checkpoint_sha256s": component_hashes,
            "official_result_keys": sorted(str(key) for key in results),
            "official_flags": result_flags,
            "sequence_frames": sequence_frames,
            "camera_coordinate_note": (
                "OpenCV optical camera: +X right, +Y down, +Z forward, meters"
            ),
            "world_coordinate_note": (
                "Only populated from official smplx_world/camera_world fields; "
                "Motius never promotes camera-space values to world-space."
            ),
        },
    )


def build_prompthmr_video_command(
    bundle: PromptHMRBundle,
    input_video: Union[str, Path],
    *,
    static_camera: bool = False,
    viser_total: int = 1500,
    viser_subsample: int = 1,
) -> PromptHMROfficialCommand:
    """Build Tyro's canonical CLI flags for official ``demo_video.py``."""

    if bundle.upstream_dir is None:
        raise ValueError("bundle.upstream_dir is required to build the command.")
    if viser_total < 1 or viser_subsample < 1:
        raise ValueError("viser_total and viser_subsample must be positive.")
    upstream = Path(bundle.upstream_dir).expanduser().resolve()
    video = Path(input_video).expanduser().resolve()
    official_stem = video.name.split(".")[0]
    if not official_stem:
        raise ValueError(f"Could not derive the official output name from {video}.")
    argv = [
        *bundle.python_command,
        "scripts/demo_video.py",
        "--input-video",
        str(video),
        "--no-run-viser",
        "--viser-total",
        str(int(viser_total)),
        "--viser-subsample",
        str(int(viser_subsample)),
    ]
    if static_camera:
        argv.append("--static-camera")
    return PromptHMROfficialCommand(
        argv=tuple(argv),
        cwd=upstream,
        output_path=upstream / "results" / official_stem / "results.pkl",
    )


@PIPELINES.register_module()
class PromptHMRPipeline(BasePipeline):
    """Run pinned official PromptHMR-Video and return the shared Motius result."""

    BUNDLE_CLS = "motius.models.prompthmr.PromptHMRBundle"

    def __init__(
        self,
        bundle: PromptHMRBundle,
        *,
        prompt_types: Optional[Sequence[str]] = None,
        tracker: str = "sam2",
    ) -> None:
        super().__init__(bundle)
        self.tracker = str(tracker).lower()
        if prompt_types is None:
            prompt_types = (
                ("box", "keypoint", "mask")
                if self.tracker == "sam2"
                else ("box", "keypoint")
            )
        self.prompt_types = tuple(str(value) for value in prompt_types)

    def build_command(self, input_video: Union[str, Path], **kwargs):
        return build_prompthmr_video_command(self.bundle, input_video, **kwargs)

    def parse_output(
        self,
        source: Union[str, Path, Mapping[str, Any]],
        *,
        original_fps: float,
        output_fps: Optional[float] = None,
        checkpoint_sha256: Optional[str] = None,
        checkpoint_sha256s: Optional[Mapping[str, str]] = None,
    ) -> MonocularCaptureResult:
        hashes = dict(checkpoint_sha256s or {})
        if checkpoint_sha256 is None:
            verified = self.bundle.verify_checkpoints()
            hashes.update(verified)
            checkpoint_sha256 = verified["video_head"]
        tracker_provenance = dict(DEFAULT_TRACKER_PROVENANCE)
        if self.tracker == "bytetrack":
            tracker_provenance = {
                "name": "Supervision ByteTrack",
                "initial_detector": "Ultralytics YOLO11x person detector",
                "upstream_config": "pipeline/config.yaml: tracker=bytetrack",
            }
        return parse_prompthmr_results(
            source,
            checkpoint_sha256=checkpoint_sha256,
            original_fps=original_fps,
            output_fps=output_fps,
            prompt_types=self.prompt_types,
            tracker_provenance=tracker_provenance,
            checkpoint_sha256s=hashes,
        )

    def replay_licensed_geometry(
        self,
        result: MonocularCaptureResult,
        model_path: Union[str, Path],
        **kwargs,
    ) -> MonocularCaptureResult:
        """Materialize geometry with a user-supplied licensed SMPL-X file."""

        from .replay import replay_prompthmr_with_licensed_model

        return replay_prompthmr_with_licensed_model(
            result,
            Path(model_path),
            **kwargs,
        )

    def __call__(
        self,
        input_video: Union[str, Path],
        *,
        original_fps: float,
        output_fps: Optional[float] = None,
        static_camera: bool = False,
        viser_total: int = 1500,
        viser_subsample: int = 1,
        reuse_existing: bool = False,
        env: Optional[Mapping[str, str]] = None,
    ) -> MonocularCaptureResult:
        video = Path(input_video).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video}")
        self.bundle.stage_official_runtime_checkpoint()
        hashes = self.bundle.verify_checkpoints()
        command = self.build_command(
            video,
            static_camera=static_camera,
            viser_total=viser_total,
            viser_subsample=viser_subsample,
        )
        if command.output_path.exists() and not reuse_existing:
            raise FileExistsError(
                f"Official output already exists at {command.output_path}. "
                "Refusing to reuse a basename-colliding or stale result; pass "
                "reuse_existing=True only after verifying its provenance."
            )
        if not command.output_path.exists():
            process_env = os.environ.copy()
            process_env["PYTHONNOUSERSITE"] = "1"
            process_env["PYTHONFAULTHANDLER"] = "1"
            process_env["OMP_NUM_THREADS"] = "1"
            process_env["MKL_NUM_THREADS"] = "1"
            if env:
                process_env.update({str(key): str(value) for key, value in env.items()})
            try:
                subprocess.run(
                    list(command.argv),
                    cwd=command.cwd,
                    env=process_env,
                    check=True,
                )
            except subprocess.CalledProcessError:
                # The pinned demo serializes the authoritative numeric
                # results.pkl before its optional MCS/GLB visualization export.
                # Some licensed installs omit the slim visualization-only
                # SMPL-X asset. Accept that late exporter failure only when the
                # complete official numeric result already exists.
                if not command.output_path.is_file():
                    raise
        if not command.output_path.is_file():
            raise RuntimeError(
                "Official PromptHMR command completed without producing "
                f"{command.output_path}."
            )
        return self.parse_output(
            command.output_path,
            original_fps=original_fps,
            output_fps=output_fps,
            checkpoint_sha256=hashes["video_head"],
            checkpoint_sha256s=hashes,
        )


__all__ = [
    "DEFAULT_CAMERA_PROVENANCE",
    "DEFAULT_DETECTOR_PROVENANCE",
    "DEFAULT_TRACKER_PROVENANCE",
    "PROMPTHMR_WORLD_COORDINATE",
    "PromptHMROfficialCommand",
    "PromptHMRPipeline",
    "build_prompthmr_video_command",
    "parse_prompthmr_results",
]
