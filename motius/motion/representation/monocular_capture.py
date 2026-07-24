"""Shared result contract for monocular human motion capture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


_VALID_HANDEDNESS = {"right", "left"}
_VALID_UNITS = {"meter"}
_PRIVATE_METADATA_KEY_TOKENS = {"path", "root", "directory", "cache"}


def _array(
    value: Optional[np.ndarray],
    *,
    name: str,
    frames: Optional[int] = None,
    trailing_shape: Optional[tuple[int, ...]] = None,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if frames is not None and (array.ndim < 1 or array.shape[0] != frames):
        raise ValueError(
            f"{name} must have {frames} frames, got shape {array.shape}."
        )
    if trailing_shape is not None and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(
            f"{name} must end in {trailing_shape}, got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _public_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _public_metadata(item)
            for key, item in value.items()
            if not any(
                token in str(key).lower()
                for token in _PRIVATE_METADATA_KEY_TOKENS
            )
        }
    if isinstance(value, (list, tuple)):
        return [_public_metadata(item) for item in value]
    if isinstance(value, Path):
        return "<local-path-redacted>"
    if isinstance(value, str) and Path(value).is_absolute():
        return "<local-path-redacted>"
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class CoordinateSystem:
    """Explicit axes and units for one capture view."""

    name: str
    up_axis: str
    forward_axis: str
    handedness: str = "right"
    units: str = "meter"
    origin: str = "unspecified"

    def __post_init__(self) -> None:
        axes = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
        if self.up_axis not in axes or self.forward_axis not in axes:
            raise ValueError("Coordinate axes must be one of ±X, ±Y, or ±Z.")
        if self.up_axis[-1] == self.forward_axis[-1]:
            raise ValueError("Up and forward axes must use different dimensions.")
        if self.handedness not in _VALID_HANDEDNESS:
            raise ValueError(f"Unsupported handedness {self.handedness!r}.")
        if self.units not in _VALID_UNITS:
            raise ValueError(f"Unsupported coordinate units {self.units!r}.")


CAMERA_OPENCV = CoordinateSystem(
    name="camera_opencv",
    up_axis="-Y",
    forward_axis="+Z",
    handedness="right",
    origin="camera_optical_center",
)

GRAVITY_WORLD_Y_UP = CoordinateSystem(
    name="gravity_world_y_up",
    up_axis="+Y",
    forward_axis="+Z",
    handedness="right",
    origin="sequence_local_world",
)


@dataclass(frozen=True)
class MonocularTrack:
    """One temporally aligned person track with native and common outputs."""

    track_id: str
    frame_ids: np.ndarray
    valid: np.ndarray
    body_model: str
    joint_names: tuple[str, ...] = ()
    poses_axis_angle: Optional[np.ndarray] = None
    shape_parameters: Optional[np.ndarray] = None
    joints_camera: Optional[np.ndarray] = None
    joints_world: Optional[np.ndarray] = None
    vertices_camera: Optional[np.ndarray] = None
    vertices_world: Optional[np.ndarray] = None
    root_translation_camera: Optional[np.ndarray] = None
    root_translation_world: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    native_parameters: Mapping[str, np.ndarray] = field(default_factory=dict)
    availability: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame_ids = np.asarray(self.frame_ids)
        valid = np.asarray(self.valid)
        if frame_ids.ndim != 1 or not np.issubdtype(frame_ids.dtype, np.integer):
            raise ValueError("frame_ids must be a one-dimensional integer array.")
        if valid.shape != frame_ids.shape or valid.dtype != np.bool_:
            raise ValueError("valid must be a boolean mask aligned to frame_ids.")
        if len(frame_ids) and np.any(np.diff(frame_ids) <= 0):
            raise ValueError("frame_ids must be strictly increasing.")
        frames = len(frame_ids)
        if not self.track_id:
            raise ValueError("track_id must be non-empty.")
        if not self.body_model:
            raise ValueError("body_model must be non-empty.")

        object.__setattr__(self, "frame_ids", frame_ids.astype(np.int64, copy=False))
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "poses_axis_angle",
            _array(
                self.poses_axis_angle,
                name="poses_axis_angle",
                frames=frames,
                trailing_shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "shape_parameters",
            _array(self.shape_parameters, name="shape_parameters"),
        )
        object.__setattr__(
            self,
            "joints_camera",
            _array(
                self.joints_camera,
                name="joints_camera",
                frames=frames,
                trailing_shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "joints_world",
            _array(
                self.joints_world,
                name="joints_world",
                frames=frames,
                trailing_shape=(3,),
            ),
        )
        for name in ("vertices_camera", "vertices_world"):
            object.__setattr__(
                self,
                name,
                _array(
                    getattr(self, name),
                    name=name,
                    frames=frames,
                    trailing_shape=(3,),
                ),
            )
        for name in ("root_translation_camera", "root_translation_world"):
            object.__setattr__(
                self,
                name,
                _array(
                    getattr(self, name),
                    name=name,
                    frames=frames,
                    trailing_shape=(3,),
                ),
            )
        if self.confidence is not None:
            confidence = _array(
                self.confidence,
                name="confidence",
                frames=frames,
            )
            object.__setattr__(self, "confidence", confidence)
        if self.joint_names:
            joint_counts = {
                value.shape[-2]
                for value in (self.joints_camera, self.joints_world)
                if value is not None
            }
            if joint_counts and joint_counts != {len(self.joint_names)}:
                raise ValueError(
                    "joint_names must match camera/world joint array sizes."
                )
        for name, value in self.native_parameters.items():
            _array(value, name=f"native_parameters[{name!r}]")

    @property
    def num_frames(self) -> int:
        return int(len(self.frame_ids))

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.num_frames else 0.0


@dataclass(frozen=True)
class MonocularCaptureResult:
    """Versioned multi-track result emitted by every monocular pipeline."""

    source_model: str
    source_revision: str
    checkpoint_sha256: str
    original_fps: float
    output_fps: float
    tracks: tuple[MonocularTrack, ...]
    camera_coordinate_system: CoordinateSystem = CAMERA_OPENCV
    world_coordinate_system: Optional[CoordinateSystem] = None
    camera_intrinsics: Optional[np.ndarray] = None
    camera_to_world: Optional[np.ndarray] = None
    frame_timestamps: Optional[np.ndarray] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.source_model or not self.source_revision:
            raise ValueError("source_model and source_revision must be non-empty.")
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.checkpoint_sha256.lower()
        ):
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest.")
        if self.original_fps <= 0 or self.output_fps <= 0:
            raise ValueError("Frame rates must be positive.")
        if not isinstance(self.tracks, tuple):
            object.__setattr__(self, "tracks", tuple(self.tracks))
        if len({track.track_id for track in self.tracks}) != len(self.tracks):
            raise ValueError("track_id values must be unique within one result.")
        max_frames = max((track.num_frames for track in self.tracks), default=0)
        if self.camera_intrinsics is not None:
            intrinsics = _array(
                self.camera_intrinsics,
                name="camera_intrinsics",
                trailing_shape=(3, 3),
            )
            if intrinsics.ndim == 3 and intrinsics.shape[0] not in {1, max_frames}:
                raise ValueError(
                    "Per-frame camera_intrinsics must align to result frames."
                )
            object.__setattr__(self, "camera_intrinsics", intrinsics)
        if self.camera_to_world is not None:
            camera_to_world = _array(
                self.camera_to_world,
                name="camera_to_world",
                frames=max_frames,
                trailing_shape=(4, 4),
            )
            object.__setattr__(self, "camera_to_world", camera_to_world)
            if self.world_coordinate_system is None:
                raise ValueError(
                    "camera_to_world requires an explicit world coordinate system."
                )
        if self.frame_timestamps is not None:
            timestamps = _array(
                self.frame_timestamps,
                name="frame_timestamps",
                frames=max_frames,
            )
            if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0):
                raise ValueError(
                    "frame_timestamps must be a strictly increasing vector."
                )
            object.__setattr__(self, "frame_timestamps", timestamps)

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    def public_manifest(self) -> dict:
        """Return provenance and shapes without local paths or restricted arrays."""

        return {
            "schema_version": self.schema_version,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "original_fps": self.original_fps,
            "output_fps": self.output_fps,
            "camera_coordinate_system": self.camera_coordinate_system.__dict__,
            "world_coordinate_system": (
                None
                if self.world_coordinate_system is None
                else self.world_coordinate_system.__dict__
            ),
            "tracks": [
                {
                    "track_id": track.track_id,
                    "frames": track.num_frames,
                    "coverage": track.coverage,
                    "body_model": track.body_model,
                    "joint_names": list(track.joint_names),
                    "availability": dict(track.availability),
                }
                for track in self.tracks
            ],
            "metadata": _public_metadata(self.metadata),
        }


def as_tracks(values: Sequence[MonocularTrack]) -> tuple[MonocularTrack, ...]:
    """Normalize any finite track sequence to the immutable public contract."""

    return tuple(values)


def save_monocular_capture_result(
    result: MonocularCaptureResult,
    path: Path,
) -> None:
    """Save a result as a pickle-free compressed NPZ artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    track_records = []
    array_fields = (
        "frame_ids",
        "valid",
        "poses_axis_angle",
        "shape_parameters",
        "joints_camera",
        "joints_world",
        "vertices_camera",
        "vertices_world",
        "root_translation_camera",
        "root_translation_world",
        "confidence",
    )
    for index, track in enumerate(result.tracks):
        prefix = f"track_{index:04d}"
        field_keys = {}
        for field_name in array_fields:
            value = getattr(track, field_name)
            if value is not None:
                key = f"{prefix}/{field_name}"
                arrays[key] = np.asarray(value)
                field_keys[field_name] = key
        native_keys = {}
        for native_index, (name, value) in enumerate(track.native_parameters.items()):
            key = f"{prefix}/native_{native_index:04d}"
            arrays[key] = np.asarray(value)
            native_keys[name] = key
        track_records.append(
            {
                "track_id": track.track_id,
                "body_model": track.body_model,
                "joint_names": list(track.joint_names),
                "field_keys": field_keys,
                "native_keys": native_keys,
                "availability": dict(track.availability),
                "metadata": dict(track.metadata),
            }
        )
    result_fields = {}
    for field_name in (
        "camera_intrinsics",
        "camera_to_world",
        "frame_timestamps",
    ):
        value = getattr(result, field_name)
        if value is not None:
            key = f"result/{field_name}"
            arrays[key] = np.asarray(value)
            result_fields[field_name] = key
    metadata = {
        "schema_version": result.schema_version,
        "source_model": result.source_model,
        "source_revision": result.source_revision,
        "checkpoint_sha256": result.checkpoint_sha256,
        "original_fps": result.original_fps,
        "output_fps": result.output_fps,
        "camera_coordinate_system": result.camera_coordinate_system.__dict__,
        "world_coordinate_system": (
            None
            if result.world_coordinate_system is None
            else result.world_coordinate_system.__dict__
        ),
        "result_fields": result_fields,
        "tracks": track_records,
        "metadata": dict(result.metadata),
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True),
        dtype=np.str_,
    )
    np.savez_compressed(path, **arrays)


def load_monocular_capture_result(
    path: Path,
    *,
    include_vertices: bool = True,
) -> MonocularCaptureResult:
    """Load a pickle-free result written by :func:`save_monocular_capture_result`.

    Set ``include_vertices=False`` for joint-only evaluation to avoid
    decompressing the substantially larger mesh arrays.
    """

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        tracks = []
        for record in metadata["tracks"]:
            values = {
                field_name: np.asarray(archive[key])
                for field_name, key in record["field_keys"].items()
                if include_vertices or not field_name.startswith("vertices_")
            }
            native_parameters = {
                name: np.asarray(archive[key])
                for name, key in record["native_keys"].items()
            }
            tracks.append(
                MonocularTrack(
                    track_id=record["track_id"],
                    body_model=record["body_model"],
                    joint_names=tuple(record["joint_names"]),
                    native_parameters=native_parameters,
                    availability=record["availability"],
                    metadata=record["metadata"],
                    **values,
                )
            )
        result_values = {
            field_name: np.asarray(archive[key])
            for field_name, key in metadata["result_fields"].items()
        }
    world_record = metadata["world_coordinate_system"]
    return MonocularCaptureResult(
        source_model=metadata["source_model"],
        source_revision=metadata["source_revision"],
        checkpoint_sha256=metadata["checkpoint_sha256"],
        original_fps=float(metadata["original_fps"]),
        output_fps=float(metadata["output_fps"]),
        tracks=tuple(tracks),
        camera_coordinate_system=CoordinateSystem(
            **metadata["camera_coordinate_system"]
        ),
        world_coordinate_system=(
            None if world_record is None else CoordinateSystem(**world_record)
        ),
        metadata=metadata["metadata"],
        schema_version=int(metadata["schema_version"]),
        **result_values,
    )


__all__ = [
    "CAMERA_OPENCV",
    "GRAVITY_WORLD_Y_UP",
    "CoordinateSystem",
    "MonocularCaptureResult",
    "MonocularTrack",
    "as_tracks",
    "load_monocular_capture_result",
    "save_monocular_capture_result",
]
