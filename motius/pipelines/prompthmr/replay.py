"""Licensed SMPL-X replay for official PromptHMR parameter outputs.

This module materializes geometry from user-supplied body-model files. The
resulting joints and vertices are explicitly marked as replayed geometry; they
are never represented as native fields from PromptHMR's ``results.pkl``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from motius.models.prompthmr.bundle import PROMPTHMR_REVISION, sha256_file
from motius.motion.representation.monocular_capture import (
    MonocularCaptureResult,
    MonocularTrack,
)


# SMPL-X uses the same first 22 articulated body joints as SMPL. These names
# match the public SMPL24 names used by Motius evaluation, excluding the final
# two SMPL hand joints that are not part of SMPL-X's first-22 body chain.
SMPL_SMPLX_BODY22_NAMES: Tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)
_VALID_GENDERS = {"neutral", "female", "male"}
_CAMERA_POSE_WIDTH = 75
_WORLD_POSE_WIDTH = 165


@dataclass(frozen=True)
class PromptHMRSMPLXParameters:
    """SMPL-X keyword parameters split exactly like pinned upstream code."""

    coordinate_space: str
    global_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    ignored_face_pose: np.ndarray
    hand_pose_source: str

    @property
    def num_frames(self) -> int:
        return int(self.global_orient.shape[0])


@dataclass(frozen=True)
class LicensedSMPLXProvenance:
    """Publishable provenance for a private, user-supplied SMPL-X file."""

    model_version: str
    gender: str
    filename: str
    sha256: str
    file_size_bytes: int
    detected_version: Optional[str] = None
    detected_gender: Optional[str] = None
    model_type: str = "SMPL-X"
    source: str = "user_supplied_licensed_file"

    def __post_init__(self) -> None:
        gender = self.gender.lower()
        if gender not in _VALID_GENDERS:
            raise ValueError(f"SMPL-X gender must be one of {sorted(_VALID_GENDERS)}.")
        if not self.model_version.strip():
            raise ValueError("SMPL-X model_version must be explicitly declared.")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("SMPL-X sha256 must be a 64-character hex digest.")
        if self.file_size_bytes < 0:
            raise ValueError("SMPL-X file_size_bytes must be non-negative.")
        object.__setattr__(self, "gender", gender)
        object.__setattr__(self, "sha256", digest)

    def as_metadata(self) -> dict:
        return {
            "source": self.source,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "gender": self.gender,
            "filename": self.filename,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "detected_version": self.detected_version,
            "detected_gender": self.detected_gender,
            "redistributed": False,
            "local_path_recorded": False,
        }


def _flat_pose(value: np.ndarray, *, width: int, name: str) -> np.ndarray:
    pose = np.asarray(value)
    if pose.ndim == 3 and pose.shape[-1] == 3:
        pose = pose.reshape(pose.shape[0], -1)
    if pose.ndim != 2 or pose.shape[1] != width:
        raise ValueError(f"{name} must have shape (frames, {width}), got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} contains non-finite values.")
    return pose


def split_prompthmr_smplx_pose(
    pose: np.ndarray,
    *,
    coordinate_space: str,
) -> PromptHMRSMPLXParameters:
    """Split official PromptHMR pose vectors for an SMPL-X forward call.

    Pinned ``phmr_vid.py`` saves camera pose as 75 values: global orientation,
    21 body joints, and three zero face joints. It saves no hand parameters.
    Pinned ``world.py`` saves 165 values but calls SMPL-X with body and hand
    slices only, explicitly replacing jaw and eye poses with zeros.
    """

    space = str(coordinate_space).lower()
    if space == "camera":
        flat = _flat_pose(
            pose,
            width=_CAMERA_POSE_WIDTH,
            name="smplx_cam.pose",
        )
        if not np.allclose(flat[:, 66:75], 0.0, atol=1e-7):
            raise ValueError(
                "Pinned phmr_vid.py writes zero jaw/eye values in "
                "smplx_cam.pose[66:75]; refusing an incompatible schema."
            )
        hands = np.zeros((len(flat), 45), dtype=flat.dtype)
        hand_source = (
            "zero_axis_angle_flat_hand; official camera results do not save hands"
        )
    elif space == "world":
        flat = _flat_pose(
            pose,
            width=_WORLD_POSE_WIDTH,
            name="smplx_world.pose",
        )
        hands = None
        hand_source = "official smplx_world.pose hand slices"
    else:
        raise ValueError("coordinate_space must be camera or world.")

    zeros_face = np.zeros((len(flat), 3), dtype=flat.dtype)
    return PromptHMRSMPLXParameters(
        coordinate_space=space,
        global_orient=flat[:, :3],
        body_pose=flat[:, 3:66],
        left_hand_pose=hands if hands is not None else flat[:, 75:120],
        right_hand_pose=hands.copy() if hands is not None else flat[:, 120:165],
        jaw_pose=zeros_face,
        leye_pose=zeros_face.copy(),
        reye_pose=zeros_face.copy(),
        ignored_face_pose=flat[:, 66:75],
        hand_pose_source=hand_source,
    )


def _safe_npz_scalar(path: Path, names: Sequence[str]) -> Optional[str]:
    if path.suffix.lower() != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in names:
                if name not in archive:
                    continue
                value = np.asarray(archive[name])
                if value.size != 1 or value.dtype.kind not in {"U", "S", "i", "u", "f"}:
                    continue
                scalar = value.reshape(-1)[0]
                if isinstance(scalar, bytes):
                    scalar = scalar.decode("utf-8", errors="replace")
                return str(scalar)
    except (OSError, ValueError, KeyError):
        return None
    return None


def inspect_licensed_smplx_model(
    model_path: Path,
    *,
    gender: str,
    model_version: str,
    expected_sha256: Optional[str] = None,
) -> LicensedSMPLXProvenance:
    """Hash and validate a local SMPL-X file without redistributing it."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Licensed SMPL-X model file does not exist: {path}")
    normalized_gender = str(gender).lower()
    if normalized_gender not in _VALID_GENDERS:
        raise ValueError(f"SMPL-X gender must be one of {sorted(_VALID_GENDERS)}.")
    if not str(model_version).strip():
        raise ValueError("model_version must be explicitly supplied.")
    upper_filename = path.name.upper()
    if upper_filename.startswith(("SMPL_", "SMPLH_")):
        raise ValueError(
            f"Expected an SMPL-X model file, not {path.name!r}."
        )

    filename_gender = next(
        (
            candidate.lower()
            for candidate in ("NEUTRAL", "FEMALE", "MALE")
            if candidate in path.name.upper()
        ),
        None,
    )
    detected_gender = _safe_npz_scalar(path, ("gender", "model_gender"))
    if detected_gender is not None:
        detected_gender = detected_gender.lower()
    for source_name, detected in (
        ("filename", filename_gender),
        ("model metadata", detected_gender),
    ):
        if detected is not None and detected != normalized_gender:
            raise ValueError(
                f"SMPL-X gender mismatch: caller declared {normalized_gender!r}, "
                f"but {source_name} indicates {detected!r}."
            )

    detected_version = _safe_npz_scalar(
        path, ("model_version", "version", "smplx_version")
    )
    if (
        detected_version is not None
        and detected_version.strip().lower() != str(model_version).strip().lower()
    ):
        raise ValueError(
            f"SMPL-X version mismatch: caller declared {model_version!r}, "
            f"model metadata indicates {detected_version!r}."
        )

    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch for {path.name}: expected "
            f"{expected_sha256.lower()}, found {digest}."
        )
    return LicensedSMPLXProvenance(
        model_version=str(model_version),
        gender=normalized_gender,
        filename=path.name,
        sha256=digest,
        file_size_bytes=path.stat().st_size,
        detected_version=detected_version,
        detected_gender=detected_gender or filename_gender,
    )


def load_licensed_smplx_model(
    model_path: Path,
    *,
    gender: str,
    model_version: str,
    expected_sha256: Optional[str] = None,
    device: Optional[str] = None,
    body_model_factory: Optional[Any] = None,
) -> tuple[Any, LicensedSMPLXProvenance]:
    """Load one user-authorized SMPL-X file and return publishable provenance."""

    path = Path(model_path).expanduser().resolve()
    provenance = inspect_licensed_smplx_model(
        path,
        gender=gender,
        model_version=model_version,
        expected_sha256=expected_sha256,
    )
    if body_model_factory is None:
        try:
            import smplx
        except ImportError as exc:
            raise ImportError(
                "Licensed SMPL-X replay requires the optional `smplx` package. "
                "Motius never downloads the model file."
            ) from exc
        body_model = smplx.SMPLX(
            str(path),
            gender=provenance.gender,
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        )
    else:
        body_model = body_model_factory(
            model_path=path,
            gender=provenance.gender,
            model_version=provenance.model_version,
            flat_hand_mean=True,
            num_betas=10,
        )
    if hasattr(body_model, "eval"):
        body_model = body_model.eval()
    if device is not None and hasattr(body_model, "to"):
        body_model = body_model.to(device)
    return body_model, provenance


def _track_space_parameters(
    track: MonocularTrack,
    space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native = track.native_parameters
    if space == "camera":
        pose = native.get("smplx_camera_pose", track.poses_axis_angle)
        betas = native.get("smplx_camera_shape", track.shape_parameters)
        translation = track.root_translation_camera
    else:
        pose = native.get("smplx_world_pose")
        betas = native.get("smplx_world_shape")
        translation = track.root_translation_world
    if pose is None or betas is None or translation is None:
        raise ValueError(
            f"Track {track.track_id!r} has no official {space}-space SMPL-X "
            "parameters to replay."
        )
    return np.asarray(pose), np.asarray(betas), np.asarray(translation)


def _valid_rows(
    value: np.ndarray,
    valid: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension.")
    if array.shape[0] == len(valid):
        return array[valid]
    if array.ndim == 1:
        return np.broadcast_to(array, (int(valid.sum()),) + array.shape).copy()
    raise ValueError(
        f"{name} must align to {len(valid)} track frames, got {array.shape}."
    )


def _model_device_dtype(body_model: Any, requested_device: Optional[str]):
    import torch

    if requested_device is not None:
        return torch.device(requested_device), torch.float32
    if hasattr(body_model, "parameters"):
        try:
            parameter = next(body_model.parameters())
        except (StopIteration, TypeError):
            pass
        else:
            return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.float32


def _replay_valid_geometry(
    body_model: Any,
    parameters: PromptHMRSMPLXParameters,
    betas: np.ndarray,
    translation: np.ndarray,
    *,
    include_vertices: bool,
    batch_size: int,
    device: Optional[str],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    frame_count = parameters.num_frames
    if frame_count < 1:
        raise ValueError("Licensed replay requires at least one valid frame.")
    betas = np.asarray(betas)
    translation = np.asarray(translation)
    if betas.ndim != 2 or betas.shape[0] != frame_count or betas.shape[1] < 10:
        raise ValueError(
            "SMPL-X betas must have shape (valid_frames, at least 10), got "
            f"{betas.shape}."
        )
    if translation.shape != (frame_count, 3):
        raise ValueError(
            "SMPL-X translation must have shape (valid_frames, 3), got "
            f"{translation.shape}."
        )

    target_device, dtype = _model_device_dtype(body_model, device)
    joints_chunks = []
    vertices_chunks = []
    arrays = {
        "global_orient": parameters.global_orient,
        "body_pose": parameters.body_pose,
        "left_hand_pose": parameters.left_hand_pose,
        "right_hand_pose": parameters.right_hand_pose,
        "jaw_pose": parameters.jaw_pose,
        "leye_pose": parameters.leye_pose,
        "reye_pose": parameters.reye_pose,
    }
    with torch.inference_mode():
        for start in range(0, frame_count, batch_size):
            stop = min(start + batch_size, frame_count)
            kwargs = {
                name: torch.as_tensor(
                    value[start:stop],
                    dtype=dtype,
                    device=target_device,
                )
                for name, value in arrays.items()
            }
            kwargs.update(
                {
                    "betas": torch.as_tensor(
                        betas[start:stop, :10],
                        dtype=dtype,
                        device=target_device,
                    ),
                    "transl": torch.as_tensor(
                        translation[start:stop],
                        dtype=dtype,
                        device=target_device,
                    ),
                    "expression": torch.zeros(
                        stop - start, 10, dtype=dtype, device=target_device
                    ),
                    "pose2rot": True,
                    "return_verts": include_vertices,
                }
            )
            output = body_model(**kwargs)
            joints = (
                output["joints"]
                if isinstance(output, Mapping)
                else getattr(output, "joints", None)
            )
            vertices = (
                output.get("vertices")
                if isinstance(output, Mapping)
                else getattr(output, "vertices", None)
            )
            if joints is None:
                raise ValueError("SMPL-X body model output did not contain joints.")
            joints = np.asarray(joints.detach().cpu().numpy())[:, :22]
            if joints.shape != (stop - start, 22, 3):
                raise ValueError(
                    "SMPL-X body model must expose at least the common first 22 "
                    f"joints, got {joints.shape}."
                )
            joints_chunks.append(joints)
            if include_vertices:
                if vertices is None:
                    raise ValueError(
                        "SMPL-X body model output did not contain requested vertices."
                    )
                vertices = np.asarray(vertices.detach().cpu().numpy())
                if (
                    vertices.ndim != 3
                    or vertices.shape[0] != stop - start
                    or vertices.shape[-1] != 3
                ):
                    raise ValueError(
                        f"Invalid SMPL-X vertices shape {vertices.shape}."
                    )
                vertices_chunks.append(vertices)

    valid_joints = np.concatenate(joints_chunks, axis=0)
    valid_vertices = (
        np.concatenate(vertices_chunks, axis=0) if include_vertices else None
    )
    return valid_joints, valid_vertices


def _materialize_track_space(
    track: MonocularTrack,
    *,
    space: str,
    body_model: Any,
    provenance: LicensedSMPLXProvenance,
    include_vertices: bool,
    batch_size: int,
    device: Optional[str],
    overwrite: bool,
) -> MonocularTrack:
    normalized_body_model = (
        track.body_model.lower().replace("-", "").replace("_", "").replace(" ", "")
    )
    if "smplx" not in normalized_body_model:
        raise ValueError(
            f"Track {track.track_id!r} declares body model {track.body_model!r}, "
            "not SMPL-X."
        )
    if "neutral" in normalized_body_model and provenance.gender != "neutral":
        raise ValueError(
            "Pinned PromptHMR uses SMPLX_NEUTRAL.npz; replay with a neutral "
            "licensed model to avoid a body-model gender mismatch."
        )
    joint_field = f"joints_{space}"
    vertex_field = f"vertices_{space}"
    if not overwrite and getattr(track, joint_field) is not None:
        raise ValueError(
            f"Track {track.track_id!r} already has {joint_field}; refusing to overwrite."
        )
    if include_vertices and not overwrite and getattr(track, vertex_field) is not None:
        raise ValueError(
            f"Track {track.track_id!r} already has {vertex_field}; refusing to overwrite."
        )

    pose, betas, translation = _track_space_parameters(track, space)
    valid = np.asarray(track.valid, dtype=np.bool_)
    valid_pose = _valid_rows(pose, valid, name=f"{space} pose")
    valid_betas = _valid_rows(betas, valid, name=f"{space} betas")
    valid_translation = _valid_rows(
        translation, valid, name=f"{space} translation"
    )
    parameters = split_prompthmr_smplx_pose(
        valid_pose,
        coordinate_space=space,
    )
    valid_joints, valid_vertices = _replay_valid_geometry(
        body_model,
        parameters,
        valid_betas,
        valid_translation,
        include_vertices=include_vertices,
        batch_size=batch_size,
        device=device,
    )

    dense_joints = np.zeros(
        (track.num_frames,) + valid_joints.shape[1:],
        dtype=valid_joints.dtype,
    )
    dense_joints[valid] = valid_joints
    dense_vertices = None
    if valid_vertices is not None:
        dense_vertices = np.zeros(
            (track.num_frames,) + valid_vertices.shape[1:],
            dtype=valid_vertices.dtype,
        )
        dense_vertices[valid] = valid_vertices

    source_text = (
        "licensed_smplx_replay: user-supplied "
        f"{provenance.filename} sha256={provenance.sha256}; valid frames only"
    )
    availability = dict(track.availability)
    availability[joint_field] = (
        source_text + "; first 22 named joints shared by public SMPL/SMPL-X"
    )
    if include_vertices:
        availability[vertex_field] = source_text + "; full SMPL-X vertex topology"
    metadata = dict(track.metadata)
    replay_metadata = dict(metadata.get("licensed_smplx_replay", {}))
    replay_metadata[space] = {
        "source": "licensed_smplx_replay",
        "valid_frames_replayed": int(valid.sum()),
        "invalid_rows": "zero placeholders; original valid mask preserved",
        "pose_mapping": (
            "global[0:3], body[3:66], face forced zero, hands forced zero"
            if space == "camera"
            else "global[0:3], body[3:66], face forced zero, hands[75:165]"
        ),
        "hand_pose_source": parameters.hand_pose_source,
        "joint_schema": "smpl_smplx_body22",
        "evaluation_body_model_alias": "smpl",
        "vertices_materialized": bool(include_vertices),
        "model": provenance.as_metadata(),
    }
    metadata["licensed_smplx_replay"] = replay_metadata

    changes = {
        joint_field: dense_joints,
        "joint_names": SMPL_SMPLX_BODY22_NAMES,
        "availability": availability,
        "metadata": metadata,
    }
    if include_vertices:
        changes[vertex_field] = dense_vertices
    return replace(track, **changes)


def replay_prompthmr_geometry(
    result: MonocularCaptureResult,
    body_model: Any,
    provenance: LicensedSMPLXProvenance,
    *,
    spaces: Sequence[str] = ("camera", "world"),
    include_vertices: bool = True,
    batch_size: int = 128,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> MonocularCaptureResult:
    """Materialize named joints/vertices while retaining the original valid mask."""

    if result.source_revision != PROMPTHMR_REVISION:
        raise ValueError(
            "Licensed replay is audited only for PromptHMR revision "
            f"{PROMPTHMR_REVISION}, got {result.source_revision!r}."
        )
    body_model_gender = getattr(body_model, "gender", None)
    if (
        isinstance(body_model_gender, str)
        and body_model_gender.lower() != provenance.gender
    ):
        raise ValueError(
            f"Loaded body-model gender {body_model_gender!r} does not match "
            f"provenance gender {provenance.gender!r}."
        )
    requested_spaces = tuple(dict.fromkeys(str(value).lower() for value in spaces))
    if not requested_spaces or any(
        value not in {"camera", "world"} for value in requested_spaces
    ):
        raise ValueError("spaces must contain camera and/or world.")

    output_tracks = []
    status: Dict[str, Dict[str, str]] = {}
    for track in result.tracks:
        updated = track
        track_status = {}
        for space in requested_spaces:
            try:
                updated = _materialize_track_space(
                    updated,
                    space=space,
                    body_model=body_model,
                    provenance=provenance,
                    include_vertices=include_vertices,
                    batch_size=batch_size,
                    device=device,
                    overwrite=overwrite,
                )
            except ValueError as exc:
                if space == "world" and (
                    updated.root_translation_world is None
                    or "smplx_world_pose" not in updated.native_parameters
                ):
                    track_status[space] = (
                        "unavailable_official_world_parameters; not synthesized"
                    )
                    continue
                raise
            track_status[space] = "licensed_smplx_replay"
        status[track.track_id] = track_status
        output_tracks.append(updated)

    metadata = dict(result.metadata)
    metadata["geometry_materialization"] = {
        "source": "licensed_smplx_replay",
        "native_official_results_fields": False,
        "upstream_revision": PROMPTHMR_REVISION,
        "model": provenance.as_metadata(),
        "requested_spaces": list(requested_spaces),
        "track_status": status,
        "joint_names": list(SMPL_SMPLX_BODY22_NAMES),
        "evaluation_body_model_alias": "smpl",
        "valid_mask_preserved": True,
        "smplx_redistributed": False,
    }
    return replace(result, tracks=tuple(output_tracks), metadata=metadata)


def replay_prompthmr_with_licensed_model(
    result: MonocularCaptureResult,
    model_path: Path,
    *,
    gender: str,
    model_version: str,
    expected_sha256: Optional[str] = None,
    spaces: Sequence[str] = ("camera", "world"),
    include_vertices: bool = True,
    batch_size: int = 128,
    device: Optional[str] = None,
    body_model_factory: Optional[Any] = None,
) -> MonocularCaptureResult:
    """Load a private model file, replay geometry, and retain only provenance."""

    body_model, provenance = load_licensed_smplx_model(
        model_path,
        gender=gender,
        model_version=model_version,
        expected_sha256=expected_sha256,
        device=device,
        body_model_factory=body_model_factory,
    )
    return replay_prompthmr_geometry(
        result,
        body_model,
        provenance,
        spaces=spaces,
        include_vertices=include_vertices,
        batch_size=batch_size,
        device=device,
    )


__all__ = [
    "LicensedSMPLXProvenance",
    "PromptHMRSMPLXParameters",
    "SMPL_SMPLX_BODY22_NAMES",
    "inspect_licensed_smplx_model",
    "load_licensed_smplx_model",
    "replay_prompthmr_geometry",
    "replay_prompthmr_with_licensed_model",
    "split_prompthmr_smplx_pose",
]
