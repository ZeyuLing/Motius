"""Materialize licensed 3DPW and EMDB annotations into canonical GT results.

The coordinate mappings in this module are protocol-selected, never inferred
from array dimensions:

* 3DPW ``cam_poses`` is the official world-to-camera transform.
* EMDB ``camera.extrinsics`` is the official world-to-camera transform.
* 3DPW validity follows the official evaluator: ``campose_valid`` and a
  non-empty ``poses2d`` observation must both hold.
* EMDB validity is exactly ``good_frames_mask``.

No benchmark data or SMPL model is bundled by Motius. Callers must provide
licensed annotations and a body-model implementation.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from motius.evaluation.monocular_capture import MonocularCaptureSample
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
    MonocularCaptureResult,
    MonocularTrack,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES


GROUND_TRUTH_REVISION = "motius_monocular_ground_truth_v1"
OFFICIAL_SOURCE_REVISIONS = {
    "3dpw_eval": "2640f244898d5503a8e3ce9825da5af3c77edb33",
    "emdb": "9a4eab677181a3789bda7ba5c36ab8cff797380c",
    "gvhmr": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
}
OFFICIAL_SOURCE_URLS = {
    "3dpw_eval": "https://github.com/miraymen/3dpw-eval",
    "emdb": "https://github.com/eth-ait/emdb",
    "gvhmr": "https://github.com/zju3dv/GVHMR",
}


class GroundTruthAnnotationError(ValueError):
    """An official annotation is missing a required field or shape."""


@dataclass(frozen=True)
class SMPLGeometry:
    """SMPL geometry in the annotation's world coordinate system."""

    vertices: np.ndarray
    joints: np.ndarray


class SMPLBodyModel(Protocol):
    """Minimal licensed-SMPL adapter used by the shared materializer."""

    model_version: str

    def materialize(
        self,
        *,
        poses_axis_angle: np.ndarray,
        betas: np.ndarray,
        translation: np.ndarray,
        gender: str,
    ) -> SMPLGeometry:
        """Return 6,890 vertices and 24 kinematic SMPL joints."""

    def fingerprint_for_gender(self, gender: str) -> str:
        """Return the SHA-256 of the exact licensed model file used."""


def sha256_file(path: Path) -> str:
    """Hash one user-supplied file without copying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_annotation(path: Path) -> dict:
    """Load a user-supplied official pickle, including Python-2 3DPW files."""

    with Path(path).open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    if not isinstance(payload, dict):
        raise GroundTruthAnnotationError(
            f"Official annotation {Path(path).name!r} must contain a dictionary."
        )
    return payload


def _require_field(mapping: Mapping, key: str, owner: str) -> object:
    if key not in mapping:
        raise GroundTruthAnnotationError(
            f"Official {owner} annotation is missing required field {key!r}."
        )
    return mapping[key]


def _require_mapping(mapping: Mapping, key: str, owner: str) -> Mapping:
    value = _require_field(mapping, key, owner)
    if not isinstance(value, Mapping):
        raise GroundTruthAnnotationError(
            f"Official {owner} field {key!r} must be a mapping."
        )
    return value


def _finite_array(
    value: object,
    *,
    shape: tuple[int, ...],
    field: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise GroundTruthAnnotationError(
            f"Official field {field!r} must have shape {shape}, got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise GroundTruthAnnotationError(
            f"Official field {field!r} must contain finite numeric values."
        )
    return array


def _rigid_world_to_camera(
    value: object,
    *,
    frames: int,
    field: str,
) -> np.ndarray:
    transforms = _finite_array(
        value,
        shape=(frames, 4, 4),
        field=field,
    ).astype(np.float64, copy=False)
    expected_bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        (frames, 4),
    )
    if not np.allclose(transforms[:, 3], expected_bottom, atol=1e-5):
        raise GroundTruthAnnotationError(
            f"Official field {field!r} must contain homogeneous transforms."
        )
    rotations = transforms[:, :3, :3]
    identities = np.einsum("fji,fjk->fik", rotations, rotations)
    if not np.allclose(identities, np.eye(3), atol=1e-3):
        raise GroundTruthAnnotationError(
            f"Official field {field!r} must contain rigid rotations."
        )
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3):
        raise GroundTruthAnnotationError(
            f"Official field {field!r} must contain proper rotations."
        )
    return transforms


def apply_world_to_camera(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    """Apply explicitly declared column-vector world-to-camera transforms."""

    points = np.asarray(points_world)
    transforms = np.asarray(world_to_camera)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points_world must have shape (frames, points, 3).")
    if transforms.shape != (points.shape[0], 4, 4):
        raise ValueError("world_to_camera must align to the point frame axis.")
    camera = (
        np.einsum("fij,fpj->fpi", transforms[:, :3, :3], points)
        + transforms[:, None, :3, 3]
    )
    return camera.astype(np.float32, copy=False)


def _normalize_gender(value: object, *, owner: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).lower()
    aliases = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
        "n": "neutral",
        "neutral": "neutral",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise GroundTruthAnnotationError(
            f"Official {owner} gender {value!r} is not male, female, or neutral."
        ) from exc


def _sequence_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _frame_slice(sample: MonocularCaptureSample, frames: int) -> slice:
    if sample.end_frame > frames:
        raise GroundTruthAnnotationError(
            f"Manifest interval [{sample.start_frame}, {sample.end_frame}) "
            f"exceeds the official annotation length {frames}."
        )
    return slice(sample.start_frame, sample.end_frame)


def _validated_geometry(
    body_model: SMPLBodyModel,
    *,
    poses: np.ndarray,
    betas: np.ndarray,
    translation: np.ndarray,
    gender: str,
) -> SMPLGeometry:
    geometry = body_model.materialize(
        poses_axis_angle=poses,
        betas=betas,
        translation=translation,
        gender=gender,
    )
    frames = len(poses)
    vertices = _finite_array(
        geometry.vertices,
        shape=(frames, 6890, 3),
        field="body_model.vertices",
    ).astype(np.float32, copy=False)
    joints = _finite_array(
        geometry.joints,
        shape=(frames, 24, 3),
        field="body_model.joints",
    ).astype(np.float32, copy=False)
    return SMPLGeometry(vertices=vertices, joints=joints)


def _fingerprint_metadata(
    *,
    annotation_path: Path,
    model_fingerprint: str | None,
    model_version: str | None,
    protocol: str,
    coordinate_space: str,
) -> dict[str, object]:
    if model_fingerprint is not None and len(model_fingerprint) != 64:
        raise ValueError("SMPL model fingerprint must be a SHA-256 digest.")
    return {
        "ground_truth_revision": GROUND_TRUTH_REVISION,
        "annotation_sha256": sha256_file(annotation_path),
        "model_sha256": model_fingerprint,
        "model_version": model_version,
        "protocol": protocol,
        "coordinate_space": coordinate_space,
        "official_source_revisions": dict(OFFICIAL_SOURCE_REVISIONS),
        "official_source_urls": dict(OFFICIAL_SOURCE_URLS),
    }


def _camera_to_world(world_to_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(world_to_camera).astype(np.float32, copy=False)


def materialize_3dpw_ground_truth(
    sample: MonocularCaptureSample,
    annotation_path: Path,
    body_model: SMPLBodyModel | None,
) -> MonocularCaptureResult:
    """Materialize one official 3DPW test person track in camera space."""

    if (
        sample.dataset.lower() != "3dpw"
        or sample.split != "test"
        or sample.protocol != "3dpw_test_camera_v1"
    ):
        raise ValueError("3DPW GT requires a 3DPW test camera-protocol sample.")
    payload = load_official_annotation(annotation_path)
    poses_by_person = _require_field(payload, "poses", "3DPW")
    betas_by_person = _require_field(payload, "betas", "3DPW")
    trans_by_person = _require_field(payload, "trans", "3DPW")
    genders = _require_field(payload, "genders", "3DPW")
    campose_valid = _require_field(payload, "campose_valid", "3DPW")
    poses2d = _require_field(payload, "poses2d", "3DPW")
    joint_positions = _require_field(payload, "jointPositions", "3DPW")
    cam_poses_raw = _require_field(payload, "cam_poses", "3DPW")
    intrinsics_raw = _require_field(payload, "cam_intrinsics", "3DPW")
    sequence = _sequence_name(_require_field(payload, "sequence", "3DPW"))
    if sequence != sample.sequence_id:
        raise GroundTruthAnnotationError(
            f"3DPW sequence {sequence!r} does not match manifest "
            f"{sample.sequence_id!r}."
        )

    person_index = sample.metadata.get("person_index")
    if not isinstance(person_index, (int, np.integer)) or isinstance(
        person_index, (bool, np.bool_)
    ):
        raise GroundTruthAnnotationError(
            "3DPW manifest metadata must contain an integer person_index."
        )
    person_index = int(person_index)
    person_fields = {
        "poses": poses_by_person,
        "betas": betas_by_person,
        "trans": trans_by_person,
        "genders": genders,
        "campose_valid": campose_valid,
        "poses2d": poses2d,
        "jointPositions": joint_positions,
    }
    for name, values in person_fields.items():
        if not isinstance(values, (list, tuple)) or person_index >= len(values):
            raise GroundTruthAnnotationError(
                f"Official 3DPW field {name!r} has no person {person_index}."
            )

    poses_all = np.asarray(poses_by_person[person_index])
    if poses_all.ndim != 2 or poses_all.shape[1] != 72:
        raise GroundTruthAnnotationError(
            "Official 3DPW poses must have shape (frames, 72)."
        )
    frames = poses_all.shape[0]
    annotation_slice = _frame_slice(sample, frames)
    poses = _finite_array(
        poses_all,
        shape=(frames, 72),
        field="poses[person_index]",
    )[annotation_slice].astype(np.float32, copy=False)
    translation = _finite_array(
        trans_by_person[person_index],
        shape=(frames, 3),
        field="trans[person_index]",
    )[annotation_slice].astype(np.float32, copy=False)
    betas_one = np.asarray(betas_by_person[person_index])
    if betas_one.ndim != 1 or betas_one.shape[0] < 10:
        raise GroundTruthAnnotationError(
            "Official 3DPW betas must be a vector with at least 10 values."
        )
    if not np.isfinite(betas_one).all():
        raise GroundTruthAnnotationError("Official 3DPW betas must be finite.")
    betas = np.repeat(
        betas_one[None, :10],
        len(poses),
        axis=0,
    ).astype(np.float32, copy=False)
    gender = _normalize_gender(genders[person_index], owner="3DPW")

    world_to_camera_all = _rigid_world_to_camera(
        cam_poses_raw,
        frames=frames,
        field="cam_poses",
    )
    world_to_camera = world_to_camera_all[annotation_slice]
    intrinsics = _finite_array(
        intrinsics_raw,
        shape=(3, 3),
        field="cam_intrinsics",
    ).astype(np.float32, copy=False)
    campose_mask = np.asarray(campose_valid[person_index])
    if (
        campose_mask.shape != (frames,)
        or (
            campose_mask.dtype != np.bool_
            and (
                not np.issubdtype(campose_mask.dtype, np.number)
                or not np.isfinite(campose_mask).all()
            )
        )
    ):
        raise GroundTruthAnnotationError(
            "Official 3DPW campose_valid must be a finite boolean-compatible "
            "vector per person."
        )
    campose_mask = campose_mask.astype(bool, copy=False)
    person_poses2d = np.asarray(poses2d[person_index])
    if person_poses2d.ndim < 2 or person_poses2d.shape[0] != frames:
        raise GroundTruthAnnotationError(
            "Official 3DPW poses2d must have the person frame axis first."
        )
    if not np.issubdtype(person_poses2d.dtype, np.number) or not np.isfinite(
        person_poses2d
    ).all():
        raise GroundTruthAnnotationError("Official 3DPW poses2d must be finite.")
    if person_poses2d.ndim != 3:
        raise GroundTruthAnnotationError(
            "Official 3DPW poses2d must have shape (frames, joints, 3) or "
            "(frames, 3, joints)."
        )
    if person_poses2d.shape[-1] == 3:
        poses2d_xyc = person_poses2d
    elif person_poses2d.shape[1] == 3:
        poses2d_xyc = np.transpose(person_poses2d, (0, 2, 1))
    else:
        raise GroundTruthAnnotationError(
            "Official 3DPW poses2d must expose x/y/confidence coordinates."
        )
    observed_2d = np.any(
        np.abs(person_poses2d.reshape(frames, -1)) > 0,
        axis=1,
    )
    valid = (campose_mask & observed_2d)[annotation_slice]

    if body_model is None:
        official_joints = np.asarray(joint_positions[person_index])
        if official_joints.shape == (frames, 72):
            official_joints = official_joints.reshape(frames, 24, 3)
        joints_world = _finite_array(
            official_joints,
            shape=(frames, 24, 3),
            field="jointPositions[person_index]",
        )[annotation_slice].astype(np.float32, copy=False)
        vertices_world = None
        model_fingerprint = None
        model_version = None
        geometry_source = "official_3dpw_jointPositions"
    else:
        geometry_world = _validated_geometry(
            body_model,
            poses=poses,
            betas=betas,
            translation=translation,
            gender=gender,
        )
        joints_world = geometry_world.joints
        vertices_world = geometry_world.vertices
        model_fingerprint = body_model.fingerprint_for_gender(gender)
        model_version = body_model.model_version
        geometry_source = "licensed_smpl_replay"
    joints_camera = apply_world_to_camera(
        joints_world,
        world_to_camera,
    )
    vertices_camera = (
        None
        if vertices_world is None
        else apply_world_to_camera(vertices_world, world_to_camera)
    )
    metadata = _fingerprint_metadata(
        annotation_path=annotation_path,
        model_fingerprint=model_fingerprint,
        model_version=model_version,
        protocol=sample.protocol,
        coordinate_space="camera_opencv",
    )
    metadata.update(
        {
            "dataset": "3DPW",
            "sequence_id": sample.sequence_id,
            "track_id": sample.track_id,
            "annotation_coordinate_space": "3dpw_sequence_world",
            "world_to_camera_field": "cam_poses",
            "world_to_camera_convention": "column_vector_T_w2c",
            "valid_mask_source": "campose_valid_and_nonzero_poses2d",
            "joint_protocol": "official_smpl24_kinematic",
            "evaluation_pelvis_indices": [0],
            "evaluation_pelvis_definition": "SMPL Pelvis joint 0",
            "evaluation_fps": float(sample.fps),
            "geometry_source": geometry_source,
            "mesh_metrics_available": vertices_camera is not None,
        }
    )
    frame_ids = np.arange(
        sample.start_frame,
        sample.end_frame,
        dtype=np.int64,
    )
    track = MonocularTrack(
        track_id=sample.track_id,
        frame_ids=frame_ids,
        valid=valid.astype(bool, copy=False),
        body_model="smpl",
        joint_names=SMPL24_NAMES,
        poses_axis_angle=poses.reshape(len(poses), 24, 3),
        shape_parameters=betas,
        joints_camera=joints_camera,
        vertices_camera=vertices_camera,
        root_translation_camera=joints_camera[:, 0],
        native_parameters={
            "smpl_trans_world": translation,
            "poses2d_xyc": poses2d_xyc[annotation_slice].astype(
                np.float32,
                copy=False,
            ),
            "cam_poses_world_to_camera": world_to_camera.astype(
                np.float32,
                copy=False,
            ),
        },
        availability={
            "camera_joints": f"{geometry_source}_transformed_by_cam_poses",
            "camera_vertices": (
                "licensed_smpl_replay_transformed_by_cam_poses"
                if vertices_camera is not None
                else "unavailable_without_licensed_smpl_replay"
            ),
            "world_geometry": "not_emitted_by_camera_protocol",
        },
        metadata={
            "evaluation_pelvis_indices": [0],
            "evaluation_pelvis_definition": "SMPL Pelvis joint 0",
            "evaluation_fps": float(sample.fps),
            "valid_mask_source": "campose_valid_and_nonzero_poses2d",
            "poses_axis_angle_space": "3dpw_sequence_world",
        },
    )
    return MonocularCaptureResult(
        source_model=(
            "3dpw-official-smpl-ground-truth"
            if body_model is not None
            else "3dpw-official-jointPositions-ground-truth"
        ),
        source_revision=GROUND_TRUTH_REVISION,
        checkpoint_sha256=model_fingerprint or metadata["annotation_sha256"],
        original_fps=float(sample.fps),
        output_fps=float(sample.fps),
        tracks=(track,),
        camera_coordinate_system=CAMERA_OPENCV,
        world_coordinate_system=GRAVITY_WORLD_Y_UP,
        camera_intrinsics=intrinsics,
        camera_to_world=_camera_to_world(world_to_camera),
        frame_timestamps=frame_ids.astype(np.float64) / sample.fps,
        metadata=metadata,
    )


def materialize_emdb_ground_truth(
    sample: MonocularCaptureSample,
    annotation_path: Path,
    body_model: SMPLBodyModel,
) -> MonocularCaptureResult:
    """Materialize one official EMDB-1 camera or EMDB-2 world target."""

    if sample.dataset.lower() != "emdb" or sample.protocol not in {
        "emdb_1_camera_v1",
        "emdb_2_global_v1",
    }:
        raise ValueError("EMDB GT requires an EMDB-1 or EMDB-2 sample.")
    payload = load_official_annotation(annotation_path)
    name = _sequence_name(_require_field(payload, "name", "EMDB"))
    if name != sample.sequence_id:
        raise GroundTruthAnnotationError(
            f"EMDB sequence {name!r} does not match manifest "
            f"{sample.sequence_id!r}."
        )
    frames_value = _require_field(payload, "n_frames", "EMDB")
    if not isinstance(frames_value, (int, np.integer)) or isinstance(
        frames_value, (bool, np.bool_)
    ):
        raise GroundTruthAnnotationError("Official EMDB n_frames must be an integer.")
    frames = int(frames_value)
    if frames <= 0:
        raise GroundTruthAnnotationError("Official EMDB n_frames must be positive.")
    annotation_slice = _frame_slice(sample, frames)
    protocol_flag = "emdb1" if sample.protocol == "emdb_1_camera_v1" else "emdb2"
    protocol_selected = _require_field(payload, protocol_flag, "EMDB")
    if not isinstance(protocol_selected, (bool, np.bool_)) or not bool(
        protocol_selected
    ):
        raise GroundTruthAnnotationError(
            f"Official EMDB field {protocol_flag!r} must select this sequence."
        )
    gender = _normalize_gender(
        _require_field(payload, "gender", "EMDB"),
        owner="EMDB",
    )
    good_frames = np.asarray(
        _require_field(payload, "good_frames_mask", "EMDB")
    )
    if good_frames.shape != (frames,) or good_frames.dtype != np.bool_:
        raise GroundTruthAnnotationError(
            "Official EMDB good_frames_mask must be a boolean n_frames vector."
        )

    camera = _require_mapping(payload, "camera", "EMDB")
    intrinsics = _finite_array(
        _require_field(camera, "intrinsics", "EMDB camera"),
        shape=(3, 3),
        field="camera.intrinsics",
    ).astype(np.float32, copy=False)
    world_to_camera_all = _rigid_world_to_camera(
        _require_field(camera, "extrinsics", "EMDB camera"),
        frames=frames,
        field="camera.extrinsics",
    )
    width = _require_field(camera, "width", "EMDB camera")
    height = _require_field(camera, "height", "EMDB camera")
    if not all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) > 0
        for value in (width, height)
    ):
        raise GroundTruthAnnotationError(
            "Official EMDB camera width and height must be positive integers."
        )

    smpl = _require_mapping(payload, "smpl", "EMDB")
    poses_root = _finite_array(
        _require_field(smpl, "poses_root", "EMDB smpl"),
        shape=(frames, 3),
        field="smpl.poses_root",
    )
    poses_body = _finite_array(
        _require_field(smpl, "poses_body", "EMDB smpl"),
        shape=(frames, 69),
        field="smpl.poses_body",
    )
    translation_all = _finite_array(
        _require_field(smpl, "trans", "EMDB smpl"),
        shape=(frames, 3),
        field="smpl.trans",
    )
    betas_one = _finite_array(
        _require_field(smpl, "betas", "EMDB smpl"),
        shape=(10,),
        field="smpl.betas",
    )
    poses_all = np.concatenate([poses_root, poses_body], axis=1)
    poses = poses_all[annotation_slice].astype(np.float32, copy=False)
    translation = translation_all[annotation_slice].astype(
        np.float32,
        copy=False,
    )
    betas = np.repeat(
        betas_one[None],
        len(poses),
        axis=0,
    ).astype(np.float32, copy=False)
    world_to_camera = world_to_camera_all[annotation_slice]
    valid = good_frames[annotation_slice]

    geometry_world = _validated_geometry(
        body_model,
        poses=poses,
        betas=betas,
        translation=translation,
        gender=gender,
    )
    is_camera_protocol = sample.protocol == "emdb_1_camera_v1"
    joints_camera = vertices_camera = None
    joints_world = vertices_world = None
    root_camera = root_world = None
    if is_camera_protocol:
        joints_camera = apply_world_to_camera(
            geometry_world.joints,
            world_to_camera,
        )
        vertices_camera = apply_world_to_camera(
            geometry_world.vertices,
            world_to_camera,
        )
        root_camera = joints_camera[:, 0]
        coordinate_space = "camera_opencv"
        availability = {
            "camera_joints": "official_smpl24_transformed_by_camera.extrinsics",
            "camera_vertices": "official_smpl_transformed_by_camera.extrinsics",
            "world_geometry": "not_emitted_by_emdb1_camera_protocol",
        }
    else:
        joints_world = geometry_world.joints
        vertices_world = geometry_world.vertices
        root_world = joints_world[:, 0]
        coordinate_space = "gravity_world_y_up"
        availability = {
            "world_joints": "official_emdb_smpl_world",
            "world_vertices": "official_emdb_smpl_world",
            "camera_geometry": "not_emitted_by_emdb2_global_protocol",
        }

    model_fingerprint = body_model.fingerprint_for_gender(gender)
    metadata = _fingerprint_metadata(
        annotation_path=annotation_path,
        model_fingerprint=model_fingerprint,
        model_version=body_model.model_version,
        protocol=sample.protocol,
        coordinate_space=coordinate_space,
    )
    metadata.update(
        {
            "dataset": "EMDB",
            "sequence_id": sample.sequence_id,
            "track_id": sample.track_id,
            "annotation_coordinate_space": "emdb_world",
            "world_to_camera_field": "camera.extrinsics",
            "world_to_camera_convention": "column_vector_T_w2c",
            "valid_mask_source": "good_frames_mask",
            "joint_protocol": "official_smpl24_kinematic",
            "evaluation_pelvis_indices": [1, 2],
            "evaluation_pelvis_definition": "mean of SMPL L_Hip and R_Hip",
            "evaluation_fps": float(sample.fps),
            "image_size": [int(width), int(height)],
        }
    )
    frame_ids = np.arange(
        sample.start_frame,
        sample.end_frame,
        dtype=np.int64,
    )
    track = MonocularTrack(
        track_id=sample.track_id,
        frame_ids=frame_ids,
        valid=valid.astype(bool, copy=False),
        body_model="smpl",
        joint_names=SMPL24_NAMES,
        poses_axis_angle=poses.reshape(len(poses), 24, 3),
        shape_parameters=betas,
        joints_camera=joints_camera,
        joints_world=joints_world,
        vertices_camera=vertices_camera,
        vertices_world=vertices_world,
        root_translation_camera=root_camera,
        root_translation_world=root_world,
        native_parameters={
            "smpl_trans_world": translation,
            "camera_extrinsics_world_to_camera": world_to_camera.astype(
                np.float32,
                copy=False,
            ),
        },
        availability=availability,
        metadata={
            "evaluation_pelvis_indices": [1, 2],
            "evaluation_pelvis_definition": "mean of SMPL L_Hip and R_Hip",
            "evaluation_fps": float(sample.fps),
            "valid_mask_source": "good_frames_mask",
            "poses_axis_angle_space": "emdb_world",
        },
    )
    return MonocularCaptureResult(
        source_model="emdb-official-smpl-ground-truth",
        source_revision=GROUND_TRUTH_REVISION,
        checkpoint_sha256=model_fingerprint,
        original_fps=float(sample.fps),
        output_fps=float(sample.fps),
        tracks=(track,),
        camera_coordinate_system=CAMERA_OPENCV,
        world_coordinate_system=GRAVITY_WORLD_Y_UP,
        camera_intrinsics=intrinsics,
        camera_to_world=_camera_to_world(world_to_camera),
        frame_timestamps=frame_ids.astype(np.float64) / sample.fps,
        metadata=metadata,
    )


def materialize_monocular_ground_truth(
    sample: MonocularCaptureSample,
    annotation_path: Path,
    body_model: SMPLBodyModel | None,
) -> MonocularCaptureResult:
    """Dispatch a manifest sample using its explicit protocol identifier."""

    if sample.protocol == "3dpw_test_camera_v1":
        return materialize_3dpw_ground_truth(
            sample,
            annotation_path,
            body_model,
        )
    if sample.protocol in {"emdb_1_camera_v1", "emdb_2_global_v1"}:
        if body_model is None:
            raise ValueError("EMDB GT materialization requires licensed SMPL.")
        return materialize_emdb_ground_truth(
            sample,
            annotation_path,
            body_model,
        )
    raise ValueError(f"Unsupported GT materialization protocol {sample.protocol!r}.")


__all__ = [
    "GROUND_TRUTH_REVISION",
    "OFFICIAL_SOURCE_REVISIONS",
    "OFFICIAL_SOURCE_URLS",
    "GroundTruthAnnotationError",
    "SMPLBodyModel",
    "SMPLGeometry",
    "apply_world_to_camera",
    "load_official_annotation",
    "materialize_3dpw_ground_truth",
    "materialize_emdb_ground_truth",
    "materialize_monocular_ground_truth",
    "sha256_file",
]
