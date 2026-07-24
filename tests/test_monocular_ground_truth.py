from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from motius.evaluation.monocular_capture import MonocularCaptureSample
from motius.evaluation.monocular_ground_truth import (
    OFFICIAL_SOURCE_REVISIONS,
    GroundTruthAnnotationError,
    SMPLGeometry,
    materialize_3dpw_ground_truth,
    materialize_emdb_ground_truth,
    sha256_file,
)
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
    load_monocular_capture_result,
    save_monocular_capture_result,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES
from tools.materialize_monocular_capture_ground_truth import (
    ROOT,
    require_repository_output_path,
)


class FakeSMPLBodyModel:
    model_version = "fake-smpl-for-tests"

    def __init__(self, fingerprint: str = "f" * 64) -> None:
        self.fingerprint = fingerprint
        self.calls = []

    def fingerprint_for_gender(self, gender: str) -> str:
        return self.fingerprint

    def materialize(
        self,
        *,
        poses_axis_angle: np.ndarray,
        betas: np.ndarray,
        translation: np.ndarray,
        gender: str,
    ) -> SMPLGeometry:
        self.calls.append(
            {
                "poses": poses_axis_angle.copy(),
                "betas": betas.copy(),
                "translation": translation.copy(),
                "gender": gender,
            }
        )
        joint_offsets = np.zeros((24, 3), dtype=np.float32)
        joint_offsets[:, 0] = np.arange(24, dtype=np.float32) / 10.0
        vertex_offsets = np.zeros((6890, 3), dtype=np.float32)
        vertex_offsets[:, 1] = np.arange(6890, dtype=np.float32) / 10000.0
        return SMPLGeometry(
            vertices=translation[:, None, :] + vertex_offsets[None],
            joints=translation[:, None, :] + joint_offsets[None],
        )


def _write_pickle(path: Path, payload: dict) -> Path:
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _world_to_camera(frames: int) -> np.ndarray:
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    transforms[:, :3, :3] = rotation
    transforms[:, :3, 3] = np.asarray([10.0, 20.0, 30.0])
    return transforms


def _transform(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return (
        np.einsum("fij,fpj->fpi", transforms[:, :3, :3], points)
        + transforms[:, None, :3, 3]
    )


def _sample(
    *,
    dataset: str,
    protocol: str,
    sequence_id: str,
    frames: int,
    metadata: dict | None = None,
) -> MonocularCaptureSample:
    return MonocularCaptureSample(
        dataset=dataset,
        split="test",
        protocol=protocol,
        sequence_id=sequence_id,
        track_id="person_0",
        input_path=Path("images"),
        input_kind="image_sequence",
        annotation_path=Path("annotation.pkl"),
        fps=30.0,
        start_frame=0,
        end_frame=frames,
        metadata=metadata or {},
    )


def _threedpw_payload(frames: int = 4) -> dict:
    poses2d = np.ones((frames, 18, 3), dtype=np.float32)
    poses2d[2] = 0.0
    joint_positions = np.arange(
        frames * 24 * 3,
        dtype=np.float32,
    ).reshape(frames, 24, 3) / 100.0
    return {
        "sequence": "courtyard",
        "poses": [np.zeros((frames, 72), dtype=np.float32)],
        "betas": [np.arange(10, dtype=np.float32)],
        "trans": [
            np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
        ],
        "genders": ["m"],
        "campose_valid": [
            np.asarray([True, False, True, True], dtype=bool)
        ],
        "poses2d": [poses2d],
        "jointPositions": [joint_positions.reshape(frames, 72)],
        "cam_poses": _world_to_camera(frames),
        "cam_intrinsics": np.asarray(
            [[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    }


def _emdb_payload(frames: int = 4) -> dict:
    return {
        "name": "P0_example",
        "n_frames": frames,
        "gender": "female",
        "emdb1": True,
        "emdb2": True,
        "good_frames_mask": np.asarray(
            [True, False, True, True],
            dtype=bool,
        ),
        "camera": {
            "intrinsics": np.asarray(
                [
                    [900.0, 0.0, 720.0],
                    [0.0, 900.0, 960.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "extrinsics": _world_to_camera(frames),
            "width": 1440,
            "height": 1920,
        },
        "smpl": {
            "poses_root": np.zeros((frames, 3), dtype=np.float32),
            "poses_body": np.zeros((frames, 69), dtype=np.float32),
            "trans": np.arange(
                frames * 3,
                dtype=np.float32,
            ).reshape(frames, 3),
            "betas": np.arange(10, dtype=np.float32),
        },
    }


def test_3dpw_materialization_uses_official_camera_pose_and_validity(
    tmp_path: Path,
):
    annotation = _write_pickle(
        tmp_path / "courtyard.pkl",
        _threedpw_payload(),
    )
    body_model = FakeSMPLBodyModel()
    sample = _sample(
        dataset="3DPW",
        protocol="3dpw_test_camera_v1",
        sequence_id="courtyard",
        frames=4,
        metadata={"person_index": 0},
    )

    result = materialize_3dpw_ground_truth(
        sample,
        annotation,
        body_model,
    )
    track = result.tracks[0]
    world_joints = body_model.materialize(
        poses_axis_angle=np.zeros((4, 72), dtype=np.float32),
        betas=np.zeros((4, 10), dtype=np.float32),
        translation=_threedpw_payload()["trans"][0],
        gender="male",
    ).joints

    np.testing.assert_allclose(
        track.joints_camera,
        _transform(world_joints, _world_to_camera(4)),
    )
    np.testing.assert_array_equal(track.valid, [True, False, False, True])
    np.testing.assert_allclose(
        result.camera_to_world,
        np.linalg.inv(_world_to_camera(4)),
    )
    assert result.camera_coordinate_system == CAMERA_OPENCV
    assert result.world_coordinate_system == GRAVITY_WORLD_Y_UP
    assert track.joints_world is track.vertices_world is None
    assert track.joints_camera.shape == (4, 24, 3)
    assert track.vertices_camera.shape == (4, 6890, 3)
    assert track.joint_names == SMPL24_NAMES
    assert track.metadata["evaluation_pelvis_indices"] == [0]
    assert result.metadata["world_to_camera_field"] == "cam_poses"
    assert result.metadata["annotation_sha256"] == sha256_file(annotation)
    assert result.metadata["model_sha256"] == "f" * 64
    assert body_model.calls[0]["gender"] == "male"
    artifact = tmp_path / "canonical_ground_truth.npz"
    save_monocular_capture_result(result, artifact)
    restored = load_monocular_capture_result(artifact)
    assert restored.public_manifest() == result.public_manifest()
    np.testing.assert_allclose(
        restored.tracks[0].vertices_camera,
        result.tracks[0].vertices_camera,
    )


def test_3dpw_joint_only_materialization_uses_official_joint_positions(
    tmp_path: Path,
):
    payload = _threedpw_payload()
    annotation = _write_pickle(tmp_path / "courtyard.pkl", payload)
    sample = _sample(
        dataset="3DPW",
        protocol="3dpw_test_camera_v1",
        sequence_id="courtyard",
        frames=4,
        metadata={"person_index": 0},
    )

    result = materialize_3dpw_ground_truth(sample, annotation, None)
    expected_world = np.asarray(payload["jointPositions"][0]).reshape(4, 24, 3)
    expected_camera = _transform(expected_world, payload["cam_poses"])

    np.testing.assert_allclose(result.tracks[0].joints_camera, expected_camera)
    assert result.tracks[0].vertices_camera is None
    assert result.metadata["geometry_source"] == "official_3dpw_jointPositions"
    assert result.metadata["mesh_metrics_available"] is False
    assert result.metadata["model_sha256"] is None
    assert result.source_model == "3dpw-official-jointPositions-ground-truth"
    assert result.tracks[0].native_parameters["poses2d_xyc"].shape == (4, 18, 3)


@pytest.mark.parametrize(
    ("protocol", "expected_space"),
    [
        ("emdb_1_camera_v1", "camera_opencv"),
        ("emdb_2_global_v1", "gravity_world_y_up"),
    ],
)
def test_emdb_protocols_use_extrinsics_world_fields_and_good_frames(
    tmp_path: Path,
    protocol: str,
    expected_space: str,
):
    payload = _emdb_payload()
    annotation = _write_pickle(tmp_path / "P0_example_data.pkl", payload)
    body_model = FakeSMPLBodyModel()
    sample = _sample(
        dataset="EMDB",
        protocol=protocol,
        sequence_id="P0_example",
        frames=4,
    )

    result = materialize_emdb_ground_truth(
        sample,
        annotation,
        body_model,
    )
    track = result.tracks[0]
    geometry_world = body_model.materialize(
        poses_axis_angle=np.zeros((4, 72), dtype=np.float32),
        betas=np.zeros((4, 10), dtype=np.float32),
        translation=payload["smpl"]["trans"],
        gender="female",
    )

    np.testing.assert_array_equal(track.valid, payload["good_frames_mask"])
    assert result.metadata["coordinate_space"] == expected_space
    assert result.metadata["world_to_camera_field"] == "camera.extrinsics"
    assert result.metadata["valid_mask_source"] == "good_frames_mask"
    assert track.metadata["evaluation_pelvis_indices"] == [1, 2]
    assert body_model.calls[0]["poses"].shape == (4, 72)
    np.testing.assert_array_equal(
        body_model.calls[0]["poses"][:, :3],
        payload["smpl"]["poses_root"],
    )
    np.testing.assert_array_equal(
        body_model.calls[0]["poses"][:, 3:],
        payload["smpl"]["poses_body"],
    )
    if protocol == "emdb_1_camera_v1":
        np.testing.assert_allclose(
            track.joints_camera,
            _transform(geometry_world.joints, payload["camera"]["extrinsics"]),
        )
        assert track.joints_world is track.vertices_world is None
    else:
        np.testing.assert_allclose(track.joints_world, geometry_world.joints)
        np.testing.assert_allclose(track.vertices_world, geometry_world.vertices)
        assert track.joints_camera is track.vertices_camera is None


def test_official_annotation_fields_are_asserted_not_inferred(
    tmp_path: Path,
):
    body_model = FakeSMPLBodyModel()
    threedpw_sample = _sample(
        dataset="3DPW",
        protocol="3dpw_test_camera_v1",
        sequence_id="courtyard",
        frames=4,
        metadata={"person_index": 0},
    )
    threedpw = _threedpw_payload()
    del threedpw["cam_poses"]
    threedpw_path = _write_pickle(tmp_path / "3dpw.pkl", threedpw)
    with pytest.raises(GroundTruthAnnotationError, match="cam_poses"):
        materialize_3dpw_ground_truth(
            threedpw_sample,
            threedpw_path,
            body_model,
        )

    emdb_sample = _sample(
        dataset="EMDB",
        protocol="emdb_1_camera_v1",
        sequence_id="P0_example",
        frames=4,
    )
    emdb = _emdb_payload()
    del emdb["camera"]["extrinsics"]
    emdb_path = _write_pickle(tmp_path / "emdb.pkl", emdb)
    with pytest.raises(GroundTruthAnnotationError, match="extrinsics"):
        materialize_emdb_ground_truth(
            emdb_sample,
            emdb_path,
            body_model,
        )

    emdb = copy.deepcopy(_emdb_payload())
    del emdb["smpl"]["poses_root"]
    emdb_path = _write_pickle(tmp_path / "emdb_missing_smpl.pkl", emdb)
    with pytest.raises(GroundTruthAnnotationError, match="poses_root"):
        materialize_emdb_ground_truth(
            emdb_sample,
            emdb_path,
            body_model,
        )


def test_fake_body_model_must_emit_official_smpl_topology(tmp_path: Path):
    class WrongJointCount(FakeSMPLBodyModel):
        def materialize(self, **kwargs) -> SMPLGeometry:
            geometry = super().materialize(**kwargs)
            return SMPLGeometry(
                vertices=geometry.vertices,
                joints=geometry.joints[:, :23],
            )

    annotation = _write_pickle(
        tmp_path / "courtyard.pkl",
        _threedpw_payload(),
    )
    sample = _sample(
        dataset="3DPW",
        protocol="3dpw_test_camera_v1",
        sequence_id="courtyard",
        frames=4,
        metadata={"person_index": 0},
    )

    with pytest.raises(
        GroundTruthAnnotationError,
        match=r"body_model\.joints.*24",
    ):
        materialize_3dpw_ground_truth(
            sample,
            annotation,
            WrongJointCount(),
        )


def test_official_source_revisions_and_output_boundary_are_pinned(
    tmp_path: Path,
):
    assert OFFICIAL_SOURCE_REVISIONS == {
        "3dpw_eval": "2640f244898d5503a8e3ce9825da5af3c77edb33",
        "emdb": "9a4eab677181a3789bda7ba5c36ab8cff797380c",
        "gvhmr": "6ec3ca39336c50492c0fae65fba2fb831fc7d866",
    }
    with pytest.raises(ValueError, match="outputs"):
        require_repository_output_path(tmp_path / "ground_truth")
    allowed = ROOT / "outputs" / "evaluation" / "monocular_ground_truth"
    assert require_repository_output_path(allowed) == allowed.resolve()
