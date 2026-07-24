import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from motius.evaluation.evaluators.monocular_capture import (
    evaluate_camera_coordinates,
    evaluate_common_joint_coordinates,
    evaluate_global_coordinates,
)
from motius.evaluation.monocular_capture import (
    MonocularCaptureSample,
    build_3dpw_test_samples,
    build_emdb_samples,
    load_monocular_capture_manifest,
    write_monocular_capture_manifest,
)
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
    CoordinateSystem,
    MonocularCaptureResult,
    MonocularTrack,
    load_monocular_capture_result,
    save_monocular_capture_result,
)
from motius.motion.representation.monocular_joints import (
    COMMON_HMR15_NAMES,
    SMPL24_NAMES,
    SOMA77_NAMES,
    select_common_hmr15,
)
from tools.eval_monocular_capture_results import evaluate_results
from tools.eval_3dpw_monocular_predictions import _associate


def _joints(frames: int = 6, joints: int = 5) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(frames, joints, 3)).astype(np.float32) * 0.1
    values[:, :, 2] += 2.0
    values[:, 0, 0] += np.linspace(0.0, 1.0, frames)
    return values


def test_monocular_result_contract_is_explicit_and_publishable():
    joints = _joints()
    track = MonocularTrack(
        track_id="person_0",
        frame_ids=np.arange(len(joints)),
        valid=np.ones(len(joints), dtype=bool),
        body_model="smpl",
        joint_names=tuple(f"joint_{index}" for index in range(joints.shape[1])),
        joints_camera=joints,
        joints_world=joints,
        root_translation_world=joints[:, 0],
        availability={"camera_joints": "native", "world_joints": "native"},
    )
    result = MonocularCaptureResult(
        source_model="test-model",
        source_revision="0123456789abcdef",
        checkpoint_sha256="a" * 64,
        original_fps=29.97,
        output_fps=30.0,
        tracks=(track,),
        world_coordinate_system=GRAVITY_WORLD_Y_UP,
        camera_to_world=np.repeat(np.eye(4)[None], len(joints), axis=0),
        metadata={
            "runtime_path": Path("/private/runtime"),
            "source_url": "https://github.com/example/model",
        },
    )

    manifest = result.public_manifest()

    assert result.camera_coordinate_system == CAMERA_OPENCV
    assert manifest["tracks"][0]["frames"] == len(joints)
    assert manifest["tracks"][0]["coverage"] == 1.0
    assert "path" not in json.dumps(manifest).lower()
    assert manifest["metadata"]["source_url"].startswith("https://")


def test_monocular_result_npz_roundtrip_is_pickle_free(tmp_path: Path):
    joints = _joints()
    result = MonocularCaptureResult(
        source_model="test-model",
        source_revision="revision",
        checkpoint_sha256="b" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            MonocularTrack(
                track_id="person_0",
                frame_ids=np.arange(len(joints)),
                valid=np.ones(len(joints), dtype=bool),
                body_model="smpl",
                joint_names=tuple(
                    f"joint_{index}" for index in range(joints.shape[1])
                ),
                joints_camera=joints,
                native_parameters={"betas": np.zeros(10, dtype=np.float32)},
            ),
        ),
    )
    path = tmp_path / "capture.npz"

    save_monocular_capture_result(result, path)
    restored = load_monocular_capture_result(path)

    assert restored.public_manifest() == result.public_manifest()
    np.testing.assert_array_equal(
        restored.tracks[0].joints_camera,
        result.tracks[0].joints_camera,
    )
    np.testing.assert_array_equal(
        restored.tracks[0].native_parameters["betas"],
        result.tracks[0].native_parameters["betas"],
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata_json"].dtype.kind == "U"


def test_monocular_result_can_skip_mesh_arrays_for_joint_only_evaluation(
    tmp_path: Path,
):
    joints = _joints()
    vertices = np.repeat(joints[:, :1], 64, axis=1)
    result = MonocularCaptureResult(
        source_model="test-model",
        source_revision="revision",
        checkpoint_sha256="b" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            MonocularTrack(
                track_id="person_0",
                frame_ids=np.arange(len(joints)),
                valid=np.ones(len(joints), dtype=bool),
                body_model="smpl",
                joint_names=tuple(
                    f"joint_{index}" for index in range(joints.shape[1])
                ),
                joints_camera=joints,
                vertices_camera=vertices,
                vertices_world=vertices,
            ),
        ),
    )
    path = tmp_path / "capture.npz"
    save_monocular_capture_result(result, path)

    restored = load_monocular_capture_result(path, include_vertices=False)

    np.testing.assert_array_equal(restored.tracks[0].joints_camera, joints)
    assert restored.tracks[0].vertices_camera is None
    assert restored.tracks[0].vertices_world is None


def test_monocular_contract_rejects_implicit_or_misaligned_coordinates():
    with pytest.raises(ValueError, match="different dimensions"):
        CoordinateSystem(
            name="bad",
            up_axis="+Y",
            forward_axis="-Y",
        )
    with pytest.raises(ValueError, match="valid"):
        MonocularTrack(
            track_id="person_0",
            frame_ids=np.arange(3),
            valid=np.ones(2, dtype=bool),
            body_model="smpl",
        )
    track = MonocularTrack(
        track_id="person_0",
        frame_ids=np.arange(3),
        valid=np.ones(3, dtype=bool),
        body_model="smpl",
    )
    with pytest.raises(ValueError, match="world coordinate"):
        MonocularCaptureResult(
            source_model="model",
            source_revision="revision",
            checkpoint_sha256="a" * 64,
            original_fps=30.0,
            output_fps=30.0,
            tracks=(track,),
            camera_to_world=np.repeat(np.eye(4)[None], 3, axis=0),
        )


def test_camera_metrics_match_similarity_and_identity_invariants():
    target = _joints()
    translated_scaled = target * 1.7 + np.asarray([0.8, -0.3, 1.1])

    result = evaluate_camera_coordinates(
        translated_scaled,
        target,
        pelvis_indices=(1, 2),
        fps=30.0,
    )
    identity = evaluate_camera_coordinates(
        target,
        target,
        pelvis_indices=(1, 2),
        fps=30.0,
    )

    assert result.means["pa_mpjpe_mm"] < 1e-4
    assert result.means["mpjpe_mm"] > 1.0
    assert all(value < 1e-8 for value in identity.means.values())


def test_global_metrics_follow_gvhmr_alignment_invariants():
    target = _joints(frames=120)
    angle = np.deg2rad(30)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    prediction = 1.2 * np.einsum("ij,tkj->tki", rotation, target)
    prediction += np.asarray([0.5, -0.2, 0.7])

    result = evaluate_global_coordinates(prediction, target)

    assert result.means["w_mpjpe_mm"] < 1e-4
    assert result.means["wa_mpjpe_mm"] < 1e-4
    assert result.protocol["chunk_frames"] == 100


def test_manifest_roundtrip_keeps_paths_relative(tmp_path: Path):
    root = tmp_path / "private"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "annotations" / "sample.pkl").write_bytes(b"annotation")
    sample = MonocularCaptureSample(
        dataset="3DPW",
        split="test",
        protocol="3dpw_test_camera_v1",
        sequence_id="courtyard",
        track_id="person_0",
        input_path=Path("images"),
        input_kind="image_sequence",
        annotation_path=Path("annotations/sample.pkl"),
        fps=30.0,
        start_frame=0,
        end_frame=8,
    )
    path = tmp_path / "manifest.json"
    write_monocular_capture_manifest(
        [sample],
        path,
        dataset_license="test license",
        source="official",
    )

    loaded = load_monocular_capture_manifest(
        path,
        data_root=root,
        require_files=True,
    )

    assert loaded == (sample,)
    assert str(root) not in path.read_text()
    with pytest.raises(ValueError, match="safe path"):
        MonocularCaptureSample(
            dataset="3DPW",
            split="test",
            protocol="3dpw_test_camera_v1",
            sequence_id="bad",
            track_id="person_0",
            input_path=root,
            input_kind="image_sequence",
            annotation_path=Path("annotation.pkl"),
            fps=30.0,
            start_frame=0,
            end_frame=1,
        )


def test_synthetic_3dpw_and_emdb_indexes(tmp_path: Path):
    threedpw = tmp_path / "3DPW"
    (threedpw / "sequenceFiles" / "test").mkdir(parents=True)
    (threedpw / "imageFiles" / "sequence_a").mkdir(parents=True)
    with (threedpw / "sequenceFiles" / "test" / "sequence_a.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sequence": "sequence_a",
                "poses": [np.zeros((4, 72)), np.zeros((3, 72))],
                "campose_valid": [np.ones(4, dtype=bool), np.ones(3, dtype=bool)],
            },
            handle,
        )

    emdb = tmp_path / "EMDB"
    sequence = emdb / "P0" / "example"
    (sequence / "images").mkdir(parents=True)
    with (sequence / "P0_example_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "name": "P0_example",
                "n_frames": 5,
                "good_frames_mask": np.ones(5, dtype=bool),
                "gender": "neutral",
                "emdb1": True,
                "emdb2": True,
            },
            handle,
        )

    threedpw_samples = build_3dpw_test_samples(threedpw)
    emdb_samples = build_emdb_samples(emdb)

    assert len(threedpw_samples) == 2
    assert threedpw_samples[1].track_id == "person_1"
    assert {sample.protocol for sample in emdb_samples} == {
        "emdb_1_camera_v1",
        "emdb_2_global_v1",
    }


def test_named_common_joint_mapping_keeps_soma_native():
    smpl = np.arange(2 * 24 * 3, dtype=np.float32).reshape(2, 24, 3)
    soma = np.arange(2 * 77 * 3, dtype=np.float32).reshape(2, 77, 3)

    smpl_common = select_common_hmr15(
        smpl,
        SMPL24_NAMES,
        body_model="smpl",
    )
    soma_common = select_common_hmr15(
        soma,
        SOMA77_NAMES,
        body_model="soma77",
    )

    assert len(SOMA77_NAMES) == 77
    assert smpl_common.shape == soma_common.shape == (
        2,
        len(COMMON_HMR15_NAMES),
        3,
    )
    assert np.array_equal(soma_common[:, 0], soma[:, SOMA77_NAMES.index("Hips")])
    with pytest.raises(ValueError, match="Missing required named joints"):
        select_common_hmr15(soma[:, :20], SOMA77_NAMES[:20], body_model="soma77")


def test_soma_to_smpl_comparison_is_joint_only_and_name_audited():
    frames = 4
    smpl = np.zeros((frames, len(SMPL24_NAMES), 3), dtype=np.float32)
    soma = np.zeros((frames, len(SOMA77_NAMES), 3), dtype=np.float32)
    common = np.arange(
        frames * len(COMMON_HMR15_NAMES) * 3,
        dtype=np.float32,
    ).reshape(frames, len(COMMON_HMR15_NAMES), 3)
    for common_index, name in enumerate(COMMON_HMR15_NAMES):
        smpl_name = {
            "pelvis": "Pelvis",
            "left_hip": "L_Hip",
            "right_hip": "R_Hip",
            "left_knee": "L_Knee",
            "right_knee": "R_Knee",
            "left_ankle": "L_Ankle",
            "right_ankle": "R_Ankle",
            "neck": "Neck",
            "head": "Head",
            "left_shoulder": "L_Shoulder",
            "right_shoulder": "R_Shoulder",
            "left_elbow": "L_Elbow",
            "right_elbow": "R_Elbow",
            "left_wrist": "L_Wrist",
            "right_wrist": "R_Wrist",
        }[name]
        soma_name = {
            "pelvis": "Hips",
            "left_hip": "LeftLeg",
            "right_hip": "RightLeg",
            "left_knee": "LeftShin",
            "right_knee": "RightShin",
            "left_ankle": "LeftFoot",
            "right_ankle": "RightFoot",
            "neck": "Neck1",
            "head": "Head",
            "left_shoulder": "LeftArm",
            "right_shoulder": "RightArm",
            "left_elbow": "LeftForeArm",
            "right_elbow": "RightForeArm",
            "left_wrist": "LeftHand",
            "right_wrist": "RightHand",
        }[name]
        smpl[:, SMPL24_NAMES.index(smpl_name)] = common[:, common_index]
        soma[:, SOMA77_NAMES.index(soma_name)] = common[:, common_index]

    result = evaluate_common_joint_coordinates(
        soma,
        SOMA77_NAMES,
        "soma77",
        smpl,
        SMPL24_NAMES,
        "smpl",
        space="camera",
    )

    assert result.means["mpjpe_mm"] == pytest.approx(0.0)
    assert result.protocol["joint_protocol"] == "common_hmr15_named_v1"
    assert result.protocol["mesh_metrics_available"] is False


def test_canonical_result_evaluator_enforces_tracks_and_coverage():
    joints = _joints(frames=6, joints=len(SMPL24_NAMES))
    valid = np.asarray([True, True, False, True, True, True])

    def result(source: str, frame_ids: np.ndarray) -> MonocularCaptureResult:
        selected = np.searchsorted(np.arange(len(joints)), frame_ids)
        return MonocularCaptureResult(
            source_model=source,
            source_revision="revision",
            checkpoint_sha256=("a" if source == "prediction" else "b") * 64,
            original_fps=30.0,
            output_fps=30.0,
            tracks=(
                MonocularTrack(
                    track_id="person_0",
                    frame_ids=frame_ids,
                    valid=valid[selected],
                    body_model="smpl",
                    joint_names=SMPL24_NAMES,
                    joints_camera=joints[selected],
                    metadata={"evaluation_pelvis_indices": [1, 2]},
                ),
            ),
        )

    target = result("target", np.arange(6))
    complete = result("prediction", np.arange(6))
    incomplete = result("prediction", np.arange(5))

    metrics = evaluate_results(
        complete,
        target,
        protocol="3dpw_test_camera_v1",
        min_coverage=1.0,
    )

    assert metrics["coverage_percent"] == pytest.approx(100.0)
    assert metrics["metrics"]["pa_mpjpe_mm"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="coverage"):
        evaluate_results(
            incomplete,
            target,
            protocol="3dpw_test_camera_v1",
            min_coverage=1.0,
        )

    second_target = replace(
        target.tracks[0],
        track_id="person_1",
    )
    multi_target = replace(
        target,
        tracks=(target.tracks[0], second_target),
    )
    partial = evaluate_results(
        complete,
        multi_target,
        protocol="3dpw_test_camera_v1",
        min_coverage=0.0,
        require_all_tracks=False,
    )
    assert partial["complete_track_coverage"] is False
    assert partial["coverage_percent"] == pytest.approx(50.0)
    assert partial["tracks"][1]["status"] == "missing_prediction"


def test_3dpw_track_association_uses_mean_bbox_iou():
    frames = np.arange(3)

    def target_track(track_id: str, x: float) -> MonocularTrack:
        poses2d = np.zeros((3, 18, 3), dtype=np.float32)
        poses2d[:, 0] = [x, 10.0, 1.0]
        poses2d[:, 1] = [x + 20.0, 40.0, 1.0]
        return MonocularTrack(
            track_id=track_id,
            frame_ids=frames,
            valid=np.ones(3, dtype=bool),
            body_model="smpl",
            native_parameters={"poses2d_xyc": poses2d},
        )

    def prediction_track(track_id: str, x: float) -> MonocularTrack:
        boxes = np.tile([x, 10.0, x + 20.0, 40.0], (3, 1))
        return MonocularTrack(
            track_id=track_id,
            frame_ids=frames,
            valid=np.ones(3, dtype=bool),
            body_model="smpl",
            native_parameters={"tracking_bboxes": boxes},
        )

    target = MonocularCaptureResult(
        source_model="target",
        source_revision="revision",
        checkpoint_sha256="a" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            target_track("person_0", 0.0),
            target_track("person_1", 100.0),
        ),
    )
    prediction = MonocularCaptureResult(
        source_model="prediction",
        source_revision="revision",
        checkpoint_sha256="b" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            prediction_track("track_right", 100.0),
            prediction_track("track_left", 0.0),
        ),
    )

    associated, report = _associate(prediction, target)

    assert [track.track_id for track in associated.tracks] == [
        "person_1",
        "person_0",
    ]
    assert report["mode"] == "hungarian_mean_2d_bbox_iou"
    assert not report["unmatched_target_tracks"]


def test_prompthmr_association_maps_resized_video_bboxes_with_intrinsics():
    frames = np.arange(3)

    def target_track(track_id: str, x: float) -> MonocularTrack:
        poses2d = np.zeros((3, 18, 3), dtype=np.float32)
        poses2d[:, 0] = [x, 20.0, 1.0]
        poses2d[:, 1] = [x + 40.0, 80.0, 1.0]
        return MonocularTrack(
            track_id=track_id,
            frame_ids=frames,
            valid=np.ones(3, dtype=bool),
            body_model="smpl",
            native_parameters={"poses2d_xyc": poses2d},
        )

    def prediction_track(track_id: str, x: float) -> MonocularTrack:
        boxes = np.tile([x / 2, 10.0, (x + 40.0) / 2, 40.0], (3, 1))
        return MonocularTrack(
            track_id=track_id,
            frame_ids=frames,
            valid=np.ones(3, dtype=bool),
            body_model="SMPL-X neutral",
            native_parameters={"tracking_bboxes": boxes},
        )

    target_k = np.array(
        [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
    )
    prediction_k = target_k.copy()
    prediction_k[:2] /= 2.0
    prediction_k[2, 2] = 1.0
    target = MonocularCaptureResult(
        source_model="target",
        source_revision="revision",
        checkpoint_sha256="a" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            target_track("person_0", 0.0),
            target_track("person_1", 100.0),
        ),
        camera_intrinsics=target_k,
    )
    prediction = MonocularCaptureResult(
        source_model="PromptHMR-Video",
        source_revision="revision",
        checkpoint_sha256="b" * 64,
        original_fps=30.0,
        output_fps=30.0,
        tracks=(
            prediction_track("track_right", 100.0),
            prediction_track("track_left", 0.0),
        ),
        camera_intrinsics=prediction_k,
    )

    associated, report = _associate(prediction, target)

    assert [track.track_id for track in associated.tracks] == [
        "person_1",
        "person_0",
    ]
    assert report["prediction_bbox_scale_to_target"] == [2.0, 2.0]
    assert [item["score"] for item in report["assignments"]] == [1.0, 1.0]
