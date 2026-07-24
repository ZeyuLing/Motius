"""Tests for the isolated PromptHMR-Video method integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

from motius.models.prompthmr import (
    PROMPTHMR_REVISION,
    PROMPTHMR_VIDEO_CHECKPOINTS,
    PromptHMRBundle,
)
from motius.motion.representation.monocular_joints import select_common_hmr15
from motius.pipelines.prompthmr import (
    LicensedSMPLXProvenance,
    PromptHMRPipeline,
    SMPL_SMPLX_BODY22_NAMES,
    build_prompthmr_video_command,
    inspect_licensed_smplx_model,
    parse_prompthmr_results,
    replay_prompthmr_geometry,
    split_prompthmr_smplx_pose,
)


VIDEO_SHA256 = PROMPTHMR_VIDEO_CHECKPOINTS["bedlam1+2"].sha256


def _official_results(*, include_world: bool = True):
    frame_count = 4
    frames = np.array([1, 3], dtype=np.int64)
    person = {
        "track_id": 7,
        "frames": frames,
        "bboxes": np.array([[10, 20, 30, 60], [12, 21, 32, 61]], dtype=np.float32),
        "detected": np.array([True, False]),
        "smplx_cam": {
            "pose": np.zeros((2, 75), dtype=np.float32),
            "shape": np.ones((2, 10), dtype=np.float32),
            "trans": np.array([[0.1, 0.2, 2.0], [0.2, 0.3, 2.1]], dtype=np.float32),
            "rotmat": np.zeros((2, 55, 3, 3), dtype=np.float32),
            "contact": np.array(
                [[True, False, True, False, False, False]] * 2
            ),
            "static_conf_logits": np.zeros((2, 6), dtype=np.float32),
        },
    }
    results = {
        "people": {7: person},
        "camera": {
            "pred_cam_R": np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy(),
            "pred_cam_T": np.zeros((frame_count, 3), dtype=np.float32),
            "img_focal": np.array(1200.0),
            "img_center": np.array([320.0, 240.0]),
        },
        "has_tracks": True,
        "has_hps_cam": True,
        "has_hps_world": include_world,
        "has_slam": True,
        "has_2d_kpts": True,
        "has_post_opt": include_world,
    }
    if include_world:
        person["smplx_world"] = {
            "pose": np.zeros((2, 165), dtype=np.float32),
            "shape": np.ones((2, 10), dtype=np.float32),
            "trans": np.array(
                [[1.0, 0.0, -2.0], [1.1, 0.0, -2.1]], dtype=np.float32
            ),
        }
        results["camera_world"] = {
            "Rwc": np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy(),
            "Twc": np.arange(frame_count * 3, dtype=np.float32).reshape(
                frame_count, 3
            ),
            "Rcw": np.broadcast_to(np.eye(3), (frame_count, 3, 3)).copy(),
            "Tcw": np.zeros((frame_count, 3), dtype=np.float32),
            "img_focal": np.array(1200.0),
            "img_center": np.array([320.0, 240.0]),
        }
    return results


def test_prompthmr_bundle_imports_without_upstream_or_weights():
    bundle = PromptHMRBundle()
    assert PROMPTHMR_REVISION == "3b566b7dbb28ce506c7ea972c18693f4c705ce8c"
    assert bundle.video_checkpoint == "bedlam1+2"
    assert bundle.expected_checkpoint_sha256 == VIDEO_SHA256
    assert bundle.resolved_video_checkpoint_path() is None


def test_prompthmr_official_video_checkpoint_hashes_are_fixed():
    assert {
        key: value.sha256
        for key, value in PROMPTHMR_VIDEO_CHECKPOINTS.items()
    } == {
        "bedlam1": "d06ae5ddc74ef74c252f4ec34e4e3092cd8fc18cba104af5aa978cdd2c669b5a",
        "bedlam1+2": "2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5",
        "bedlam2": "631433bf4dfd548dc5c6e2df037e11a11ce4a83c37367ee0f31b2f1627aa06d9",
    }


def test_prompthmr_runtime_patch_is_exact_and_reversible(tmp_path: Path):
    runtime = tmp_path / "upstream"
    wrapper = runtime / "pipeline/utils_detectron2.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        '            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))\n'
        "            image.to(self.cfg.MODEL.DEVICE)\n"
        "\n"
        '            inputs = {"image": image, "height": height, "width": width}\n'
    )
    patcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "patch_prompthmr_runtime.py"
    )

    subprocess.run(
        [sys.executable, str(patcher), "--runtime-root", str(runtime)],
        check=True,
    )
    assert "image = image.to(self.cfg.MODEL.DEVICE)" in wrapper.read_text()

    subprocess.run(
        [
            sys.executable,
            str(patcher),
            "--runtime-root",
            str(runtime),
            "--restore",
        ],
        check=True,
    )
    assert "image.to(self.cfg.MODEL.DEVICE)" in wrapper.read_text()


def test_prompthmr_bundle_accepts_only_explicit_audited_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    upstream = tmp_path / "upstream"
    (upstream / "scripts").mkdir(parents=True)
    (upstream / "scripts/demo_video.py").write_text("")
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.name", "Test"],
        check=True,
    )
    wrapper = upstream / "pipeline/utils_detectron2.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("image.to(device)\n")
    subprocess.run(["git", "-C", str(upstream), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "commit", "-qm", "fixture"],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    wrapper.write_text("image = image.to(device)\n")
    monkeypatch.setenv("MOTIUS_PROMPTHMR_AUDITED_PATCH", "1")
    monkeypatch.setattr(
        "motius.models.prompthmr.bundle.PROMPTHMR_REVISION",
        revision,
    )

    assert PromptHMRBundle(
        upstream_dir=str(upstream)
    ).verify_upstream_revision() == upstream.resolve()


def test_prompthmr_checkpoint_hashes_are_cached_by_file_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    video = tmp_path / "video.ckpt"
    image = tmp_path / "image.ckpt"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    calls = []

    def fake_sha256(path: Path) -> str:
        calls.append(path)
        if path == video:
            return VIDEO_SHA256
        return "b" * 64

    monkeypatch.setattr(
        "motius.models.prompthmr.bundle.sha256_file",
        fake_sha256,
    )
    bundle = PromptHMRBundle(
        video_checkpoint_path=str(video),
        image_checkpoint_path=str(image),
    )

    first = bundle.verify_checkpoints()
    second = bundle.verify_checkpoints()

    assert first == second
    assert calls == [video, image]


def test_prompthmr_command_builder_matches_official_tyro_cli(tmp_path: Path):
    upstream = tmp_path / "upstream"
    bundle = PromptHMRBundle(
        upstream_dir=str(upstream),
        python_command=("conda", "run", "-n", "phmr_pt2.4", "python"),
    )
    command = build_prompthmr_video_command(
        bundle,
        tmp_path / "clip.v1.mp4",
        static_camera=True,
        viser_total=100,
        viser_subsample=3,
    )
    assert command.cwd == upstream.resolve()
    assert command.argv[:5] == (
        "conda",
        "run",
        "-n",
        "phmr_pt2.4",
        "python",
    )
    assert command.argv[5:8] == (
        "scripts/demo_video.py",
        "--input-video",
        str((tmp_path / "clip.v1.mp4").resolve()),
    )
    assert "--no-run-viser" in command.argv
    assert "--static-camera" in command.argv
    assert command.output_path == upstream.resolve() / "results/clip/results.pkl"


def test_prompthmr_parser_maps_official_camera_and_world_fields():
    result = parse_prompthmr_results(
        _official_results(include_world=True),
        checkpoint_sha256=VIDEO_SHA256,
        checkpoint_sha256s={
            "video_head": VIDEO_SHA256,
            "image_model": "f" * 64,
        },
        original_fps=60.0,
        output_fps=30.0,
    )
    assert result.source_revision == PROMPTHMR_REVISION
    assert result.checkpoint_sha256 == VIDEO_SHA256
    assert result.camera_coordinate_system.name == "camera_opencv"
    assert result.world_coordinate_system.name == "prompthmr_gravity_world"
    assert result.world_coordinate_system.forward_axis == "-Z"
    assert result.camera_intrinsics.shape == (3, 3)
    assert result.camera_to_world.shape == (4, 4, 4)
    np.testing.assert_array_equal(
        result.camera_to_world[:, :3, 3],
        _official_results()["camera_world"]["Twc"],
    )
    np.testing.assert_allclose(result.frame_timestamps, [0, 1 / 30, 2 / 30, 3 / 30])

    track = result.tracks[0]
    assert track.track_id == "7"
    np.testing.assert_array_equal(track.frame_ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(track.valid, [False, True, False, True])
    assert track.poses_axis_angle.shape == (4, 25, 3)
    np.testing.assert_allclose(track.root_translation_camera[1], [0.1, 0.2, 2.0])
    np.testing.assert_allclose(track.root_translation_world[3], [1.1, 0.0, -2.1])
    assert track.joints_camera is None
    assert track.joints_world is None
    assert track.vertices_world is None
    assert "not saved" in track.availability["joints_camera"]
    assert "smplx_world_pose" in track.native_parameters
    assert result.metadata["prompt_types"] == ["box", "keypoint", "mask"]
    assert result.metadata["checkpoint_sha256s"]["image_model"] == "f" * 64


def test_prompthmr_parser_does_not_invent_world_space():
    result = parse_prompthmr_results(
        _official_results(include_world=False),
        checkpoint_sha256=VIDEO_SHA256,
        original_fps=30.0,
    )
    assert result.world_coordinate_system is None
    assert result.camera_to_world is None
    assert result.tracks[0].root_translation_world is None
    assert "not synthesized" in result.tracks[0].availability[
        "root_translation_world"
    ]


def test_prompthmr_parser_rejects_inconsistent_component_hash():
    with pytest.raises(ValueError, match="video_head"):
        parse_prompthmr_results(
            _official_results(),
            checkpoint_sha256=VIDEO_SHA256,
            checkpoint_sha256s={"video_head": "0" * 64},
            original_fps=30.0,
        )


def test_prompthmr_pipeline_parser_needs_no_official_runtime():
    pipeline = PromptHMRPipeline(PromptHMRBundle())
    result = pipeline.parse_output(
        _official_results(include_world=False),
        checkpoint_sha256=VIDEO_SHA256,
        original_fps=30.0,
    )
    assert result.num_tracks == 1
    assert result.tracks[0].body_model == "SMPL-X neutral"


def test_prompthmr_camera_pose_split_matches_pinned_upstream():
    pose = np.zeros((2, 75), dtype=np.float32)
    pose[:, :66] = np.arange(66, dtype=np.float32)
    split = split_prompthmr_smplx_pose(pose, coordinate_space="camera")
    np.testing.assert_array_equal(split.global_orient, pose[:, :3])
    np.testing.assert_array_equal(split.body_pose, pose[:, 3:66])
    np.testing.assert_array_equal(split.ignored_face_pose, pose[:, 66:75])
    np.testing.assert_array_equal(split.left_hand_pose, np.zeros((2, 45)))
    np.testing.assert_array_equal(split.right_hand_pose, np.zeros((2, 45)))
    np.testing.assert_array_equal(split.jaw_pose, np.zeros((2, 3)))
    assert "official camera results do not save hands" in split.hand_pose_source


def test_prompthmr_camera_pose_split_rejects_nonzero_face_schema():
    pose = np.zeros((1, 75), dtype=np.float32)
    pose[:, 66] = 1.0
    with pytest.raises(ValueError, match="phmr_vid.py"):
        split_prompthmr_smplx_pose(pose, coordinate_space="camera")


def test_prompthmr_world_pose_split_matches_pinned_upstream():
    pose = np.arange(2 * 165, dtype=np.float32).reshape(2, 165)
    split = split_prompthmr_smplx_pose(pose, coordinate_space="world")
    np.testing.assert_array_equal(split.global_orient, pose[:, :3])
    np.testing.assert_array_equal(split.body_pose, pose[:, 3:66])
    np.testing.assert_array_equal(split.ignored_face_pose, pose[:, 66:75])
    np.testing.assert_array_equal(split.left_hand_pose, pose[:, 75:120])
    np.testing.assert_array_equal(split.right_hand_pose, pose[:, 120:165])
    np.testing.assert_array_equal(split.leye_pose, np.zeros((2, 3)))


def test_licensed_smplx_file_provenance_checks_gender_version_and_hash(
    tmp_path: Path,
):
    model_path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(
        model_path,
        gender=np.array("neutral"),
        model_version=np.array("1.1"),
    )
    provenance = inspect_licensed_smplx_model(
        model_path,
        gender="neutral",
        model_version="1.1",
    )
    assert provenance.filename == "SMPLX_NEUTRAL.npz"
    assert provenance.detected_gender == "neutral"
    assert provenance.detected_version == "1.1"
    assert provenance.file_size_bytes == model_path.stat().st_size
    assert len(provenance.sha256) == 64
    assert provenance.as_metadata()["local_path_recorded"] is False

    with pytest.raises(ValueError, match="gender mismatch"):
        inspect_licensed_smplx_model(
            model_path,
            gender="female",
            model_version="1.1",
        )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect_licensed_smplx_model(
            model_path,
            gender="neutral",
            model_version="1.1",
            expected_sha256="0" * 64,
        )


class _FakeSMPLX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({key: value.detach().cpu() for key, value in kwargs.items() if torch.is_tensor(value)})
        translation = kwargs["transl"]
        batch = len(translation)
        joint_offset = torch.arange(
            22, dtype=translation.dtype, device=translation.device
        ).reshape(1, 22, 1)
        vertex_offset = torch.arange(
            5, dtype=translation.dtype, device=translation.device
        ).reshape(1, 5, 1)
        return SimpleNamespace(
            joints=translation[:, None, :] + joint_offset,
            vertices=translation[:, None, :] + vertex_offset,
        )


def test_licensed_smplx_replay_materializes_valid_frames_only():
    parsed = parse_prompthmr_results(
        _official_results(include_world=True),
        checkpoint_sha256=VIDEO_SHA256,
        original_fps=30.0,
    )
    fake_model = _FakeSMPLX()
    provenance = LicensedSMPLXProvenance(
        model_version="1.1",
        gender="neutral",
        filename="SMPLX_NEUTRAL.npz",
        sha256="e" * 64,
        file_size_bytes=123,
    )
    replayed = replay_prompthmr_geometry(
        parsed,
        fake_model,
        provenance,
        spaces=("camera", "world"),
        include_vertices=True,
        batch_size=8,
    )

    track = replayed.tracks[0]
    np.testing.assert_array_equal(track.valid, parsed.tracks[0].valid)
    assert track.joint_names == SMPL_SMPLX_BODY22_NAMES
    assert track.joints_camera.shape == (4, 22, 3)
    assert track.joints_world.shape == (4, 22, 3)
    assert track.vertices_camera.shape == (4, 5, 3)
    assert track.vertices_world.shape == (4, 5, 3)
    np.testing.assert_array_equal(track.joints_camera[[0, 2]], 0.0)
    np.testing.assert_array_equal(track.vertices_world[[0, 2]], 0.0)
    np.testing.assert_allclose(track.joints_camera[1, 0], [0.1, 0.2, 2.0])
    np.testing.assert_allclose(track.joints_world[3, 0], [1.1, 0.0, -2.1])
    assert track.availability["joints_camera"].startswith(
        "licensed_smplx_replay"
    )
    assert track.metadata["licensed_smplx_replay"]["world"]["source"] == (
        "licensed_smplx_replay"
    )
    assert len(fake_model.calls) == 2
    assert all(call["global_orient"].shape[0] == 2 for call in fake_model.calls)
    geometry_meta = replayed.metadata["geometry_materialization"]
    assert geometry_meta["native_official_results_fields"] is False
    assert geometry_meta["valid_mask_preserved"] is True
    assert geometry_meta["evaluation_body_model_alias"] == "smpl"
    assert geometry_meta["model"]["sha256"] == "e" * 64
    common = select_common_hmr15(
        track.joints_camera,
        track.joint_names,
        body_model=track.body_model,
    )
    assert common.shape == (4, 15, 3)


def test_licensed_replay_never_synthesizes_missing_world_geometry():
    parsed = parse_prompthmr_results(
        _official_results(include_world=False),
        checkpoint_sha256=VIDEO_SHA256,
        original_fps=30.0,
    )
    provenance = LicensedSMPLXProvenance(
        model_version="1.1",
        gender="neutral",
        filename="SMPLX_NEUTRAL.npz",
        sha256="e" * 64,
        file_size_bytes=123,
    )
    replayed = replay_prompthmr_geometry(
        parsed,
        _FakeSMPLX(),
        provenance,
        spaces=("world",),
    )
    assert replayed.tracks[0].joints_world is None
    status = replayed.metadata["geometry_materialization"]["track_status"]["7"]
    assert status["world"] == "unavailable_official_world_parameters; not synthesized"


def test_licensed_replay_cli_uses_pickle_free_motius_artifact():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools/materialize_prompthmr_smplx.py").read_text()

    assert ".motius.npz" in script
    assert "save_monocular_capture_result" in script
    assert "pickle.dump" not in script
