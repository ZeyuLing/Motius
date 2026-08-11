import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import motius.models.gvhmr.bundle as bundle_module
from motius.models.gvhmr import (
    GVHMR_ARTIFACT_FORMAT,
    GVHMRBundle,
    OFFICIAL_RUNTIME_REVISION,
    sha256_file,
)
from motius.models.gvhmr.bundle import _bbox_xys_from_xyxy
from motius.models.gvhmr.parity import capture_gvhmr_trace
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES
from motius.pipelines.gvhmr import GVHMRPipeline, parse_gvhmr_output
from motius.registry import MODEL_BUNDLES, PIPELINES


ASSETS = (
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "inputs/checkpoints/yolo/yolov8x.pt",
)


def make_artifact(root: Path) -> Path:
    digests = {}
    for index, relative in enumerate(ASSETS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset-{index}".encode())
        digests[relative] = sha256_file(path)
    (root / "inputs/checkpoints/body_models/smplx").mkdir(parents=True)
    (root / "gvhmr_config.json").write_text(
        json.dumps(
            {
                "artifact_format": GVHMR_ARTIFACT_FORMAT,
                "source_revision": OFFICIAL_RUNTIME_REVISION,
                "sha256": digests,
            }
        )
    )
    return root


def official_payload(frames: int = 3) -> dict:
    body_pose = np.arange(frames * 63, dtype=np.float32).reshape(frames, 63)
    betas = np.arange(frames * 10, dtype=np.float32).reshape(frames, 10) / 100
    return {
        "smpl_params_global": {
            "global_orient": np.full((frames, 3), 0.25, dtype=np.float32),
            "body_pose": body_pose,
            "betas": betas,
            "transl": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        },
        "smpl_params_incam": {
            "global_orient": np.full((frames, 3), -0.5, dtype=np.float32),
            "body_pose": body_pose.copy(),
            "betas": betas.copy(),
            "transl": (
                np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) + 10
            ),
        },
        "K_fullimg": np.repeat(
            np.eye(3, dtype=np.float32)[None],
            frames,
            axis=0,
        ),
    }


def converted_payload(frames: int = 3) -> dict:
    nested = official_payload(frames)
    flat = {}
    for prefix in ("smpl_params_global", "smpl_params_incam"):
        for name, value in nested[prefix].items():
            flat[f"{prefix}_{name}"] = value
    flat.update(
        {
            "K_fullimg": nested["K_fullimg"],
            "runtime_revision": np.asarray(OFFICIAL_RUNTIME_REVISION),
            "checkpoint_sha256": np.asarray("b" * 64),
            "valid": np.asarray([True, False, True]),
            "frame_ids": np.asarray([4, 5, 6]),
            "joints_camera": np.zeros((frames, 24, 3), dtype=np.float32),
            "joints_world": np.ones((frames, 24, 3), dtype=np.float32),
            "vertices_camera": np.zeros((frames, 6890, 3), dtype=np.float32),
            "vertices_world": np.ones((frames, 6890, 3), dtype=np.float32),
        }
    )
    return flat


def test_repo_native_artifact_registers_and_loads(tmp_path: Path):
    artifact = make_artifact(tmp_path / "artifact")
    assert MODEL_BUNDLES.get("GVHMRBundle") is GVHMRBundle
    assert PIPELINES.get("GVHMRPipeline") is GVHMRPipeline

    pipeline = GVHMRPipeline.from_pretrained(
        str(artifact),
        bundle_kwargs={"python_executable": sys.executable},
    )

    assert pipeline.bundle.artifact_root == artifact
    pipeline.bundle.validate_runtime(require_checkpoint=True)
    assert pipeline.bundle.checkpoint_sha256 == sha256_file(
        artifact / ASSETS[0]
    )


def test_artifact_rejects_revision_and_checkpoint_drift(tmp_path: Path):
    artifact = make_artifact(tmp_path / "artifact")
    config_path = artifact / "gvhmr_config.json"
    config = json.loads(config_path.read_text())
    config["source_revision"] = "wrong"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source revision"):
        GVHMRBundle(artifact_root=artifact)

    artifact = make_artifact(tmp_path / "second")
    bundle = GVHMRBundle(
        artifact_root=artifact,
        python_executable=sys.executable,
    )
    (artifact / ASSETS[0]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _ = bundle.checkpoint_sha256


def test_parser_maps_documented_official_demo_fields():
    payload = official_payload()
    result = parse_gvhmr_output(
        payload,
        checkpoint_sha256="a" * 64,
        original_fps=29.97,
    )
    track = result.tracks[0]

    assert result.source_revision == OFFICIAL_RUNTIME_REVISION
    assert result.checkpoint_sha256 == "a" * 64
    assert result.original_fps == pytest.approx(29.97)
    assert result.output_fps == 30.0
    assert result.camera_coordinate_system == CAMERA_OPENCV
    assert result.world_coordinate_system == GRAVITY_WORLD_Y_UP
    assert result.camera_to_world is None
    np.testing.assert_array_equal(track.valid, np.ones(3, dtype=bool))
    np.testing.assert_allclose(
        track.poses_axis_angle[:, 0],
        payload["smpl_params_global"]["global_orient"],
    )
    np.testing.assert_allclose(
        track.root_translation_camera,
        payload["smpl_params_incam"]["transl"],
    )
    assert track.joints_camera is None


def test_parser_loads_materialized_npz_provenance(tmp_path: Path):
    source = tmp_path / "motius_monocular_capture.npz"
    np.savez_compressed(source, **converted_payload())

    result = parse_gvhmr_output(source, original_fps=60.0)
    track = result.tracks[0]

    assert result.checkpoint_sha256 == "b" * 64
    assert track.joint_names == SMPL24_NAMES
    assert track.vertices_world.shape == (3, 6890, 3)
    np.testing.assert_array_equal(track.valid, [True, False, True])
    np.testing.assert_array_equal(track.frame_ids, [4, 5, 6])
    np.testing.assert_allclose(result.frame_timestamps, [4 / 30, 5 / 30, 6 / 30])


def test_parser_safely_loads_raw_official_pt(tmp_path: Path):
    payload = official_payload()
    tensor_payload = {
        name: (
            {key: torch.from_numpy(value) for key, value in group.items()}
            if isinstance(group, dict)
            else torch.from_numpy(group)
        )
        for name, group in payload.items()
    }
    tensor_payload["net_outputs"] = {"ignored_tensor": torch.ones(1)}
    source = tmp_path / "hmr4d_results.pt"
    torch.save(tensor_payload, source)

    result = parse_gvhmr_output(
        source,
        checkpoint_sha256="c" * 64,
        original_fps=24.0,
    )

    assert result.tracks[0].num_frames == 3
    assert "net_outputs" not in result.tracks[0].native_parameters


def test_parser_rejects_missing_hash_and_revision_mismatch():
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        parse_gvhmr_output(official_payload(), original_fps=30.0)

    converted = converted_payload()
    converted["runtime_revision"] = np.asarray("wrong")
    with pytest.raises(ValueError, match="revision"):
        parse_gvhmr_output(converted, original_fps=30.0)


def test_repo_native_command_and_preseed_cache(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    output_root = repository / "outputs" / "gvhmr"
    artifact = make_artifact(tmp_path / "artifact")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    boxes = np.asarray(
        [[10.0, 20.0, 70.0, 180.0], [20.0, 40.0, 220.0, 140.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        _bbox_xys_from_xyxy(boxes),
        [[40.0, 100.0, 192.0], [120.0, 90.0, 320.0]],
    )
    monkeypatch.setattr(bundle_module, "_REPO_ROOT", repository)
    bundle = GVHMRBundle(
        artifact_root=artifact,
        python_executable=sys.executable,
    )

    def fake_run(command, **kwargs):
        assert command[:3] == [
            sys.executable,
            "-m",
            "motius.models.gvhmr.vendor.official_demo",
        ]
        assert kwargs["cwd"] == artifact
        assert kwargs["env"]["MOTIUS_GVHMR_ARTIFACT_ROOT"] == str(artifact)
        assert ".git" not in " ".join(command)
        run_root = Path(
            next(
                item.split("=", 1)[1]
                for item in command
                if item.startswith("--output_root=")
            )
        )
        cache = torch.load(
            run_root / video.stem / "preprocess/bbx.pt",
            map_location="cpu",
            weights_only=True,
        )
        np.testing.assert_array_equal(cache["bbx_xyxy"].numpy(), boxes)
        np.testing.assert_allclose(
            cache["bbx_xys"].numpy(),
            _bbox_xys_from_xyxy(boxes),
        )
        result = run_root / video.stem / "hmr4d_results.pt"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"result")

    monkeypatch.setattr(bundle_module.subprocess, "run", fake_run)
    result = bundle.run_official_demo(video, output_root, bbox_xyxy=boxes)

    assert result.name == "hmr4d_results.pt"


def test_save_pretrained_emits_complete_hf_contract(
    tmp_path: Path,
    monkeypatch,
):
    repository = tmp_path / "repo"
    artifact = make_artifact(tmp_path / "artifact")
    output = repository / "outputs" / "gvhmr"
    monkeypatch.setattr(bundle_module, "_REPO_ROOT", repository)
    bundle = GVHMRBundle(
        artifact_root=artifact,
        python_executable=sys.executable,
    )

    bundle.save_pretrained(str(output))

    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index["pipeline_class"].endswith("GVHMRPipeline")
    assert model_index["tasks"] == ["monocular_motion_capture"]
    assert "GVHMRPipeline.from_pretrained" in (output / "README.md").read_text()
    for relative in ASSETS:
        assert sha256_file(output / relative) == sha256_file(artifact / relative)


def test_outputs_and_setup_are_productized(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repo"
    monkeypatch.setattr(bundle_module, "_REPO_ROOT", repository)
    with pytest.raises(ValueError, match="outputs"):
        bundle_module._ensure_output_root(tmp_path / "elsewhere")

    root = Path(__file__).resolve().parents[1]
    setup = (root / "tools/setup_gvhmr_env.sh").read_text()
    assert "git clone" not in setup
    assert "No external GVHMR source checkout" in setup
    assert "motius.models.gvhmr.vendor.official_demo" in (
        root / "motius/models/gvhmr/bundle.py"
    ).read_text()
    assert '"utils/body_model/*.pth"' in (root / "pyproject.toml").read_text()


def test_every_vendored_python_directory_is_wheel_discoverable():
    root = Path(__file__).resolve().parents[1]
    source_root = root / "motius/models/gvhmr/vendor/hmr4d"
    missing = sorted(
        {
            source.parent
            for source in source_root.rglob("*.py")
            if not (source.parent / "__init__.py").is_file()
        }
    )
    assert not missing


def test_vendored_two_view_fallback_is_deterministic():
    root = Path(__file__).resolve().parents[1]
    solver = (
        root
        / "motius/models/gvhmr/vendor/hmr4d/utils/preproc/relpose/"
        "solver_two_view.py"
    ).read_text()
    assert "if relative_pose is None" in solver
    assert "np.eye(4, dtype=np.float32)" in solver


def test_optional_render_does_not_require_ffmpeg_executable():
    root = Path(__file__).resolve().parents[1]
    demo = (
        root / "motius/models/gvhmr/vendor/official_demo.py"
    ).read_text()
    assert 'shutil.which("ffmpeg") is not None' in demo
    assert "Skipped because the ffmpeg executable is absent" in demo


def test_gvhmr_parity_contract_covers_every_inference_boundary(tmp_path: Path):
    frames = 2
    bbox = {
        "bbx_xyxy": torch.zeros(frames, 4),
        "bbx_xys": torch.zeros(frames, 3),
    }
    data = {
        "length": torch.tensor(frames),
        "bbx_xys": bbox["bbx_xys"],
        "kp2d": torch.zeros(frames, 17, 3),
        "K_fullimg": torch.eye(3).repeat(frames, 1, 1),
        "cam_angvel": torch.zeros(frames, 6),
        "f_imgseq": torch.zeros(frames, 1024),
    }
    prediction = official_payload(frames)
    output = capture_gvhmr_trace(
        tmp_path / "outputs/gvhmr.npz",
        name="gvhmr-test",
        bbox=bbox,
        data=data,
        prediction=prediction,
    )

    from motius.utils.monocular_parity import MonocularParityTrace

    trace = MonocularParityTrace.load(output)
    assert trace.stage_names == (
        "01_tracking",
        "02_camera",
        "03_visual_features",
        "04_model_input",
        "05_model_output",
    )
