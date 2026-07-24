import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from motius.models.gvhmr import (
    GVHMRBundle,
    OFFICIAL_RUNTIME_REVISION,
    sha256_file,
)
from motius.models.gvhmr.bundle import _bbox_xys_from_xyxy
from motius.motion.representation.monocular_capture import (
    CAMERA_OPENCV,
    GRAVITY_WORLD_Y_UP,
)
from motius.motion.representation.monocular_joints import SMPL24_NAMES
from motius.pipelines.gvhmr import GVHMRPipeline, parse_gvhmr_output
from motius.registry import MODEL_BUNDLES, PIPELINES
from tools.patch_gvhmr_runtime import patch_runtime


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


def test_packages_register_without_official_runtime():
    assert MODEL_BUNDLES.get("GVHMRBundle") is GVHMRBundle
    assert PIPELINES.get("GVHMRPipeline") is GVHMRPipeline
    bundle = GVHMRBundle(runtime_root="/definitely/missing")
    with pytest.raises(FileNotFoundError, match="runtime not found"):
        bundle.validate_runtime(require_checkpoint=False)


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
        track.poses_axis_angle[:, 1:].reshape(3, 63),
        payload["smpl_params_global"]["body_pose"],
    )
    np.testing.assert_allclose(
        track.root_translation_camera,
        payload["smpl_params_incam"]["transl"],
    )
    assert track.joints_camera is None
    assert (
        track.metadata["valid_mask_source"]
        == "official_dense_output_after_bbox_interpolation"
    )
    assert result.metadata["camera_to_world"] == "not_emitted_by_official_demo"


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


def test_checkpoint_checksum_is_computed_from_file(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"gvhmr")
    assert (
        sha256_file(checkpoint)
        == "5628dae8b036ede4a3dae7be50e9cbbb272b11e32f87bf5422cd9e721453acba"
    )


def test_official_bbox_conversion_and_preseed_cache(
    tmp_path: Path,
    monkeypatch,
):
    boxes = np.asarray(
        [
            [10.0, 20.0, 70.0, 180.0],
            [20.0, 40.0, 220.0, 140.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        _bbox_xys_from_xyxy(boxes),
        [[40.0, 100.0, 192.0], [120.0, 90.0, 320.0]],
    )

    runtime = tmp_path / "runtime"
    (runtime / "tools/demo").mkdir(parents=True)
    (runtime / "tools/demo/demo.py").write_text("")
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    output_root = tmp_path / "outputs"
    bundle = GVHMRBundle(
        runtime_root=runtime,
        python_executable=sys.executable,
    )
    monkeypatch.setattr(bundle, "validate_runtime", lambda **kwargs: None)

    def fake_run(command, **kwargs):
        run_root = Path(
            next(item.split("=", 1)[1] for item in command if item.startswith("--output_root="))
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

    monkeypatch.setattr("motius.models.gvhmr.bundle.subprocess.run", fake_run)

    result = bundle.run_official_demo(
        video,
        output_root,
        bbox_xyxy=boxes,
    )

    assert result.name == "hmr4d_results.pt"


def test_setup_defaults_keep_runtime_under_outputs():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "tools/setup_gvhmr_env.sh").read_text()

    assert "outputs/tmp/gvhmr/upstream" in setup
    assert "outputs/tmp/gvhmr/conda-env" in setup
    assert "/ref_repo/GVHMR" not in setup


def test_runtime_patch_handles_missing_two_view_pose_and_restores(tmp_path: Path):
    solver = (
        tmp_path
        / "hmr4d/utils/preproc/relpose/solver_two_view.py"
    )
    solver.parent.mkdir(parents=True)
    original = """\
def solve(answer):
        # cam2_from_cam1 means T_0_to_1 in our language
        Rt = answer.cam2_from_cam1.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4)
        T[:3] = Rt
        return T
"""
    solver.write_text(original)
    demo = tmp_path / "tools/demo/demo.py"
    demo.parent.mkdir(parents=True)
    demo_original = """\
import cv2
import torch

if __name__ == "__main__":
    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
"""
    demo.write_text(demo_original)

    patch_runtime(tmp_path)
    patched = solver.read_text()
    demo_patched = demo.read_text()
    assert "if relative_pose is None" in patched
    assert "np.eye(4, dtype=np.float32)" in patched
    assert 'os.environ.get("MOTIUS_GVHMR_SKIP_RENDER")' in demo_patched
    assert "import os" in demo_patched
    patch_runtime(tmp_path)
    assert solver.read_text() == patched
    assert demo.read_text() == demo_patched

    patch_runtime(tmp_path, restore=True)
    assert solver.read_text() == original
    assert demo.read_text() == demo_original
