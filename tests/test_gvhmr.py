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
from motius.pipelines.gvhmr.preprocessing import _bbox_xys_from_xyxy
from motius.pipelines.gvhmr.parity import capture_gvhmr_trace
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


def test_native_network_registration_forward_and_checkpoint(tmp_path):
    assert MODEL_BUNDLES.get("GVHMRBundle") is GVHMRBundle
    assert PIPELINES.get("GVHMRPipeline") is GVHMRPipeline
    torch.manual_seed(7)
    bundle=GVHMRBundle(config=dict(latent_dim=32,num_layers=1,num_heads=4))
    inputs=dict(length=torch.tensor([10,8]),obs=torch.randn(2,10,17,3),
        f_cliffcam=torch.randn(2,10,3),f_cam_angvel=torch.randn(2,10,6),f_imgseq=torch.randn(2,10,1024))
    with torch.no_grad():expected=bundle.predict_motion(**inputs)
    bundle.save_pretrained(tmp_path)
    loaded=GVHMRBundle.from_pretrained(str(tmp_path))
    with torch.no_grad():actual=loaded.predict_motion(**inputs)
    assert set(actual)=={"pred_context","pred_x","pred_cam","static_conf_logits"}
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert actual["pred_x"].shape==(2,10,151)
    assert loaded.checkpoint_sha256==sha256_file(tmp_path/"gvhmr.safetensors")
    assert not hasattr(bundle,"run_official_demo")


def test_native_checkpoint_revision_and_shape_validation(tmp_path):
    bundle=GVHMRBundle(config=dict(latent_dim=32,num_layers=1,num_heads=4))
    bundle.save_pretrained(tmp_path)
    config=tmp_path/"gvhmr_config.json";meta=json.loads(config.read_text())
    meta["source_revision"]="unsupported";config.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="revision"):
        GVHMRBundle.from_pretrained(str(tmp_path))
