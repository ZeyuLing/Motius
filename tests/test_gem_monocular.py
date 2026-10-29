import json
import os
from pathlib import Path

import numpy as np
import pytest

from motius.models.gem_smpl import GemSmplBundle
from motius.configuration.gem_smpl_provenance import GEM_SMPL_ARTIFACT_FORMAT
from motius.configuration import gem_smpl_provenance as gem_smpl_runtime
from motius.models.gem_x import GemXBundle
from motius.configuration.gem_x_provenance import GEM_X_ARTIFACT_FORMAT
from motius.configuration import gem_x_provenance as gem_x_runtime
from motius.motion.representation.monocular_joints import (
    COMMON_HMR15_NAMES,
    SMPL24_NAMES,
    SOMA77_NAMES,
    select_common_hmr15,
)
from motius.pipelines.gem_smpl.parser import (
    parse_gem_smpl_file,
    parse_gem_smpl_output,
)
from motius.pipelines.gem_smpl.export_native import _apply_sparse_vertex_map
from motius.pipelines.gem_smpl import GemSmplPipeline
from motius.pipelines.gem_x.parser import parse_gem_x_file, parse_gem_x_output
from motius.pipelines.gem_x import GemXPipeline


def _smpl_group(frames: int) -> dict[str, np.ndarray]:
    return {
        "body_pose": np.zeros((frames, 63), dtype=np.float32),
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "transl": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        "betas": np.zeros((frames, 10), dtype=np.float32),
    }


def _soma_group(frames: int) -> dict[str, np.ndarray]:
    return {
        "body_pose": np.zeros((frames, 228), dtype=np.float32),
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "transl": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3),
        "identity_coeffs": np.zeros((frames, 45), dtype=np.float32),
        "scale_params": np.zeros((frames, 69), dtype=np.float32),
    }


def _artifact(
    root: Path,
    *,
    filename: str,
    artifact_format: str,
    source_revision: str,
) -> Path:
    root.mkdir()
    (root / filename).write_text(
        json.dumps(
            {
                "artifact_format": artifact_format,
                "source_revision": source_revision,
                "sha256": {},
            }
        )
    )
    return root


def test_pinned_revisions_and_checkpoints_are_independent():
    assert gem_smpl_runtime.SOURCE_REVISION == "16bebf402d8893184249ee206d957b8248cd8310"
    assert gem_x_runtime.SOURCE_REVISION == "32992550dba114c62243fb55e361311972dce8f9"
    assert gem_smpl_runtime.CHECKPOINT_SHA256 == (
        "1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
    )
    assert gem_x_runtime.CHECKPOINT_SHA256 == (
        "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
    )





def test_gem_smpl_parser_preserves_official_fields_and_named_smpl_joints():
    frames = 3
    payload = {
        "body_params_incam": _smpl_group(frames),
        "body_params_global": _smpl_group(frames),
        "K_fullimg": np.eye(3, dtype=np.float32),
        "joints_camera": np.arange(frames * 24 * 3, dtype=np.float32).reshape(
            frames, 24, 3
        ),
        "joints_world": np.zeros((frames, 24, 3), dtype=np.float32),
    }

    result = parse_gem_smpl_output(payload, original_fps=29.97)
    track = result.tracks[0]

    assert result.source_revision == gem_smpl_runtime.SOURCE_REVISION
    assert track.body_model == "smpl"
    assert track.poses_axis_angle.shape == (frames, 22, 3)
    assert track.shape_parameters.shape == (frames, 10)
    assert track.joint_names == SMPL24_NAMES
    assert track.vertices_camera is track.vertices_world is None
    assert track.availability["pve"] == "unavailable_without_exported_vertices"
    assert "body_params_incam.betas" in track.native_parameters
    common = select_common_hmr15(
        track.joints_camera,
        track.joint_names,
        body_model=track.body_model,
    )
    assert common.shape == (frames, len(COMMON_HMR15_NAMES), 3)


def test_gem_x_parser_keeps_soma_native_and_disallows_cross_topology_pve():
    frames = 2
    payload = {
        "body_params_incam": _soma_group(frames),
        "body_params_global": _soma_group(frames),
        "K_fullimg": np.repeat(np.eye(3, dtype=np.float32)[None], frames, axis=0),
        "joints_camera": np.arange(frames * 77 * 3, dtype=np.float32).reshape(
            frames, 77, 3
        ),
        "joints_world": np.zeros((frames, 77, 3), dtype=np.float32),
    }

    result = parse_gem_x_output(payload, original_fps=24.0)
    track = result.tracks[0]

    assert result.source_revision == gem_x_runtime.SOURCE_REVISION
    assert track.body_model == "soma77"
    assert track.poses_axis_angle.shape == (frames, 77, 3)
    assert track.shape_parameters is None
    assert track.joint_names == SOMA77_NAMES
    assert track.vertices_camera is track.vertices_world is None
    assert track.availability["pve"] == "not_comparable_to_smpl_topology"
    assert track.native_parameters["body_params_incam.identity_coeffs"].shape == (
        frames,
        45,
    )
    assert track.native_parameters["body_params_incam.scale_params"].shape == (
        frames,
        69,
    )
    common = select_common_hmr15(
        track.joints_camera,
        track.joint_names,
        body_model=track.body_model,
    )
    assert common.shape == (frames, len(COMMON_HMR15_NAMES), 3)


def test_numeric_npz_parsers_require_no_official_runtime(tmp_path: Path):
    frames = 2
    smpl_path = tmp_path / "smpl.npz"
    smpl_arrays = {
        f"body_params_incam.{key}": value
        for key, value in _smpl_group(frames).items()
    }
    smpl_arrays["joints_camera"] = np.zeros((frames, 24, 3), dtype=np.float32)
    np.savez(smpl_path, **smpl_arrays)

    soma_path = tmp_path / "soma.npz"
    soma_arrays = {
        f"body_params_global.{key}": value
        for key, value in _soma_group(frames).items()
    }
    soma_arrays["joints_world"] = np.zeros((frames, 77, 3), dtype=np.float32)
    np.savez(soma_path, **soma_arrays)

    assert parse_gem_smpl_file(smpl_path, original_fps=30).tracks[0].num_frames == frames
    soma = parse_gem_x_file(soma_path, original_fps=30)
    assert soma.tracks[0].num_frames == frames
    assert soma.tracks[0].body_model == "soma77"


def test_gem_smpl_sparse_vertex_map_matches_dense_batch_definition():
    import torch

    dense = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    vertices = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)

    actual = _apply_sparse_vertex_map(dense.to_sparse(), vertices)
    expected = torch.stack([dense @ frame for frame in vertices])

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)




















def test_native_gem_public_geometry_assets_are_materialized():
    root=Path(__file__).resolve().parents[1]/'motius/configuration/gem_smpl/body'
    for name in ('coco_aug_dict.pth','smpl_neutral_J_regressor.pt','smplx2smpl_sparse.pt'):
        assert (root/name).stat().st_size>256
