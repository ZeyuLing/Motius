import json
import os
from pathlib import Path

import numpy as np
import pytest

from motius.models.gem_smpl import GemSmplBundle
from motius.models.gem_smpl.bundle import GEM_SMPL_ARTIFACT_FORMAT
from motius.models.gem_smpl import runtime as gem_smpl_runtime
from motius.models.gem_x import GemXBundle
from motius.models.gem_x.bundle import GEM_X_ARTIFACT_FORMAT
from motius.models.gem_x import runtime as gem_x_runtime
from motius.models.gem_runtime import (
    PACKAGE_ROOT,
    ensure_outputs_path,
    process_env,
    project_root,
    resolve_python,
)
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
    assert gem_smpl_runtime.VENDORED_RUNTIME_ROOT.is_dir()
    assert gem_x_runtime.VENDORED_RUNTIME_ROOT.is_dir()
    assert (
        gem_smpl_runtime.VENDORED_RUNTIME_ROOT
        / "scripts/demo/demo_smpl_hpe.py"
    ).is_file()
    assert (
        gem_x_runtime.VENDORED_RUNTIME_ROOT / "scripts/demo/demo_soma.py"
    ).is_file()


def test_external_runtime_checkouts_are_rejected(tmp_path: Path):
    for runtime in (gem_smpl_runtime, gem_x_runtime):
        with pytest.raises(ValueError, match="external runtime checkout"):
            runtime.verify_runtime_checkout(tmp_path)
        assert runtime.verify_runtime_checkout() == runtime.VENDORED_RUNTIME_ROOT


def test_gem_methods_use_canonical_motius_bundle_pipeline_layers(tmp_path: Path):
    smpl_root = _artifact(
        tmp_path / "gem_smpl",
        filename="gem_smpl_config.json",
        artifact_format=GEM_SMPL_ARTIFACT_FORMAT,
        source_revision=gem_smpl_runtime.SOURCE_REVISION,
    )
    x_root = _artifact(
        tmp_path / "gem_x",
        filename="gem_x_config.json",
        artifact_format=GEM_X_ARTIFACT_FORMAT,
        source_revision=gem_x_runtime.SOURCE_REVISION,
    )
    smpl_bundle = GemSmplBundle(
        artifact_root=smpl_root,
        python_executable="/usr/bin/python3",
    )
    x_bundle = GemXBundle(
        artifact_root=x_root,
        python_executable="/usr/bin/python3",
    )

    smpl_pipeline = GemSmplPipeline(smpl_bundle)
    x_pipeline = GemXPipeline(x_bundle)

    assert smpl_pipeline.bundle is smpl_bundle
    assert x_pipeline.bundle is x_bundle
    assert smpl_bundle.source_revision == gem_smpl_runtime.SOURCE_REVISION
    assert x_bundle.source_revision == gem_x_runtime.SOURCE_REVISION
    assert hasattr(smpl_pipeline, "infer_monocular_motion_capture")
    assert hasattr(x_pipeline, "infer_monocular_motion_capture")

    assert GemSmplBundle.from_pretrained(
        str(smpl_root),
        python_executable="/usr/bin/python3",
    ).artifact_root == smpl_root
    assert GemXBundle.from_pretrained(
        str(x_root),
        python_executable="/usr/bin/python3",
    ).artifact_root == x_root


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


def test_outputs_contract_and_external_setup_scripts_are_removed(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    assert not (root / "motius/models/gem_smpl/setup_runtime.sh").exists()
    assert not (root / "motius/models/gem_x/setup_runtime.sh").exists()
    assert ensure_outputs_path(root / "outputs/gem", method="GEM") == (
        root / "outputs/gem"
    )
    with pytest.raises(ValueError, match="outputs"):
        ensure_outputs_path(tmp_path / "gem", method="GEM")


def test_outputs_contract_uses_caller_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs/gem/run"

    assert project_root() == tmp_path
    assert ensure_outputs_path(output, method="GEM") == output
    with pytest.raises(ValueError, match="outputs"):
        ensure_outputs_path(tmp_path / "run", method="GEM")


def test_outputs_contract_honors_explicit_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MOTIUS_PROJECT_ROOT", str(tmp_path))

    assert project_root() == tmp_path
    assert ensure_outputs_path(tmp_path / "outputs/run", method="GEM") == (
        tmp_path / "outputs/run"
    )


def test_subprocess_environment_does_not_inherit_external_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    external = tmp_path / "external_checkout"
    vendor = tmp_path / "vendor"
    artifact = tmp_path / "artifact"
    monkeypatch.setenv("PYTHONPATH", str(external))

    env = process_env(
        vendor_root=vendor,
        artifact_root=artifact,
        method_prefix="GEM_TEST",
    )

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(vendor.resolve()),
        str(PACKAGE_ROOT),
    ]
    assert str(external) not in env["PYTHONPATH"]
    assert env["MOTIUS_GEM_TEST_ARTIFACT_ROOT"] == str(artifact)


def test_explicit_python_path_is_resolved_before_runtime_chdir(tmp_path: Path):
    base_python = tmp_path / "base" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("#!/bin/sh\n")
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base_python)

    original = Path.cwd()
    try:
        os.chdir(tmp_path)
        resolved = resolve_python(Path("venv/bin/python"))
    finally:
        os.chdir(original)

    assert resolved == str(python)


def test_every_vendored_python_directory_is_wheel_discoverable():
    root = Path(__file__).resolve().parents[1]
    for vendor in (
        root / "motius/models/gem_smpl/vendor",
        root / "motius/models/gem_x/vendor",
    ):
        missing = []
        for directory in sorted({path.parent for path in vendor.rglob("*.py")}):
            # Third-party roots with a dash are shipped as package data and put
            # on sys.path by the isolated runner.
            if any("-" in part for part in directory.relative_to(vendor).parts):
                continue
            if not (directory / "__init__.py").is_file():
                missing.append(str(directory.relative_to(root)))
        assert not missing, missing


def test_gem_x_runner_patches_the_demo_function_globals():
    source = (
        Path(__file__).resolve().parents[1]
        / "motius/models/gem_x/vendor/runner.py"
    ).read_text()

    assert 'demo_globals = namespace["main"].__globals__' in source
    assert 'namespace["render_incam"]' not in source
