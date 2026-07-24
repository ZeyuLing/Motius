from pathlib import Path

import numpy as np

from motius.models.gem_smpl import GemSmplBundle
from motius.models.gem_smpl import runtime as gem_smpl_runtime
from motius.models.gem_x import GemXBundle
from motius.models.gem_x import runtime as gem_x_runtime
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


def test_pinned_revisions_checkpoints_and_commands_are_independent(tmp_path: Path):
    smpl_root = tmp_path / "gem_smpl"
    soma_root = tmp_path / "gem_x"
    for root in (smpl_root, soma_root):
        python = root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.touch()

    smpl_command = gem_smpl_runtime.build_demo_command(
        runtime_root=smpl_root,
        video=tmp_path / "clip.mp4",
        output_root=tmp_path / "smpl_output",
        checkpoint=tmp_path / "gem_smpl.ckpt",
        static_camera=True,
    )
    soma_command = gem_x_runtime.build_demo_command(
        runtime_root=soma_root,
        video=tmp_path / "clip.mp4",
        output_root=tmp_path / "soma_output",
        checkpoint=tmp_path / "gem_soma.ckpt",
        static_camera=True,
    )

    assert gem_smpl_runtime.SOURCE_REVISION == "16bebf402d8893184249ee206d957b8248cd8310"
    assert gem_x_runtime.SOURCE_REVISION == "32992550dba114c62243fb55e361311972dce8f9"
    assert gem_smpl_runtime.CHECKPOINT_SHA256 == (
        "1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
    )
    assert gem_x_runtime.CHECKPOINT_SHA256 == (
        "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
    )
    assert "demo_smpl_hpe.py" in smpl_command[1]
    assert "--video" in smpl_command
    assert "--ckpt_path" in smpl_command and "--no_render" in smpl_command
    assert "demo_soma.py" in soma_command[1]
    assert "--ckpt" in soma_command and "--ckpt_path" not in soma_command
    assert smpl_command[0] != soma_command[0]


def test_runtime_python_preserves_virtualenv_symlink(tmp_path: Path):
    system_python = tmp_path / "system_python"
    system_python.touch()
    for runtime in (gem_smpl_runtime, gem_x_runtime):
        root = tmp_path / runtime.__name__.replace(".", "_")
        python = root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(system_python)

        resolved = runtime.runtime_python(root)

        assert resolved == python.absolute()
        assert resolved != system_python.resolve()


def test_gem_methods_use_canonical_motius_bundle_pipeline_layers(tmp_path: Path):
    smpl_bundle = GemSmplBundle(
        runtime_root=str(tmp_path / "gem_smpl"),
        checkpoint=str(tmp_path / "gem_smpl.ckpt"),
        python_executable="/usr/bin/python3",
    )
    x_bundle = GemXBundle(
        runtime_root=str(tmp_path / "gem_x"),
        checkpoint=str(tmp_path / "gem_soma.ckpt"),
        python_executable="/usr/bin/python3",
    )

    smpl_pipeline = GemSmplPipeline(smpl_bundle)
    x_pipeline = GemXPipeline(x_bundle)

    assert smpl_pipeline.bundle is smpl_bundle
    assert x_pipeline.bundle is x_bundle
    assert smpl_bundle.source_revision == gem_smpl_runtime.SOURCE_REVISION
    assert x_bundle.source_revision == gem_x_runtime.SOURCE_REVISION


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


def test_setup_runtime_defaults_stay_under_outputs():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "motius/models/gem_smpl/setup_runtime.sh",
        "motius/models/gem_x/setup_runtime.sh",
    ):
        script = (root / relative).read_text()
        assert "/outputs/tmp/" in script
        assert "$PWD/.external" not in script
