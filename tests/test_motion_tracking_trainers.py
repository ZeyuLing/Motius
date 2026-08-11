import argparse
import os
from pathlib import Path
import io
import sys
import tarfile
from types import SimpleNamespace

import pytest

from motius.trainers.protomotions import train as protomotions_train_module
from motius.trainers.sonic import train as sonic_train_module
from motius.trainers.beyondmimic.export_policy import build_export_argv
from motius.trainers.beyondmimic.prepare_motion import build_prepare_argv
from motius.trainers.beyondmimic.train import (
    BeyondMimicTrainer,
    build_launch as beyondmimic_launch,
)
from motius.trainers.beyondmimic.vendor.scripts.rsl_rl import cli_args
from motius.trainers.protomotions.train import (
    DEFAULT_EXPERIMENT,
    ProtoMotionsTrainer,
    _training_argv as protomotions_argv,
)
from motius.trainers.sonic.train import (
    OUTPUT_ROOT as SONIC_OUTPUT_ROOT,
    SonicTrainer,
    _exported_onnx_pair,
    _training_argv as sonic_argv,
)
from tools.train_motion_tracking import load_command
from tools.download_beyondmimic_assets import _is_selected, extract_assets


ROOT = Path(__file__).resolve().parents[1]


def _assert_materialized_asset(relative_path: str) -> None:
    path = ROOT / relative_path
    assert path.is_file(), f"missing trainer asset: {path}"
    prefix = path.read_bytes()[:128]
    assert not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
    assert path.stat().st_size > 256


@pytest.mark.parametrize(
    "relative_path",
    [
        (
            "motius/trainers/sonic/vendor/gear_sonic/data/assets/"
            "robot_description/meshes/g1/pelvis.STL"
        ),
        (
            "motius/trainers/protomotions/vendor/protomotions/data/assets/"
            "mesh/G1/pelvis.stl"
        ),
        (
            "motius/trainers/protomotions/vendor/protomotions/data/assets/"
            "usd/g1_holo_compat/configuration/g1_holo_compat_robot.usd"
        ),
    ],
)
def test_public_g1_training_assets_are_materialized(relative_path):
    _assert_materialized_asset(relative_path)


def test_sonic_training_source_and_defaults_are_repo_local():
    source = ROOT / "motius/trainers/sonic/vendor/gear_sonic/train_agent_trl.py"
    assert source.is_file()
    assert len(SonicTrainer.upstream_commit) == 40
    argv = sonic_argv(["num_envs=16"])
    assert "+exp=manager/universal_token/all_modes/sonic_release" in argv
    assert f"base_dir={SONIC_OUTPUT_ROOT}" in argv


def test_sonic_export_selects_one_matching_encoder_decoder_pair(tmp_path):
    exported = tmp_path / "exported"
    exported.mkdir()
    old_encoder = exported / "model_step_000003_encoder.onnx"
    old_decoder = exported / "model_step_000003_decoder.onnx"
    new_encoder = exported / "model_step_000006_encoder.onnx"
    new_decoder = exported / "model_step_000006_decoder.onnx"
    for path in (old_encoder, old_decoder, new_encoder, new_decoder):
        path.write_bytes(b"onnx")
    old_encoder.touch()
    old_decoder.touch()
    new_encoder.touch()
    new_decoder.touch()

    assert _exported_onnx_pair(tmp_path) == (new_encoder, new_decoder)


def test_sonic_launcher_restores_process_context(tmp_path, monkeypatch):
    original_argv = ["embedding-process", "--preserve"]
    observed = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setattr(
        sonic_train_module.runpy,
        "run_module",
        lambda *args, **kwargs: observed.update(
            cwd=Path.cwd(),
            argv=tuple(sys.argv),
        ),
    )

    sonic_train_module.main([])

    assert observed["cwd"] == sonic_train_module.VENDOR_ROOT
    assert observed["argv"][0] == "gear_sonic.train_agent_trl"
    assert Path.cwd() == tmp_path
    assert sys.argv is original_argv


def test_protomotions_training_source_and_defaults_are_repo_local(tmp_path, monkeypatch):
    source = ROOT / "motius/trainers/protomotions/vendor/protomotions/train_agent.py"
    exporter = (
        ROOT
        / "motius/trainers/protomotions/vendor/deployment/export_bm_tracker_onnx.py"
    )
    assert source.is_file()
    assert exporter.is_file()
    assert DEFAULT_EXPERIMENT.is_file()
    assert len(ProtoMotionsTrainer.upstream_commit) == 40
    monkeypatch.chdir(tmp_path)
    argv = protomotions_argv(["--motion-file", "motions.pt"])
    assert str((tmp_path / "motions.pt").resolve()) in argv
    assert str(DEFAULT_EXPERIMENT) in argv


def test_protomotions_launcher_restores_process_context(tmp_path, monkeypatch):
    original_argv = ["embedding-process", "--preserve"]
    observed = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", original_argv)
    monkeypatch.setenv("MOTIUS_PROTOMOTIONS_OUTPUT_ROOT", "existing-output")
    monkeypatch.setattr(
        protomotions_train_module.runpy,
        "run_module",
        lambda *args, **kwargs: observed.update(
            cwd=Path.cwd(),
            argv=tuple(sys.argv),
            output_root=os.environ["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"],
        ),
    )

    protomotions_train_module.main([])

    assert observed["cwd"] == protomotions_train_module.VENDOR_ROOT
    assert observed["argv"][0] == "protomotions.train_agent"
    assert observed["output_root"] == str(protomotions_train_module.OUTPUT_ROOT)
    assert Path.cwd() == tmp_path
    assert sys.argv is original_argv
    assert os.environ["MOTIUS_PROTOMOTIONS_OUTPUT_ROOT"] == "existing-output"


def test_beyondmimic_training_source_and_local_motion_are_repo_local(tmp_path):
    vendor = ROOT / "motius/trainers/beyondmimic/vendor"
    assert (vendor / "scripts/rsl_rl/train.py").is_file()
    assert (
        vendor
        / "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"
    ).is_file()
    assert (vendor / "UPSTREAM_COMMIT").read_text().strip() == (
        BeyondMimicTrainer.upstream_commit
    )
    motion = tmp_path / "motion.npz"
    motion.touch()
    launch = beyondmimic_launch(
        [
            "--motion_file",
            str(motion),
            "--output-root",
            "outputs/training/beyondmimic",
            "--asset-root",
            "checkpoints/robots",
        ],
        cwd=tmp_path,
    )
    assert launch.argv[:2] == ("--task", "Tracking-Flat-G1-v0")
    assert str(motion) in launch.argv
    assert launch.output_root == (
        tmp_path / "outputs/training/beyondmimic"
    ).resolve()
    assert launch.asset_root == (tmp_path / "checkpoints/robots").resolve()
    assert launch.isaac_asset_dir == (
        tmp_path
        / "outputs/training/beyondmimic"
        / "isaac_assets/g1"
    ).resolve()
    tracking_cfg = (
        vendor
        / "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
    ).read_text()
    assert "class LocalFlatTerrainImporter" in tracking_cfg
    assert "class_type=LocalFlatTerrainImporter" in tracking_cfg
    assert "(512.0, 512.0)" in tracking_cfg
    assert "2.0e6" not in tracking_cfg
    train_source = (vendor / "scripts/rsl_rl/train.py").read_text()
    assert "env_cfg.wait_for_textures = False" in train_source
    assert "env_cfg.commands.motion.debug_vis = False" in train_source
    assert "env_cfg.scene.contact_forces.debug_vis = False" in train_source
    play_source = (vendor / "scripts/rsl_rl/play.py").read_text()
    assert "env_cfg.wait_for_textures = False" in play_source
    assert "env_cfg.commands.motion.debug_vis = False" in play_source
    assert "env_cfg.scene.contact_forces.debug_vis = False" in play_source
    assert "env_cfg.commands.motion.motion_file = str(motion_file)" in play_source


def test_beyondmimic_auto_resume_is_scoped_to_output_root(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.touch()
    run = tmp_path / "out/logs/rsl_rl/g1_flat/2026-07-29_lafan"
    run.mkdir(parents=True)
    checkpoint = run / "model_1500.pt"
    checkpoint.touch()
    launch = beyondmimic_launch(
        [
            "--motion_file",
            str(motion),
            "--output-root",
            str(tmp_path / "out"),
            "--auto-resume",
        ],
        cwd=tmp_path,
    )
    assert launch.resumed_from == checkpoint
    assert ("--resume", "True") == launch.argv[-6:-4]
    assert launch.argv[-4:] == (
        "--load_run",
        run.name,
        "--checkpoint",
        checkpoint.name,
    )


def test_beyondmimic_cli_applies_experiment_and_save_interval():
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    parsed = parser.parse_args(
        [
            "--experiment_name",
            "reproduction",
            "--save_interval",
            "1",
        ]
    )
    config = SimpleNamespace(
        seed=0,
        resume=False,
        load_run=None,
        load_checkpoint=None,
        run_name="",
        experiment_name="g1_flat",
        save_interval=500,
        logger="tensorboard",
    )

    updated = cli_args.update_rsl_rl_cfg(config, parsed)

    assert updated.experiment_name == "reproduction"
    assert updated.save_interval == 1


def test_beyondmimic_prepare_and_export_paths_are_absolute(tmp_path):
    prepare_argv, asset_root = build_prepare_argv(
        [
            "--input_file",
            "input.csv",
            "--output_file",
            "outputs/motion.npz",
            "--asset-root",
            "checkpoints/robots",
        ],
        cwd=tmp_path,
    )
    assert str((tmp_path / "input.csv").resolve()) in prepare_argv
    assert str((tmp_path / "outputs/motion.npz").resolve()) in prepare_argv
    assert asset_root == (tmp_path / "checkpoints/robots").resolve()

    export_argv, output_root, export_assets, isaac_asset_dir = (
        build_export_argv(
        [
            "--motion_file",
            "outputs/motion.npz",
            "--load_run",
            "run",
            "--checkpoint",
            "model_500.pt",
        ],
        cwd=tmp_path,
        )
    )
    assert export_argv[:2] == ("--task", "Tracking-Flat-G1-v0")
    assert export_argv[-1] == "--export_only"
    assert output_root == (
        tmp_path / "outputs/training/beyondmimic"
    ).resolve()
    assert export_assets == (tmp_path / "checkpoints/robots").resolve()
    assert isaac_asset_dir == (
        output_root / "isaac_assets/g1"
    ).resolve()


def test_beyondmimic_asset_extractor_only_writes_g1_files(tmp_path):
    archive = tmp_path / "assets.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in (
            ("unitree_description/urdf/g1/main.urdf", b"<robot/>"),
            ("unitree_description/meshes/g1/pelvis.STL", b"solid"),
            ("unitree_description/meshes/go2/base.obj", b"skip"),
            ("../escape.txt", b"skip"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    output = tmp_path / "checkpoints/robots"
    assert extract_assets(archive, output) == 2
    assert (output / "unitree_description/urdf/g1/main.urdf").is_file()
    assert not (output / "unitree_description/meshes/go2/base.obj").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert _is_selected("unitree_description/meshes/g1/head_link.STL")
    assert not _is_selected("../../escape")


@pytest.mark.parametrize(
    ("config", "method_env", "expected_module"),
    [
        (
            "sonic_g1_bones_seed.yaml",
            {"G1_MOTION_DIR": "/motions/g1", "SMPL_MOTION_DIR": "/motions/smpl"},
            "motius.trainers.sonic.train",
        ),
        (
            "protomotions_g1_bones_seed.yaml",
            {"G1_MOTION_FILE": "/motions/g1.pt"},
            "motius.trainers.protomotions.train",
        ),
        (
            "beyondmimic_g1_lafan1.yaml",
            {"BEYONDMIMIC_MOTION_FILE": "/motions/lafan1_g1.npz"},
            "motius.trainers.beyondmimic.train",
        ),
    ],
)
def test_tracking_training_configs_build_commands(
    config, method_env, expected_module, monkeypatch
):
    for key, value in method_env.items():
        monkeypatch.setenv(key, value)
    command = load_command(
        ROOT / "configs/motion_tracking" / config,
        num_processes=1,
    )
    assert command[1:3] == ["-m", expected_module]
