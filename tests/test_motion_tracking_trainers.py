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
from motius.trainers.protomotions.train import (
    ProtoMotionsTrainer,
)
from motius.trainers.sonic.train import (
    OUTPUT_ROOT as SONIC_OUTPUT_ROOT,
    SonicTrainer,
)
from tools.train_motion_tracking import load_command
from tools.download_beyondmimic_assets import _is_selected, extract_assets


ROOT = Path(__file__).resolve().parents[1]


def _assert_materialized_asset(relative_path: str) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        pytest.skip("Download the public G1 assets with tools/download_sonic_assets.py")
    prefix = path.read_bytes()[:128]
    assert not prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
    assert path.stat().st_size > 256


def test_public_g1_training_assets_are_materialized():
    _assert_materialized_asset("checkpoints/robots/sonic/robot_description/meshes/g1/pelvis.STL")


def test_sonic_public_trainer_uses_framework_objective():
    from motius.trainers.base_trainer import BaseTrainer
    from motius.trainers.sonic import SONICOnlinePPOTrainer
    assert SonicTrainer is SONICOnlinePPOTrainer
    assert issubclass(SonicTrainer, BaseTrainer)
    assert len(SonicTrainer.upstream_commit) == 40
    assert not (ROOT / "motius/trainers/sonic/vendor").exists()


def test_sonic_cli_accepts_native_configuration(tmp_path):
    args = sonic_train_module.build_parser().parse_args([
        "--checkpoint", str(tmp_path / "last.pt"),
        "--motion-file", str(tmp_path / "robot.pkl"),
        "--smpl-motion-file", str(tmp_path / "smpl.pkl"),
        "--max-iterations", "2", "--auto-resume",
    ])
    assert args.checkpoint == tmp_path / "last.pt"
    assert args.motion_file == tmp_path / "robot.pkl"
    assert args.smpl_motion_file == tmp_path / "smpl.pkl"
    assert args.max_iterations == 2 and args.auto_resume


def test_protomotions_public_trainer_uses_framework_objective():
    from motius.trainers.base_trainer import BaseTrainer
    from motius.trainers.protomotions import ProtoMotionsPPOTrainer

    assert ProtoMotionsTrainer is ProtoMotionsPPOTrainer
    assert issubclass(ProtoMotionsTrainer, BaseTrainer)
    assert not (ROOT / "motius/trainers/protomotions/vendor").exists()


def test_protomotions_cli_accepts_native_configuration(tmp_path):
    args = protomotions_train_module.build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "last.ckpt"),
            "--motion-file",
            str(tmp_path / "reference.motion"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--max-iterations",
            "2",
            "--auto-resume",
            "--headless",
        ]
    )
    assert args.checkpoint == tmp_path / "last.ckpt"
    assert args.motion_file == tmp_path / "reference.motion"
    assert args.max_iterations == 2 and args.auto_resume


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
            {"G1_MOTION_DIR": "/motions/g1", "SMPL_MOTION_DIR": "/motions/smpl", "SONIC_CHECKPOINT": "/weights/last.pt"},
            "motius.trainers.sonic.train",
        ),
        (
            "protomotions_g1_bones_seed.yaml",
            {
                "G1_MOTION_FILE": "/motions/g1.pt",
                "PROTOMOTIONS_CHECKPOINT": "/weights/last.ckpt",
            },
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
