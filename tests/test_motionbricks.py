"""Contract tests for the MotionBricks wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from motius.models.motionbricks import MotionBricksBundle
from motius.motion.representation import get_spec
from motius.pipelines.motionbricks import MotionBricksPipeline


def test_motionbricks_representation_specs():
    assert get_spec("motionbricks_g1_414").dim == 414
    assert get_spec("motionbricks_g1_413").dim == 413
    assert get_spec("motionbricks_g1_418").dim == 418
    assert get_spec("MotionBricks-G1-414").name == "motionbricks_g1_414"


def test_motionbricks_assets_are_packaged():
    bundle = MotionBricksBundle(load_model=False)
    assert bundle.asset_root.joinpath("skeletons/g1/g1.xml").is_file()
    assert bundle.asset_root.joinpath("skeletons/g1/scene_29dof.xml").is_file()


def test_motionbricks_missing_checkpoint_error(tmp_path: Path):
    bundle = MotionBricksBundle(checkpoint_dir=tmp_path, load_model=False)
    with pytest.raises(FileNotFoundError, match="missing:"):
        bundle.validate_checkpoints()


def test_motionbricks_lfs_pointer_error(tmp_path: Path):
    layout = MotionBricksBundle(checkpoint_dir=tmp_path, load_model=False).required_checkpoint_files
    pointer = "version https://git-lfs.github.com/spec/v1\n"
    for path in (layout.clip, layout.vqvae, layout.pose, layout.root_model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pointer)

    with pytest.raises(FileNotFoundError, match="git-lfs pointers"):
        MotionBricksBundle(checkpoint_dir=tmp_path, load_model=False).validate_checkpoints()


def test_motionbricks_pipeline_from_pretrained_without_loading(tmp_path: Path):
    pipe = MotionBricksPipeline.from_pretrained(
        str(tmp_path),
        bundle_kwargs={"load_model": False, "device": "cpu"},
    )
    assert pipe.representation == "motionbricks_g1_414"
    assert pipe.fps == 30


def test_motionbricks_vendored_namespace_imports_light_dataset():
    from motius.models.motionbricks.network.data.synthetic_dataset import (
        SyntheticMotionDataset,
        collate_batch,
    )

    dataset = SyntheticMotionDataset(feat_dim=418, num_samples=2, min_frames=3, max_frames=4)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["motion"].shape[0] == 2
    assert batch["motion"].shape[-1] == 418
    assert batch["motion_pad_mask"].dtype == torch.bool
