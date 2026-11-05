"""Regression checks for native runtime ownership and integration boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("warp")
from motius.pipelines.motionbricks.runtime_engine import (
    MotionBricksEngine,
    _fp32_inference,
)
from motius.pipelines.motionbricks.scene_states import _resolve_clip_path


def test_fp32_scope_preserves_cudnn_and_restores_settings_after_failure():
    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="inference failed"), _fp32_inference():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.backends.cudnn.enabled == original[2]
            raise RuntimeError("inference failed")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.enabled == original[2]
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.enabled,
        ) = original


def test_interaction_clip_resolves_against_scene_assets(tmp_path):
    character = SimpleNamespace(asset_pool=SimpleNamespace(assets_root=tmp_path))
    assert (
        _resolve_clip_path(character, Path("motions/sitting.bvh"))
        == tmp_path / "motions/sitting.bvh"
    )
    absolute = tmp_path / "elsewhere.bvh"
    assert _resolve_clip_path(character, absolute) == absolute


def test_runtime_requires_cuda_without_importing_upstream():
    with pytest.raises(ValueError, match="CUDA bundle"):
        MotionBricksEngine(
            SimpleNamespace(bundle=SimpleNamespace(device=torch.device("cpu")))
        )
