from pathlib import Path

import pytest
import torch

from motius.models.clip_artifact import artifact_clip_load, load_openai_clip


def _integration_weights() -> Path:
    weights = Path(
        "outputs/hf_checkpoint_artifacts/mdm_humanml3d_v2/clip.safetensors"
    )
    if not weights.is_file():
        pytest.skip("artifact CLIP integration weights are not available")
    return weights


def test_load_openai_clip_from_safetensors_without_clip_download(monkeypatch):
    weights = _integration_weights()
    monkeypatch.setattr(
        "clip.clip._download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLIP network download attempted")
        ),
    )

    model = load_openai_clip(weights, device="cpu")
    assert model.visual.input_resolution == 224
    assert model.token_embedding.weight.shape == (49408, 512)
    assert next(model.parameters()).dtype == torch.float32


def test_artifact_clip_load_patches_only_expected_model(monkeypatch):
    weights = _integration_weights()
    monkeypatch.setattr(
        "clip.clip._download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLIP network download attempted")
        ),
    )

    import clip

    with artifact_clip_load(weights, expected_name="ViT-B/32"):
        model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    assert preprocess is None
    assert model.token_embedding.weight.shape == (49408, 512)
