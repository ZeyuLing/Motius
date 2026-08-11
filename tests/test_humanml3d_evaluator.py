from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

import motius.models.humanml3d_evaluator.bundle as bundle_module


class _TinyEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)


def test_bundle_loads_exported_safetensors_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_module, "MovementConvEncoder", _TinyEncoder)
    monkeypatch.setattr(bundle_module, "TextEncoderBiGRUCo", _TinyEncoder)
    monkeypatch.setattr(bundle_module, "MotionEncoderBiGRUCo", _TinyEncoder)
    monkeypatch.setattr(
        bundle_module, "HumanML3DWordVectorizer", lambda path: SimpleNamespace()
    )
    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(load=lambda name: SimpleNamespace())
    )

    (tmp_path / "glove").mkdir()
    (tmp_path / "stats").mkdir()
    np.save(tmp_path / "stats" / "mean.npy", np.zeros(263, dtype=np.float32))
    np.save(tmp_path / "stats" / "std.npy", np.ones(263, dtype=np.float32))
    expected = _TinyEncoder().state_dict()
    weights = {
        f"{module_name}.{key}": value
        for module_name in ("movement_encoder", "text_encoder", "motion_encoder")
        for key, value in expected.items()
    }
    weights = {key: value.clone() for key, value in weights.items()}
    save_file(weights, tmp_path / "model.safetensors")

    bundle = bundle_module.HumanML3DMatchingBundle(
        artifact_dir=str(tmp_path), device="cpu"
    )

    assert torch.equal(
        bundle.movement_encoder.projection.weight,
        expected["projection.weight"],
    )
    assert torch.equal(bundle.mean, torch.zeros(263))
    assert torch.equal(bundle.std, torch.ones(263))
