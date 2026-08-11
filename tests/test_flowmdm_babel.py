from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

from motius.models.flowmdm.bundle import _maybe_download_hub
from motius.models.flowmdm.network.model.cfg_sampler import ClassifierFreeSampleModel
from motius.pipelines.flowmdm import FlowMDMPipeline


def _pipe(dataset):
    pipe = object.__new__(FlowMDMPipeline)
    pipe.bundle = SimpleNamespace(dataset=dataset)
    return pipe


def test_babel_lengths_preserve_official_protocol_values():
    pipe = _pipe("babel")
    assert pipe.resolve_length(31) == 31
    assert pipe.resolve_length(42) == 42
    assert pipe.resolve_length(120) == 120


@pytest.mark.parametrize("length", [0, 29, 201])
def test_babel_lengths_reject_outside_protocol(length):
    with pytest.raises(ValueError, match=r"\[30, 200\]"):
        _pipe("babel").resolve_length(length)


def test_babel_sequential_lengths_preserve_long_official_actions():
    pipe = _pipe("babel")
    assert pipe.resolve_sequential_length(201) == 201
    assert pipe.resolve_sequential_length(1657) == 1657
    with pytest.raises(ValueError, match=">= 30"):
        pipe.resolve_sequential_length(29)


def test_humanml_length_policy_is_unchanged():
    pipe = _pipe("humanml")
    assert pipe.resolve_length(31) == 40
    assert pipe.resolve_length(123) == 120
    assert pipe.resolve_length(220) == 196


def test_embedding_cache_is_cleared_on_wrapped_model():
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_mask_prob = 0.1
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.register_buffer("emb_hash", torch.tensor(123))
            self.register_buffer("emb_forcemask_hash", torch.tensor(456))

    inner = Inner()
    wrapper = ClassifierFreeSampleModel(inner)
    FlowMDMPipeline._clear_embedding_cache(SimpleNamespace(model=wrapper))

    assert inner.emb_hash.item() == -1
    assert inner.emb_forcemask_hash.item() == -1
    assert "emb_hash" not in vars(wrapper)
    assert "emb_forcemask_hash" not in vars(wrapper)


def test_flowmdm_hub_resolution_returns_downloaded_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(snapshot),
    )
    assert _maybe_download_hub("org/flowmdm", tmp_path / "missing") == snapshot
