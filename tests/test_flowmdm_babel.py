from types import SimpleNamespace

import huggingface_hub
import pytest

from motius.models.flowmdm.bundle import _maybe_download_hub
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


def test_humanml_length_policy_is_unchanged():
    pipe = _pipe("humanml")
    assert pipe.resolve_length(31) == 40
    assert pipe.resolve_length(123) == 120
    assert pipe.resolve_length(220) == 196


def test_flowmdm_hub_resolution_returns_downloaded_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: str(snapshot),
    )
    assert _maybe_download_hub("org/flowmdm", tmp_path / "missing") == snapshot
