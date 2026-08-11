import inspect

import pytest

from motius.models.dart.bundle import DARTBundle
from motius.models.vimogen.bundle import _resolve_wan_dir


def test_dart_defers_licensed_body_model_loading():
    parameter = inspect.signature(DARTBundle.__init__).parameters["load_dataset"]
    assert parameter.default is False


def test_vimogen_requires_artifact_local_wan_runtime(tmp_path):
    with pytest.raises(FileNotFoundError, match="bundled Wan"):
        _resolve_wan_dir(None, tmp_path, {})


def test_vimogen_accepts_minimal_artifact_local_wan_runtime(tmp_path):
    wan = tmp_path / "wan"
    tokenizer = wan / "google" / "umt5-xxl"
    tokenizer.mkdir(parents=True)
    (wan / "config.json").write_text("{}")
    (wan / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"weights")
    (tokenizer / "tokenizer_config.json").write_text("{}")

    assert _resolve_wan_dir(None, tmp_path, {}) == wan.resolve()
