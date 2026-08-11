import json

import pytest

from motius.models.mld.bundle import MLDBundle
from motius.models.motionlcm.bundle import MotionLCMBundle
from motius.models.motionstreamer.bundle import MotionStreamerBundle


@pytest.mark.parametrize(
    ("bundle_cls", "config_name", "config_key", "component_path"),
    [
        (
            MLDBundle,
            "mld_config.json",
            "text_encoder_name",
            "text_encoder/sentence-t5-large",
        ),
        (
            MotionLCMBundle,
            "motionlcm_config.json",
            "text_encoder_name",
            "text_encoder/sentence-t5-large",
        ),
        (
            MotionStreamerBundle,
            "ms_config.json",
            "text_model_name",
            "text_encoder/sentence-t5-xxl",
        ),
    ],
)
def test_from_pretrained_uses_bundled_text_encoder(
    tmp_path,
    monkeypatch,
    bundle_cls,
    config_name,
    config_key,
    component_path,
):
    config = {
        "config": {},
        config_key: component_path,
        "text_encoder_stored_in_artifact": True,
    }
    (tmp_path / config_name).write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / component_path).mkdir(parents=True)

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(bundle_cls, "__init__", fake_init)
    bundle_cls.from_pretrained(str(tmp_path))

    assert captured[config_key] == str(tmp_path / component_path)


@pytest.mark.parametrize(
    ("bundle_cls", "config_name", "config_key", "component_path"),
    [
        (
            MLDBundle,
            "mld_config.json",
            "text_encoder_name",
            "text_encoder/sentence-t5-large",
        ),
        (
            MotionLCMBundle,
            "motionlcm_config.json",
            "text_encoder_name",
            "text_encoder/sentence-t5-large",
        ),
        (
            MotionStreamerBundle,
            "ms_config.json",
            "text_model_name",
            "text_encoder/sentence-t5-xxl",
        ),
    ],
)
def test_from_pretrained_rejects_missing_bundled_text_encoder(
    tmp_path,
    bundle_cls,
    config_name,
    config_key,
    component_path,
):
    config = {
        "config": {},
        config_key: component_path,
        "text_encoder_stored_in_artifact": True,
    }
    (tmp_path / config_name).write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="artifact is incomplete"):
        bundle_cls.from_pretrained(str(tmp_path))
