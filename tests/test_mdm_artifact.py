import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from motius.models.mdm.bundle import MDMBundle


class _FakeMDM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = torch.nn.Linear(3, 2)
        self.clip_model = torch.nn.Linear(4, 3)
        self.clip_version = "ViT-B/32"


def _bundle() -> MDMBundle:
    bundle = MDMBundle.__new__(MDMBundle)
    torch.nn.Module.__init__(bundle)
    bundle.net = _FakeMDM()
    bundle.guidance_param = 2.5
    bundle._args = Namespace(dataset="humanml", latent_dim=512)
    bundle.register_buffer("mean", torch.zeros(263))
    bundle.register_buffer("std", torch.ones(263))
    return bundle


def test_mdm_export_bundles_clip_and_direct_load_manifest(tmp_path: Path):
    bundle = _bundle()
    bundle.save_pretrained(str(tmp_path))

    model_state = load_file(str(tmp_path / "model.safetensors"))
    clip_state = load_file(str(tmp_path / "clip.safetensors"))
    assert "denoiser.weight" in model_state
    assert not any(key.startswith("clip_model.") for key in model_state)
    assert set(clip_state) == {"bias", "weight"}

    config = json.loads((tmp_path / "mdm_config.json").read_text())
    assert config["format"] == "motius-mdm-v2"
    assert config["components"]["clip"]["stored_in_artifact"] is True

    model_index = json.loads((tmp_path / "model_index.json").read_text())
    assert model_index["pipeline_class"] == (
        "motius.pipelines.mdm.pipeline.MDMPipeline"
    )
    assert "clip.safetensors" in model_index["required_files"]
    assert "infer_text_to_motion" in model_index["api"]["task_methods"]


def test_mdm_artifact_without_clip_fails_closed(tmp_path: Path):
    (tmp_path / "mdm_config.json").write_text(
        json.dumps(
            {
                "guidance_param": 2.5,
                "config": {"dataset": "humanml"},
                "components": {
                    "clip": {
                        "stored_in_artifact": False,
                        "path": "clip.safetensors",
                    }
                },
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="bundled CLIP"):
        MDMBundle.from_pretrained(str(tmp_path))
