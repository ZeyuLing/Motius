"""MotionStreamer-272 text-motion evaluator.

This is a plain-PyTorch port of the ACTOR/TEMOS encoders bundled with the
MotionStreamer evaluator. Parameter names intentionally match the public
checkpoint converted by ``tools/export_evaluator_hf.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .evaluator_network import ActorAgnosticEncoder, DistilbertActorAgnosticEncoder


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


@MODEL_BUNDLES.register_module()
class MotionStreamer272Evaluator(ModelBundle):
    """Self-contained MotionStreamer evaluator for raw MotionStreamer-272 data."""

    def __init__(self, artifact_dir: str, *, device: str = "cuda") -> None:
        super().__init__()
        artifact = Path(artifact_dir).expanduser().resolve()
        config = json.loads((artifact / "config.json").read_text())
        preprocessor = json.loads((artifact / "preprocessor_config.json").read_text())
        from safetensors.torch import load_file
        from transformers import AutoConfig

        tokenizer_dir = artifact / preprocessor.get("tokenizer", "tokenizer")
        model_config = AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)
        if config.get("text_attn_implementation") is not None:
            model_config._attn_implementation = config["text_attn_implementation"]
        latent_dim = int(config.get("latent_dim", 256))
        num_layers = int(config.get("num_layers", 4))
        num_heads = int(config.get("num_heads", 4))
        self.text_encoder = DistilbertActorAgnosticEncoder(
            model_config,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.motion_encoder = ActorAgnosticEncoder(
            nfeats=int(config.get("motion_nfeats", 272)),
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=int(preprocessor.get("max_motion_length", 300)),
        )
        for name, dtype in config.get("neural_dtype", {}).items():
            if name in {"text_encoder", "motion_encoder"}:
                getattr(self, name).to(dtype=getattr(torch, dtype))
        state = load_file(artifact / config.get("weights_file", "model.safetensors"))
        self.text_encoder.load_state_dict(
            {
                key.removeprefix("text_encoder."): value
                for key, value in state.items()
                if key.startswith("text_encoder.")
            },
            strict=True,
        )
        self.motion_encoder.load_state_dict(
            {
                key.removeprefix("motion_encoder."): value
                for key, value in state.items()
                if key.startswith("motion_encoder.")
            },
            strict=True,
        )
        mean_path = artifact / preprocessor.get("mean", "stats/mean.npy")
        std_path = artifact / preprocessor.get("std", "stats/std.npy")
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)))
        self.register_buffer("std", torch.from_numpy(np.load(std_path)))
        self.max_motion_length = int(preprocessor.get("max_motion_length", 300))
        self.unit_length = int(preprocessor.get("unit_length", 4))
        self.artifact_dir = artifact
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(resolved_device).eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.preprocessor_config = preprocessor
        self.evaluator_config = config
        self._save_ckpt_modules = ["text_encoder", "motion_encoder"]
        self._frozen_modules = ["text_encoder", "motion_encoder"]

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(str(_download_artifact(pretrained_model_name_or_path)), **kwargs)

    def train(self, mode: bool = True):
        return super().train(False)

    def save_pretrained(self, save_directory, **kwargs):
        from safetensors.torch import save_file
        from motius.configuration.motionstreamer_evaluator import export_resources

        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        state = {
            root + "." + name: value.detach().cpu().contiguous()
            for root in ["text_encoder", "motion_encoder"]
            for name, value in getattr(self, root).state_dict().items()
        }
        save_file(state, str(target / "model.safetensors"))
        (target / "stats").mkdir(exist_ok=True)
        np.save(target / "stats/mean.npy", self.mean.detach().cpu().numpy())
        np.save(target / "stats/std.npy", self.std.detach().cpu().numpy())
        export_resources(self, target)
        return str(target)

    def encode_text_tokens(self, tokens):
        return self.text_encoder(tokens).loc

    def encode_motion_features(self, features, lengths):
        return self.motion_encoder(features, lengths).loc


__all__ = ["MotionStreamer272Evaluator"]
