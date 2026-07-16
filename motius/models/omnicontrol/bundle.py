"""Motius bundle for the released OmniControl HumanML3D checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import build_diffusion, build_model, load_model_wo_clip, normalize_config


_ASSETS = Path(__file__).resolve().parent / "assets"


def _resolve_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _load_weights(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "model_avg" in value:
        return value["model_avg"]
    if isinstance(value, dict) and "model" in value:
        return value["model"]
    return value


@MODEL_BUNDLES.register_module()
class OmniControlBundle(ModelBundle):
    """Text-conditioned arbitrary-joint 3D position control on HML263."""

    def __init__(
        self,
        config: Optional[dict] = None,
        weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        raw_mean_path: Optional[str] = None,
        raw_std_path: Optional[str] = None,
        guidance_param: float = 2.5,
        respacing: str = "",
    ):
        super().__init__()
        self.config = normalize_config(config)
        self.guidance_param = float(guidance_param)
        self.respacing = str(respacing)

        arrays = {}
        for name, supplied, default in (
            ("mean", mean_path, _ASSETS / "Mean.npy"),
            ("std", std_path, _ASSETS / "Std.npy"),
            ("raw_mean", raw_mean_path, _ASSETS / "Mean_raw.npy"),
            ("raw_std", raw_std_path, _ASSETS / "Std_raw.npy"),
        ):
            arrays[name] = np.load(str(supplied or default)).astype(np.float32)
        if arrays["mean"].shape != (263,) or arrays["std"].shape != (263,):
            raise ValueError("OmniControl motion statistics must be 263-dimensional")
        if arrays["raw_mean"].shape != (66,) or arrays["raw_std"].shape != (66,):
            raise ValueError("OmniControl joint statistics must be 66-dimensional")
        for name, value in arrays.items():
            self.register_buffer(name, torch.from_numpy(value), persistent=False)

        self.net = build_model(self.config)
        if weights_path:
            load_model_wo_clip(self.net, _load_weights(Path(weights_path)))
        self.diffusion = build_diffusion(
            self.config,
            mean=arrays["mean"],
            std=arrays["std"],
            raw_mean=arrays["raw_mean"],
            raw_std=arrays["raw_std"],
            respacing=self.respacing,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = _resolve_artifact(str(pretrained_model_name_or_path))
        if path.is_file():
            return cls(weights_path=str(path), **kwargs)
        metadata_path = path / "omnicontrol_config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        weights = path / "model.safetensors"
        if not weights.exists():
            candidates = sorted(path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No OmniControl weights found under {path}")
            weights = candidates[0]
        def optional(name):
            value = path / name
            return str(value) if value.exists() else None
        return cls(
            config=metadata.get("config"),
            weights_path=str(weights),
            mean_path=optional("Mean.npy"),
            std_path=optional("Std.npy"),
            raw_mean_path=optional("Mean_raw.npy"),
            raw_std_path=optional("Std_raw.npy"),
            guidance_param=kwargs.pop("guidance_param", metadata.get("guidance_param", 2.5)),
            respacing=kwargs.pop("respacing", metadata.get("respacing", "")),
            **kwargs,
        )

    def normalize(self, motion):
        return (motion - self.mean) / self.std

    def denormalize(self, motion):
        return motion * self.std + self.mean

    @property
    def device(self):
        return self.mean.device

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use OmniControlPipeline for inference")


__all__ = ["OmniControlBundle"]
