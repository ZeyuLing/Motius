"""Motius bundle for the released OmniControl HumanML3D checkpoint."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.clip_artifact import artifact_clip_load
from motius.registry import MODEL_BUNDLES

from .network import CMDM
from motius.configuration.omnicontrol import normalize_config


_ASSETS = Path(__file__).resolve().parents[2] / "motion" / "assets" / "omnicontrol"


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


def build_model(config=None):
    cfg = normalize_config(config)
    return CMDM(
        modeltype="",
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=cfg["latent_dim"],
        ff_size=1024,
        num_layers=cfg["layers"],
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep="hml_vec",
        cond_mode=cfg["cond_mode"],
        cond_mask_prob=cfg["cond_mask_prob"],
        action_emb="tensor",
        arch=cfg["arch"],
        emb_trans_dec=cfg["emb_trans_dec"],
        clip_version="ViT-B/32",
        dataset="humanml",
    )


def load_model_wo_clip(model, state_dict):
    state = dict(state_dict)
    for key in list(state):
        if "sequence_pos_encoder.pe" in key:
            state.pop(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key
        for key in missing
        if not key.startswith("clip_model.") and "sequence_pos_encoder.pe" not in key
    ]
    invalid_unexpected = [key for key in unexpected if not key.startswith("rot2xyz.")]
    if invalid_missing:
        raise RuntimeError(
            f"missing OmniControl checkpoint keys: {invalid_missing[:8]}"
        )
    if invalid_unexpected:
        raise RuntimeError(
            f"unexpected OmniControl checkpoint keys: {invalid_unexpected[:8]}"
        )


@MODEL_BUNDLES.register_module()
class OmniControlBundle(ModelBundle):
    """Text-conditioned arbitrary-joint 3D position control on HML263."""

    SUPPORTED_TASKS = (
        "text_to_motion",
        "temporal_motion_completion",
        "kinematic_motion_control",
    )

    def __init__(
        self,
        config: Optional[dict] = None,
        weights_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        raw_mean_path: Optional[str] = None,
        raw_std_path: Optional[str] = None,
        guidance_param: float = 2.5,
        respacing: str = "",
        device: Optional[str | torch.device] = None,
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

        clip_context = (
            artifact_clip_load(clip_weights_path, expected_name="ViT-B/32")
            if clip_weights_path
            else nullcontext()
        )
        with clip_context:
            self.net = build_model(self.config)
        if weights_path:
            load_model_wo_clip(self.net, _load_weights(Path(weights_path)))
        if device is not None:
            self.to(torch.device(device))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = _resolve_artifact(str(pretrained_model_name_or_path))
        if path.is_file():
            return cls(weights_path=str(path), **kwargs)
        metadata_path = path / "omnicontrol_config.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        weights = path / metadata.get("weights", "model.safetensors")
        if not weights.exists():
            candidates = [path / "model.pt"] if (path / "model.pt").is_file() else []
            if not candidates:
                raise FileNotFoundError(f"No OmniControl weights found under {path}")
            weights = candidates[0]

        def optional(name):
            value = path / name
            return str(value) if value.exists() else None

        return cls(
            config=metadata.get("config"),
            weights_path=str(weights),
            clip_weights_path=kwargs.pop(
                "clip_weights_path",
                optional(metadata.get("clip_weights", "clip.safetensors")),
            ),
            mean_path=optional("Mean.npy"),
            std_path=optional("Std.npy"),
            raw_mean_path=optional("Mean_raw.npy"),
            raw_std_path=optional("Std_raw.npy"),
            guidance_param=kwargs.pop(
                "guidance_param", metadata.get("guidance_param", 2.5)
            ),
            respacing=kwargs.pop("respacing", metadata.get("respacing", "")),
            **kwargs,
        )

    def save_pretrained(
        self, save_directory: str, safe_serialization: bool = True, **kwargs
    ):
        from motius.utils.gaussian_motion_checkpoint import (
            save_gaussian_motion_artifact,
        )

        return save_gaussian_motion_artifact(
            self,
            save_directory,
            method="omnicontrol",
            safe_serialization=safe_serialization,
        )

    @property
    def device(self):
        return self.mean.device

    def denoise(self, motion, timesteps, **model_kwargs):
        return self.net(motion, timesteps, **model_kwargs)

    forward = denoise


__all__ = ["OmniControlBundle"]
