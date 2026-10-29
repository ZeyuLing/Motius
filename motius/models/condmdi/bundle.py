"""CondMDI model bundle with artifact-local OpenAI CLIP weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.clip_artifact import artifact_clip_load
from motius.registry import MODEL_BUNDLES

from .network import MDM_UNET
from motius.configuration.condmdi import normalize_config


def _resolve_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _load_weights(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_avg" in checkpoint:
        return checkpoint["model_avg"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_model(config=None):
    cfg = normalize_config(config)
    if cfg["dataset"] != "humanml":
        raise ValueError("The released CondMDI artifact supports HumanML3D only")
    return MDM_UNET(
        modeltype="",
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=cfg["latent_dim"],
        ff_size=cfg["ff_size"],
        num_layers=cfg["layers"],
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep="hml_vec",
        cond_mode="text",
        cond_mask_prob=cfg["cond_mask_prob"],
        action_emb="tensor",
        arch=cfg["arch"],
        emb_trans_dec=cfg["emb_trans_dec"],
        clip_version="ViT-B/32",
        dataset="humanml",
        dim_mults=cfg["dim_mults"],
        adagn=cfg["unet_adagn"],
        zero=cfg["unet_zero"],
        unet_out_mult=cfg["out_mult"],
        xz_only=cfg["xz_only"],
        keyframe_conditioned=cfg["keyframe_conditioned"],
        keyframe_selection_scheme=cfg["keyframe_selection_scheme"],
        zero_keyframe_loss=cfg["zero_keyframe_loss"],
    )


def load_model_wo_clip(model, state_dict):
    state = dict(state_dict)
    state.pop("sequence_pos_encoder.pe", None)
    state.pop("embed_timestep.sequence_pos_encoder.pe", None)
    state = {k: v for k, v in state.items() if not k.startswith("clip_model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected CondMDI checkpoint keys: {unexpected[:8]}")
    invalid_missing = [
        k
        for k in missing
        if not k.startswith("clip_model.") and "sequence_pos_encoder" not in k
    ]
    if invalid_missing:
        raise RuntimeError(f"missing CondMDI checkpoint keys: {invalid_missing[:8]}")


@MODEL_BUNDLES.register_module()
class CondMDIBundle(ModelBundle):
    """Flexible motion in-betweening bundle for absolute-root HML263."""

    def __init__(
        self,
        config: Optional[dict] = None,
        weights_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        mean_abs_path: Optional[str] = None,
        std_abs_path: Optional[str] = None,
        guidance_param: float = 2.5,
        respacing: str = "",
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.config = normalize_config(config)
        self.guidance_param = float(guidance_param)
        self.respacing = str(respacing)
        if not clip_weights_path:
            raise ValueError("CondMDI requires artifact-local CLIP weights")
        with artifact_clip_load(clip_weights_path, expected_name="ViT-B/32"):
            self.net = build_model(self.config)
        if weights_path:
            load_model_wo_clip(self.net, _load_weights(Path(weights_path)))

        if not mean_abs_path or not std_abs_path:
            raise ValueError(
                "CondMDI requires artifact-local Mean_abs_3d.npy and Std_abs_3d.npy"
            )
        mean = np.load(mean_abs_path).astype(np.float32)
        std = np.load(std_abs_path).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(
                f"invalid CondMDI stats shapes: mean={mean.shape}, std={std.shape}"
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(
                "CondMDI normalization statistics must be finite with positive std"
            )
        self.register_buffer("mean_abs", torch.from_numpy(mean), persistent=False)
        self.register_buffer("std_abs", torch.from_numpy(std), persistent=False)
        if device is not None:
            self.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = _resolve_artifact(str(pretrained_model_name_or_path))
        config_path = path / "condmdi_config.json"
        if not config_path.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        metadata = json.loads(config_path.read_text())
        weights = path / "model.safetensors"
        if not weights.exists():
            weights = path / "model.pt"
        clip_weights = path / "clip.safetensors"
        if not clip_weights.exists():
            clip_weights = path / "clip.pt"
        if not clip_weights.exists():
            raise FileNotFoundError(
                f"The CondMDI artifact is incomplete: missing {path / 'clip.safetensors'}"
            )
        return cls(
            config=metadata["config"],
            weights_path=str(weights),
            clip_weights_path=str(clip_weights),
            mean_abs_path=str(path / "Mean_abs_3d.npy"),
            std_abs_path=str(path / "Std_abs_3d.npy"),
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
            method="condmdi",
            safe_serialization=safe_serialization,
        )

    @property
    def device(self) -> torch.device:
        return self.mean_abs.device

    def denoise(self, motion, timesteps, **model_kwargs):
        return self.net(motion, timesteps, **model_kwargs)

    forward = denoise


__all__ = ["CondMDIBundle"]
