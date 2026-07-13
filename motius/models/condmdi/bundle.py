"""CondMDI model bundle.

The released artifact stores only the CondMDI UNet weights and normalization
statistics. OpenAI CLIP remains the frozen text encoder, matching the official
implementation; no source checkout or dataset files are required at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import build_diffusion, build_model, load_model_wo_clip, normalize_config


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


@MODEL_BUNDLES.register_module()
class CondMDIBundle(ModelBundle):
    """Flexible motion in-betweening bundle for absolute-root HML263."""

    def __init__(
        self,
        config: Optional[dict] = None,
        weights_path: Optional[str] = None,
        mean_abs_path: Optional[str] = None,
        std_abs_path: Optional[str] = None,
        guidance_param: float = 2.5,
        respacing: str = "",
    ):
        super().__init__()
        self.config = normalize_config(config)
        self.guidance_param = float(guidance_param)
        self.respacing = str(respacing)
        self.net = build_model(self.config)
        if weights_path:
            load_model_wo_clip(self.net, _load_weights(Path(weights_path)))
        self.diffusion = build_diffusion(self.config, respacing=self.respacing)

        if not mean_abs_path or not std_abs_path:
            raise ValueError("CondMDI requires artifact-local Mean_abs_3d.npy and Std_abs_3d.npy")
        mean = np.load(mean_abs_path).astype(np.float32)
        std = np.load(std_abs_path).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(f"invalid CondMDI stats shapes: mean={mean.shape}, std={std.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("CondMDI normalization statistics must be finite with positive std")
        self.register_buffer("mean_abs", torch.from_numpy(mean), persistent=False)
        self.register_buffer("std_abs", torch.from_numpy(std), persistent=False)

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
        return cls(
            config=metadata["config"],
            weights_path=str(weights),
            mean_abs_path=str(path / "Mean_abs_3d.npy"),
            std_abs_path=str(path / "Std_abs_3d.npy"),
            guidance_param=kwargs.pop("guidance_param", metadata.get("guidance_param", 2.5)),
            respacing=kwargs.pop("respacing", metadata.get("respacing", "")),
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_type": "condmdi",
            "library_name": "motius",
            "tasks": ["text-to-motion", "motion-control", "motion-in-betweening"],
            "native_motion_representation": "HumanML3D-263 absolute-root",
            "guidance_param": self.guidance_param,
            "respacing": self.respacing,
            "config": self.config,
        }
        (save_dir / "condmdi_config.json").write_text(json.dumps(metadata, indent=2) + "\n")

        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.net.state_dict().items()
            if not key.startswith("clip_model.") and "sequence_pos_encoder.pe" not in key
        }
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state, str(save_dir / "model.safetensors"))
        else:
            torch.save(state, save_dir / "model.pt")
        np.save(save_dir / "Mean_abs_3d.npy", self.mean_abs.detach().cpu().numpy())
        np.save(save_dir / "Std_abs_3d.npy", self.std_abs.detach().cpu().numpy())
        return str(save_dir)

    def normalize_absolute(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean_abs) / self.std_abs

    def denormalize_absolute(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std_abs + self.mean_abs

    @property
    def device(self) -> torch.device:
        return self.mean_abs.device

    def forward(self, *args, **kwargs):  # pragma: no cover - pipeline owns sampling
        raise NotImplementedError("Use CondMDIPipeline for inference")


__all__ = ["CondMDIBundle"]
