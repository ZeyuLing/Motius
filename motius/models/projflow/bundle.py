"""Self-contained ProjFlow model bundle."""

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

from .network.acmdm import ACMDM_models


_ASSETS = Path(__file__).resolve().parent / "assets"


def _resolve_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "ema_acmdm" in value:
        return value["ema_acmdm"]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported ProjFlow checkpoint payload in {path}")
    return value


@MODEL_BUNDLES.register_module()
class ProjFlowBundle(ModelBundle):
    """ACMDM Raw Flow prior plus ProjFlow sampler and artifact-local CLIP."""

    SUPPORTED_TASKS = (
        "temporal_motion_completion",
        "kinematic_motion_control",
        "part_level_motion_control",
    )

    def __init__(
        self,
        *,
        model_id: str = "ACMDM-Raw-Flow-S-PatchSize22",
        weights_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        guidance_scale: float = 2.5,
        num_steps: int = 100,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        if model_id not in ACMDM_models:
            raise ValueError(f"Unsupported ProjFlow ACMDM model: {model_id}")
        self.model_id = str(model_id)
        self.guidance_scale = float(guidance_scale)
        self.num_steps = int(num_steps)

        mean = np.load(mean_path or (_ASSETS / "joints_mean.npy")).astype(np.float32)
        std = np.load(std_path or (_ASSETS / "joints_std.npy")).astype(np.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("ProjFlow joint statistics must each have shape (3,)")
        if np.any(std <= 0):
            raise ValueError("ProjFlow joint standard deviations must be positive")
        self.register_buffer("joints_mean", torch.from_numpy(mean), persistent=False)
        self.register_buffer("joints_std", torch.from_numpy(std), persistent=False)

        clip_context = (
            artifact_clip_load(clip_weights_path, expected_name="ViT-B/32")
            if clip_weights_path
            else nullcontext()
        )
        with clip_context:
            self.net = ACMDM_models[self.model_id](input_dim=3, cond_mode="text")
        if weights_path:
            missing, unexpected = self.net.load_state_dict(
                _load_weights(Path(weights_path)),
                strict=False,
            )
            invalid_missing = [key for key in missing if not key.startswith("clip_model.")]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "ProjFlow checkpoint does not match ACMDM: "
                    f"missing={invalid_missing}, unexpected={unexpected}"
                )
        if device is not None:
            self.to(torch.device(device))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = _resolve_artifact(str(pretrained_model_name_or_path))
        if path.is_file():
            return cls(weights_path=str(path), **kwargs)
        config_path = path / "projflow_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        def optional(name: str):
            value = path / name
            return str(value) if value.is_file() else None

        weights = optional(config.get("weights", "model.safetensors"))
        if weights is None:
            candidates = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.tar"))
            if not candidates:
                raise FileNotFoundError(f"No ProjFlow ACMDM weights found under {path}")
            weights = str(candidates[0])
        clip_weights = kwargs.pop(
            "clip_weights_path",
            optional(config.get("clip_weights", "clip.safetensors")),
        )
        if clip_weights is None:
            raise FileNotFoundError(
                "ProjFlow artifact is incomplete: clip.safetensors is required"
            )
        return cls(
            model_id=kwargs.pop("model_id", config.get("model_id", "ACMDM-Raw-Flow-S-PatchSize22")),
            weights_path=weights,
            clip_weights_path=clip_weights,
            mean_path=optional(config.get("mean", "joints_mean.npy")),
            std_path=optional(config.get("std", "joints_std.npy")),
            guidance_scale=kwargs.pop("guidance_scale", config.get("guidance_scale", 2.5)),
            num_steps=kwargs.pop("num_steps", config.get("num_steps", 100)),
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": "projflow",
            "library_name": "motius",
            "model_id": self.model_id,
            "weights": "model.safetensors" if safe_serialization else "model.pt",
            "clip_weights": "clip.safetensors" if safe_serialization else "clip.pt",
            "mean": "joints_mean.npy",
            "std": "joints_std.npy",
            "guidance_scale": self.guidance_scale,
            "num_steps": self.num_steps,
            "native_motion_representation": "HumanML3D SMPL-22 joint positions",
            "fps": 20,
        }
        (save_dir / "projflow_config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        model_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.net.state_dict().items()
            if not key.startswith("clip_model.")
        }
        clip_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.net.clip_model.state_dict().items()
        }
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(model_state, str(save_dir / "model.safetensors"))
            save_file(clip_state, str(save_dir / "clip.safetensors"))
        else:
            torch.save(model_state, save_dir / "model.pt")
            torch.save(clip_state, save_dir / "clip.pt")
        np.save(save_dir / "joints_mean.npy", self.joints_mean.detach().cpu().numpy())
        np.save(save_dir / "joints_std.npy", self.joints_std.detach().cpu().numpy())
        model_index = {
            "_class_name": "ProjFlowPipeline",
            "library_name": "motius",
            "pipeline_class": "motius.pipelines.projflow.ProjFlowPipeline",
            "bundle_class": "motius.models.projflow.ProjFlowBundle",
            "tasks": list(self.SUPPORTED_TASKS),
            "required_files": [
                "projflow_config.json",
                config["weights"],
                config["clip_weights"],
                "joints_mean.npy",
                "joints_std.npy",
            ],
        }
        (save_dir / "model_index.json").write_text(
            json.dumps(model_index, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(save_dir)

    @property
    def device(self) -> torch.device:
        return self.joints_mean.device

    def normalize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        return (joints - self.joints_mean) / self.joints_std

    def denormalize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        return joints * self.joints_std + self.joints_mean

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


__all__ = ["ProjFlowBundle"]
