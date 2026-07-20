"""MotionGPT3 ModelBundle.

The official runtime is vendored under
``motius.models.motiongpt3.network``. The default artifact under
``checkpoints/motiongpt3`` contains the checkpoint, configs, GPT2
adapter, and HML263 stats needed for independent inference.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "motiongpt3"


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a Hugging Face Hub model repo id to a local snapshot directory."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _install_legacy_aliases() -> None:
    for alias, target in (
        ("motGPT", "motius.models.motiongpt3.network.motGPT"),
        ("mot_code", "motius.models.motiongpt3.network.mot_code"),
    ):
        sys.modules.setdefault(alias, importlib.import_module(target))


def _rewrite_config_targets(value):
    """Translate artifact paths from the pre-Motius package layout."""

    if isinstance(value, str):
        for marker in ("models.motion.motiongpt3", "models.motiongpt3"):
            index = value.find(marker)
            if index >= 0:
                suffix = value[index + len(marker) :]
                return f"motius.models.motiongpt3{suffix}"
        return value
    if isinstance(value, dict):
        return {key: _rewrite_config_targets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_config_targets(item) for item in value]
    return value


@contextlib.contextmanager
def _cwd(path: Path):
    import os

    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _DummyDataModule:
    name = "humanml3d"
    njoints = 22
    fps = 20
    is_mm = False

    def __init__(self, mean_path: Path, std_path: Path):
        self.mean_path = Path(mean_path).resolve()
        self.std_path = Path(std_path).resolve()
        self.mean = torch.from_numpy(np.load(self.mean_path)).float()
        self.std = torch.from_numpy(np.load(self.std_path)).float()

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(features.device, features.dtype)
        std = self.std.to(features.device, features.dtype)
        return features * std + mean

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*features.shape[:2], self.njoints, 3), device=features.device, dtype=features.dtype)


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    for param in module.parameters(recurse=True):
        return param.dtype
    return torch.float32


def _convert_head_mask_to_5d(module: torch.nn.Module, head_mask: torch.Tensor, num_hidden_layers: int) -> torch.Tensor:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != 5:
        raise ValueError(f"head_mask.dim != 5, instead {head_mask.dim()}")
    return head_mask.to(dtype=_module_dtype(module))


def _get_head_mask(module: torch.nn.Module, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False):
    if head_mask is None:
        return [None] * num_hidden_layers
    head_mask = _convert_head_mask_to_5d(module, head_mask, num_hidden_layers)
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask


def _patch_motiongpt3_transformers_compat(model: torch.nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "MoTGPT2Model":
            continue
        if not hasattr(module, "model_parallel"):
            module.model_parallel = False
        if not hasattr(module, "device_map"):
            module.device_map = None
        if not hasattr(module, "get_head_mask"):
            module.get_head_mask = types.MethodType(_get_head_mask, module)
            patched += 1
    return patched


def _load_cfg(
    artifact: Path,
    cfg_path: Path,
    checkpoint: Path,
    guidance_scale: float,
    runtime_dir: Optional[Path],
):
    _install_legacy_aliases()
    from motius.models.motiongpt3.network.motGPT.config import get_module_config

    with _cwd(artifact):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg_assets = OmegaConf.load(str(artifact / "configs" / "assets.yaml"))
        cfg_base = OmegaConf.load(str(artifact / "configs" / "default.yaml"))
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(str(cfg_path)))
        if not cfg_exp.FULL_CONFIG:
            cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
        cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg = OmegaConf.create(
        _rewrite_config_targets(OmegaConf.to_container(cfg, resolve=False))
    )
    cfg.DEBUG = False
    cfg.DEVICE = [0]
    cfg.TEST.CHECKPOINTS = str(checkpoint)
    cfg.METRIC.TYPE = []
    cfg.model.params.metrics_dict = []
    cfg.model.params.guidance_scale = guidance_scale
    cfg.lm_ablation.model_guidance_scale = guidance_scale
    cfg.FOLDER = str(artifact)
    cfg.TIME = _dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg.FOLDER_EXP = str(runtime_dir or (artifact / "_runtime"))

    mot_gpt2 = artifact / "deps" / "mot-gpt2"
    if "lm" in cfg and "mot_vae_gpt2" in cfg.lm:
        cfg.lm.mot_vae_gpt2.params.model_path = str(mot_gpt2)
    return cfg


@MODEL_BUNDLES.register_module()
class MotionGPT3Bundle(ModelBundle):
    """MotionGPT3 text-to-motion bundle for HumanML3D-263 generation."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        cfg: Optional[str] = None,
        checkpoint: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        guidance_scale: float = 7.5,
        runtime_dir: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        _install_legacy_aliases()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        cfg_path = Path(cfg or artifact / "configs" / "test.yaml").resolve()
        checkpoint = Path(checkpoint or artifact / "motiongpt3.ckpt").resolve()
        mean_path = Path(mean_path or artifact / "assets" / "meta" / "mean.npy").resolve()
        std_path = Path(std_path or artifact / "assets" / "meta" / "std.npy").resolve()
        runtime_path = Path(runtime_dir).resolve() if runtime_dir else None

        from motius.models.motiongpt3.network.motGPT.models.base import BaseModel
        from motius.models.motiongpt3.network.motGPT.models.build_model import build_model

        BaseModel.configure_metrics = lambda self: setattr(self, "metrics", torch.nn.Module())

        self.cfg = _load_cfg(
            artifact,
            cfg_path,
            checkpoint,
            guidance_scale=guidance_scale,
            runtime_dir=runtime_path,
        )
        self.datamodule = _DummyDataModule(mean_path, std_path)
        model = build_model(self.cfg, self.datamodule).eval()
        patched = _patch_motiongpt3_transformers_compat(model)
        state = torch.load(str(checkpoint), map_location="cpu")["state_dict"]
        load_result = model.load_state_dict(state, strict=False)
        if load_result is None:
            missing, unexpected = [], []
        else:
            missing, unexpected = load_result
        self.load_report = {
            "missing": len(missing),
            "unexpected": len(unexpected),
            "compat_patched": patched,
        }
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(resolved_device).eval()
        self.guidance_scale = float(guidance_scale)
        self.register_buffer("mean", self.datamodule.mean.float(), persistent=True)
        self.register_buffer("std", self.datamodule.std.float(), persistent=True)
        self.to_device(resolved_device)

    def to_device(self, device):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.datamodule.mean = self.mean.detach().cpu()
        self.datamodule.std = self.std.detach().cpu()
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std.to(motion_263) + self.mean.to(motion_263)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "motiongpt3.ckpt").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / "motiongpt3.ckpt").exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use MotionGPT3Pipeline task methods for inference.")
