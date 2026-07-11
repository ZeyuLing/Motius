"""MLD ModelBundle.

Wraps a native Motius implementation of Motion Latent Diffusion (MLD;
Chen et al., CVPR 2023) for HumanML3D text-to-motion. The runtime reuses the
MLD VAE, latent denoiser, and SentenceT5 text wrapper already vendored for
MotionLCM, but drives them with the original DDIM latent diffusion sampler.

Representation: **HumanML3D-263** (263-dim, 20 fps, 22 joints). The VAE output
is de-normalised with the embedded HumanML3D training Mean/Std before it is
returned by the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_MEAN = _REPO_ROOT / "checkpoints" / "mdm" / "humanml_trans_enc_512" / "Mean.npy"
_DEFAULT_STD = _REPO_ROOT / "checkpoints" / "mdm" / "humanml_trans_enc_512" / "Std.npy"
_DEFAULT_TEXT_ENCODER = "sentence-transformers/sentence-t5-large"
_DIM_POSE = 263

_VAE_DEFAULTS = {
    "nfeats": _DIM_POSE,
    "latent_dim": [1, 256],
    "ff_size": 1024,
    "num_layers": 9,
    "num_heads": 4,
    "dropout": 0.1,
    "arch": "encoder_decoder",
    "normalize_before": False,
    "activation": "gelu",
    "position_embedding": "learned",
}

_DENOISER_DEFAULTS = {
    "latent_dim": [1, 256],
    "ff_size": 1024,
    "num_layers": 9,
    "num_heads": 4,
    "dropout": 0.1,
    "normalize_before": False,
    "activation": "gelu",
    "flip_sin_to_cos": True,
    "return_intermediate_dec": False,
    "position_embedding": "learned",
    "arch": "trans_enc",
    "freq_shift": 0,
    "text_encoded_dim": 768,
    "time_cond_proj_dim": None,
}

_SCHEDULER_DEFAULTS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "clip_sample": False,
    "set_alpha_to_one": False,
    "steps_offset": 1,
    "eta": 0.0,
}


def _load_state_dict(path: str | Path) -> dict:
    p = str(path)
    if p.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


def _strip_prefix(sd: dict, prefix: Optional[str]) -> dict:
    if prefix is None:
        return dict(sd)
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _deep_update(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _infer_raw_config(vae_ckpt: str | Path, denoiser_ckpt: str | Path) -> dict:
    vae_sd = _strip_prefix(_load_state_dict(vae_ckpt), "vae.")
    den_sd = _strip_prefix(_load_state_dict(denoiser_ckpt), "denoiser.")
    cfg = {"vae": {}, "denoiser": {}, "scheduler": {}}

    token = vae_sd.get("global_motion_token")
    if token is not None:
        latent_size = int(token.shape[0]) // 2
        hidden_dim = int(token.shape[1])
        latent_pre = vae_sd.get("latent_pre.weight")
        if latent_pre is not None:
            cfg["vae"]["latent_dim"] = [
                latent_size,
                int(latent_pre.shape[0]),
                hidden_dim,
            ]
        else:
            cfg["vae"]["latent_dim"] = [latent_size, hidden_dim]
        cfg["denoiser"]["latent_dim"] = list(cfg["vae"]["latent_dim"])

    cond = den_sd.get("time_embedding.cond_proj.weight")
    cfg["denoiser"]["time_cond_proj_dim"] = int(cond.shape[1]) if cond is not None else None
    return cfg


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


@MODEL_BUNDLES.register_module()
class MLDBundle(ModelBundle):
    """MLD text-to-motion bundle (HumanML3D-263, SentenceT5 text encoder)."""

    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        vae_ckpt: Optional[str] = None,
        denoiser_ckpt: Optional[str] = None,
        config: Optional[dict] = None,
        vae_weights_path: Optional[str] = None,
        denoiser_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        text_encoder_name: str = _DEFAULT_TEXT_ENCODER,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        load_text_encoder: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        use_artifact = vae_weights_path is not None
        if use_artifact and denoiser_weights_path is None:
            raise ValueError("MLD artifact load is missing denoiser_weights_path.")

        if model_ckpt is not None:
            vae_ckpt = vae_ckpt or model_ckpt
            denoiser_ckpt = denoiser_ckpt or model_ckpt
        if not use_artifact and (vae_ckpt is None or denoiser_ckpt is None):
            raise ValueError(
                "MLDBundle requires artifact weights or a raw MLD checkpoint."
            )

        from motius.models.motionlcm.network import (
            MldDenoiser,
            MldTextEncoder,
            MldVae,
        )

        self.text_encoder_name = text_encoder_name
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)

        cfg = dict(config) if config is not None else {}
        if not use_artifact:
            cfg = _deep_update(cfg, _infer_raw_config(str(vae_ckpt), str(denoiser_ckpt)))
        vae_cfg = {**_VAE_DEFAULTS, **cfg.get("vae", {})}
        den_cfg = {**_DENOISER_DEFAULTS, **cfg.get("denoiser", {})}
        sched_cfg = {**_SCHEDULER_DEFAULTS, **cfg.get("scheduler", {})}
        self._vae_cfg = vae_cfg
        self._den_cfg = den_cfg
        self._sched_cfg = sched_cfg

        vae = MldVae(
            nfeats=vae_cfg["nfeats"],
            latent_dim=list(vae_cfg["latent_dim"]),
            ff_size=vae_cfg["ff_size"],
            num_layers=vae_cfg["num_layers"],
            num_heads=vae_cfg["num_heads"],
            dropout=vae_cfg["dropout"],
            arch=vae_cfg["arch"],
            normalize_before=vae_cfg["normalize_before"],
            activation=vae_cfg["activation"],
            position_embedding=vae_cfg["position_embedding"],
        )

        denoiser = MldDenoiser(
            latent_dim=list(den_cfg["latent_dim"]),
            ff_size=den_cfg["ff_size"],
            num_layers=den_cfg["num_layers"],
            num_heads=den_cfg["num_heads"],
            dropout=den_cfg["dropout"],
            normalize_before=den_cfg["normalize_before"],
            activation=den_cfg["activation"],
            flip_sin_to_cos=den_cfg["flip_sin_to_cos"],
            return_intermediate_dec=den_cfg["return_intermediate_dec"],
            position_embedding=den_cfg["position_embedding"],
            arch=den_cfg["arch"],
            freq_shift=den_cfg["freq_shift"],
            text_encoded_dim=den_cfg["text_encoded_dim"],
            time_cond_proj_dim=den_cfg["time_cond_proj_dim"],
            is_controlnet=False,
        )

        if use_artifact:
            self._load_module(vae, vae_weights_path, prefix=None)
            self._load_module(denoiser, denoiser_weights_path, prefix=None)
        else:
            self._load_module(vae, str(vae_ckpt), prefix="vae.")
            self._load_module(denoiser, str(denoiser_ckpt), prefix="denoiser.")

        vae.eval()
        denoiser.eval()
        self.vae = vae
        self.denoiser = denoiser
        self.njoints = _DIM_POSE
        self.nfeats = _DIM_POSE
        self.scheduler = self._build_scheduler(sched_cfg)

        self.text_encoder = None
        if load_text_encoder:
            self.text_encoder = MldTextEncoder(modelpath=text_encoder_name, last_hidden_state=False)
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        mean = np.load(str(mean_path or _DEFAULT_MEAN)).astype(np.float32)
        std = np.load(str(std_path or _DEFAULT_STD)).astype(np.float32)
        if mean.shape != (_DIM_POSE,) or std.shape != (_DIM_POSE,):
            raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)

    @staticmethod
    def _build_scheduler(sched_cfg: dict):
        from diffusers import DDIMScheduler

        return DDIMScheduler(
            num_train_timesteps=sched_cfg["num_train_timesteps"],
            beta_start=sched_cfg["beta_start"],
            beta_end=sched_cfg["beta_end"],
            beta_schedule=sched_cfg["beta_schedule"],
            clip_sample=sched_cfg["clip_sample"],
            set_alpha_to_one=sched_cfg["set_alpha_to_one"],
            steps_offset=sched_cfg["steps_offset"],
        )

    @staticmethod
    def _load_module(module, path: str, prefix: Optional[str]) -> None:
        sd = _strip_prefix(_load_state_dict(path), prefix)
        module.load_state_dict(sd, strict=True)

    def config_dict(self) -> dict:
        return {
            "vae": dict(self._vae_cfg),
            "denoiser": dict(self._den_cfg),
            "scheduler": dict(self._sched_cfg),
        }

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)
        cfg = {
            "model_type": "mld",
            "text_encoder_name": self.text_encoder_name,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "config": self.config_dict(),
        }
        (save_dir / "mld_config.json").write_text(json.dumps(cfg, indent=2))

        def _cpu_state(m):
            return {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(_cpu_state(self.vae), str(save_dir / "vae.safetensors"))
            save_file(_cpu_state(self.denoiser), str(save_dir / "denoiser.safetensors"))
        else:
            torch.save(_cpu_state(self.vae), str(save_dir / "vae.pt"))
            torch.save(_cpu_state(self.denoiser), str(save_dir / "denoiser.pt"))

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "mld_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "mld_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())

        def _w(name):
            st = path / f"{name}.safetensors"
            return st if st.exists() else path / f"{name}.pt"

        text_encoder_name = kwargs.pop(
            "text_encoder_name", meta.get("text_encoder_name", _DEFAULT_TEXT_ENCODER))
        guidance_scale = kwargs.pop("guidance_scale", meta.get("guidance_scale", 7.5))
        num_inference_steps = kwargs.pop(
            "num_inference_steps", meta.get("num_inference_steps", 50))
        return cls(
            config=meta["config"],
            vae_weights_path=str(_w("vae")),
            denoiser_weights_path=str(_w("denoiser")),
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            text_encoder_name=text_encoder_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )

    def to_device(self, device):
        device = torch.device(device)
        self.vae.to(device)
        self.denoiser.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded (load_text_encoder=False).")
        return self.text_encoder(list(captions))

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use MLDPipeline.infer_t2m for inference.")
