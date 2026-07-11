"""MotionLCM ModelBundle.

Wraps the Motius-native MotionLCM implementation (Dai et al., ECCV 2024): a
latent **consistency** model distilled from a pretrained MLD latent diffusion
model that samples motion latents in just 1-4 steps. The MLD motion VAE, latent
consistency denoiser and text encoder live in
``motius.models.motionlcm.network``.

Components (see ``motius.models.motionlcm.network``):

* ``self.vae`` — :class:`MldVae`, a Transformer VAE that encodes a 263-dim
  HumanML3D motion into ``latent_size`` latent tokens and decodes back. The
  released HumanML3D checkpoint uses ``latent_dim=[16, 32, 256]`` (16 tokens,
  32-dim code, 256-dim transformer).
* ``self.denoiser`` — :class:`MldDenoiser`, the latent consistency model
  (``time_cond_proj_dim=256``); guidance is folded into the timestep
  conditioning (distilled CFG) so inference needs **no** second unconditional
  pass.
* ``self.scheduler`` — diffusers ``LCMScheduler`` driving the few-step
  consistency sampling.
* ``self.text_encoder`` — :class:`MldTextEncoder` (frozen ``sentence-t5-large``
  SentenceTransformer; reloaded by name, **not** stored in the artifact, exactly
  like CLIP in MDM / MoMask).
* ``mean`` / ``std`` — 263-dim HumanML3D denorm stats (``register_buffer``).

Representation: **HumanML3D-263** (263-dim, 20 fps, 22 joints). After
de-normalising the VAE output with ``Mean`` / ``Std`` the raw 263 vectors feed
directly into ``HumanML263Evaluator``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Repo root: motius/models/motionlcm/bundle.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The released assets contain two architecture families:
#
# * ``*_v1.ckpt``: latent_dim=[1, 256], used by the benchmark runner.
# * non-v1 ``*.ckpt``: latent_dim=[16, 32, 256], a separate checkpoint family.
#
# The Motius model-zoo artifact follows the v1 benchmark path. Explicit
# raw-checkpoint construction still infers the latent shape from checkpoint
# tensors so both families can be loaded safely.

# Standard Guo et al. HumanML3D-263 *training* stats. MLD / MotionLCM normalise
# the 263 motion with these before the VAE and de-normalise the VAE output with
# them (``HumanML3DDataModule.feats2joints``), so they MUST match the data
# convention the VAE was trained on. We verified by VAE auto-encoding: these
# stats give ~15 mm reconstruction MPJPE, whereas the (mislabelled, tiny-std)
# ``CondMDI/.../Mean.npy`` gives ~480 mm. They are embedded into the artifact so
# the checkpoint never depends on an external Mean/Std file.
_DEFAULT_MEAN = _REPO_ROOT / "checkpoints" / "mdm" / "humanml_trans_enc_512" / "Mean.npy"
_DEFAULT_STD = _REPO_ROOT / "checkpoints" / "mdm" / "humanml_trans_enc_512" / "Std.npy"

# Text encoder (sentence-t5-large, 768-dim). Resolved by name via HF hub.
_DEFAULT_TEXT_ENCODER = "sentence-transformers/sentence-t5-large"

# 263-dim HumanML3D feature.
_DIM_POSE = 263

# MLD VAE defaults (mirror ``configs/modules/motion_vae.yaml`` + released ckpt).
_VAE_DEFAULTS = {
    "nfeats": _DIM_POSE,
    "latent_dim": [16, 32, 256],
    "ff_size": 1024,
    "num_layers": 9,
    "num_heads": 4,
    "dropout": 0.1,
    "arch": "encoder_decoder",
    "normalize_before": False,
    "activation": "gelu",
    "position_embedding": "learned",
}
# Latent consistency denoiser defaults (mirror ``configs/modules/denoiser.yaml``
# + ``unet_time_cond_proj_dim=256`` from ``configs/motionlcm_t2m.yaml``).
_DENOISER_DEFAULTS = {
    "latent_dim": [16, 32, 256],
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
    "time_cond_proj_dim": 256,
}
# LCMScheduler defaults (mirror ``configs/modules/scheduler.yaml``).
_SCHEDULER_DEFAULTS = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "clip_sample": False,
    "set_alpha_to_one": False,
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


def _infer_raw_config(vae_ckpt: str | Path, denoiser_ckpt: str | Path) -> dict:
    """Infer MotionLCM module config from raw Lightning checkpoints.

    Upstream configs are not sufficient because the released folder contains
    both one-token and sixteen-token latent checkpoints. The tensors themselves
    are the reliable contract.
    """
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


def _deep_update(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


@MODEL_BUNDLES.register_module()
class MotionLCMBundle(ModelBundle):
    """MotionLCM text-to-motion bundle (HumanML3D-263, sentence-t5-large text)."""

    def __init__(
        self,
        # --- explicit raw upstream weights (converter/debug only) --------- #
        vae_ckpt: Optional[str] = None,
        denoiser_ckpt: Optional[str] = None,
        # --- self-contained Motius artifact ---------------------------- #
        config: Optional[dict] = None,
        vae_weights_path: Optional[str] = None,
        denoiser_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        # --- shared ------------------------------------------------------- #
        text_encoder_name: str = _DEFAULT_TEXT_ENCODER,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 1,
        load_text_encoder: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Construct the MotionLCM bundle.

        Two weight sources are supported (mirroring the MoMask / T2M-GPT
        bundles):

        * **Raw upstream checkpoints** — the released lightning ``.ckpt`` files
          from the released MotionLCM experiment directories (state dict in
          ``["state_dict"]`` with ``vae.*`` / ``denoiser.*`` prefixes). This is
          what :func:`scripts.eval.convert_motionlcm_checkpoint` consumes; the
          bundle never guesses a raw-checkout location.
        * **Self-contained Motius artifact** — ``config`` plus
          ``vae_weights_path`` / ``denoiser_weights_path`` (safetensors) +
          ``Mean.npy`` / ``Std.npy``, as produced by :meth:`save_pretrained` /
          consumed by :meth:`from_pretrained`. The text backbone is never
          stored; it is reloaded by ``text_encoder_name``.
        """
        super().__init__()
        use_artifact = vae_weights_path is not None
        if use_artifact and denoiser_weights_path is None:
            raise ValueError("MotionLCM artifact load is missing denoiser_weights_path.")
        if not use_artifact and (vae_ckpt is None or denoiser_ckpt is None):
            raise ValueError(
                "MotionLCMBundle requires artifact weights or explicit "
                "vae_ckpt and denoiser_ckpt for raw checkpoint conversion."
            )

        from .network import MldDenoiser, MldTextEncoder, MldVae

        self.text_encoder_name = text_encoder_name
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)

        raw_vae_ckpt = str(vae_ckpt) if vae_ckpt is not None else None
        raw_denoiser_ckpt = str(denoiser_ckpt) if denoiser_ckpt is not None else None

        cfg = dict(config) if config is not None else {}
        if not use_artifact:
            cfg = _deep_update(cfg, _infer_raw_config(raw_vae_ckpt, raw_denoiser_ckpt))
        vae_cfg = {**_VAE_DEFAULTS, **cfg.get("vae", {})}
        den_cfg = {**_DENOISER_DEFAULTS, **cfg.get("denoiser", {})}
        sched_cfg = {**_SCHEDULER_DEFAULTS, **cfg.get("scheduler", {})}
        self._vae_cfg = vae_cfg
        self._den_cfg = den_cfg
        self._sched_cfg = sched_cfg

        # ----- MLD motion VAE --------------------------------------------- #
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

        # ----- latent consistency denoiser -------------------------------- #
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

        # ----- weight loading --------------------------------------------- #
        if use_artifact:
            self._load_module(vae, vae_weights_path, prefix=None)
            self._load_module(denoiser, denoiser_weights_path, prefix=None)
        else:
            self._load_module(vae, raw_vae_ckpt, prefix="vae.")
            self._load_module(
                denoiser, raw_denoiser_ckpt, prefix="denoiser.")

        vae.eval()
        denoiser.eval()
        self.vae = vae
        self.denoiser = denoiser
        self.njoints = _DIM_POSE
        self.nfeats = _DIM_POSE

        # ----- LCM scheduler (diffusers, not an nn.Module) ---------------- #
        self.scheduler = self._build_scheduler(sched_cfg)

        # ----- text encoder (frozen, reloadable; not stored in artifact) -- #
        self.text_encoder = None
        if load_text_encoder:
            self.text_encoder = self._build_text_encoder(text_encoder_name)

        # ----- normalization buffers (263-dim) ---------------------------- #
        mean = np.load(str(mean_path or _DEFAULT_MEAN)).astype(np.float32)
        std = np.load(str(std_path or _DEFAULT_STD)).astype(np.float32)
        if mean.shape != (_DIM_POSE,) or std.shape != (_DIM_POSE,):
            raise ValueError(
                f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _build_scheduler(sched_cfg: dict):
        from diffusers import LCMScheduler

        return LCMScheduler(
            num_train_timesteps=sched_cfg["num_train_timesteps"],
            beta_start=sched_cfg["beta_start"],
            beta_end=sched_cfg["beta_end"],
            beta_schedule=sched_cfg["beta_schedule"],
            clip_sample=sched_cfg["clip_sample"],
            set_alpha_to_one=sched_cfg["set_alpha_to_one"],
        )

    @staticmethod
    def _build_text_encoder(name: str):
        """Load + freeze the sentence-t5-large text encoder (by name)."""
        from .network import MldTextEncoder

        enc = MldTextEncoder(modelpath=name, last_hidden_state=False)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        return enc

    @staticmethod
    def _load_module(module, path: str, prefix: Optional[str]) -> None:
        """Load weights from a safetensors artifact or a raw lightning ``.ckpt``.

        ``prefix`` selects + strips the sub-state-dict inside a raw checkpoint
        (``'vae.'`` / ``'denoiser.'``). For a safetensors artifact pass
        ``prefix=None``.
        """
        sd = _strip_prefix(_load_state_dict(path), prefix)
        module.load_state_dict(sd, strict=True)

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, raw-checkout-independent)
    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return {
            "vae": dict(self._vae_cfg),
            "denoiser": dict(self._den_cfg),
            "scheduler": dict(self._sched_cfg),
        }

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """Export a self-contained Motius MotionLCM artifact.

        Layout::

            <dir>/motionlcm_config.json  # arch config (vae / denoiser / scheduler)
            <dir>/vae.safetensors        # MLD motion VAE weights
            <dir>/denoiser.safetensors   # latent consistency denoiser weights
            <dir>/Mean.npy, Std.npy      # 263-dim denorm stats

        The sentence-t5-large text encoder is reloaded by name
        (``text_encoder_name``) and is **not** stored (exactly like CLIP in MDM).
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)

        cfg = {
            "model_type": "motionlcm",
            "text_encoder_name": self.text_encoder_name,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "config": self.config_dict(),
        }
        (save_dir / "motionlcm_config.json").write_text(json.dumps(cfg, indent=2))

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
        """Load a self-contained Motius MotionLCM artifact (local dir or HF Hub id)."""
        path = Path(pretrained_model_name_or_path)
        if not (path / "motionlcm_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "motionlcm_config.json"
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
            "num_inference_steps", meta.get("num_inference_steps", 1))
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

    # ------------------------------------------------------------------
    # device / forward helpers
    # ------------------------------------------------------------------
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
        """sentence-t5-large features ``(B, 1, 768)``."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded (load_text_encoder=False).")
        return self.text_encoder(list(captions))

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        """Un-standardize HumanML3D-263 features back to physical scale."""
        return motion_263 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use MotionLCMPipeline.infer_t2m for inference.")
