"""MotionStreamer ModelBundle.

Wraps the Motius-native MotionStreamer implementation (causal TAE + LLaMA
autoregressive transformer + per-token diffusion head; see
``motius.models.motionstreamer.network``) behind a clean
``ModelBundle`` interface.

The text encoder is SentenceT5-XXL (loaded by name via ``sentence_transformers``,
frozen, and -- like CLIP in MDM -- not duplicated into the Motius artifact).

Representation: MotionStreamer-272 (272-dim, 30 fps).
Generation path: text -> SentenceT5 -> LLaMA AR (CFG, per-token diffusion
sampling) -> latent tokens -> TAE decoder -> 272-dim motion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Repo root: motius/models/motionstreamer/bundle.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a HuggingFace Hub repo id to a local snapshot dir.

    Returns ``local`` unchanged if it is already a directory; otherwise tries
    ``snapshot_download(name_or_path)`` and returns the cached path.
    """
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local

# MotionStreamer-272 normalization (released ``humanml3d_272/mean_std``). These
# travel inside the Motius artifact so a checkpoint is self-contained.
_DEFAULT_MEAN = _REPO_ROOT / "checkpoints/motionstreamer/t2m_humanml272/Mean.npy"
_DEFAULT_STD = _REPO_ROOT / "checkpoints/motionstreamer/t2m_humanml272/Std.npy"

# SentenceT5-XXL: resolved by HF/sentence-transformers name. The released demo
# used a local ``sentencet5-xxl/`` dir, identical to this hub model.
_DEFAULT_TEXT_MODEL = "sentence-transformers/sentence-t5-xxl"

# Architecture defaults (MotionStreamer ``options/option_transformer.py`` +
# ``demo_t2m.py`` / ``eval_t2m.py``).
_TAE_DEFAULTS = {
    "hidden_size": 1024,
    "down_t": 2,
    "stride_t": 2,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "latent_dim": 16,
    "clip_range": [-30, 20],
}
_AR_DEFAULTS = {
    "config_name": "Normal_size",   # n_layer=12, n_head=12, n_embd=768
    "block_size": 78,
    "num_diffusion_head_layers": 9,
    "latent_dim": 16,
    "t5_xxl_dim": 768,
}


def _strip_module_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        out[nk] = v
    return out


@MODEL_BUNDLES.register_module()
class MotionStreamerBundle(ModelBundle):
    """MotionStreamer text-to-motion bundle (272-dim, SentenceT5-XXL text)."""

    def __init__(
        self,
        tae_path: Optional[str] = None,
        ar_path: Optional[str] = None,
        text_model_name: str = _DEFAULT_TEXT_MODEL,
        guidance_param: float = 4.0,
        config: Optional[dict] = None,
        tae_weights_path: Optional[str] = None,
        ar_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        load_text_model: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Construct the MotionStreamer bundle.

        Two weight sources are supported:

        * **Explicit raw upstream checkpoints** — pass ``tae_path`` (``.pth``
          with a ``net`` key) and ``ar_path`` (``.pth`` with a ``trans`` key).
          These are intended for converter/debug code only; the bundle never
          guesses a raw checkpoint location.
        * **Self-contained Motius artifact** — pass ``config`` plus
          ``tae_weights_path`` / ``ar_weights_path`` (safetensors). This is what
          :meth:`from_pretrained` / :meth:`save_pretrained` use.
        """
        super().__init__()
        if tae_weights_path is None and tae_path is None:
            raise ValueError(
                "MotionStreamerBundle requires tae_weights_path from an "
                "Motius artifact or an explicit raw tae_path for conversion."
            )
        if ar_weights_path is None and ar_path is None:
            raise ValueError(
                "MotionStreamerBundle requires ar_weights_path from an "
                "Motius artifact or an explicit raw ar_path for conversion."
            )

        from .network import LLaMAHF, LLaMAHFConfig, Causal_HumanTAE

        self.guidance_param = float(guidance_param)
        self.text_model_name = text_model_name

        cfg = dict(config) if config is not None else {}
        tae_cfg = {**_TAE_DEFAULTS, **cfg.get("tae", {})}
        ar_cfg = {**_AR_DEFAULTS, **cfg.get("ar", {})}
        self._tae_cfg = tae_cfg
        self._ar_cfg = ar_cfg

        build_device = torch.device(device) if device is not None else torch.device("cpu")

        # --- TAE -------------------------------------------------------- #
        tae = Causal_HumanTAE(
            hidden_size=tae_cfg["hidden_size"],
            down_t=tae_cfg["down_t"],
            stride_t=tae_cfg["stride_t"],
            depth=tae_cfg["depth"],
            dilation_growth_rate=tae_cfg["dilation_growth_rate"],
            activation=tae_cfg["activation"],
            latent_dim=tae_cfg["latent_dim"],
            clip_range=list(tae_cfg["clip_range"]),
        )

        # --- AR transformer (+ diffusion head) -------------------------- #
        llama_cfg = LLaMAHFConfig.from_name(ar_cfg["config_name"])
        llama_cfg.block_size = ar_cfg["block_size"]
        llama_cfg.T5_xxl_dim = ar_cfg["t5_xxl_dim"]
        ar = LLaMAHF(
            llama_cfg,
            ar_cfg["num_diffusion_head_layers"],
            ar_cfg["latent_dim"],
            build_device,
        )

        # --- load weights ---------------------------------------------- #
        if tae_weights_path is not None:
            self._load_weights(tae, tae_weights_path, key=None)
        else:
            self._load_weights(tae, str(tae_path), key="net")
        if ar_weights_path is not None:
            self._load_weights(ar, ar_weights_path, key=None)
        else:
            self._load_weights(ar, str(ar_path), key="trans")

        tae.eval()
        ar.eval()
        self.tae = tae
        self.ar = ar

        self.nfeats = 272

        # --- text encoder (frozen, reloadable; not stored in artifact) -- #
        self.text_model = None
        if load_text_model:
            self.text_model = self._build_text_model(text_model_name, build_device)

        # --- normalization buffers (272-dim) --------------------------- #
        mean_p = Path(mean_path) if mean_path else Path(_DEFAULT_MEAN)
        std_p = Path(std_path) if std_path else Path(_DEFAULT_STD)
        mean = np.load(str(mean_p)).astype(np.float32)
        std = np.load(str(std_p)).astype(np.float32)
        if mean.shape != (272,) or std.shape != (272,):
            raise ValueError(
                f"expected 272-dim mean/std, got {mean.shape} and {std.shape}"
            )
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_weights(module, path: str, key: Optional[str]) -> None:
        p = str(path)
        if p.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(p)
        else:
            ckpt = torch.load(p, map_location="cpu")
            sd = ckpt[key] if (key is not None and isinstance(ckpt, dict) and key in ckpt) else ckpt
        sd = _strip_module_prefix(sd)
        module.load_state_dict(sd, strict=True)

    @staticmethod
    def _build_text_model(name: str, device: torch.device):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name, device=str(device))
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, raw-checkout-independent)
    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return {
            "tae": dict(self._tae_cfg),
            "ar": dict(self._ar_cfg),
        }

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """Export a self-contained Motius MotionStreamer artifact.

        Layout::

            <dir>/ms_config.json        # tae + ar arch config, text model name, guidance
            <dir>/tae.safetensors       # causal TAE weights
            <dir>/ar.safetensors        # LLaMA AR + diffusion-head weights
            <dir>/Mean.npy, Std.npy     # 272-dim denorm stats

        The SentenceT5-XXL text encoder is *not* duplicated; it is reloaded by
        name (``text_model_name``).
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)

        cfg = {
            "model_type": "motionstreamer",
            "guidance_param": self.guidance_param,
            "text_model_name": self.text_model_name,
            "config": self.config_dict(),
        }
        (save_dir / "ms_config.json").write_text(json.dumps(cfg, indent=2))

        def _cpu_state(m):
            return {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(_cpu_state(self.tae), str(save_dir / "tae.safetensors"))
            save_file(_cpu_state(self.ar), str(save_dir / "ar.safetensors"))
        else:
            torch.save(_cpu_state(self.tae), str(save_dir / "tae.pt"))
            torch.save(_cpu_state(self.ar), str(save_dir / "ar.pt"))

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a self-contained Motius MotionStreamer artifact.

        ``pretrained_model_name_or_path`` may be a local directory **or** a
        HuggingFace Hub repo id (e.g.
        ``"ZeyuLing/hftrainer-motionstreamer-humanml272"``), which is fetched
        via ``snapshot_download``.
        """
        path = Path(pretrained_model_name_or_path)
        if not (path / "ms_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "ms_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())
        tae_w = path / "tae.safetensors"
        if not tae_w.exists():
            tae_w = path / "tae.pt"
        ar_w = path / "ar.safetensors"
        if not ar_w.exists():
            ar_w = path / "ar.pt"
        guidance_param = kwargs.pop("guidance_param", meta.get("guidance_param", 4.0))
        text_model_name = kwargs.pop(
            "text_model_name", meta.get("text_model_name", _DEFAULT_TEXT_MODEL)
        )
        return cls(
            config=meta["config"],
            tae_weights_path=str(tae_w),
            ar_weights_path=str(ar_w),
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            guidance_param=guidance_param,
            text_model_name=text_model_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # device / forward helpers
    # ------------------------------------------------------------------
    def to_device(self, device):
        device = torch.device(device)
        self.tae.to(device)
        self.ar.to(device)
        if self.text_model is not None:
            self.text_model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def denormalize(self, motion_272: torch.Tensor) -> torch.Tensor:
        """Un-standardize MotionStreamer-272 features back to physical scale."""
        return motion_272 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use MotionStreamerPipeline.infer_t2m for inference.")
