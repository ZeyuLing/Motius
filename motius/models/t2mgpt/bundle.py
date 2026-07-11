"""T2M-GPT ModelBundle.

Wraps the Motius-native T2M-GPT implementation (Zhang et al., CVPR 2023): a
VQ-VAE motion tokenizer + a cross-conditional GPT that autoregressively predicts
motion tokens from a CLIP text embedding. The runtime imports only the local
``motius.models.t2mgpt.network`` package and self-contained
Motius artifacts; raw upstream checkpoints are accepted only when explicit
paths are provided by conversion scripts.

The reproduction preserves exact numerical parity with the released checkpoint
and the gold-standard ``scripts/eval/t2mgpt_infer_hml3d263.py`` script (same
seed/caption -> bit-identical 263-dim output).

Components
---------
* ``self.vqvae`` — :class:`HumanVQVAE` (Encoder/Decoder + EMA-reset quantizer),
  263-dim HumanML3D input, ``nb_code=512`` codebook.
* ``self.gpt`` — :class:`Text2Motion_Transformer` (9-layer cross-conditional GPT,
  ``embed_dim=1024``, ``clip_dim=512``).
* ``self.clip_model`` — CLIP ViT-B/32 text encoder (frozen, stored in the
  Motius artifact as ``clip.safetensors`` for offline ``from_pretrained``).
* ``mean`` / ``std`` — 263-dim HumanML3D denorm stats (register_buffer).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]

# 263-dim HumanML3D denorm stats used by the parity script. These are embedded
# into the self-contained artifact so the reproduced checkpoint never depends on
# an external Mean/Std file.
_DEFAULT_MEAN = _REPO_ROOT / "data/statistic/humanml3d_263/Mean.npy"
_DEFAULT_STD = _REPO_ROOT / "data/statistic/humanml3d_263/Std.npy"

# CLIP text encoder (clip_dim = 512). New artifacts store its state dict and
# use this name only as a fallback for legacy/lightweight exports.
_DEFAULT_CLIP_NAME = "ViT-B/32"

# VQ-VAE defaults — match the official README HumanML3D config (and the
# ``build_t2mgpt_args`` helper in ``scripts/eval/t2mgpt_infer_hml3d263.py``).
_VQVAE_DEFAULTS = {
    "dataname": "t2m",           # -> 263-dim input (251 only for KIT)
    "nb_code": 512,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "vq_act": "relu",
    "norm": None,
    "quantizer": "ema_reset",
    "mu": 0.99,
}
# GPT (Text2Motion_Transformer) defaults — official 9-layer config.
_GPT_DEFAULTS = {
    "embed_dim_gpt": 1024,
    "clip_dim": 512,
    "block_size": 51,
    "num_layers": 9,
    "n_head_gpt": 16,
    "drop_out_rate": 0.1,
    "ff_rate": 4,
}


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a HuggingFace Hub repo id to a local snapshot dir (or return local)."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


@MODEL_BUNDLES.register_module()
class T2MGPTBundle(ModelBundle):
    """T2M-GPT text-to-motion bundle (HumanML3D-263, CLIP ViT-B/32 text)."""

    def __init__(
        self,
        vq_path: Optional[str] = None,
        gpt_path: Optional[str] = None,
        clip_name: str = _DEFAULT_CLIP_NAME,
        config: Optional[dict] = None,
        vq_weights_path: Optional[str] = None,
        gpt_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        load_clip: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Construct the bundle.

        Two weight sources are supported:

        * **Self-contained Motius artifact** — ``config`` plus
          ``vq_weights_path`` / ``gpt_weights_path`` (safetensors), as produced
          by :meth:`save_pretrained` / consumed by :meth:`from_pretrained`.
        * **Explicit raw upstream checkpoints** — ``vq_path`` (a ``net_last.pth``
          with a ``net`` key) and ``gpt_path`` (a ``net_best_fid.pth`` with a
          ``trans`` key). These are intended for converter scripts only; this
          bundle never guesses an upstream checkout location.
        """
        super().__init__()
        from .network import HumanVQVAE, Text2Motion_Transformer

        self.clip_name = clip_name
        cfg = dict(config) if config is not None else {}
        vq_cfg = {**_VQVAE_DEFAULTS, **cfg.get("vqvae", {})}
        gpt_cfg = {**_GPT_DEFAULTS, **cfg.get("gpt", {})}
        self._vq_cfg = vq_cfg
        self._gpt_cfg = gpt_cfg

        # --- VQ-VAE tokenizer ------------------------------------------- #
        # ``HumanVQVAE``/``QuantizeEMAReset`` read these attributes off ``args``.
        vq_args = SimpleNamespace(
            dataname=vq_cfg["dataname"],
            quantizer=vq_cfg["quantizer"],
            mu=vq_cfg["mu"],
        )
        vqvae = HumanVQVAE(
            vq_args,
            vq_cfg["nb_code"],
            vq_cfg["code_dim"],
            vq_cfg["output_emb_width"],
            vq_cfg["down_t"],
            vq_cfg["stride_t"],
            vq_cfg["width"],
            vq_cfg["depth"],
            vq_cfg["dilation_growth_rate"],
            activation=vq_cfg["vq_act"],
            norm=vq_cfg["norm"],
        )

        # --- cross-conditional GPT -------------------------------------- #
        gpt = Text2Motion_Transformer(
            num_vq=vq_cfg["nb_code"],
            embed_dim=gpt_cfg["embed_dim_gpt"],
            clip_dim=gpt_cfg["clip_dim"],
            block_size=gpt_cfg["block_size"],
            num_layers=gpt_cfg["num_layers"],
            n_head=gpt_cfg["n_head_gpt"],
            drop_out_rate=gpt_cfg["drop_out_rate"],
            fc_rate=gpt_cfg["ff_rate"],
        )

        # --- load weights ----------------------------------------------- #
        if vq_weights_path is not None:
            self._load_weights(vqvae, vq_weights_path, key=None)
        else:
            if vq_path is None:
                raise ValueError(
                    "T2MGPTBundle requires vq_weights_path from a Motius "
                    "artifact or an explicit raw vq_path for conversion."
                )
            self._load_weights(vqvae, str(vq_path), key="net")
        if gpt_weights_path is not None:
            self._load_weights(gpt, gpt_weights_path, key=None)
        else:
            if gpt_path is None:
                raise ValueError(
                    "T2MGPTBundle requires gpt_weights_path from a Motius "
                    "artifact or an explicit raw gpt_path for conversion."
                )
            self._load_weights(gpt, str(gpt_path), key="trans")

        vqvae.eval()
        gpt.eval()
        self.vqvae = vqvae
        self.gpt = gpt
        self.nfeats = 263

        # --- CLIP text encoder (frozen, stored in new artifacts) ---------- #
        self.clip_model = None
        if load_clip:
            self.clip_model = self._build_clip(
                clip_name,
                device,
                weights_path=clip_weights_path,
            )

        # --- normalization buffers (263-dim) ---------------------------- #
        mean = np.load(str(mean_path or _DEFAULT_MEAN)).astype(np.float32)
        std = np.load(str(std_path or _DEFAULT_STD)).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_weights(module, path: str, key: Optional[str]) -> None:
        """Load weights from a safetensors file or a raw upstream ``.pth``.

        ``key`` selects the sub-dict inside a raw checkpoint (``'net'`` for the
        VQ-VAE, ``'trans'`` for the GPT). When loading a safetensors artifact,
        pass ``key=None``.
        """
        p = str(path)
        if p.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(p)
        else:
            ckpt = torch.load(p, map_location="cpu")
            sd = ckpt[key] if (key is not None and isinstance(ckpt, dict) and key in ckpt) else ckpt
        module.load_state_dict(sd, strict=True)

    @staticmethod
    def _build_clip(name: str, device: Optional[str], weights_path: Optional[str] = None):
        """Load + freeze CLIP exactly like the parity script.

        ``clip.load`` -> ``clip.model.convert_weights`` (fp16) -> ``eval`` ->
        freeze. Keeping this identical to ``t2mgpt_infer_hml3d263.py`` is what
        makes generation bit-identical.
        """
        import clip

        dev = device
        if dev is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        if weights_path is not None:
            if str(weights_path).endswith(".safetensors"):
                from safetensors.torch import load_file

                state = load_file(str(weights_path), device="cpu")
            else:
                state = torch.load(str(weights_path), map_location="cpu")
            clip_model = clip.model.build_model(dict(state)).to(dev)
        else:
            clip_model, _ = clip.load(name, device=dev, jit=False)
            clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, repo-independent)
    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return {"vqvae": dict(self._vq_cfg), "gpt": dict(self._gpt_cfg)}

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        include_clip: bool = True,
        **kwargs,
    ):
        """Export a self-contained Motius T2M-GPT artifact.

        Layout::

            <dir>/t2mgpt_config.json    # vqvae + gpt arch config, clip name
            <dir>/vq.safetensors        # HumanVQVAE (encoder/decoder/quantizer)
            <dir>/gpt.safetensors       # Text2Motion_Transformer weights
            <dir>/clip.safetensors      # CLIP ViT-B/32 text encoder weights
            <dir>/Mean.npy, Std.npy     # 263-dim denorm stats
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)
        if include_clip and self.clip_model is None:
            self.clip_model = self._build_clip(
                self.clip_name,
                str(self.device) if hasattr(self, "mean") else None,
            )
        clip_artifact = (
            ("clip.safetensors" if safe_serialization else "clip.pt")
            if include_clip
            else None
        )

        cfg = {
            "model_type": "t2mgpt",
            "format": "motius-t2mgpt-v1",
            "clip_name": self.clip_name,
            "components": {
                "clip": {
                    "name": self.clip_name,
                    "stored_in_artifact": bool(include_clip),
                    "path": clip_artifact,
                }
            },
            "config": self.config_dict(),
        }
        (save_dir / "t2mgpt_config.json").write_text(json.dumps(cfg, indent=2))
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "T2MGPTPipeline",
                    "_library_name": "motius",
                    "model_type": "t2mgpt",
                    "format": "motius-t2mgpt-v1",
                    "bundle_class": "motius.models.t2mgpt.bundle.T2MGPTBundle",
                    "pipeline_class": "motius.pipelines.t2mgpt.pipeline.T2MGPTPipeline",
                    "artifacts": {
                        "vqvae": "vq.safetensors" if safe_serialization else "vq.pt",
                        "gpt": "gpt.safetensors" if safe_serialization else "gpt.pt",
                        "clip": clip_artifact,
                        "mean": "Mean.npy",
                        "std": "Std.npy",
                    },
                    "components": {
                        "clip": {
                            "name": self.clip_name,
                            "stored_in_artifact": bool(include_clip),
                            "path": clip_artifact,
                        }
                    },
                    "external_components": {
                        "clip": {
                            "name": self.clip_name,
                            "stored_in_artifact": bool(include_clip),
                            "path": clip_artifact,
                        }
                    },
                    "api": {
                        "from_pretrained": (
                            "motius.models.t2mgpt.T2MGPTBundle"
                            ".from_pretrained"
                        ),
                        "from_config": (
                            "motius.models.t2mgpt.T2MGPTBundle"
                            ".from_config"
                        ),
                    },
                },
                indent=2,
            )
        )

        def _cpu_state(m):
            return {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(_cpu_state(self.vqvae), str(save_dir / "vq.safetensors"))
            save_file(_cpu_state(self.gpt), str(save_dir / "gpt.safetensors"))
            if include_clip:
                save_file(_cpu_state(self.clip_model), str(save_dir / "clip.safetensors"))
        else:
            torch.save(_cpu_state(self.vqvae), str(save_dir / "vq.pt"))
            torch.save(_cpu_state(self.gpt), str(save_dir / "gpt.pt"))
            if include_clip:
                torch.save(_cpu_state(self.clip_model), str(save_dir / "clip.pt"))

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a self-contained Motius T2M-GPT artifact (local dir or HF Hub id)."""
        path = Path(pretrained_model_name_or_path)
        if not (path / "t2mgpt_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "t2mgpt_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())
        vq_w = path / "vq.safetensors"
        if not vq_w.exists():
            vq_w = path / "vq.pt"
        gpt_w = path / "gpt.safetensors"
        if not gpt_w.exists():
            gpt_w = path / "gpt.pt"
        clip_w = path / "clip.safetensors"
        if not clip_w.exists():
            clip_w = path / "clip.pt"
        clip_name = kwargs.pop("clip_name", meta.get("clip_name", _DEFAULT_CLIP_NAME))
        load_clip = kwargs.get("load_clip", True)
        return cls(
            config=meta["config"],
            vq_weights_path=str(vq_w),
            gpt_weights_path=str(gpt_w),
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            clip_name=clip_name,
            clip_weights_path=str(clip_w) if load_clip and clip_w.exists() else None,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # device / forward helpers
    # ------------------------------------------------------------------
    def to_device(self, device):
        device = torch.device(device)
        self.vqvae.to(device)
        self.gpt.to(device)
        if self.clip_model is not None:
            self.clip_model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        """CLIP ViT-B/32 sentence features ``(B, 512)`` (float32).

        Mirrors the parity script: ``clip.tokenize(..., truncate=True)`` ->
        ``clip_model.encode_text`` -> ``.float()``.
        """
        import clip

        if self.clip_model is None:
            raise RuntimeError("CLIP text encoder not loaded (load_clip=False).")
        device = next(self.clip_model.parameters()).device
        text = clip.tokenize(list(captions), truncate=True).to(device)
        return self.clip_model.encode_text(text).float()

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        """Un-standardize HumanML3D-263 features back to physical scale."""
        return motion_263 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use T2MGPTPipeline.infer_t2m for inference.")
