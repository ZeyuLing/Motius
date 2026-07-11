"""MoMask ModelBundle.

Wraps the Motius-native MoMask implementation (Guo et al., CVPR 2024) behind
a clean ``ModelBundle`` interface. The RVQ-VAE tokenizer, masked generative
transformer, residual transformer and length estimator live in
``motius.models.momask.network`` while preserving numerical parity
with the released HumanML3D checkpoints.

Architecture (see ``motius.models.momask.network``):

* **RVQVAE** (``vq_model``) — non-causal 1D-conv encoder/decoder with a
  **6-quantizer residual VQ** codebook (nb_code=512, code_dim=512). Tokenizes
  the 263-dim HumanML3D feature into a ``(T, 6)`` token grid.
* **MaskTransformer** (``t2m_transformer``) — confidence-based masked
  iterative decoder of the base (q=0) token map with classifier-free guidance.
* **ResidualTransformer** (``res_transformer``) — autoregressively predicts
  quantizers ``q=1..5`` conditioned on the lower layers.
* **LengthEstimator** (``length_estimator``, optional) — samples a motion
  length (in tokens) from the CLIP text embedding when no length is given.
* **CLIP ViT-B/32** text encoder — frozen, lives inside the two transformers,
  and is stored once in the Motius artifact as ``clip.safetensors``.

Representation: **HumanML3D-263** (263-dim, 20 fps, 22 joints). After
de-normalising with the RVQ-VAE training ``Mean`` / ``Std`` the raw 263 vectors
feed directly into ``HumanML263Evaluator``.
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

# Repo root: motius/models/momask/bundle.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_DATASET = "t2m"
_DEFAULT_VQ_NAME = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
_DEFAULT_T2M_NAME = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
_DEFAULT_RES_NAME = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
_DEFAULT_LEN_NAME = "length_estimator"

_CLIP_VERSION = "ViT-B/32"

# 263-dim HumanML3D feature.
_DIM_POSE = 263

# RVQ-VAE defaults (mirrors weights/t2m/rvq_nq6_*/opt.txt).
_VQ_DEFAULTS = {
    "nb_code": 512,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "vq_act": "relu",
    "vq_norm": None,
    "num_quantizers": 6,
    "shared_codebook": False,
    "quantize_dropout_prob": 0.2,
    "mu": 0.99,
}
# MaskTransformer defaults (mirrors weights/t2m/t2m_nlayer8_*/opt.txt).
_TRANS_DEFAULTS = {
    "latent_dim": 384,
    "ff_size": 1024,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.2,
    "cond_drop_prob": 0.1,
    "clip_dim": 512,
}
# ResidualTransformer defaults (mirrors weights/t2m/tres_nlayer8_*/opt.txt).
_RES_DEFAULTS = {
    "latent_dim": 384,
    "ff_size": 1024,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.2,
    "cond_drop_prob": 0.2,
    "share_weight": True,
    "clip_dim": 512,
}
# LengthEstimator defaults (LengthEstimator(512, 50)).
_LEN_DEFAULTS = {"input_size": 512, "output_size": 50}


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


@MODEL_BUNDLES.register_module()
class MoMaskBundle(ModelBundle):
    """MoMask text-to-motion bundle (HumanML3D-263, CLIP ViT-B/32 text)."""

    def __init__(
        self,
        # --- explicit raw upstream weights (converter/debug only) ---------
        weights_root: Optional[str] = None,
        dataset_name: str = _DEFAULT_DATASET,
        vq_name: str = _DEFAULT_VQ_NAME,
        t2m_name: str = _DEFAULT_T2M_NAME,
        res_name: str = _DEFAULT_RES_NAME,
        len_name: str = _DEFAULT_LEN_NAME,
        # --- self-contained Motius artifact ----------------------------
        config: Optional[dict] = None,
        vq_weights_path: Optional[str] = None,
        t2m_weights_path: Optional[str] = None,
        res_weights_path: Optional[str] = None,
        length_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        # --- shared -------------------------------------------------------
        clip_version: str = _CLIP_VERSION,
        load_length_estimator: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Construct the MoMask bundle.

        Two weight sources are supported:

        * **Raw upstream checkpoints** — the released ``.tar`` files under
          ``<weights_root>/<dataset_name>/<name>/model/`` plus
          ``meta/{mean,std}.npy``. This is what
          :func:`scripts.eval.convert_momask_checkpoint` consumes; the bundle
          never guesses a raw-checkout location.
        * **Self-contained Motius artifact** — ``config`` plus
          ``*_weights_path`` (safetensors) + ``clip.safetensors`` +
          ``Mean.npy`` / ``Std.npy``, as produced by :meth:`save_pretrained` /
          consumed by :meth:`from_pretrained`.
        """
        super().__init__()
        use_artifact = vq_weights_path is not None
        if use_artifact:
            missing = [
                name
                for name, value in (
                    ("t2m_weights_path", t2m_weights_path),
                    ("res_weights_path", res_weights_path),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"MoMask artifact load is missing: {', '.join(missing)}")
        elif weights_root is None:
            raise ValueError(
                "MoMaskBundle requires artifact weights or an explicit "
                "weights_root for raw checkpoint conversion."
            )

        from .network import (
            LengthEstimator,
            MaskTransformer,
            ResidualTransformer,
            RVQVAE,
        )

        self.clip_version = clip_version
        self._clip_weights_path = clip_weights_path
        effective_clip_version = clip_weights_path or clip_version
        # ``opt.device`` controls CLIP precision (cuda -> fp16 via
        # clip.model.convert_weights), matching the parity script. We pick the
        # final device up-front so CLIP text features are bit-identical.
        if device is not None:
            opt_device = torch.device(device)
        elif torch.cuda.is_available():
            opt_device = torch.device("cuda")
        else:
            opt_device = torch.device("cpu")

        cfg = dict(config) if config is not None else {}
        vq_cfg = {**_VQ_DEFAULTS, **cfg.get("vq", {})}
        trans_cfg = {**_TRANS_DEFAULTS, **cfg.get("t2m", {})}
        res_cfg = {**_RES_DEFAULTS, **cfg.get("res", {})}
        len_cfg = {**_LEN_DEFAULTS, **cfg.get("length", {})}
        self._vq_cfg = vq_cfg
        self._trans_cfg = trans_cfg
        self._res_cfg = res_cfg
        self._len_cfg = len_cfg

        # ----- RVQ-VAE ----------------------------------------------------
        vq_args = SimpleNamespace(
            num_quantizers=vq_cfg["num_quantizers"],
            shared_codebook=vq_cfg["shared_codebook"],
            quantize_dropout_prob=vq_cfg["quantize_dropout_prob"],
            mu=vq_cfg["mu"],
        )
        vq_model = RVQVAE(
            vq_args,
            _DIM_POSE,
            vq_cfg["nb_code"],
            vq_cfg["code_dim"],
            vq_cfg["output_emb_width"],
            vq_cfg["down_t"],
            vq_cfg["stride_t"],
            vq_cfg["width"],
            vq_cfg["depth"],
            vq_cfg["dilation_growth_rate"],
            vq_cfg["vq_act"],
            vq_cfg["vq_norm"],
        )

        num_tokens = vq_cfg["nb_code"]
        num_quantizers = vq_cfg["num_quantizers"]
        code_dim = vq_cfg["code_dim"]

        # ----- MaskTransformer (base tokens) ------------------------------
        trans_opt = SimpleNamespace(
            num_tokens=num_tokens,
            num_quantizers=num_quantizers,
            code_dim=code_dim,
            device=opt_device,
        )
        t2m_transformer = MaskTransformer(
            code_dim=code_dim,
            cond_mode="text",
            latent_dim=trans_cfg["latent_dim"],
            ff_size=trans_cfg["ff_size"],
            num_layers=trans_cfg["n_layers"],
            num_heads=trans_cfg["n_heads"],
            dropout=trans_cfg["dropout"],
            clip_dim=trans_cfg["clip_dim"],
            cond_drop_prob=trans_cfg["cond_drop_prob"],
            clip_version=effective_clip_version,
            opt=trans_opt,
        )

        # ----- ResidualTransformer (quantizers 1..Q-1) --------------------
        res_opt = SimpleNamespace(
            num_tokens=num_tokens,
            num_quantizers=num_quantizers,
            code_dim=code_dim,
            device=opt_device,
        )
        res_transformer = ResidualTransformer(
            code_dim=code_dim,
            cond_mode="text",
            latent_dim=res_cfg["latent_dim"],
            ff_size=res_cfg["ff_size"],
            num_layers=res_cfg["n_layers"],
            num_heads=res_cfg["n_heads"],
            dropout=res_cfg["dropout"],
            clip_dim=res_cfg["clip_dim"],
            shared_codebook=vq_cfg["shared_codebook"],
            cond_drop_prob=res_cfg["cond_drop_prob"],
            share_weight=res_cfg["share_weight"],
            clip_version=effective_clip_version,
            opt=res_opt,
        )

        # ----- LengthEstimator (optional) ---------------------------------
        length_estimator = None
        if load_length_estimator:
            length_estimator = LengthEstimator(
                len_cfg["input_size"], len_cfg["output_size"]
            )

        # ----- weight loading ---------------------------------------------
        if use_artifact:
            self._load_module(vq_model, vq_weights_path, key=None, allow_clip_missing=False)
            self._load_module(t2m_transformer, t2m_weights_path, key=None, allow_clip_missing=True)
            self._load_module(res_transformer, res_weights_path, key=None, allow_clip_missing=True)
            if length_estimator is not None and length_weights_path is not None:
                self._load_module(length_estimator, length_weights_path, key=None, allow_clip_missing=False)
        else:
            root = Path(weights_root)
            vq_dir = root / dataset_name / vq_name
            t2m_dir = root / dataset_name / t2m_name
            res_dir = root / dataset_name / res_name
            len_dir = root / dataset_name / len_name
            self._load_module(
                vq_model, str(vq_dir / "model" / "net_best_fid.tar"),
                key=("vq_model", "net"), allow_clip_missing=False,
            )
            self._load_module(
                t2m_transformer, str(t2m_dir / "model" / "latest.tar"),
                key=("t2m_transformer", "trans"), allow_clip_missing=True,
            )
            self._load_module(
                res_transformer, str(res_dir / "model" / "net_best_fid.tar"),
                key=("res_transformer",), allow_clip_missing=True,
            )
            if length_estimator is not None:
                self._load_module(
                    length_estimator, str(len_dir / "model" / "finest.tar"),
                    key=("estimator",), allow_clip_missing=False,
                )
            if mean_path is None:
                mean_path = str(vq_dir / "meta" / "mean.npy")
            if std_path is None:
                std_path = str(vq_dir / "meta" / "std.npy")

        vq_model.eval()
        t2m_transformer.eval()
        res_transformer.eval()
        if length_estimator is not None:
            length_estimator.eval()

        self.vq_model = vq_model
        self.t2m_transformer = t2m_transformer
        self.res_transformer = res_transformer
        self.length_estimator = length_estimator
        self.njoints = _DIM_POSE
        self.nfeats = 1

        # ----- normalization buffers (263-dim) ----------------------------
        if mean_path is None or std_path is None:
            raise ValueError("MoMaskBundle needs Mean/Std (mean_path/std_path).")
        mean = np.load(str(mean_path)).astype(np.float32)
        std = np.load(str(std_path)).astype(np.float32)
        if mean.shape != (_DIM_POSE,) or std.shape != (_DIM_POSE,):
            raise ValueError(
                f"expected 263-dim mean/std, got {mean.shape} and {std.shape}"
            )
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)
        elif opt_device.type == "cuda":
            self.to_device(opt_device)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_state(ckpt, key):
        """Pull the relevant sub-state-dict out of a raw MoMask ``.tar``."""
        if key is None:
            return ckpt
        if isinstance(ckpt, dict):
            for k in key:
                if k in ckpt:
                    return ckpt[k]
        return ckpt

    def _load_module(self, module, path: str, key, allow_clip_missing: bool):
        p = str(path)
        if p.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(p)
        else:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            sd = self._extract_state(ckpt, key)
        missing, unexpected = module.load_state_dict(sd, strict=False)
        assert len(unexpected) == 0, f"unexpected keys loading {p}: {unexpected[:5]}"
        if allow_clip_missing:
            bad = [k for k in missing if not k.startswith("clip_model.")]
            assert not bad, f"unexpected missing keys loading {p}: {bad[:5]}"
        else:
            assert not missing, f"unexpected missing keys loading {p}: {missing[:5]}"

    @staticmethod
    def _state_dict_no_clip(module) -> dict:
        return {
            k: v.detach().cpu().contiguous()
            for k, v in module.state_dict().items()
            if not k.startswith("clip_model.")
        }

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, raw-checkout-independent)
    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return {
            "vq": dict(self._vq_cfg),
            "t2m": dict(self._trans_cfg),
            "res": dict(self._res_cfg),
            "length": dict(self._len_cfg),
        }

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        include_clip: bool = True,
        **kwargs,
    ):
        """Export a self-contained Motius MoMask artifact.

        Layout::

            <dir>/momask_config.json   # arch config for all sub-modules
            <dir>/vq.safetensors       # RVQ-VAE weights
            <dir>/t2m_trans.safetensors  # MaskTransformer non-CLIP weights
            <dir>/res_trans.safetensors  # ResidualTransformer non-CLIP weights
            <dir>/length_est.safetensors # LengthEstimator (if present)
            <dir>/clip.safetensors       # shared CLIP ViT-B/32 text encoder
            <dir>/Mean.npy, Std.npy    # 263-dim denorm stats
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)
        clip_artifact = (
            ("clip.safetensors" if safe_serialization else "clip.pt")
            if include_clip
            else None
        )

        cfg = {
            "model_type": "momask",
            "format": "motius-momask-v1",
            "clip_version": self.clip_version,
            "has_length_estimator": self.length_estimator is not None,
            "components": {
                "clip": {
                    "name": self.clip_version,
                    "stored_in_artifact": bool(include_clip),
                    "path": clip_artifact,
                }
            },
            "config": self.config_dict(),
        }
        (save_dir / "momask_config.json").write_text(json.dumps(cfg, indent=2))
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MoMaskPipeline",
                    "_library_name": "motius",
                    "model_type": "momask",
                    "format": "motius-momask-v1",
                    "bundle_class": "motius.models.momask.bundle.MoMaskBundle",
                    "pipeline_class": "motius.pipelines.momask.pipeline.MoMaskPipeline",
                    "artifacts": {
                        "vq": "vq.safetensors" if safe_serialization else "vq.pt",
                        "t2m_transformer": "t2m_trans.safetensors"
                        if safe_serialization else "t2m_trans.pt",
                        "res_transformer": "res_trans.safetensors"
                        if safe_serialization else "res_trans.pt",
                        "length_estimator": (
                            "length_est.safetensors"
                            if safe_serialization and self.length_estimator is not None
                            else (
                                "length_est.pt"
                                if self.length_estimator is not None
                                else None
                            )
                        ),
                        "clip": clip_artifact,
                        "mean": "Mean.npy",
                        "std": "Std.npy",
                    },
                    "components": {
                        "clip": {
                            "name": self.clip_version,
                            "stored_in_artifact": bool(include_clip),
                            "path": clip_artifact,
                        }
                    },
                    "external_components": {
                        "clip": {
                            "name": self.clip_version,
                            "stored_in_artifact": bool(include_clip),
                            "path": clip_artifact,
                        }
                    },
                    "api": {
                        "from_pretrained": (
                            "motius.models.momask.MoMaskBundle"
                            ".from_pretrained"
                        ),
                        "from_config": (
                            "motius.models.momask.MoMaskBundle"
                            ".from_config"
                        ),
                    },
                },
                indent=2,
            )
        )

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(self.vq_model.state_dict(), str(save_dir / "vq.safetensors"))
            save_file(self._state_dict_no_clip(self.t2m_transformer), str(save_dir / "t2m_trans.safetensors"))
            save_file(self._state_dict_no_clip(self.res_transformer), str(save_dir / "res_trans.safetensors"))
            if self.length_estimator is not None:
                save_file(self.length_estimator.state_dict(), str(save_dir / "length_est.safetensors"))
            if include_clip:
                save_file(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in self.t2m_transformer.clip_model.state_dict().items()
                    },
                    str(save_dir / "clip.safetensors"),
                )
        else:
            torch.save(self.vq_model.state_dict(), str(save_dir / "vq.pt"))
            torch.save(self._state_dict_no_clip(self.t2m_transformer), str(save_dir / "t2m_trans.pt"))
            torch.save(self._state_dict_no_clip(self.res_transformer), str(save_dir / "res_trans.pt"))
            if self.length_estimator is not None:
                torch.save(self.length_estimator.state_dict(), str(save_dir / "length_est.pt"))
            if include_clip:
                torch.save(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in self.t2m_transformer.clip_model.state_dict().items()
                    },
                    str(save_dir / "clip.pt"),
                )

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a self-contained Motius MoMask artifact (local dir or HF Hub id)."""
        path = Path(pretrained_model_name_or_path)
        if not (path / "momask_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "momask_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())

        def _w(name):
            st = path / f"{name}.safetensors"
            return st if st.exists() else path / f"{name}.pt"

        load_length = kwargs.pop("load_length_estimator", meta.get("has_length_estimator", True))
        length_w = _w("length_est")
        clip_w = _w("clip")
        clip_version = kwargs.pop("clip_version", meta.get("clip_version", _CLIP_VERSION))
        return cls(
            config=meta["config"],
            vq_weights_path=str(_w("vq")),
            t2m_weights_path=str(_w("t2m_trans")),
            res_weights_path=str(_w("res_trans")),
            length_weights_path=str(length_w) if length_w.exists() else None,
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            clip_weights_path=str(clip_w) if clip_w.exists() else None,
            clip_version=clip_version,
            load_length_estimator=load_length and length_w.exists(),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # device / forward helpers
    # ------------------------------------------------------------------
    def to_device(self, device):
        device = torch.device(device)
        self.vq_model.to(device)
        self.t2m_transformer.to(device)
        self.res_transformer.to(device)
        if self.length_estimator is not None:
            self.length_estimator.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        """CLIP ViT-B/32 sentence features ``(B, 512)`` (via the t2m transformer)."""
        return self.t2m_transformer.encode_text(list(captions))

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        """Un-standardize HumanML3D-263 features back to physical scale."""
        return motion_263 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use MoMaskPipeline.infer_t2m for inference.")
