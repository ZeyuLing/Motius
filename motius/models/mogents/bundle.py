"""MoGenTS ModelBundle.

Wraps the Motius-native MoGenTS implementation (Yuan et al., NeurIPS 2024).
MoGenTS extends MoMask-style masked token generation with **dual spatial-
temporal modeling**:

* ``vq_model`` - dual-stream RVQ-VAE. It tokenizes HumanML3D-263 into a 1D
  auxiliary token stream ``(T, Q)`` and a 2D spatial-temporal token grid
  ``(T, 6, Q)``.
* ``mask_transformer_aux`` / ``mask_transformer_ts`` - masked iterative
  decoders for the 1D and 2D base token maps.
* ``res_transformer_aux`` / ``res_transformer_ts`` - residual decoders for
  quantizers ``q=1..5`` in each stream.
* ``length_estimator`` - optional MoMask-compatible CLIP length predictor.

Representation: **HumanML3D-263** (263-dim, 20 fps, 22 joints). The generated
motion returned by the VQ-VAE is de-normalized with the stored ``Mean`` / ``Std``.
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

_DEFAULT_DATASET = "humanml3d"
_DEFAULT_VQ_NAME = "pretrain_vq"
_DEFAULT_MTRANS_NAME = "pretrain_mtrans"
_DEFAULT_RTRANS_NAME = "pretrain_rtrans"
_DEFAULT_LEN_NAME = "length_estimator"
_DEFAULT_MASK_CKPT = "net_best_fid.tar"
_DEFAULT_RES_CKPT = "net_best_fid.tar"
_DEFAULT_VQ_CKPT = "net_best_fid.tar"
_DEFAULT_LEN_CKPT = "finest.tar"
_CLIP_VERSION = "ViT-B/32"
_DIM_POSE = 263
_UNIT_LENGTH = 4
_N_JOINT_GROUPS = 6

_VQ_DEFAULTS = {
    "dataset_name": _DEFAULT_DATASET,
    "code_dim1d": 512,
    "nb_code1d": 512,
    # Official HumanML3D pretrained command uses these 2D-code settings.
    "code_dim2d": 1024,
    "nb_code2d": 256,
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
_MASK_DEFAULTS = {
    "latent_dim": 384,
    "ff_size": 1024,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.2,
    "cond_drop_prob": 0.1,
    "clip_dim": 512,
    "attnj": True,
    "attnt": True,
}
_RES_DEFAULTS = {
    "latent_dim": 384,
    "ff_size": 1024,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.2,
    "cond_drop_prob": 0.01,
    "clip_dim": 512,
    "attnj": True,
    "attnt": True,
    "share_weight": True,
}
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
class MoGenTSBundle(ModelBundle):
    """MoGenTS text-to-motion bundle (HumanML3D-263, CLIP ViT-B/32 text)."""

    def __init__(
        self,
        # --- explicit raw upstream weights ---------------------------------
        weights_root: Optional[str] = None,
        dataset_name: str = _DEFAULT_DATASET,
        vq_name: str = _DEFAULT_VQ_NAME,
        mtrans_name: str = _DEFAULT_MTRANS_NAME,
        rtrans_name: str = _DEFAULT_RTRANS_NAME,
        len_name: str = _DEFAULT_LEN_NAME,
        length_root: Optional[str] = "checkpoints",
        vq_ckpt_name: str = _DEFAULT_VQ_CKPT,
        mask_ckpt_name: str = _DEFAULT_MASK_CKPT,
        res_ckpt_name: str = _DEFAULT_RES_CKPT,
        len_ckpt_name: str = _DEFAULT_LEN_CKPT,
        # --- self-contained Motius artifact -----------------------------
        config: Optional[dict] = None,
        vq_weights_path: Optional[str] = None,
        mask_aux_weights_path: Optional[str] = None,
        mask_ts_weights_path: Optional[str] = None,
        res_aux_weights_path: Optional[str] = None,
        res_ts_weights_path: Optional[str] = None,
        length_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        # --- shared ---------------------------------------------------------
        clip_version: str = _CLIP_VERSION,
        load_generation_models: bool = True,
        load_length_estimator: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        use_artifact = vq_weights_path is not None
        if use_artifact:
            if load_generation_models:
                missing = [
                    name
                    for name, value in (
                        ("mask_aux_weights_path", mask_aux_weights_path),
                        ("mask_ts_weights_path", mask_ts_weights_path),
                        ("res_aux_weights_path", res_aux_weights_path),
                        ("res_ts_weights_path", res_ts_weights_path),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError(f"MoGenTS artifact load is missing: {', '.join(missing)}")
        elif weights_root is None:
            raise ValueError(
                "MoGenTSBundle requires artifact weights or an explicit "
                "weights_root for raw checkpoint conversion."
            )

        from .network import (
            LengthEstimator,
            MaskTransformer,
            MaskTransformer2D,
            ResidualTransformer,
            ResidualTransformer2D,
            RVQVAE,
        )

        if device is not None:
            opt_device = torch.device(device)
        elif torch.cuda.is_available():
            opt_device = torch.device("cuda")
        else:
            opt_device = torch.device("cpu")

        self.clip_version = clip_version
        self._clip_weights_path = clip_weights_path
        effective_clip_version = clip_weights_path or clip_version

        cfg = dict(config) if config is not None else {}
        vq_cfg = {**_VQ_DEFAULTS, **cfg.get("vq", {})}
        mask_cfg = {**_MASK_DEFAULTS, **cfg.get("mask", {})}
        res_cfg = {**_RES_DEFAULTS, **cfg.get("res", {})}
        len_cfg = {**_LEN_DEFAULTS, **cfg.get("length", {})}
        vq_cfg["dataset_name"] = dataset_name if not use_artifact else vq_cfg.get("dataset_name", dataset_name)
        self._vq_cfg = vq_cfg
        self._mask_cfg = mask_cfg
        self._res_cfg = res_cfg
        self._len_cfg = len_cfg

        vq_args = SimpleNamespace(**vq_cfg)
        vq_model = RVQVAE(
            vq_args,
            _DIM_POSE,
            vq_cfg["down_t"],
            vq_cfg["stride_t"],
            vq_cfg["width"],
            vq_cfg["depth"],
            vq_cfg["dilation_growth_rate"],
            vq_cfg["vq_act"],
            vq_cfg["vq_norm"],
        )

        mask_transformer_aux = None
        mask_transformer_ts = None
        res_transformer_aux = None
        res_transformer_ts = None
        if load_generation_models:
            mask_opt = SimpleNamespace(
                num_tokens1d=vq_cfg["nb_code1d"],
                num_tokens2d=vq_cfg["nb_code2d"],
                num_quantizers=vq_cfg["num_quantizers"],
                device=opt_device,
                attnj=mask_cfg["attnj"],
                attnt=mask_cfg["attnt"],
            )
            mask_transformer_aux = MaskTransformer(
                code_dim=vq_cfg["code_dim1d"],
                cond_mode="text",
                latent_dim=mask_cfg["latent_dim"],
                ff_size=mask_cfg["ff_size"],
                num_layers=mask_cfg["n_layers"],
                num_heads=mask_cfg["n_heads"],
                dropout=mask_cfg["dropout"],
                clip_dim=mask_cfg["clip_dim"],
                cond_drop_prob=mask_cfg["cond_drop_prob"],
                clip_version=effective_clip_version,
                opt=mask_opt,
            )
            mask_transformer_ts = MaskTransformer2D(
                code_dim=vq_cfg["code_dim2d"],
                cond_mode="text",
                latent_dim=mask_cfg["latent_dim"],
                ff_size=mask_cfg["ff_size"],
                num_layers=mask_cfg["n_layers"],
                num_heads=mask_cfg["n_heads"],
                dropout=mask_cfg["dropout"],
                clip_dim=mask_cfg["clip_dim"],
                cond_drop_prob=mask_cfg["cond_drop_prob"],
                clip_version=effective_clip_version,
                opt=mask_opt,
            )

            res_opt = SimpleNamespace(
                num_tokens1d=vq_cfg["nb_code1d"],
                num_tokens2d=vq_cfg["nb_code2d"],
                num_quantizers=vq_cfg["num_quantizers"],
                device=opt_device,
                attnj=res_cfg["attnj"],
                attnt=res_cfg["attnt"],
            )
            res_transformer_aux = ResidualTransformer(
                code_dim=vq_cfg["code_dim1d"],
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
            res_transformer_ts = ResidualTransformer2D(
                code_dim=vq_cfg["code_dim2d"],
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

        length_estimator = None
        if load_generation_models and load_length_estimator:
            length_estimator = LengthEstimator(len_cfg["input_size"], len_cfg["output_size"])

        if use_artifact:
            self._load_module(vq_model, vq_weights_path, key=None, allow_clip_missing=False)
            if load_generation_models:
                self._load_module(mask_transformer_aux, mask_aux_weights_path, key=None, allow_clip_missing=True)
                self._load_module(mask_transformer_ts, mask_ts_weights_path, key=None, allow_clip_missing=True)
                self._load_module(res_transformer_aux, res_aux_weights_path, key=None, allow_clip_missing=True)
                self._load_module(res_transformer_ts, res_ts_weights_path, key=None, allow_clip_missing=True)
            if length_estimator is not None and length_weights_path is not None:
                self._load_module(length_estimator, length_weights_path, key=None, allow_clip_missing=False)
        else:
            root = Path(weights_root)
            vq_dir = root / dataset_name / vq_name
            mask_dir = root / dataset_name / mtrans_name
            res_dir = root / dataset_name / rtrans_name
            self._load_module(
                vq_model,
                str(vq_dir / "model" / vq_ckpt_name),
                key=("vq_model", "net"),
                allow_clip_missing=False,
            )
            if load_generation_models:
                self._load_module(
                    mask_transformer_aux,
                    str(mask_dir / "model" / mask_ckpt_name),
                    key=("t2m_transformer_aux",),
                    allow_clip_missing=True,
                )
                self._load_module(
                    mask_transformer_ts,
                    str(mask_dir / "model" / mask_ckpt_name),
                    key=("t2m_transformer_ts",),
                    allow_clip_missing=True,
                )
                self._load_module(
                    res_transformer_aux,
                    str(res_dir / "model" / res_ckpt_name),
                    key=("res_transformer_aux",),
                    allow_clip_missing=True,
                )
                self._load_module(
                    res_transformer_ts,
                    str(res_dir / "model" / res_ckpt_name),
                    key=("res_transformer_ts",),
                    allow_clip_missing=True,
                )
            if length_estimator is not None:
                if length_weights_path is None:
                    if length_root is None:
                        raise ValueError(
                            "load_length_estimator=True needs length_weights_path "
                            "or length_root."
                        )
                    length_weights_path = str(
                        Path(length_root)
                        / dataset_name
                        / len_name
                        / "model"
                        / len_ckpt_name
                    )
                self._load_module(
                    length_estimator,
                    str(length_weights_path),
                    key=("estimator",),
                    allow_clip_missing=False,
                )
            if mean_path is None:
                mean_path = str(vq_dir / "meta" / "mean.npy")
            if std_path is None:
                std_path = str(vq_dir / "meta" / "std.npy")

        for module in (
            vq_model,
            mask_transformer_aux,
            mask_transformer_ts,
            res_transformer_aux,
            res_transformer_ts,
        ):
            if module is None:
                continue
            module.eval()
        if length_estimator is not None:
            length_estimator.eval()

        self.vq_model = vq_model
        self.mask_transformer_aux = mask_transformer_aux
        self.mask_transformer_ts = mask_transformer_ts
        self.res_transformer_aux = res_transformer_aux
        self.res_transformer_ts = res_transformer_ts
        self.length_estimator = length_estimator
        self.njoints = _DIM_POSE
        self.nfeats = 1
        self.unit_length = _UNIT_LENGTH
        self.n_joint_groups = _N_JOINT_GROUPS

        if mean_path is None or std_path is None:
            raise ValueError("MoGenTSBundle needs Mean/Std (mean_path/std_path).")
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

    @staticmethod
    def _extract_state(ckpt, key):
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

    def config_dict(self) -> dict:
        return {
            "vq": dict(self._vq_cfg),
            "mask": dict(self._mask_cfg),
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
        """Export a self-contained Motius MoGenTS artifact."""
        import os

        if any(
            module is None
            for module in (
                self.mask_transformer_aux,
                self.mask_transformer_ts,
                self.res_transformer_aux,
                self.res_transformer_ts,
            )
        ):
            raise ValueError("save_pretrained requires load_generation_models=True")

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)
        clip_artifact = (
            ("clip.safetensors" if safe_serialization else "clip.pt")
            if include_clip
            else None
        )

        cfg = {
            "model_type": "mogents",
            "format": "motius-mogents-v1",
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
        (save_dir / "mogents_config.json").write_text(json.dumps(cfg, indent=2))
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MoGenTSPipeline",
                    "_library_name": "motius",
                    "model_type": "mogents",
                    "format": "motius-mogents-v1",
                    "bundle_class": "motius.models.mogents.bundle.MoGenTSBundle",
                    "pipeline_class": "motius.pipelines.mogents.pipeline.MoGenTSPipeline",
                    "artifacts": {
                        "vq": "vq.safetensors" if safe_serialization else "vq.pt",
                        "mask_aux": "mask_aux.safetensors" if safe_serialization else "mask_aux.pt",
                        "mask_ts": "mask_ts.safetensors" if safe_serialization else "mask_ts.pt",
                        "res_aux": "res_aux.safetensors" if safe_serialization else "res_aux.pt",
                        "res_ts": "res_ts.safetensors" if safe_serialization else "res_ts.pt",
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
                    "api": {
                        "from_pretrained": (
                            "motius.models.mogents.MoGenTSBundle"
                            ".from_pretrained"
                        ),
                        "from_config": (
                            "motius.models.mogents.MoGenTSBundle"
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
            save_file(self._state_dict_no_clip(self.mask_transformer_aux), str(save_dir / "mask_aux.safetensors"))
            save_file(self._state_dict_no_clip(self.mask_transformer_ts), str(save_dir / "mask_ts.safetensors"))
            save_file(self._state_dict_no_clip(self.res_transformer_aux), str(save_dir / "res_aux.safetensors"))
            save_file(self._state_dict_no_clip(self.res_transformer_ts), str(save_dir / "res_ts.safetensors"))
            if self.length_estimator is not None:
                save_file(self.length_estimator.state_dict(), str(save_dir / "length_est.safetensors"))
            if include_clip:
                save_file(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in self.mask_transformer_ts.clip_model.state_dict().items()
                    },
                    str(save_dir / "clip.safetensors"),
                )
        else:
            torch.save(self.vq_model.state_dict(), str(save_dir / "vq.pt"))
            torch.save(self._state_dict_no_clip(self.mask_transformer_aux), str(save_dir / "mask_aux.pt"))
            torch.save(self._state_dict_no_clip(self.mask_transformer_ts), str(save_dir / "mask_ts.pt"))
            torch.save(self._state_dict_no_clip(self.res_transformer_aux), str(save_dir / "res_aux.pt"))
            torch.save(self._state_dict_no_clip(self.res_transformer_ts), str(save_dir / "res_ts.pt"))
            if self.length_estimator is not None:
                torch.save(self.length_estimator.state_dict(), str(save_dir / "length_est.pt"))
            if include_clip:
                torch.save(
                    {
                        k: v.detach().cpu().contiguous()
                        for k, v in self.mask_transformer_ts.clip_model.state_dict().items()
                    },
                    str(save_dir / "clip.pt"),
                )

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "mogents_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "mogents_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())

        def _w(name):
            st = path / f"{name}.safetensors"
            return st if st.exists() else path / f"{name}.pt"

        load_length = kwargs.pop("load_length_estimator", meta.get("has_length_estimator", True))
        load_generation = kwargs.pop("load_generation_models", True)
        length_w = _w("length_est")
        clip_w = _w("clip")
        clip_version = kwargs.pop("clip_version", meta.get("clip_version", _CLIP_VERSION))
        return cls(
            config=meta["config"],
            vq_weights_path=str(_w("vq")),
            mask_aux_weights_path=str(_w("mask_aux")),
            mask_ts_weights_path=str(_w("mask_ts")),
            res_aux_weights_path=str(_w("res_aux")),
            res_ts_weights_path=str(_w("res_ts")),
            length_weights_path=str(length_w) if length_w.exists() else None,
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            clip_weights_path=str(clip_w) if clip_w.exists() else None,
            clip_version=clip_version,
            load_generation_models=load_generation,
            load_length_estimator=load_length and length_w.exists(),
            **kwargs,
        )

    def to_device(self, device):
        device = torch.device(device)
        self.vq_model.to(device)
        if self.mask_transformer_aux is not None:
            self.mask_transformer_aux.to(device)
        if self.mask_transformer_ts is not None:
            self.mask_transformer_ts.to(device)
        if self.res_transformer_aux is not None:
            self.res_transformer_aux.to(device)
        if self.res_transformer_ts is not None:
            self.res_transformer_ts.to(device)
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
        if self.mask_transformer_ts is None:
            raise RuntimeError("encode_text requires load_generation_models=True")
        return self.mask_transformer_ts.encode_text(list(captions))

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError("Use MoGenTSPipeline.infer_t2m for inference.")
