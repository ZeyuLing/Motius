"""Self-contained MaskControl model bundle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.models.momask.network import LengthEstimator, ResidualTransformer, RVQVAE
from motius.registry import MODEL_BUNDLES

from .network import CONTROL_JOINT_IDS, MaskControlTransformer


_DIM_POSE = 263
_CLIP_VERSION = "ViT-B/32"
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
_CONTROL_DEFAULTS = {
    "latent_dim": 384,
    "ff_size": 1024,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.2,
    "cond_drop_prob": 0.1,
    "clip_dim": 512,
    "control_joint_ids": list(CONTROL_JOINT_IDS),
}
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
_LEN_DEFAULTS = {"input_size": 512, "output_size": 50}


def _resolve_hub_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path))


def _load_state(path: str, keys: Sequence[str] = ()) -> dict:
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _load_module(
    module: torch.nn.Module,
    path: str,
    *,
    keys: Sequence[str] = (),
    allowed_missing_prefixes: Sequence[str] = (),
) -> None:
    state = _load_state(path, keys)
    missing, unexpected = module.load_state_dict(state, strict=False)
    bad_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"failed loading {path}: missing={bad_missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


@MODEL_BUNDLES.register_module()
class MaskControlBundle(ModelBundle):
    """MaskControl HumanML3D bundle.

    The artifact stores the retrained base transformer and logits regularizer,
    RVQ-VAE, residual transformer, optional length estimator, CLIP ViT-B/32,
    and HumanML3D normalization statistics.  No upstream checkout is needed at
    inference time.
    """

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        control_weights_path: str,
        vq_weights_path: str,
        residual_weights_path: str,
        mean_path: str,
        std_path: str,
        length_weights_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        clip_version: str = _CLIP_VERSION,
        raw_control_checkpoint: bool = False,
        load_length_estimator: bool = True,
        device: Optional[str | torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        cfg = dict(config or {})
        vq_cfg = {**_VQ_DEFAULTS, **cfg.get("vq", {})}
        control_cfg = {**_CONTROL_DEFAULTS, **cfg.get("control", {})}
        residual_cfg = {**_RES_DEFAULTS, **cfg.get("residual", {})}
        length_cfg = {**_LEN_DEFAULTS, **cfg.get("length", {})}
        self._vq_cfg = vq_cfg
        self._control_cfg = control_cfg
        self._residual_cfg = residual_cfg
        self._length_cfg = length_cfg
        self.clip_version = clip_version

        if device is not None:
            initial_device = torch.device(device)
        elif torch.cuda.is_available():
            initial_device = torch.device("cuda")
        else:
            initial_device = torch.device("cpu")
        effective_clip = clip_weights_path or clip_version

        mean = np.asarray(np.load(mean_path), dtype=np.float32)
        std = np.asarray(np.load(std_path), dtype=np.float32)
        if mean.shape != (_DIM_POSE,) or std.shape != (_DIM_POSE,):
            raise ValueError(
                f"MaskControl mean/std must be 263-dim, got {mean.shape}/{std.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("MaskControl normalization statistics are invalid")

        vq_opt = SimpleNamespace(
            num_quantizers=vq_cfg["num_quantizers"],
            shared_codebook=vq_cfg["shared_codebook"],
            quantize_dropout_prob=vq_cfg["quantize_dropout_prob"],
            mu=vq_cfg["mu"],
        )
        vq_model = RVQVAE(
            vq_opt,
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
        _load_module(
            vq_model,
            vq_weights_path,
            keys=("vq_model", "net"),
        )

        common_opt = SimpleNamespace(
            num_tokens=vq_cfg["nb_code"],
            num_quantizers=vq_cfg["num_quantizers"],
            code_dim=vq_cfg["code_dim"],
            joints_num=22,
            device=initial_device,
        )
        control_model = MaskControlTransformer(
            code_dim=vq_cfg["code_dim"],
            cond_mode="text",
            vq_model=vq_model,
            mean=torch.from_numpy(mean),
            std=torch.from_numpy(std),
            control_joint_ids=control_cfg["control_joint_ids"],
            latent_dim=control_cfg["latent_dim"],
            ff_size=control_cfg["ff_size"],
            num_layers=control_cfg["n_layers"],
            num_heads=control_cfg["n_heads"],
            dropout=control_cfg["dropout"],
            clip_dim=control_cfg["clip_dim"],
            cond_drop_prob=control_cfg["cond_drop_prob"],
            clip_version=effective_clip,
            opt=common_opt,
        )
        control_state = _load_state(
            control_weights_path,
            ("ct2m_transformer",),
        )
        if raw_control_checkpoint:
            control_state = {
                key: value
                for key, value in control_state.items()
                if not key.startswith("vq_model.")
            }
        missing, unexpected = control_model.load_state_dict(
            control_state, strict=False
        )
        bad_missing = [
            key
            for key in missing
            if key not in {"mean", "std", "mask_emb_vq"}
            and not key.startswith(("clip_model.", "vq_model."))
        ]
        if bad_missing or unexpected:
            raise RuntimeError(
                "failed loading MaskControl transformer: "
                f"missing={bad_missing[:8]}, unexpected={unexpected[:8]}"
            )

        residual_opt = SimpleNamespace(**vars(common_opt))
        residual_model = ResidualTransformer(
            code_dim=vq_cfg["code_dim"],
            cond_mode="text",
            latent_dim=residual_cfg["latent_dim"],
            ff_size=residual_cfg["ff_size"],
            num_layers=residual_cfg["n_layers"],
            num_heads=residual_cfg["n_heads"],
            dropout=residual_cfg["dropout"],
            clip_dim=residual_cfg["clip_dim"],
            shared_codebook=vq_cfg["shared_codebook"],
            cond_drop_prob=residual_cfg["cond_drop_prob"],
            share_weight=residual_cfg["share_weight"],
            clip_version=effective_clip,
            opt=residual_opt,
        )
        _load_module(
            residual_model,
            residual_weights_path,
            keys=("res_transformer",),
            allowed_missing_prefixes=("clip_model.",),
        )

        length_estimator = None
        if load_length_estimator and length_weights_path:
            length_estimator = LengthEstimator(
                length_cfg["input_size"], length_cfg["output_size"]
            )
            _load_module(
                length_estimator,
                length_weights_path,
                keys=("estimator",),
            )

        for module in (vq_model, control_model, residual_model, length_estimator):
            if module is not None:
                module.eval()
                module.requires_grad_(False)
        self.vq_model = vq_model
        self.control_model = control_model
        self.residual_model = residual_model
        self.length_estimator = length_estimator
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)
        if device is not None or initial_device.type == "cuda":
            self.to_device(initial_device)

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def to_device(self, device):
        device = torch.device(device)
        self.vq_model.to(device)
        self.control_model.to(device)
        self.residual_model.to(device)
        if self.length_estimator is not None:
            self.length_estimator.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std + self.mean

    def generate(
        self,
        captions: Optional[Sequence[str]],
        frame_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_residual = bool(kwargs.pop("use_residual", True))
        return self.control_model.sample_motion(
            captions,
            frame_lengths,
            targets,
            target_mask,
            residual_model=self.residual_model if use_residual else None,
            **kwargs,
        )

    def config_dict(self) -> dict:
        return {
            "vq": dict(self._vq_cfg),
            "control": dict(self._control_cfg),
            "residual": dict(self._residual_cfg),
            "length": dict(self._length_cfg),
        }

    @staticmethod
    def _state_without(module: nn.Module, prefixes: Sequence[str]) -> dict:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in module.state_dict().items()
            if not any(key.startswith(prefix) for prefix in prefixes)
        }

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        include_clip: bool = True,
        **kwargs,
    ) -> str:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "safetensors" if safe_serialization else "pt"
        metadata = {
            "model_type": "maskcontrol",
            "format": "motius-maskcontrol-v1",
            "clip_version": self.clip_version,
            "has_length_estimator": self.length_estimator is not None,
            "config": self.config_dict(),
        }
        (save_dir / "maskcontrol_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MaskControlPipeline",
                    "_library_name": "motius",
                    "model_type": "maskcontrol",
                    "format": "motius-maskcontrol-v1",
                    "bundle_class": "motius.models.maskcontrol.MaskControlBundle",
                    "pipeline_class": "motius.pipelines.maskcontrol.MaskControlPipeline",
                    "tasks": [
                        "text-to-motion",
                        "temporal-control",
                        "body-part-timeline",
                        "sequential-generation",
                    ],
                    "motion_representation": "HumanML3D-263",
                    "artifacts": {
                        "control": f"control.{suffix}",
                        "vq": f"vq.{suffix}",
                        "residual": f"residual.{suffix}",
                        "length_estimator": (
                            f"length.{suffix}"
                            if self.length_estimator is not None
                            else None
                        ),
                        "clip": f"clip.{suffix}" if include_clip else None,
                        "mean": "Mean.npy",
                        "std": "Std.npy",
                    },
                },
                indent=2,
            )
            + "\n"
        )

        control_state = self._state_without(
            self.control_model, ("clip_model.", "vq_model.")
        )
        residual_state = self._state_without(
            self.residual_model, ("clip_model.",)
        )
        states = {
            "control": control_state,
            "vq": self.vq_model.state_dict(),
            "residual": residual_state,
        }
        if self.length_estimator is not None:
            states["length"] = self.length_estimator.state_dict()
        if include_clip:
            states["clip"] = self.control_model.clip_model.state_dict()
        if safe_serialization:
            from safetensors.torch import save_file

            for name, state in states.items():
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in state.items()
                    },
                    str(save_dir / f"{name}.safetensors"),
                )
        else:
            for name, state in states.items():
                torch.save(state, save_dir / f"{name}.pt")
        np.save(save_dir / "Mean.npy", self.mean.detach().cpu().numpy())
        np.save(save_dir / "Std.npy", self.std.detach().cpu().numpy())
        return str(save_dir)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = _resolve_hub_path(pretrained_model_name_or_path)
        config_path = path / "maskcontrol_config.json"
        if not config_path.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        metadata = json.loads(config_path.read_text())

        def artifact(name: str) -> Path:
            safe = path / f"{name}.safetensors"
            return safe if safe.exists() else path / f"{name}.pt"

        clip_path = artifact("clip")
        length_path = artifact("length")
        return cls(
            config=metadata["config"],
            control_weights_path=str(artifact("control")),
            vq_weights_path=str(artifact("vq")),
            residual_weights_path=str(artifact("residual")),
            length_weights_path=(
                str(length_path)
                if metadata.get("has_length_estimator") and length_path.exists()
                else None
            ),
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            clip_weights_path=str(clip_path) if clip_path.exists() else None,
            clip_version=metadata.get("clip_version", _CLIP_VERSION),
            raw_control_checkpoint=False,
            **kwargs,
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use MaskControlPipeline for inference")


__all__ = ["MaskControlBundle"]
