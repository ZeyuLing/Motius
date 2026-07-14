"""Motius bundle for the official MotionCLR HumanML3D checkpoint.

The network source is adapted from IDEA-Research/MotionCLR and remains under
the bundled IDEA License 1.0. Runtime loading never imports the reference
checkout.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


MOTIONCLR_REPO_ID = "EvanTHU/MotionCLR"
MOTIONCLR_SOURCE_REPOSITORY = "https://github.com/IDEA-Research/MotionCLR"
MOTIONCLR_SOURCE_REVISION = "a6f44a791940682fe335c82f1b436bae05a1cebb"
MOTIONCLR_CHECKPOINT_SHA256 = (
    "5852e139bbe45f5ca45b67b72cc54ab02b7da7ae18b42f27ea630a715c5c2b5f"
)
MOTIONCLR_MEAN_SHA256 = (
    "0bdb5ba69a3a9e34d71990db15bc535ebc024c8d95ddb5574196f96058faa7d3"
)
MOTIONCLR_STD_SHA256 = (
    "487855309295f986d08e96d65e415fb6b2a94211ac34ce444007e84cba8f33bb"
)
MOTIONCLR_DIM = 263
MOTIONCLR_FPS = 20.0
MOTIONCLR_MAX_FRAMES = 196
_ARTIFACT_FORMAT = "motius-motionclr-v1"

DEFAULT_NETWORK_CONFIG = {
    "input_feats": MOTIONCLR_DIM,
    "base_dim": 512,
    "dim_mults": [2, 2, 2, 2],
    "dims": None,
    "adagn": True,
    "zero": True,
    "dropout": 0.1,
    "no_eff": True,
    "time_dim": 512,
    "latent_dim": 512,
    "cond_mask_prob": 0.1,
    "clip_dim": 512,
    "clip_version": "ViT-B/32",
    "text_latent_dim": 256,
    "text_ff_size": 2048,
    "text_num_heads": 4,
    "activation": "gelu",
    "num_text_layers": 4,
    "self_attention": True,
    "vis_attn": False,
}


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(
            str(path), map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:  # torch < 2.0
        return torch.load(str(path), map_location="cpu")
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(str(path), map_location="cpu", weights_only=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_official_hashes(layout: Mapping[str, Path]) -> None:
    expected = {
        "checkpoint": MOTIONCLR_CHECKPOINT_SHA256,
        "mean": MOTIONCLR_MEAN_SHA256,
        "std": MOTIONCLR_STD_SHA256,
    }
    for name, digest in expected.items():
        actual = _sha256(layout[name])
        if actual != digest:
            raise ValueError(
                f"Official MotionCLR {name} SHA256 mismatch: expected {digest}, got {actual}"
            )


def _load_state(path: str | Path, *, use_ema: bool = True) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"MotionCLR checkpoint must contain a mapping: {path}")
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        state = dict(payload)
    elif use_ema and isinstance(payload.get("model_ema"), Mapping):
        state = dict(payload["model_ema"])
    elif isinstance(payload.get("encoder"), Mapping):
        state = dict(payload["encoder"])
    elif isinstance(payload.get("state_dict"), Mapping):
        state = dict(payload["state_dict"])
    else:
        raise ValueError(
            f"MotionCLR checkpoint {path} has no model_ema, encoder, or state_dict"
        )
    state.pop("n_averaged", None)
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def _parse_official_options(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    options: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        try:
            value = ast.literal_eval(raw_value.strip())
        except (SyntaxError, ValueError):
            value = raw_value.strip()
        options[key.strip()] = value
    return options


def _network_config_from_options(options: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_NETWORK_CONFIG)
    direct = (
        "base_dim",
        "dim_mults",
        "dropout",
        "time_dim",
        "latent_dim",
        "cond_mask_prob",
        "self_attention",
        "vis_attn",
        "text_latent_dim",
    )
    for key in direct:
        if key in options:
            config[key] = options[key]
    if "dim_pose" in options:
        config["input_feats"] = int(options["dim_pose"])
    config["adagn"] = not bool(options.get("no_adagn", not config["adagn"]))
    config["no_eff"] = bool(options.get("no_eff", config["no_eff"]))
    config["dim_mults"] = [int(value) for value in config["dim_mults"]]
    return config


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Optional[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return None


def _official_layout(path: Path) -> dict[str, Optional[Path]]:
    if path.is_file():
        checkpoint = path
        roots = [path.parent, path.parent.parent]
    else:
        checkpoint = _first_existing(path, ("model/latest.tar", "latest.tar"))
        roots = [path]
    mean = std = options = None
    for root in roots:
        mean = mean or _first_existing(root, ("meta/mean.npy", "mean.npy", "Mean.npy"))
        std = std or _first_existing(root, ("meta/std.npy", "std.npy", "Std.npy"))
        options = options or _first_existing(root, ("opt.txt",))
    return {"checkpoint": checkpoint, "mean": mean, "std": std, "options": options}


def _download_hub_layout(repo_id: str) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    files = set(list_repo_files(repo_id))
    if "motionclr_config.json" in files:
        path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "motionclr_config.json",
                    "model_index.json",
                    "model.safetensors",
                    "model.pt",
                    "Mean.npy",
                    "Std.npy",
                    "LICENSE*",
                    "ATTRIBUTIONS*",
                    "README*",
                    "clip/**",
                ],
            )
        )
        return {"artifact": path}

    def choose(*names: str) -> str:
        for name in names:
            if name in files:
                return name
        raise FileNotFoundError(
            f"{repo_id} is not a complete MotionCLR repository; missing one of {names}"
        )

    # Download one copy of each large/small asset. EvanTHU/MotionCLR publishes
    # duplicate flat and nested model files, each checkpoint being about 6.7 GB.
    names = {
        "checkpoint": choose("model/latest.tar", "latest.tar"),
        "mean": choose("meta/mean.npy", "mean.npy", "Mean.npy"),
        "std": choose("meta/std.npy", "std.npy", "Std.npy"),
        "options": choose("opt.txt"),
    }
    return {
        key: Path(hf_hub_download(repo_id=repo_id, filename=name))
        for key, name in names.items()
    }


def _load_stats(
    value: Optional[np.ndarray | torch.Tensor],
    path: Optional[str | Path],
    name: str,
) -> torch.Tensor:
    if value is None:
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"MotionCLR requires a valid {name} statistics file")
        value = np.load(str(path))
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (MOTIONCLR_DIM,):
        raise ValueError(f"MotionCLR {name} must have shape (263,), got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"MotionCLR {name} contains non-finite values")
    if name.lower().startswith("std") and torch.any(tensor <= 0):
        raise ValueError("MotionCLR Std must be strictly positive")
    return tensor


def _resolve_dtype(value: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if value is None or isinstance(value, torch.dtype):
        return value
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MotionCLR torch_dtype: {value}") from exc


@MODEL_BUNDLES.register_module()
class MotionCLRBundle(ModelBundle):
    """Official MotionCLR T2M bundle for normalized HumanML3D-263 motion."""

    SUPPORTED_TASKS = {
        "text_to_motion": "HumanML3D-263 text-to-motion generation at 20 fps",
    }

    def __init__(
        self,
        network_config: Optional[Mapping[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        mean: Optional[np.ndarray | torch.Tensor] = None,
        std: Optional[np.ndarray | torch.Tensor] = None,
        use_ema: bool = True,
        load_model: bool = True,
        load_clip: bool = True,
        clip_path: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        torch_dtype: Optional[str | torch.dtype] = None,
        diffuser_name: str = "dpmsolver",
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        network: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        config = dict(DEFAULT_NETWORK_CONFIG)
        config.update(dict(network_config or {}))
        config["dim_mults"] = [int(value) for value in config["dim_mults"]]
        if int(config["input_feats"]) != MOTIONCLR_DIM:
            raise ValueError("MotionCLR artifacts must use HumanML3D-263 input_feats=263")
        if not bool(config["no_eff"]):
            raise NotImplementedError(
                "Only no_eff=True is supported because the released source omits "
                "the LinearCrossAttention implementation."
            )
        self.network_config = config
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.weights_path = str(weights_path) if weights_path else None
        self.use_ema = bool(use_ema)
        self.load_clip = bool(load_clip)
        self.clip_path = str(clip_path) if clip_path else None
        self.diffuser_name = str(diffuser_name)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.inference_dtype = _resolve_dtype(torch_dtype)
        self.register_buffer("mean", _load_stats(mean, mean_path, "Mean"), persistent=True)
        self.register_buffer("std", _load_stats(std, std_path, "Std"), persistent=True)

        self.network: Optional[nn.Module] = network
        if load_model and self.network is None:
            from .network import MotionCLR

            self.network = MotionCLR(
                **self.network_config,
                load_clip=self.load_clip,
                clip_path=self.clip_path,
            )
            source = self.weights_path or self.checkpoint_path
            if source is None:
                raise FileNotFoundError(
                    "MotionCLR load_model=True requires checkpoint_path or weights_path"
                )
            state = _load_state(source, use_ema=self.use_ema)
            if self.weights_path:
                incompatible = self.network.load_state_dict(state, strict=False)
                bad_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith("clip_model.")
                ]
                if bad_missing or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "Invalid MotionCLR artifact state_dict; missing "
                        f"{bad_missing}, unexpected {incompatible.unexpected_keys}"
                    )
            else:
                self.network.load_state_dict(state, strict=True)
        if self.network is not None:
            self.network.eval()
            if self.inference_dtype is not None:
                self.network.to(dtype=self.inference_dtype)
                # OpenAI CLIP's LayerNorm evaluates in float32. Casting its
                # parameters to half breaks on current PyTorch releases.
                clip_model = getattr(self.network, "clip_model", None)
                if clip_model is not None:
                    clip_model.float()
        if device is not None:
            self.to_device(device)

    @property
    def fps(self) -> float:
        return MOTIONCLR_FPS

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def to_device(self, device: str | torch.device):
        self.to(torch.device(device))
        return self

    def normalize(self, motion_263):
        """Normalize physical-scale HumanML3D-263 features."""
        if isinstance(motion_263, np.ndarray):
            return (
                motion_263 - self.mean.detach().cpu().numpy()
            ) / self.std.detach().cpu().numpy()
        motion = torch.as_tensor(motion_263, device=self.device)
        return (motion - self.mean) / self.std

    def denormalize(self, motion_263):
        """Restore normalized HumanML3D-263 features to physical scale."""
        if isinstance(motion_263, np.ndarray):
            return (
                motion_263 * self.std.detach().cpu().numpy()
                + self.mean.detach().cpu().numpy()
            )
        motion = torch.as_tensor(motion_263, device=self.device)
        return motion * self.std + self.mean

    def save_pretrained(
        self,
        save_directory: str | Path,
        safe_serialization: bool = True,
        **kwargs,
    ):
        """Save a self-contained Motius artifact with weights and HML3D stats."""
        del kwargs
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.network is not None:
            state = {
                key: value.detach().cpu().contiguous()
                for key, value in self.network.state_dict().items()
                if not key.startswith("clip_model.")
            }
        else:
            source = self.weights_path or self.checkpoint_path
            if source is None or not Path(source).is_file():
                raise FileNotFoundError(
                    "Cannot save MotionCLR: no loaded network or source checkpoint"
                )
            state = {
                key: value.detach().cpu().contiguous()
                for key, value in _load_state(source, use_ema=self.use_ema).items()
                if not key.startswith("clip_model.")
            }

        weight_name = "model.safetensors" if safe_serialization else "model.pt"
        weight_path = save_dir / weight_name
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state, str(weight_path))
        else:
            torch.save(state, str(weight_path))
        np.save(save_dir / "Mean.npy", self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(save_dir / "Std.npy", self.std.detach().cpu().numpy().astype(np.float32))

        stored_clip = False
        if self.clip_path and Path(self.clip_path).is_file():
            clip_dir = save_dir / "clip"
            clip_dir.mkdir(exist_ok=True)
            shutil.copy2(self.clip_path, clip_dir / "ViT-B-32.pt")
            stored_clip = True

        metadata = {
            "model_type": "motionclr",
            "format": _ARTIFACT_FORMAT,
            "source_repository": MOTIONCLR_SOURCE_REPOSITORY,
            "source_revision": MOTIONCLR_SOURCE_REVISION,
            "official_files": {
                "checkpoint_sha256": MOTIONCLR_CHECKPOINT_SHA256,
                "mean_sha256": MOTIONCLR_MEAN_SHA256,
                "std_sha256": MOTIONCLR_STD_SHA256,
            },
            "bundle_class": "motius.models.motionclr.MotionCLRBundle",
            "pipeline_class": "motius.pipelines.motionclr.MotionCLRPipeline",
            "network": self.network_config,
            "weights": weight_name,
            "statistics": {"mean": "Mean.npy", "std": "Std.npy"},
            "text_encoder": {
                "name": "ViT-B/32",
                "stored_in_artifact": stored_clip,
                "path": "clip/ViT-B-32.pt" if stored_clip else None,
            },
            "load_clip": bool(stored_clip or self.load_clip),
            "use_ema": self.use_ema,
            "inference": {
                "diffuser_name": self.diffuser_name,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "torch_dtype": (
                    str(self.inference_dtype).removeprefix("torch.")
                    if self.inference_dtype is not None
                    else "float32"
                ),
                "fps": self.fps,
                "max_frames": MOTIONCLR_MAX_FRAMES,
            },
            "capabilities": self.SUPPORTED_TASKS,
            "license": "IDEA License 1.0 (non-commercial research)",
        }
        (save_dir / "motionclr_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MotionCLRPipeline",
                    "_library_name": "motius",
                    "model_type": "motionclr",
                    "format": _ARTIFACT_FORMAT,
                    "bundle_class": metadata["bundle_class"],
                    "pipeline_class": metadata["pipeline_class"],
                    "source_revision": MOTIONCLR_SOURCE_REVISION,
                    "capabilities": self.SUPPORTED_TASKS,
                },
                indent=2,
            )
            + "\n"
        )
        license_path = Path(__file__).with_name("LICENSE")
        if license_path.is_file():
            shutil.copy2(license_path, save_dir / "LICENSE")
        return str(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """Load EvanTHU's official layout or a self-contained Motius artifact."""
        verify_hashes = bool(kwargs.pop("verify_hashes", True))
        source = str(pretrained_model_name_or_path)
        path = Path(source)
        hub_layout: Optional[dict[str, Path]] = None
        if not path.exists():
            hub_layout = _download_hub_layout(source)
            if "artifact" in hub_layout:
                path = hub_layout["artifact"]

        config_file = path / "motionclr_config.json" if path.is_dir() else None
        if config_file is not None and config_file.is_file():
            metadata = json.loads(config_file.read_text())
            required = {
                "weights": path / metadata.get("weights", "model.safetensors"),
                "mean": path / metadata.get("statistics", {}).get("mean", "Mean.npy"),
                "std": path / metadata.get("statistics", {}).get("std", "Std.npy"),
            }
            missing = [name for name, value in required.items() if not value.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete MotionCLR artifact {path}; missing {', '.join(missing)}"
                )
            inference = metadata.get("inference", {})
            defaults = {
                "network_config": metadata["network"],
                "weights_path": str(required["weights"]),
                "mean_path": str(required["mean"]),
                "std_path": str(required["std"]),
                "load_clip": metadata.get("load_clip", True),
                "use_ema": metadata.get("use_ema", True),
                "diffuser_name": inference.get("diffuser_name", "dpmsolver"),
                "num_inference_steps": inference.get("num_inference_steps", 10),
                "guidance_scale": inference.get("guidance_scale", 2.5),
                "torch_dtype": inference.get("torch_dtype", "float32"),
            }
            text_encoder = metadata.get("text_encoder", {})
            local_clip = text_encoder.get("path")
            if local_clip:
                clip_file = path / local_clip
                if not clip_file.is_file():
                    raise FileNotFoundError(
                        f"Incomplete MotionCLR artifact {path}; missing text encoder {local_clip}"
                    )
                defaults["clip_path"] = str(clip_file)
            defaults.update(kwargs)
            return cls(**defaults)

        if hub_layout is None:
            layout = _official_layout(path)
        else:
            layout = hub_layout
        missing = [
            name
            for name in ("checkpoint", "mean", "std", "options")
            if layout.get(name) is None or not Path(layout[name]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete official MotionCLR layout at {path}; missing {', '.join(missing)}"
            )
        if verify_hashes:
            _verify_official_hashes(layout)
        config = _network_config_from_options(_parse_official_options(layout["options"]))
        defaults = {
            "network_config": config,
            "checkpoint_path": str(layout["checkpoint"]),
            "mean_path": str(layout["mean"]),
            "std_path": str(layout["std"]),
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def forward(self, *args, **kwargs):  # pragma: no cover - pipeline owns sampling
        if self.network is None:
            raise RuntimeError("MotionCLR network was not loaded")
        return self.network(*args, **kwargs)


__all__ = [
    "DEFAULT_NETWORK_CONFIG",
    "MOTIONCLR_DIM",
    "MOTIONCLR_FPS",
    "MOTIONCLR_MAX_FRAMES",
    "MOTIONCLR_REPO_ID",
    "MOTIONCLR_SOURCE_REPOSITORY",
    "MOTIONCLR_SOURCE_REVISION",
    "MotionCLRBundle",
]
