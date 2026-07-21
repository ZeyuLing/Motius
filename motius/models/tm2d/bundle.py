"""Self-contained Motius bundle for the released TM2D checkpoint."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import (
    AudioMotionTransformer,
    TextMotionTransformer,
    VQDecoder,
    VQEncoder,
    VectorQuantizer,
)
from .tokenizer import TM2DTokenizer


TM2D_REPO_ID = "ZeyuLing/Motius-TM2D-HumanML3D-AISTPP"
TM2D_SOURCE_REPOSITORY = "https://github.com/Garfield-kh/TM2D"
TM2D_SOURCE_REVISION = "98bef9571419b6459927630d5d96f8450898687e"
TM2D_ARTIFACT_FORMAT = "motius-tm2d-v1"

DEFAULT_TM2D_CONFIG: dict[str, Any] = {
    "fps": 60.0,
    "motion_representation": "tm2d_humanml24_287",
    "normalized_motion_dim": 287,
    "vq_encoder_dim": 283,
    "codebook_size": 1024,
    "latent_dim": 1024,
    "code_stride": 8,
    "motion_vocabulary_size": 1027,
    "motion_start_id": 1024,
    "motion_end_id": 1025,
    "motion_pad_id": 1026,
    "text_vocabulary_size": 4201,
    "text_length_id": 4199,
    "text_pad_id": 4200,
    "max_text_tokens": 20,
    "audio_feature_dim": 438,
    "audio_feature_fps": 7.5,
    "audio_chunk_length": 50,
    "audio_chunk_overlap": 1,
    "default_music_seed_token": 423,
    "d_model": 512,
    "d_inner": 1024,
    "n_encoder_layers": 4,
    "n_decoder_layers": 4,
    "n_head": 8,
    "d_k": 64,
    "d_v": 64,
    "dropout": 0.1,
    "max_source_length": 100,
    "max_target_length": 100,
}


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"TM2D component must contain a state dict: {path}")
    return dict(payload)


def _resolve_artifact(
    path_or_repo: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(path_or_repo).expanduser()
    if (path / "tm2d_config.json").is_file():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "tm2d_config.json",
                "model_index.json",
                "vq_encoder.safetensors",
                "quantizer.safetensors",
                "vq_decoder.safetensors",
                "audio_transformer.safetensors",
                "text_transformer.safetensors",
                "mean.npy",
                "std.npy",
                "vocab.json",
                "ATTRIBUTIONS.md",
                "README.md",
            ],
        )
    )


@MODEL_BUNDLES.register_module()
class TM2DBundle(ModelBundle):
    """TM2D text/music encoders and shared 24-joint motion tokenizer."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        vocabulary: Mapping[str, int] | None = None,
        mean: np.ndarray | torch.Tensor | None = None,
        std: np.ndarray | torch.Tensor | None = None,
        component_weights: Mapping[str, str | Path] | None = None,
        provenance: Mapping[str, Any] | None = None,
        strict: bool = True,
    ):
        super().__init__()
        self.config = copy.deepcopy(dict(config or DEFAULT_TM2D_CONFIG))
        self.provenance = copy.deepcopy(dict(provenance or {}))
        self.vq_encoder = VQEncoder(
            self.config["vq_encoder_dim"],
            self.config["latent_dim"],
            int(round(np.log2(self.config["code_stride"]))),
        )
        self.quantizer = VectorQuantizer(
            self.config["codebook_size"], self.config["latent_dim"]
        )
        self.vq_decoder = VQDecoder(
            self.config["latent_dim"],
            self.config["normalized_motion_dim"],
            n_resblocks=3,
            n_up=int(round(np.log2(self.config["code_stride"]))),
        )
        self.audio_transformer = AudioMotionTransformer(self.config)
        self.text_transformer = TextMotionTransformer(self.config)

        dimension = self.config["normalized_motion_dim"]
        mean_tensor = torch.as_tensor(
            np.zeros(dimension, dtype=np.float32) if mean is None else mean,
            dtype=torch.float32,
        )
        std_tensor = torch.as_tensor(
            np.ones(dimension, dtype=np.float32) if std is None else std,
            dtype=torch.float32,
        )
        if mean_tensor.shape != (dimension,) or std_tensor.shape != (dimension,):
            raise ValueError(f"TM2D mean/std must have shape ({dimension},)")
        if torch.any(std_tensor <= 0):
            raise ValueError("TM2D std must be positive")
        self.register_buffer("motion_mean", mean_tensor, persistent=False)
        self.register_buffer("motion_std", std_tensor, persistent=False)

        vocabulary = vocabulary or {"sos": 0, "eos": 1, "unk": 2}
        self.tokenizer = TM2DTokenizer(
            vocabulary,
            max_text_tokens=self.config["max_text_tokens"],
            length_id=self.config["text_length_id"],
            pad_id=self.config["text_pad_id"],
        )
        self.load_reports: dict[str, dict[str, list[str]]] = {}
        for name, path in (component_weights or {}).items():
            module = getattr(self, name)
            incompatible = module.load_state_dict(_load_state(path), strict=strict)
            self.load_reports[name] = {
                "missing": list(incompatible.missing_keys),
                "unexpected": list(incompatible.unexpected_keys),
            }
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.text_transformer.parameters()).device

    @property
    def fps(self) -> float:
        return float(self.config["fps"])

    @property
    def code_stride(self) -> int:
        return int(self.config["code_stride"])

    def normalize_motion(self, motion):
        return (motion - self.motion_mean) / self.motion_std

    def denormalize_motion(self, motion):
        return motion * self.motion_std + self.motion_mean

    def encode_motion(self, motion):
        """Encode normalized 287-D motion through the official 283-D VQ input."""

        motion = torch.as_tensor(motion, dtype=torch.float32, device=self.device)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if motion.ndim != 3 or motion.shape[-1] != self.config["normalized_motion_dim"]:
            raise ValueError("motion must have shape (T,287) or (B,T,287)")
        normalized = self.normalize_motion(motion)
        latent = self.vq_encoder(normalized[..., : self.config["vq_encoder_dim"]])
        return self.quantizer.indices(latent)

    def decode_tokens(self, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        normalized = self.vq_decoder(self.quantizer.lookup(tokens))
        return self.denormalize_motion(normalized)

    def save_pretrained(self, save_directory: str, **_kwargs):
        from safetensors.torch import save_file

        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_format": TM2D_ARTIFACT_FORMAT,
            "model_type": "tm2d",
            "source_repository": TM2D_SOURCE_REPOSITORY,
            "source_revision": TM2D_SOURCE_REVISION,
            "provenance": self.provenance,
            "config": self.config,
        }
        (output / "tm2d_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "TM2DPipeline",
                    "_motius_bundle": "TM2DBundle",
                    "artifact_format": TM2D_ARTIFACT_FORMAT,
                    "tasks": ["text-to-motion", "music-to-dance"],
                },
                indent=2,
            )
            + "\n"
        )
        components = {
            "vq_encoder": self.vq_encoder,
            "quantizer": self.quantizer,
            "vq_decoder": self.vq_decoder,
            "audio_transformer": self.audio_transformer,
            "text_transformer": self.text_transformer,
        }
        for name, module in components.items():
            save_file(
                {
                    key: value.detach().cpu().clone().contiguous()
                    for key, value in module.state_dict().items()
                },
                str(output / f"{name}.safetensors"),
            )
        np.save(output / "mean.npy", self.motion_mean.cpu().numpy())
        np.save(output / "std.npy", self.motion_std.cpu().numpy())
        (output / "vocab.json").write_text(
            json.dumps(self.tokenizer.vocabulary, indent=2, sort_keys=True) + "\n"
        )
        attribution = Path(__file__).resolve().parent / "ATTRIBUTIONS.md"
        if attribution.is_file():
            shutil.copy2(attribution, output / attribution.name)
        return str(output)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        artifact = _resolve_artifact(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        metadata = json.loads((artifact / "tm2d_config.json").read_text())
        if metadata.get("artifact_format") != TM2D_ARTIFACT_FORMAT:
            raise ValueError(
                f"Unsupported TM2D artifact format: {metadata.get('artifact_format')!r}"
            )
        component_weights = {
            name: artifact / f"{name}.safetensors"
            for name in (
                "vq_encoder",
                "quantizer",
                "vq_decoder",
                "audio_transformer",
                "text_transformer",
            )
        }
        return cls(
            metadata["config"],
            vocabulary=json.loads((artifact / "vocab.json").read_text()),
            mean=np.load(artifact / "mean.npy"),
            std=np.load(artifact / "std.npy"),
            component_weights=component_weights,
            provenance=metadata.get("provenance"),
            **kwargs,
        )


__all__ = [
    "DEFAULT_TM2D_CONFIG",
    "TM2D_REPO_ID",
    "TM2D_SOURCE_REPOSITORY",
    "TM2D_SOURCE_REVISION",
    "TM2DBundle",
]
