"""Self-contained model bundle for the released EDGE AIST++ checkpoint."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import DanceDecoder, EDGE_REPR_DIM


EDGE_REPO_ID = "ZeyuLing/Motius-EDGE-AISTPP"
EDGE_SOURCE_REPOSITORY = "https://github.com/Stanford-TML/EDGE"
EDGE_SOURCE_REVISION = "17c3428669ed6733edd9d8c66f7dc62060b8e46d"
EDGE_OFFICIAL_CHECKPOINT_SHA256 = (
    "28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7"
)
EDGE_ARTIFACT_FORMAT = "motius-edge-v1"

DEFAULT_EDGE_CONFIG: dict[str, Any] = {
    "fps": 30.0,
    "window_frames": 150,
    "overlap_frames": 75,
    "representation_dim": EDGE_REPR_DIM,
    "motion_representation": "EDGE-151 (contacts4 + root3 + SMPL24 local rot6d144)",
    "music_representation": "Jukebox layer 66, 4800-D at 30 fps",
    "network": {
        "nfeats": EDGE_REPR_DIM,
        "seq_len": 150,
        "latent_dim": 512,
        "ff_size": 1024,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.1,
        "cond_feature_dim": 4_800,
        "use_rotary": True,
    },
    "sampling": {
        "train_timesteps": 1_000,
        "sampling_steps": 50,
        "schedule": "cosine",
        "prediction_type": "sample",
        "eta": 1.0,
        "guidance_weight": 2.0,
    },
}


def _resolve_artifact(
    path_or_repo: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(path_or_repo).expanduser()
    if (path / "edge_config.json").is_file():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "edge_config.json",
                "model_index.json",
                "model.safetensors",
                "normalizer.npz",
                "LICENSE",
                "ATTRIBUTIONS.md",
                "README.md",
            ],
        )
    )


@MODEL_BUNDLES.register_module()
class EDGEBundle(ModelBundle):
    """EDGE diffusion decoder and its checkpoint-specific MinMax normalizer."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        network: nn.Module | None = None,
        weights_path: str | Path | None = None,
        normalizer_scale=None,
        normalizer_min=None,
        provenance: Mapping[str, Any] | None = None,
        strict: bool = True,
    ):
        super().__init__()
        self.config = copy.deepcopy(dict(config or DEFAULT_EDGE_CONFIG))
        self.provenance = copy.deepcopy(dict(provenance or {}))
        self.network = network or DanceDecoder(**self.config["network"])
        if weights_path is not None:
            from safetensors.torch import load_file

            state = load_file(str(weights_path), device="cpu")
            incompatible = self.network.load_state_dict(state, strict=strict)
            self.load_report = {
                "missing": list(incompatible.missing_keys),
                "unexpected": list(incompatible.unexpected_keys),
            }
        else:
            self.load_report = {"missing": [], "unexpected": []}
        scale = torch.as_tensor(normalizer_scale, dtype=torch.float32)
        offset = torch.as_tensor(normalizer_min, dtype=torch.float32)
        if scale.shape != (EDGE_REPR_DIM,) or offset.shape != (EDGE_REPR_DIM,):
            raise ValueError("EDGE normalizer_scale and normalizer_min must have shape (151,)")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("EDGE normalizer_scale must be finite and positive")
        if not torch.isfinite(offset).all():
            raise ValueError("EDGE normalizer_min must be finite")
        self.register_buffer("normalizer_scale", scale)
        self.register_buffer("normalizer_min", offset)
        self.network.eval()

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    @property
    def fps(self) -> float:
        return float(self.config.get("fps", 30.0))

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(-1, 1)
        return (value - self.normalizer_min.to(value)) / self.normalizer_scale.to(value)

    def save_pretrained(self, save_directory: str, **_kwargs):
        from safetensors.torch import save_file

        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_format": EDGE_ARTIFACT_FORMAT,
            "model_type": "edge",
            "source_repository": EDGE_SOURCE_REPOSITORY,
            "source_revision": EDGE_SOURCE_REVISION,
            "official_checkpoint_sha256": EDGE_OFFICIAL_CHECKPOINT_SHA256,
            "provenance": self.provenance,
            "config": self.config,
        }
        (output / "edge_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "EDGEPipeline",
                    "_motius_bundle": "EDGEBundle",
                    "artifact_format": EDGE_ARTIFACT_FORMAT,
                },
                indent=2,
            )
            + "\n"
        )
        save_file(
            {
                # Rotary embeddings are shared by decoder layers in the source
                # model. Clone every tensor so safetensors receives independent
                # storage while preserving the exact state-dict keys.
                key: value.detach().cpu().clone()
                for key, value in self.network.state_dict().items()
            },
            str(output / "model.safetensors"),
        )
        np.savez(
            output / "normalizer.npz",
            scale=self.normalizer_scale.detach().cpu().numpy(),
            min=self.normalizer_min.detach().cpu().numpy(),
        )
        source_dir = Path(__file__).resolve().parent
        for name in ("LICENSE", "ATTRIBUTIONS.md"):
            shutil.copy2(source_dir / name, output / name)
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
        metadata = json.loads((artifact / "edge_config.json").read_text())
        if metadata.get("artifact_format") != EDGE_ARTIFACT_FORMAT:
            raise ValueError(f"Unsupported EDGE artifact: {metadata.get('artifact_format')}")
        with np.load(artifact / "normalizer.npz", allow_pickle=False) as stats:
            scale = np.asarray(stats["scale"], dtype=np.float32)
            offset = np.asarray(stats["min"], dtype=np.float32)
        return cls(
            config=metadata["config"],
            weights_path=artifact / "model.safetensors",
            normalizer_scale=scale,
            normalizer_min=offset,
            provenance=metadata.get("provenance"),
            **kwargs,
        )


__all__ = [
    "DEFAULT_EDGE_CONFIG",
    "EDGE_ARTIFACT_FORMAT",
    "EDGE_OFFICIAL_CHECKPOINT_SHA256",
    "EDGE_REPO_ID",
    "EDGE_SOURCE_REPOSITORY",
    "EDGE_SOURCE_REVISION",
    "EDGEBundle",
]
