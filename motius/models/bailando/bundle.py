"""Self-contained Motius bundle for the released Bailando checkpoint."""

from __future__ import annotations

import copy
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import CrossCondGPT2AC, SepVQVAER


BAILANDO_REPO_ID = "ZeyuLing/Motius-Bailando-AISTPP"
BAILANDO_SOURCE_REPOSITORY = "https://github.com/lisiyao21/Bailando"
BAILANDO_SOURCE_REVISION = "cc90b98bff81c9709570db413c9610c2562e27ca"
BAILANDO_ARTIFACT_FORMAT = "motius-bailando-v1"


def _half_config(sample_length: int = 240) -> dict[str, Any]:
    return {
        "levels": 1,
        "downs_t": [3],
        "strides_t": [2],
        "emb_width": 512,
        "l_bins": 512,
        "l_mu": 0.99,
        "commit": 0.02,
        "hvqvae_multipliers": [1],
        "width": 512,
        "depth": 3,
        "m_conv": 1.0,
        "dilation_growth_rate": 3,
        "sample_length": sample_length,
        "use_bottleneck": True,
        "joint_channel": 3,
        "vqvae_reverse_decoder_dilation": True,
    }


DEFAULT_BAILANDO_CONFIG: dict[str, Any] = {
    "fps": 60.0,
    "code_downsample": 8,
    "motion_representation": "aistpp_smpl24_joints",
    "default_initial_codes": [423, 12],
    "vqvae": {
        "up_half": _half_config(),
        "down_half": {**_half_config(), "acc": 1.0},
        "use_bottleneck": True,
        "joint_channel": 3,
    },
    "gpt": {
        "block_size": 29,
        "base": {
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "vocab_size_up": 512,
            "vocab_size_down": 512,
            "block_size": 29,
            "n_layer": 6,
            "n_head": 12,
            "n_embd": 768,
            "n_music": 438,
            "n_music_emb": 768,
        },
        "head": {
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "vocab_size": 512,
            "block_size": 29,
            "n_layer": 6,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size_up": 512,
            "vocab_size_down": 512,
        },
        "critic_net": {
            "embd_pdrop": 0.0,
            "resid_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "block_size": 29,
            "n_layer": 3,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size_up": 1,
            "vocab_size_down": 1,
        },
        "n_music": 438,
        "n_music_emb": 768,
    },
}


def _namespace(value):
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _load_torch_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device="cpu"))
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        payload = payload["model"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"Bailando checkpoint must contain a state dict: {path}")
    state = dict(payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _resolve_artifact(
    path_or_repo: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(path_or_repo).expanduser()
    if (path / "bailando_config.json").is_file():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "bailando_config.json",
                "model_index.json",
                "vqvae.safetensors",
                "gpt.safetensors",
                "LICENSE",
                "ATTRIBUTIONS.md",
                "README.md",
            ],
        )
    )


@MODEL_BUNDLES.register_module()
class BailandoBundle(ModelBundle):
    """Pose VQ-VAE and actor-critic GPT used by Bailando inference."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        vqvae: nn.Module | None = None,
        gpt: nn.Module | None = None,
        vqvae_weights: str | None = None,
        gpt_weights: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        strict: bool = True,
    ):
        super().__init__()
        self.config = copy.deepcopy(dict(config or DEFAULT_BAILANDO_CONFIG))
        self.provenance = copy.deepcopy(dict(provenance or {}))
        self.vqvae = vqvae or SepVQVAER(_namespace(self.config["vqvae"]))
        self.gpt = gpt or CrossCondGPT2AC(_namespace(self.config["gpt"]))
        self.load_reports: dict[str, dict[str, list[str]]] = {}
        if vqvae_weights:
            self.load_reports["vqvae"] = self._load_component(
                self.vqvae, vqvae_weights, strict=strict
            )
        if gpt_weights:
            self.load_reports["gpt"] = self._load_component(
                self.gpt, gpt_weights, strict=strict
            )
        self.vqvae.eval()
        self.gpt.eval()

    @staticmethod
    def _load_component(module, path, *, strict: bool):
        incompatible = module.load_state_dict(_load_torch_state(path), strict=strict)
        return {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }

    @property
    def device(self) -> torch.device:
        return next(self.gpt.parameters()).device

    @property
    def fps(self) -> float:
        return float(self.config.get("fps", 60.0))

    @property
    def code_downsample(self) -> int:
        return int(self.config.get("code_downsample", 8))

    @property
    def default_initial_codes(self) -> tuple[int, int]:
        values = self.config.get("default_initial_codes", [423, 12])
        return int(values[0]), int(values[1])

    def save_pretrained(self, save_directory: str, **_kwargs):
        from safetensors.torch import save_file

        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_format": BAILANDO_ARTIFACT_FORMAT,
            "model_type": "bailando",
            "source_repository": BAILANDO_SOURCE_REPOSITORY,
            "source_revision": BAILANDO_SOURCE_REVISION,
            "provenance": self.provenance,
            "config": self.config,
        }
        (output / "bailando_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "BailandoPipeline",
                    "_motius_bundle": "BailandoBundle",
                    "artifact_format": BAILANDO_ARTIFACT_FORMAT,
                },
                indent=2,
            )
            + "\n"
        )
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in self.vqvae.state_dict().items()
            },
            str(output / "vqvae.safetensors"),
        )
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in self.gpt.state_dict().items()
            },
            str(output / "gpt.safetensors"),
        )
        source_dir = Path(__file__).resolve().parent
        for name in ("LICENSE", "ATTRIBUTIONS.md"):
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, output / name)
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
        metadata = json.loads((artifact / "bailando_config.json").read_text())
        if metadata.get("artifact_format") != BAILANDO_ARTIFACT_FORMAT:
            raise ValueError(
                "Unsupported Bailando artifact format: "
                f"{metadata.get('artifact_format')!r}"
            )
        return cls(
            config=metadata["config"],
            provenance=metadata.get("provenance"),
            vqvae_weights=str(artifact / "vqvae.safetensors"),
            gpt_weights=str(artifact / "gpt.safetensors"),
            **kwargs,
        )


__all__ = [
    "BAILANDO_REPO_ID",
    "BAILANDO_SOURCE_REPOSITORY",
    "BAILANDO_SOURCE_REVISION",
    "BailandoBundle",
    "DEFAULT_BAILANDO_CONFIG",
]
