"""Self-contained official HumanML3D text-motion matching evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "evaluators" / "humanml3d_263"


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _first_existing(root: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"HumanML3D evaluator artifact is missing: {', '.join(candidates)}"
    )


def _prefixed_state_dict(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    marker = f"{prefix}."
    values = {
        key[len(marker) :]: value
        for key, value in state_dict.items()
        if key.startswith(marker)
    }
    if not values:
        raise KeyError(f"Evaluator weights are missing the {prefix!r} module")
    return values


@MODEL_BUNDLES.register_module()
class HumanML3DMatchingBundle(ModelBundle):
    """Official Guo et al. HumanML3D-263 matching network.

    Public inputs are unnormalized HumanML3D-263 motions and raw English text.
    Both embedding methods preserve input order, including variable-length
    batches, so downstream candidate batching cannot silently break pairs.
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        *,
        device: str = "cuda",
        spacy_model: str = "en_core_web_sm",
        max_text_length: int = 20,
        max_motion_length: int = 196,
        unit_length: int = 4,
    ) -> None:
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).expanduser().resolve()
        safetensors_path = artifact / "model.safetensors"
        checkpoint_path = None
        if not safetensors_path.is_file():
            checkpoint_path = _first_existing(
                artifact,
                (
                    "model.pt",
                    "text_mot_match.tar",
                    "text_mot_match/model/finest.tar",
                    "t2m/text_mot_match/model/finest.tar",
                    "checkpoints/t2m/text_mot_match/model/finest.tar",
                ),
            )
        mean_path = _first_existing(
            artifact,
            (
                "stats/mean.npy",
                "meta/mean.npy",
                "t2m/Comp_v6_KLD005/meta/mean.npy",
                "checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy",
            ),
        )
        std_path = _first_existing(
            artifact,
            (
                "stats/std.npy",
                "meta/std.npy",
                "t2m/Comp_v6_KLD005/meta/std.npy",
                "checkpoints/t2m/Comp_v6_KLD005/meta/std.npy",
            ),
        )

        self.movement_encoder = MovementConvEncoder(259, 512, 512)
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=15,
            hidden_size=512,
            output_size=512,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(512, 1024, 512)
        if safetensors_path.is_file():
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path, device="cpu")
            module_states = {
                name: _prefixed_state_dict(state_dict, name)
                for name in ("movement_encoder", "text_encoder", "motion_encoder")
            }
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            module_states = {
                name: checkpoint[name]
                for name in ("movement_encoder", "text_encoder", "motion_encoder")
            }
        self.movement_encoder.load_state_dict(
            module_states["movement_encoder"], strict=True
        )
        self.text_encoder.load_state_dict(module_states["text_encoder"], strict=True)
        self.motion_encoder.load_state_dict(
            module_states["motion_encoder"], strict=True
        )
        self._save_ckpt_modules.extend(
            ["movement_encoder", "text_encoder", "motion_encoder"]
        )
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)))
        self.register_buffer("std", torch.from_numpy(np.load(std_path)))
        self.spacy_model = spacy_model
        self.max_text_length = int(max_text_length)
        self.max_motion_length = int(max_motion_length)
        self.unit_length = int(unit_length)
        self.artifact_dir = artifact
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(resolved_device).eval()

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        import json

        artifact = _download_artifact(pretrained_model_name_or_path)
        properties = artifact / "matching_config.json"
        settings = json.loads(properties.read_text()) if properties.is_file() else {}
        settings.update(kwargs)
        return cls(
            artifact_dir=str(artifact),
            **settings,
        )

    def save_pretrained(self, save_directory, safe_serialization=True, **kwargs):
        """Save the actual current neural roots, original-precision stats and vocabulary."""
        from motius.configuration.humanml_evaluator import export_matching_resources

        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        states = {}
        for name in ["movement_encoder", "text_encoder", "motion_encoder"]:
            module = getattr(self, name)
            while hasattr(module, "module"):
                module = module.module
            states[name] = {
                key: value.detach().cpu().contiguous()
                for key, value in module.state_dict().items()
            }
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(
                {
                    f"{name}.{key}": value
                    for name, state in states.items()
                    for key, value in state.items()
                },
                directory / "model.safetensors",
            )
        else:
            torch.save(states, directory / "model.pt")
            (directory / "model.safetensors").unlink(missing_ok=True)
        (directory / "stats").mkdir(exist_ok=True)
        np.save(directory / "stats/mean.npy", self.mean.detach().cpu().numpy())
        np.save(directory / "stats/std.npy", self.std.detach().cpu().numpy())
        export_matching_resources(
            self, directory, "model.safetensors" if safe_serialization else "model.pt"
        )
        return save_directory

    def train(self, mode: bool = True):
        return super().train(False)

    def encode_text_tokens(self, word_embeddings, pos_one_hot, lengths):
        return self.text_encoder(word_embeddings, pos_one_hot, lengths)

    def encode_motion_features(self, normalized_motion, token_lengths):
        movement = self.movement_encoder(normalized_motion[..., :-4]).detach()
        return self.motion_encoder(movement, token_lengths)


__all__ = ["HumanML3DMatchingBundle"]
