"""Self-contained model bundle for UniMuMo inference."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .generator import DelayedPattern, UniMuMoGenerator, generate_parallel
from .motion_codec import UniMuMoMotionCodec


UNIMUMO_REPO_ID = "ZeyuLing/Motius-UniMuMo"
UNIMUMO_SOURCE_REPOSITORY = "https://github.com/hanyangclarence/UniMuMo"
UNIMUMO_SOURCE_REVISION = "a75ddac791ff6806b5bd511d1ce887a1980e20d5"
UNIMUMO_ARTIFACT_FORMAT = "motius-unimumo-v1"

DEFAULT_UNIMUMO_CONFIG: dict[str, Any] = {
    "sample_rate": 32_000,
    "motion_fps": 60.0,
    "code_fps": 50.0,
    "motion_dim": 263,
    "motion_representation": "humanml3d_263",
    "default_duration_seconds": 10.0,
    "max_duration_seconds": 10.0,
    "motion_codec": {
        "motion_dim": 263,
        "latent_dim": 128,
        "encoder_channels": [256, 224, 192, 144, 128],
        "decoder_channels": [128, 144, 192, 224, 256],
        "motion_fps": 60.0,
        "code_fps": 50.0,
        "dilation_growth_rate": 2,
        "depth_per_block": 6,
        "activation": "relu",
        "norm": None,
        "pre_quant_multiplier": 4,
        "post_quant_multiplier": 4,
    },
    "generator": {
        "num_codebooks": 4,
        "codebook_size": 2048,
        "dimension": 1024,
        "hidden_dimension": 4096,
        "num_heads": 16,
        "num_layers": 24,
        "dropout": 0.1,
        "attention_dropout": 0.0,
        "bias_attention": False,
        "bias_ffn": False,
        "output_bias": False,
        "norm_first": True,
        "max_period": 10_000.0,
        "position_scale": 1.0,
    },
    "text_hidden_size": 768,
    "caption_max_length": 256,
    "caption_num_beams": 4,
}


class _UniMuMoCore(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.motion_codec = UniMuMoMotionCodec(dict(config["motion_codec"]))
        self.generator = UniMuMoGenerator(dict(config["generator"]))
        text_hidden = int(config["text_hidden_size"])
        generator_hidden = int(config["generator"]["dimension"])
        self.text_condition_projection = nn.Linear(text_hidden, generator_hidden)
        self.caption_context_projection = nn.Linear(generator_hidden, text_hidden)


def _resolve_artifact(
    path_or_repo: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(path_or_repo).expanduser()
    if (path / "unimumo_config.json").is_file():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_patterns=[
                "unimumo_config.json",
                "model_index.json",
                "core*.safetensors",
                "core*.safetensors.index.json",
                "mean.npy",
                "std.npy",
                "audio_codec/**",
                "text_encoder/**",
                "captioner/**",
                "tokenizer/**",
                "ATTRIBUTIONS.md",
                "README.md",
            ],
        )
    )


def _adapt_audio_codec_state_dict(
    state: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Bridge old and parametrized weight-norm checkpoint key layouts."""

    replacements = (
        (".parametrizations.weight.original0", ".weight_g"),
        (".parametrizations.weight.original1", ".weight_v"),
        (".weight_g", ".parametrizations.weight.original0"),
        (".weight_v", ".parametrizations.weight.original1"),
    )
    mapped: dict[str, torch.Tensor] = {}
    unexpected = []
    for source_key, value in state.items():
        destination = source_key if source_key in target else None
        if destination is None:
            for old, new in replacements:
                if old in source_key:
                    candidate = source_key.replace(old, new)
                    if candidate in target:
                        destination = candidate
                        break
        if destination is None:
            unexpected.append(source_key)
            continue
        if destination in mapped:
            raise RuntimeError(
                "UniMuMo audio codec checkpoint has duplicate mappings for "
                f"{destination}"
            )
        if target[destination].shape != value.shape:
            raise RuntimeError(
                "UniMuMo audio codec tensor shape mismatch: "
                f"{source_key} {tuple(value.shape)} -> {destination} "
                f"{tuple(target[destination].shape)}"
            )
        mapped[destination] = value
    missing = sorted(set(target) - set(mapped))
    if missing or unexpected:
        raise RuntimeError(
            "UniMuMo audio codec checkpoint mismatch: "
            f"missing={missing}, unexpected={sorted(unexpected)}"
        )
    return mapped


def _load_audio_codec(
    artifact: Path,
    *,
    torch_dtype: torch.dtype | None,
) -> nn.Module:
    from safetensors.torch import load_file
    from transformers import EncodecConfig, EncodecModel

    directory = artifact / "audio_codec"
    config = EncodecConfig.from_pretrained(directory, local_files_only=True)
    model = EncodecModel(config)
    checkpoint = directory / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = _adapt_audio_codec_state_dict(
        load_file(str(checkpoint)), model.state_dict()
    )
    model.load_state_dict(state, strict=True)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    return model


@MODEL_BUNDLES.register_module()
class UniMuMoBundle(ModelBundle):
    """Frozen UniMuMo codecs, dual-stream LM, and captioning models."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        audio_codec: nn.Module,
        text_encoder: nn.Module,
        captioner: nn.Module,
        tokenizer: Any,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        provenance: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.config = copy.deepcopy(dict(config))
        self.provenance = copy.deepcopy(dict(provenance or {}))
        self.audio_codec = audio_codec
        self.text_encoder = text_encoder
        self.captioner = captioner
        self.tokenizer = tokenizer
        self.core = _UniMuMoCore(self.config)

        motion_dim = int(self.config["motion_dim"])
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        if mean_tensor.shape != (motion_dim,) or std_tensor.shape != (motion_dim,):
            raise ValueError(f"UniMuMo mean/std must have shape ({motion_dim},)")
        if torch.any(std_tensor <= 0):
            raise ValueError("UniMuMo motion std must be positive")
        self.register_buffer("motion_mean", mean_tensor, persistent=False)
        self.register_buffer("motion_std", std_tensor, persistent=False)
        self.requires_grad_(False)
        self.eval()

    @property
    def motion_codec(self) -> UniMuMoMotionCodec:
        return self.core.motion_codec

    @property
    def generator(self) -> UniMuMoGenerator:
        return self.core.generator

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @property
    def sample_rate(self) -> int:
        return int(self.config["sample_rate"])

    @property
    def motion_fps(self) -> float:
        return float(self.config["motion_fps"])

    @property
    def code_fps(self) -> float:
        return float(self.config["code_fps"])

    def normalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.motion_mean) / self.motion_std

    def denormalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.motion_std + self.motion_mean

    def _prepare_waveform(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        if waveform.ndim == 1:
            waveform = waveform[None, None]
        elif waveform.ndim == 2:
            waveform = waveform[:, None]
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape (N), (B,N), or (B,1,N)")
        return waveform

    @torch.inference_mode()
    def encode_audio(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform)
        frame_multiple = self.sample_rate // int(self.code_fps)
        joint_multiple = frame_multiple * 5
        target_length = waveform.shape[-1] // joint_multiple * joint_multiple
        if target_length < joint_multiple:
            raise ValueError("waveform is too short for UniMuMo")
        encoded = self.audio_codec.encode(waveform[..., :target_length])
        codes = encoded.audio_codes
        if codes.shape[0] != 1:
            raise ValueError("Chunked Encodec artifacts are not supported")
        return codes[0].contiguous()

    @torch.inference_mode()
    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes[None]
        if codes.ndim != 3:
            raise ValueError("audio codes must have shape (K,T) or (B,K,T)")
        decoded = self.audio_codec.decode(codes[None], [None])
        return decoded.audio_values

    def _zero_audio_embeddings(self, batch: int, motion_frames: int) -> torch.Tensor:
        samples = int(round(motion_frames / self.motion_fps * self.sample_rate))
        waveform = torch.zeros(
            (batch, 1, samples), dtype=self.motion_mean.dtype, device=self.device
        )
        return self.audio_codec.encoder(waveform)

    @torch.inference_mode()
    def encode_motion(self, motion: torch.Tensor | np.ndarray) -> torch.Tensor:
        motion = torch.as_tensor(motion, dtype=torch.float32, device=self.device)
        if motion.ndim == 2:
            motion = motion[None]
        if motion.ndim != 3 or motion.shape[-1] != int(self.config["motion_dim"]):
            raise ValueError("motion must have shape (T,263) or (B,T,263)")
        frame_multiple = int(round(self.motion_fps / self.code_fps * 5))
        frames = motion.shape[1] // frame_multiple * frame_multiple
        if frames < frame_multiple:
            raise ValueError("motion is too short for UniMuMo")
        normalized = self.normalize_motion(motion[:, :frames])
        music_embeddings = self._zero_audio_embeddings(len(normalized), frames)
        motion_embeddings = self.motion_codec.encode_embeddings(
            normalized, music_embeddings
        )
        return self.audio_codec.quantizer.encode(motion_embeddings).permute(1, 0, 2)

    def _decode_code_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_codec.quantizer.decode(codes.permute(1, 0, 2))

    @torch.inference_mode()
    def decode_motion(
        self,
        music_codes: torch.Tensor,
        motion_codes: torch.Tensor,
    ) -> torch.Tensor:
        music_codes = torch.as_tensor(
            music_codes, dtype=torch.long, device=self.device
        )
        motion_codes = torch.as_tensor(
            motion_codes, dtype=torch.long, device=self.device
        )
        if music_codes.ndim == 2:
            music_codes = music_codes[None]
        if motion_codes.ndim == 2:
            motion_codes = motion_codes[None]
        if music_codes.shape != motion_codes.shape:
            raise ValueError("music and motion codes must share shape")
        music_embeddings = self._decode_code_embeddings(music_codes)
        motion_embeddings = self._decode_code_embeddings(motion_codes)
        normalized = self.motion_codec.decode_embeddings(
            music_embeddings, motion_embeddings
        )
        return self.denormalize_motion(normalized)

    def _encode_descriptions(
        self, descriptions: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(descriptions), padding=True, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.text_encoder(**encoded).last_hidden_state
        hidden = self.core.text_condition_projection(hidden)
        mask = encoded["attention_mask"].bool()
        return hidden * mask[..., None], mask

    @staticmethod
    def _split_descriptions(
        descriptions: Sequence[str], mode: str
    ) -> tuple[list[str], list[str]]:
        music, motion = [], []
        for description in descriptions:
            parts = description.split("<separation>")
            music_text, motion_text = parts[0].strip(), parts[-1].strip()
            if mode == "music2motion":
                music_text = ""
            elif mode == "motion2music":
                motion_text = ""
            elif mode != "music_motion":
                raise ValueError(f"Unsupported generation mode: {mode!r}")
            music.append(music_text)
            motion.append(motion_text)
        return music, motion

    def text_condition(
        self, descriptions: Sequence[str], mode: str = "music_motion"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        music_text, motion_text = self._split_descriptions(descriptions, mode)
        return self._condition_from_streams(music_text, motion_text)

    def _condition_from_streams(
        self, music_text: Sequence[str], motion_text: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(music_text) != len(motion_text):
            raise ValueError("music and motion text batches must have equal length")
        music_hidden, music_mask = self._encode_descriptions(music_text)
        motion_hidden, motion_mask = self._encode_descriptions(motion_text)
        hidden = torch.cat((music_hidden, motion_hidden), dim=1)
        mask = torch.zeros(
            (len(music_text), 2, hidden.shape[1]),
            dtype=torch.bool,
            device=self.device,
        )
        mask[:, 0, : music_mask.shape[1]] = music_mask
        mask[:, 1, music_mask.shape[1] :] = motion_mask
        return hidden, mask

    def cfg_text_condition(
        self, descriptions: Sequence[str], mode: str = "music_motion"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode conditional and null prompts with identical padding lengths."""

        music_text, motion_text = self._split_descriptions(descriptions, mode)
        batch = len(descriptions)
        hidden, mask = self._condition_from_streams(
            music_text + [""] * batch,
            motion_text + [""] * batch,
        )
        return hidden[:batch], mask[:batch], hidden[batch:], mask[batch:]

    def generate_codes(
        self,
        descriptions: Sequence[str],
        *,
        timesteps: int,
        mode: str = "music_motion",
        music_codes: torch.Tensor | None = None,
        motion_codes: torch.Tensor | None = None,
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 250,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition, condition_mask, unconditional, unconditional_mask = (
            self.cfg_text_condition(descriptions, mode)
        )
        return generate_parallel(
            self.generator,
            condition=condition,
            condition_mask=condition_mask,
            unconditional_condition=unconditional,
            unconditional_mask=unconditional_mask,
            timesteps=timesteps,
            music_codes=music_codes,
            motion_codes=motion_codes,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

    @torch.inference_mode()
    def caption(self, codes: torch.Tensor, *, modality: str) -> list[str]:
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes[None]
        if modality not in {"music", "motion"}:
            raise ValueError("modality must be 'music' or 'motion'")
        pattern = DelayedPattern(
            codes.shape[-1], tuple(range(self.generator.num_codebooks))
        )
        sequence, _ = pattern.build(
            codes, special_token=self.generator.special_token_id, valid_only=True
        )
        empty = torch.full_like(sequence, self.generator.special_token_id)
        music_sequence, motion_sequence = (
            (sequence, empty) if modality == "music" else (empty, sequence)
        )
        condition, condition_mask = self.text_condition(
            ["<separation>"] * len(codes)
        )
        context = self.generator.forward_features(
            music_sequence,
            motion_sequence,
            condition=condition,
            condition_mask=condition_mask,
            caption_mode=True,
        )
        stream_length = sequence.shape[-1]
        attention_mask = torch.zeros(
            (len(codes), stream_length * 2), dtype=torch.long, device=self.device
        )
        if modality == "music":
            attention_mask[:, :stream_length] = 1
        else:
            attention_mask[:, stream_length:] = 1
        from transformers.modeling_outputs import BaseModelOutput

        encoder_outputs = BaseModelOutput(
            last_hidden_state=self.core.caption_context_projection(context)
        )
        output = self.captioner.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=int(self.config["caption_max_length"]),
            num_beams=int(self.config["caption_num_beams"]),
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def save_pretrained(self, save_directory: str, **_kwargs):
        from huggingface_hub import save_torch_model

        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_format": UNIMUMO_ARTIFACT_FORMAT,
            "model_type": "unimumo",
            "source_repository": UNIMUMO_SOURCE_REPOSITORY,
            "source_revision": UNIMUMO_SOURCE_REVISION,
            "provenance": self.provenance,
            "config": self.config,
        }
        (output / "unimumo_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "UniMuMoPipeline",
                    "_motius_bundle": "UniMuMoBundle",
                    "artifact_format": UNIMUMO_ARTIFACT_FORMAT,
                    "tasks": [
                        "text-to-music-motion",
                        "text-to-motion",
                        "text-to-music",
                        "music-to-motion",
                        "motion-to-music",
                        "music-to-text",
                        "motion-to-text",
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        save_torch_model(
            self.core,
            output,
            filename_pattern="core{suffix}.safetensors",
            max_shard_size="2GB",
            safe_serialization=True,
        )
        self.audio_codec.save_pretrained(output / "audio_codec")
        self.text_encoder.save_pretrained(output / "text_encoder")
        self.captioner.save_pretrained(output / "captioner")
        self.tokenizer.save_pretrained(output / "tokenizer")
        np.save(output / "mean.npy", self.motion_mean.cpu().numpy())
        np.save(output / "std.npy", self.motion_std.cpu().numpy())
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
        torch_dtype: torch.dtype | None = None,
        **_kwargs,
    ):
        from huggingface_hub import load_torch_model
        from transformers import (
            AutoTokenizer,
            T5Config,
            T5EncoderModel,
        )
        from transformers import T5ForConditionalGeneration

        artifact = _resolve_artifact(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        metadata = json.loads((artifact / "unimumo_config.json").read_text())
        if metadata.get("artifact_format") != UNIMUMO_ARTIFACT_FORMAT:
            raise ValueError(
                f"Unsupported UniMuMo artifact: {metadata.get('artifact_format')!r}"
            )
        model_kwargs = {"local_files_only": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        caption_config = T5Config.from_pretrained(
            artifact / "captioner", local_files_only=True
        )
        # Early converted artifacts inherited the encoder-only flag because
        # T5EncoderModel mutates its config during construction. The caption
        # weights are unchanged; restore the seq2seq contract while loading.
        caption_config.is_encoder_decoder = True
        bundle = cls(
            metadata["config"],
            audio_codec=_load_audio_codec(artifact, torch_dtype=torch_dtype),
            text_encoder=T5EncoderModel.from_pretrained(
                artifact / "text_encoder", **model_kwargs
            ),
            captioner=T5ForConditionalGeneration.from_pretrained(
                artifact / "captioner", config=caption_config, **model_kwargs
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                artifact / "tokenizer", local_files_only=True, use_fast=False
            ),
            mean=np.load(artifact / "mean.npy"),
            std=np.load(artifact / "std.npy"),
            provenance=metadata.get("provenance"),
        )
        report = load_torch_model(
            bundle.core,
            artifact,
            # huggingface_hub applies strict=True to every shard individually,
            # so a valid multi-shard checkpoint is rejected for keys living in
            # the other shards. Validate the complete loaded key set below.
            strict=False,
            safe=True,
            filename_pattern="core{suffix}.safetensors",
        )
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "UniMuMo core checkpoint mismatch: "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )
        bundle.requires_grad_(False)
        bundle.eval()
        return bundle


__all__ = [
    "DEFAULT_UNIMUMO_CONFIG",
    "UNIMUMO_ARTIFACT_FORMAT",
    "UNIMUMO_REPO_ID",
    "UNIMUMO_SOURCE_REPOSITORY",
    "UNIMUMO_SOURCE_REVISION",
    "UniMuMoBundle",
]
