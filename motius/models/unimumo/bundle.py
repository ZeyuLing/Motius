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
from .generator import UniMuMoGenerator
from .motion_codec import UniMuMoMotionCodec
UNIMUMO_REPO_ID = 'ZeyuLing/Motius-UniMuMo'
UNIMUMO_SOURCE_REPOSITORY = 'https://github.com/hanyangclarence/UniMuMo'
UNIMUMO_SOURCE_REVISION = 'a75ddac791ff6806b5bd511d1ce887a1980e20d5'
UNIMUMO_ARTIFACT_FORMAT = 'motius-unimumo-v2'
DEFAULT_UNIMUMO_CONFIG: dict[str, Any] = {'sample_rate': 32000, 'motion_fps': 60.0, 'code_fps': 50.0, 'motion_dim': 263, 'motion_representation': 'humanml3d_263', 'default_duration_seconds': 10.0, 'max_duration_seconds': 10.0, 'motion_codec': {'motion_dim': 263, 'latent_dim': 128, 'encoder_channels': [256, 224, 192, 144, 128], 'decoder_channels': [128, 144, 192, 224, 256], 'motion_fps': 60.0, 'code_fps': 50.0, 'dilation_growth_rate': 2, 'depth_per_block': 6, 'activation': 'relu', 'norm': None, 'pre_quant_multiplier': 4, 'post_quant_multiplier': 4}, 'generator': {'num_codebooks': 4, 'codebook_size': 2048, 'dimension': 1024, 'hidden_dimension': 4096, 'num_heads': 16, 'num_layers': 24, 'dropout': 0.0, 'attention_dropout': 0.0, 'bias_attention': False, 'bias_ffn': False, 'output_bias': False, 'norm_first': True, 'max_period': 10000.0, 'position_scale': 1.0}, 'text_hidden_size': 768, 'caption_max_length': 256, 'caption_num_beams': 4}

class _UniMuMoCore(nn.Module):

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.motion_codec = UniMuMoMotionCodec(dict(config['motion_codec']))
        self.generator = UniMuMoGenerator(dict(config['generator']))
        text_hidden = int(config['text_hidden_size'])
        generator_hidden = int(config['generator']['dimension'])
        self.text_condition_projection = nn.Linear(text_hidden, generator_hidden)
        self.caption_context_projection = nn.Linear(generator_hidden, text_hidden)

    def forward(self, music_codes, motion_codes, *, condition=None, condition_mask=None, project_condition=False, caption_mode=False):
        if project_condition:
            condition = self.text_condition_projection(condition)
        if caption_mode:
            features = self.generator.forward_features(music_codes, motion_codes, condition=condition, condition_mask=condition_mask, caption_mode=True)
            return self.caption_context_projection(features)
        return self.generator(music_codes, motion_codes, condition=condition, condition_mask=condition_mask)

def _resolve_artifact(path_or_repo: str, *, revision: str | None=None, cache_dir: str | Path | None=None, local_files_only: bool=False) -> Path:
    path = Path(path_or_repo).expanduser()
    if (path / 'unimumo_config.json').is_file():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=path_or_repo, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only, allow_patterns=['unimumo_config.json', 'model_index.json', 'core*.safetensors', 'core*.safetensors.index.json', 'mean.npy', 'std.npy', 'audio_codec/**', 'text_encoder/**', 'captioner/**', 'tokenizer/**', 'ATTRIBUTIONS.md', 'README.md']))

def _adapt_audio_codec_state_dict(state: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Bridge old and parametrized weight-norm checkpoint key layouts."""
    replacements = (('.parametrizations.weight.original0', '.weight_g'), ('.parametrizations.weight.original1', '.weight_v'), ('.weight_g', '.parametrizations.weight.original0'), ('.weight_v', '.parametrizations.weight.original1'))
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
            raise RuntimeError(f'UniMuMo audio codec checkpoint has duplicate mappings for {destination}')
        if target[destination].shape != value.shape:
            raise RuntimeError(f'UniMuMo audio codec tensor shape mismatch: {source_key} {tuple(value.shape)} -> {destination} {tuple(target[destination].shape)}')
        mapped[destination] = value
    missing = sorted(set(target) - set(mapped))
    if missing or unexpected:
        raise RuntimeError(f'UniMuMo audio codec checkpoint mismatch: missing={missing}, unexpected={sorted(unexpected)}')
    return mapped

def _load_audio_codec(artifact: Path, *, torch_dtype: torch.dtype | None) -> nn.Module:
    from safetensors.torch import load_file
    from transformers import EncodecConfig, EncodecModel
    directory = artifact / 'audio_codec'
    config = EncodecConfig.from_pretrained(directory, local_files_only=True)
    model = EncodecModel(config)
    checkpoint = directory / 'model.safetensors'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = _adapt_audio_codec_state_dict(load_file(str(checkpoint)), model.state_dict())
    model.load_state_dict(state, strict=True)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    return model

@MODEL_BUNDLES.register_module()
class UniMuMoBundle(ModelBundle):
    """Frozen UniMuMo codecs, dual-stream LM, and captioning models."""

    def __init__(self, config: Mapping[str, Any], *, audio_codec: nn.Module, text_encoder: nn.Module, captioner: nn.Module, tokenizer_path: Any, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor, provenance: Mapping[str, Any] | None=None):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.config = copy.deepcopy(dict(config))
        self.provenance = copy.deepcopy(dict(provenance or {}))
        self.audio_codec = audio_codec
        self.text_encoder = text_encoder
        self.captioner = captioner
        self.core = _UniMuMoCore(self.config)
        motion_dim = int(self.config['motion_dim'])
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        if mean_tensor.shape != (motion_dim,) or std_tensor.shape != (motion_dim,):
            raise ValueError(f'UniMuMo mean/std must have shape ({motion_dim},)')
        if torch.any(std_tensor <= 0):
            raise ValueError('UniMuMo motion std must be positive')
        self.register_buffer('motion_mean', mean_tensor, persistent=False)
        self.register_buffer('motion_std', std_tensor, persistent=False)
        self.requires_grad_(False)
        # Preserve the released caption-stage parameter flags: ATen can select
        # different matmul folding paths even under no_grad when these differ.
        self.captioner.requires_grad_(True)
        self.core.caption_context_projection.requires_grad_(True)
        for layer in self.generator.layers:layer.caption_attention.requires_grad_(True)
        self.eval()

    @property
    def motion_codec(self) -> UniMuMoMotionCodec:
        return self.unwrap_network(self.core).motion_codec

    @property
    def generator(self) -> UniMuMoGenerator:
        return self.unwrap_network(self.core).generator

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @property
    def sample_rate(self) -> int:
        return int(self.config['sample_rate'])

    @property
    def motion_fps(self) -> float:
        return float(self.config['motion_fps'])

    @property
    def code_fps(self) -> float:
        return float(self.config['code_fps'])

    def save_pretrained(self, save_directory: str, **_kwargs):
        from huggingface_hub import save_torch_model
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {'artifact_format': UNIMUMO_ARTIFACT_FORMAT, 'model_type': 'unimumo', 'source_repository': UNIMUMO_SOURCE_REPOSITORY, 'source_revision': UNIMUMO_SOURCE_REVISION, 'provenance': self.provenance, 'config': self.config}
        (output / 'unimumo_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (output / 'model_index.json').write_text(json.dumps({'_class_name': 'UniMuMoPipeline', '_motius_bundle': 'UniMuMoBundle', 'artifact_format': UNIMUMO_ARTIFACT_FORMAT, 'tasks': ['text-to-music-motion', 'text-to-motion', 'text-to-music', 'music-to-motion', 'motion-to-music', 'music-to-text', 'motion-to-text']}, indent=2) + '\n')
        save_torch_model(self.core, output, filename_pattern='core{suffix}.safetensors', max_shard_size='2GB', safe_serialization=True)
        self.audio_codec.save_pretrained(output / 'audio_codec')
        self.text_encoder.save_pretrained(output / 'text_encoder')
        self.captioner.save_pretrained(output / 'captioner')
        if self.tokenizer_path is not None:
            shutil.copytree(self.tokenizer_path, output / 'tokenizer', dirs_exist_ok=True)
        np.save(output / 'mean.npy', self.motion_mean.cpu().numpy())
        np.save(output / 'std.npy', self.motion_std.cpu().numpy())
        attribution = Path(__file__).resolve().parent / 'ATTRIBUTIONS.md'
        if attribution.is_file():
            shutil.copy2(attribution, output / attribution.name)
        return str(output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, revision: str | None=None, cache_dir: str | Path | None=None, local_files_only: bool=False, torch_dtype: torch.dtype | None=None, **_kwargs):
        from huggingface_hub import load_torch_model
        from transformers import AutoTokenizer, T5Config, T5EncoderModel
        from transformers import T5ForConditionalGeneration
        artifact = _resolve_artifact(pretrained_model_name_or_path, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
        metadata = json.loads((artifact / 'unimumo_config.json').read_text())
        if metadata.get('artifact_format') not in [UNIMUMO_ARTIFACT_FORMAT,'motius-unimumo-v1']:
            raise ValueError(f"Unsupported UniMuMo artifact: {metadata.get('artifact_format')!r}")
        config=copy.deepcopy(metadata['config'])
        if metadata.get('artifact_format')=='motius-unimumo-v1':
            # The v1 inference converter wrote an unused 0.1 dropout default;
            # the released MusicGen architecture uses 0.0 for all these layers.
            config['generator']['dropout']=0.0
        model_kwargs = {'local_files_only': True}
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype
        caption_config = T5Config.from_pretrained(artifact / 'captioner', local_files_only=True)
        caption_config.is_encoder_decoder = True
        bundle = cls(config, audio_codec=_load_audio_codec(artifact, torch_dtype=torch_dtype), text_encoder=T5EncoderModel.from_pretrained(artifact / 'text_encoder', **model_kwargs), captioner=T5ForConditionalGeneration.from_pretrained(artifact / 'captioner', config=caption_config, **model_kwargs), tokenizer_path=artifact / 'tokenizer', mean=np.load(artifact / 'mean.npy'), std=np.load(artifact / 'std.npy'), provenance=metadata.get('provenance'))
        report = load_torch_model(bundle.core, artifact, strict=False, safe=True, filename_pattern='core{suffix}.safetensors')
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(f'UniMuMo core checkpoint mismatch: missing={report.missing_keys}, unexpected={report.unexpected_keys}')
        bundle.requires_grad_(False)
        bundle.eval()
        return bundle

    @staticmethod
    def unwrap_network(model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def predict_codes(self, *args, **kwargs):
        return self.core(*args, **kwargs)

    def encode_text_tokens(self, **kwargs):
        return self.text_encoder(**kwargs).last_hidden_state

    def forward(self, *args, **kwargs):
        return self.predict_codes(*args, **kwargs)
__all__ = ['DEFAULT_UNIMUMO_CONFIG', 'UNIMUMO_ARTIFACT_FORMAT', 'UNIMUMO_REPO_ID', 'UNIMUMO_SOURCE_REPOSITORY', 'UNIMUMO_SOURCE_REVISION', 'UniMuMoBundle']
