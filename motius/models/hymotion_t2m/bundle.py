"""HyMotion-T2M Bundle: text-to-motion generation via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - predict_flow()          -- single forward through the transformer
  - encode_text()           -- lazy-load text encoder and encode text

Unlike HyMotion-M2M, this bundle does NOT use VACE conditioning.
The input to the transformer is just x_t (motion_dim), not
[x_t, vace_context] (motion_dim * 4).
"""

from __future__ import annotations

import json
import os.path as osp
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


_DEFAULT_TEXT_ENCODER_CFG = {
    'type': 'HYTextModel',
    'llm_type': 'qwen3',
    'max_length_llm': 128,
    'sentence_emb_type': 'clipl',
    'max_length_sentence_emb': 77,
    'enable_llm_padding': True,
}
_ARTIFACT_FORMAT = 'motius-hymotion-t2m-v1'
# Repo root: motius/models/hymotion_t2m/bundle.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DTYPE_ALIASES = {
    'fp32': torch.float32,
    'float32': torch.float32,
    'torch.float32': torch.float32,
    'fp16': torch.float16,
    'float16': torch.float16,
    'torch.float16': torch.float16,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'torch.bfloat16': torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert length list to boolean mask. (B,) -> (B, max_len)."""
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


def _get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a HuggingFace Hub repo id to a local snapshot dir."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


def _resolve_dtype(dtype: Optional[Any]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = _DTYPE_ALIASES.get(dtype)
        if resolved is not None:
            return resolved
    raise ValueError(f'Unsupported text dtype: {dtype!r}')


def _dtype_name(dtype: Optional[Any]) -> Optional[str]:
    dtype = _resolve_dtype(dtype)
    if dtype is None:
        return None
    if dtype is torch.float32:
        return 'fp32'
    if dtype is torch.float16:
        return 'fp16'
    if dtype is torch.bfloat16:
        return 'bf16'
    return str(dtype)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _is_local_relative_path(value: Optional[str]) -> bool:
    if not value:
        return False
    path = str(value)
    return '://' not in path and not Path(path).is_absolute()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    repo_path = _REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def _resolve_artifact_path(value: Optional[str], artifact_dir: Path) -> Optional[str]:
    if not value or not _is_local_relative_path(value):
        return value
    return str(artifact_dir / value)


def _copy_pretrained_tree(src: Path, dst: Path, ignore_patterns: Optional[tuple[str, ...]] = None) -> None:
    if not src.exists():
        raise FileNotFoundError(f'pretrained component not found: {src}')
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*(ignore_patterns or ())) if ignore_patterns else None
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def _ensure_motion_transformer_registered() -> None:
    # Needed when users set HFTRAINER_SKIP_AUTOREGISTER=1 for lightweight
    # inference/imports.
    import motius.models.hymotion_t2m.network  # noqa: F401


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@MODEL_BUNDLES.register_module()
class HyMotionT2MBundle(ModelBundle):
    """ModelBundle for HyMotion-T2M text-to-motion generation.

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer``  (the HunyuanMotionMMDiT).  Auxiliary objects
    (M2MLoss, null embeddings, mean/std buffers) are created directly in
    ``__init__`` as regular attributes.

    Key difference from HyMotion-M2M: NO VACE conditioning.
    input_dim = motion_dim (not motion_dim * 4).
    """

    def __init__(
        self,
        motion_transformer: dict,
        # ----- optional text encoder (lazy-loaded at encode_text time) -----
        text_encoder: Optional[dict] = None,
        # ----- mean / std for normalisation -----
        mean_std_dir: Optional[str] = None,
        # ----- model hyperparams -----
        motion_type: str = 'smpl_22',
        pred_type: str = 'velocity',
        uncondition_mode: bool = False,
        losses_cfg: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        infer_noise_scheduler_cfg: Optional[dict] = None,
        cond_mask_prob: float = 0.1,
        enable_special_game_feat: bool = False,
        train_null_embeddings: bool = True,
        train_special_game_embeddings: bool = True,
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- self-contained artifact loading --------------------------
        motion_weights_path: Optional[str] = None,
        bundle_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        text_dtype: Optional[Any] = None,
        device: Optional[str] = None,
        # ----- SMPL body model path (optional; skipped if None) -----
        body_model_path: Optional[str] = None,
    ):
        super().__init__()

        # ---- build trainable module via _build_modules ----
        _ensure_motion_transformer_registered()
        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_type = motion_type
        self.pred_type = pred_type
        self.uncondition_mode = uncondition_mode
        self.cond_mask_prob = cond_mask_prob
        self._losses_cfg = deepcopy(losses_cfg or {})
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg or {'method': 'euler'})
        self._infer_noise_scheduler_cfg = deepcopy(
            infer_noise_scheduler_cfg or {'validation_steps': 50}
        )

        # ---- text encoder config (lazy-loaded) ----
        from motius.configuration.hymotion_text import hymotion_text_precision
        self._text_encoder_cfg = hymotion_text_precision(text_encoder, motion_type)
        self._text_dtype = _resolve_dtype(text_dtype)

        # ---- null embeddings for classifier-free guidance ----
        # Zero default; actual values loaded from pretrained checkpoint.
        self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim))
        self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim))
        self.special_game_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim))
        self.special_game_ctxt_feat = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim))
        self.enable_special_game_feat = bool(enable_special_game_feat)
        self.train_null_embeddings = bool(train_null_embeddings)
        self.train_special_game_embeddings = bool(train_special_game_embeddings)
        if not self.train_null_embeddings:
            self.null_vtxt_feat.requires_grad_(False)
            self.null_ctxt_input.requires_grad_(False)
        if not self.train_special_game_embeddings:
            self.special_game_vtxt_feat.requires_grad_(False)
            self.special_game_ctxt_feat.requires_grad_(False)

        # ---- mean / std buffers ----
        self._load_mean_std(mean_std_dir, mean_path=mean_path, std_path=std_path)

        # ---- Motion flow loss (same velocity / x1 objective as HYMotion T2M) ----

        # ---- SMPL body model (optional for FK losses / decode) ----
        self._body_model_path = body_model_path

        # ---- store vtxt/ctxt dims for later ----
        self._vtxt_input_dim = vtxt_input_dim
        self._ctxt_input_dim = ctxt_input_dim

        # ---- infer params ----
        self.validation_steps = self._infer_noise_scheduler_cfg.get(
            'validation_steps', 50
        )

        if motion_weights_path is not None:
            self._load_artifact_weights(
                motion_weights_path,
                bundle_weights_path=bundle_weights_path,
            )

        if device is not None:
            self.to(torch.device(device))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_mean_std(
        self,
        mean_std_dir: Optional[str],
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
    ) -> None:
        if mean_path is not None and std_path is not None:
            mean = torch.from_numpy(np.load(str(mean_path))).float()
            std = torch.from_numpy(np.load(str(std_path))).float()
            std = torch.where(std < 1e-3, torch.zeros_like(std), std)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        elif mean_std_dir is not None and osp.isdir(mean_std_dir):
            mean = torch.from_numpy(
                np.load(osp.join(mean_std_dir, 'Mean.npy'))
            ).float()
            std = torch.from_numpy(
                np.load(osp.join(mean_std_dir, 'Std.npy'))
            ).float()
            # Zero-out near-zero std dims (matching official HY-Motion-1.0)
            # These dims are effectively constant and should produce zero after normalization
            std = torch.where(std < 1e-3, torch.zeros_like(std), std)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        else:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('std', torch.ones(1))

    @staticmethod
    def _load_tensor_file(weights_path: str) -> dict[str, Tensor]:
        """Load a PyTorch or safetensors state dict on CPU."""
        p = str(weights_path)
        if p.endswith('.safetensors'):
            from safetensors.torch import load_file

            return load_file(p)
        return torch.load(p, map_location='cpu')

    def _load_artifact_weights(
        self,
        weights_path: str,
        bundle_weights_path: Optional[str] = None,
    ) -> None:
        state = self._load_tensor_file(weights_path)
        mt_prefix = 'motion_transformer.'
        mt_state = {
            k[len(mt_prefix):]: v
            for k, v in state.items()
            if k.startswith(mt_prefix)
        }
        if not mt_state:
            expected = set(self.motion_transformer.state_dict())
            if expected and expected.issubset(state):
                mt_state = {key: state[key] for key in expected}
            else:
                raise ValueError(
                    f'No compatible motion transformer weights found in {weights_path}'
                )
        self.motion_transformer.load_state_dict(mt_state, strict=True)

        bundle_state = state
        if bundle_weights_path is not None:
            bundle_state = self._load_tensor_file(bundle_weights_path)
        for name in ('null_vtxt_feat', 'null_ctxt_input'):
            if name not in bundle_state:
                source = bundle_weights_path or weights_path
                raise ValueError(f'{source} is missing {name}')
            target = getattr(self, name)
            target.data.copy_(bundle_state[name].to(target.device))
        for name in ('special_game_vtxt_feat', 'special_game_ctxt_feat'):
            if name in bundle_state:
                target = getattr(self, name)
                target.data.copy_(bundle_state[name].to(target.device))
            elif self.enable_special_game_feat:
                raise ValueError(f'{bundle_weights_path or weights_path} is missing {name}')


    # ------------------------------------------------------------------
    # Atomic forward functions (shared by Trainer and Pipeline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, text: dict) -> Dict[str, Tensor]:
        """Lazy-load text encoder and encode text to vtxt/ctxt.

        Returns dict with keys: text_vec_raw, text_ctxt_raw, text_ctxt_raw_length.
        """
        device = _get_module_device(self)
        if not hasattr(self, '_text_encoder') or self._text_encoder is None:
            if self._text_encoder_cfg is None:
                raise RuntimeError(
                    'No text_encoder config provided; cannot encode text.'
                )
            from motius.models.hymotion_t2m.network.text_encoder import (
                HYTextModel,
            )
            cfg = deepcopy(self._text_encoder_cfg)
            cfg.pop('type', None)
            cfg.pop('prompt_template', None)
            cfg.pop('sentence_padding', None)
            if self._text_dtype is not None:
                cfg['torch_dtype'] = self._text_dtype
                cfg['llm_torch_dtype'] = self._text_dtype
                cfg['sentence_torch_dtype'] = self._text_dtype
            elif 'torch_dtype' in cfg:
                cfg['torch_dtype'] = _resolve_dtype(cfg['torch_dtype'])
            self._text_encoder = HYTextModel(**cfg).to(device).eval()
        else:
            self._text_encoder = self._text_encoder.to(device).eval()
        vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
        return {
            'text_vec_raw': vtxt.to(device),
            'text_ctxt_raw': ctxt.to(device),
            'text_ctxt_raw_length': ctxt_len.to(device),
        }

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O
    # ------------------------------------------------------------------

    def _artifact_text_encoder_cfg(self) -> dict:
        cfg = deepcopy(self._text_encoder_cfg) if self._text_encoder_cfg else deepcopy(_DEFAULT_TEXT_ENCODER_CFG)
        if self._text_dtype is not None:
            cfg['torch_dtype'] = _dtype_name(self._text_dtype)
        return cfg

    @staticmethod
    def _text_encoder_cfg_is_packaged(cfg: dict) -> bool:
        return bool(cfg.get('llm_model_path') and cfg.get('sentence_emb_model_path'))

    def text_encoder_components(self, cfg: Optional[dict] = None, stored: Optional[bool] = None) -> dict:
        """Return text-encoder component metadata for this artifact."""
        cfg = deepcopy(cfg) if cfg is not None else self._artifact_text_encoder_cfg()
        if stored is None:
            stored = self._text_encoder_cfg_is_packaged(cfg)
        return {
            'llm': {
                'type': cfg.get('llm_type', _DEFAULT_TEXT_ENCODER_CFG['llm_type']),
                'stored_in_artifact': bool(stored),
                'path': cfg.get('llm_model_path'),
            },
            'sentence': {
                'type': cfg.get(
                    'sentence_emb_type',
                    _DEFAULT_TEXT_ENCODER_CFG['sentence_emb_type'],
                ),
                'stored_in_artifact': bool(stored),
                'path': cfg.get('sentence_emb_model_path'),
            },
        }

    def external_text_encoder_components(self) -> dict:
        """Backward-compatible alias for component metadata."""
        return self.text_encoder_components()

    def text_encoder_requires_external_weights(self) -> bool:
        """Return True only for legacy lightweight artifacts without encoder paths."""
        return not self._text_encoder_cfg_is_packaged(self._artifact_text_encoder_cfg())

    def config_dict(self) -> dict:
        """Architecture and inference metadata persisted in HYMotion artifacts."""
        text_encoder_cfg = self._artifact_text_encoder_cfg()
        text_encoder_components = self.text_encoder_components(text_encoder_cfg)
        return {
            'format': _ARTIFACT_FORMAT,
            'motion_transformer': self.get_module_build_cfg('motion_transformer'),
            'text_encoder': text_encoder_cfg,
            'text_encoder_components': text_encoder_components,
            'external_text_encoder_components': text_encoder_components,
            'motion_type': self.motion_type,
            'pred_type': self.pred_type,
            'uncondition_mode': self.uncondition_mode,
            'losses_cfg': deepcopy(self._losses_cfg),
            'noise_scheduler_cfg': deepcopy(self._noise_scheduler_cfg),
            'infer_noise_scheduler_cfg': deepcopy(self._infer_noise_scheduler_cfg),
            'cond_mask_prob': self.cond_mask_prob,
            'enable_special_game_feat': self.enable_special_game_feat,
            'train_null_embeddings': self.train_null_embeddings,
            'train_special_game_embeddings': self.train_special_game_embeddings,
            'vtxt_input_dim': self._vtxt_input_dim,
            'ctxt_input_dim': self._ctxt_input_dim,
            'body_model_path': self._body_model_path,
        }

    def _copy_text_encoder_artifacts(self, save_dir: Path, text_encoder_cfg: dict, text_encoder_subdir: str) -> dict:
        from motius.configuration.hymotion_text import export_text_networks
        return export_text_networks(self, save_dir, text_encoder_cfg, text_encoder_subdir)

    @staticmethod
    def _resolve_text_encoder_cfg_paths(cfg: Optional[dict], artifact_dir: Path) -> Optional[dict]:
        if cfg is None:
            return None
        out = deepcopy(cfg)
        packaged_llm = artifact_dir / 'text_encoder' / 'llm'
        packaged_sentence = artifact_dir / 'text_encoder' / 'sentence'
        if packaged_llm.exists() and 'llm_model_path' not in out:
            out['llm_model_path'] = str(packaged_llm)
            out['llm_tokenizer_path'] = str(packaged_llm)
        if packaged_sentence.exists() and 'sentence_emb_model_path' not in out:
            out['sentence_emb_model_path'] = str(packaged_sentence)
            out['sentence_emb_tokenizer_path'] = str(packaged_sentence)
        for key in (
            'llm_model_path',
            'llm_tokenizer_path',
            'sentence_emb_model_path',
            'sentence_emb_tokenizer_path',
        ):
            out[key] = _resolve_artifact_path(out.get(key), artifact_dir)
        return out

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        include_text_encoder: bool = True,
        text_encoder_subdir: str = 'text_encoder',
        **kwargs,
    ):
        """Export a self-contained Motius HYMotion-T2M artifact.

        Layout::

            <dir>/hymotion_t2m_config.json
            <dir>/motion_transformer.safetensors
            <dir>/Mean.npy, Std.npy
            <dir>/text_encoder/llm/
            <dir>/text_encoder/sentence/
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        from motius.configuration.hymotion_text import copy_hymotion_notices
        copy_hymotion_notices(save_dir, wooden=True)
        config = self.config_dict()
        text_encoder_cfg = deepcopy(config['text_encoder'])
        if include_text_encoder:
            text_encoder_cfg = self._copy_text_encoder_artifacts(
                save_dir,
                text_encoder_cfg,
                text_encoder_subdir=text_encoder_subdir,
            )
            components = self.text_encoder_components(text_encoder_cfg, stored=True)
        else:
            components = self.text_encoder_components(text_encoder_cfg, stored=False)
        config['text_encoder'] = _jsonable(text_encoder_cfg)
        config['text_encoder_components'] = _jsonable(components)
        config['external_text_encoder_components'] = _jsonable(components)

        meta = {
            'model_type': 'hymotion_t2m',
            'format': _ARTIFACT_FORMAT,
            'variant': variant,
            'config': _jsonable(config),
            'pipeline_class': 'motius.pipelines.hymotion_t2m.hymotion_t2m_pipeline.HyMotionT2MPipeline',
            'bundle_class': 'motius.models.hymotion_t2m.bundle.HyMotionT2MBundle',
            'weights': {
                'motion_transformer': 'motion_transformer.safetensors'
                if safe_serialization else 'motion_transformer.pt',
                'mean': 'Mean.npy',
                'std': 'Std.npy',
            },
            'components': {
                'text_encoder': _jsonable(components),
            },
            'external_components': _jsonable(components),
        }
        if include_text_encoder:
            meta['weights']['text_encoder'] = text_encoder_subdir
        (save_dir / 'hymotion_t2m_config.json').write_text(
            json.dumps(meta, indent=2)
        )
        model_index = {
            '_class_name': 'HyMotionT2MPipeline',
            '_library_name': 'motius',
            'model_type': 'hymotion_t2m',
            'format': _ARTIFACT_FORMAT,
            'bundle_class': meta['bundle_class'],
            'pipeline_class': meta['pipeline_class'],
            'artifacts': meta['weights'],
            'components': meta['components'],
            'external_components': meta['external_components'],
            'api': {
                'from_pretrained': (
                    'motius.models.hymotion_t2m.HyMotionT2MBundle'
                    '.from_pretrained'
                ),
                'from_config': (
                    'motius.models.hymotion_t2m.HyMotionT2MBundle'
                    '.from_config'
                ),
            },
        }
        (save_dir / 'model_index.json').write_text(json.dumps(model_index, indent=2))
        readme = save_dir / 'README.md'
        if not readme.exists():
            readme.write_text(
                '# HYMotion T2M Motius Artifact\n\n'
                'This repository stores a self-contained Motius HYMotion T2M '
                'artifact, including the motion transformer, classifier-free '
                'null embeddings, normalization stats, and frozen text encoder '
                'weights. Load it with '
                '`HyMotionT2MBundle.from_pretrained(...)` and run it through '
                '`HyMotionT2MPipeline`.\n'
            )

        network = self.motion_transformer
        while hasattr(network, 'module'):
            network = network.module
        state = {
            f'motion_transformer.{k}': v.detach().cpu().contiguous()
            for k, v in network.state_dict().items()
        }
        state['null_vtxt_feat'] = self.null_vtxt_feat.detach().cpu().contiguous()
        state['null_ctxt_input'] = self.null_ctxt_input.detach().cpu().contiguous()
        state['special_game_vtxt_feat'] = self.special_game_vtxt_feat.detach().cpu().contiguous()
        state['special_game_ctxt_feat'] = self.special_game_ctxt_feat.detach().cpu().contiguous()

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state, str(save_dir / 'motion_transformer.safetensors'))
        else:
            torch.save(state, str(save_dir / 'motion_transformer.pt'))

        np.save(str(save_dir / 'Mean.npy'), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / 'Std.npy'), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        """Construct from a raw bundle config, artifact metadata, or config file path."""
        if isinstance(cfg, (str, Path)):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / 'hymotion_t2m_config.json'
            cfg = _read_json(cfg_path)
        cfg_dict = cls._to_plain_dict(cfg)
        if cfg_dict.get('model_type') == 'hymotion_t2m' and 'config' in cfg_dict:
            cfg_dict = deepcopy(cfg_dict['config'])
        cfg_dict.pop('format', None)
        cfg_dict.pop('external_text_encoder_components', None)
        cfg_dict.pop('text_encoder_components', None)
        return super().from_config(cfg_dict, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[str] = None,
        text_dtype: Optional[Any] = None,
        **kwargs,
    ):
        """Load a HYMotion-T2M artifact from a local dir or HuggingFace Hub id."""
        path = Path(pretrained_model_name_or_path)
        if not (path / 'hymotion_t2m_config.json').exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / 'hymotion_t2m_config.json'
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())
        artifact_weights = meta.get('weights', {})
        weights = path / artifact_weights.get(
            'motion_transformer', 'motion_transformer.safetensors'
        )
        if not weights.exists() and weights.suffix == '.safetensors':
            weights = path / 'motion_transformer.pt'
        bundle_weights = artifact_weights.get('bundle_params')
        bundle_weights_path = (
            str(path / bundle_weights) if bundle_weights is not None else None
        )

        cfg = deepcopy(meta['config'])
        cfg.pop('format', None)
        cfg.pop('external_text_encoder_components', None)
        cfg.pop('text_encoder_components', None)
        if 'text_encoder' in cfg:
            cfg['text_encoder'] = cls._resolve_text_encoder_cfg_paths(
                cfg['text_encoder'],
                path,
            )
        if text_dtype is not None:
            cfg['text_dtype'] = text_dtype
        cfg.update(kwargs)
        return cls(
            motion_weights_path=str(weights),
            bundle_weights_path=bundle_weights_path,
            mean_path=str(path / 'Mean.npy'),
            std_path=str(path / 'Std.npy'),
            device=device,
            **cfg,
        )


    def predict_flow(
        self,
        x_input: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
    ) -> Tensor:
        """Single forward pass through the MMDiT transformer.

        Args:
            x_input: noisy motion x_t, shape (B, L, motion_dim).
                     Unlike M2M, this is NOT concatenated with VACE context.
            ctxt_input: token-level text embeddings, (B, Lc, Dc).
            vtxt_input: sentence-level text embeddings, (B, 1, Dv).
            timesteps: diffusion timesteps, (B,).
            x_mask_temporal: (B, L) boolean mask for motion sequence.
            ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.

        Returns:
            Model prediction, shape (B, L, motion_dim).
        """
        return self.motion_transformer(
            x=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
