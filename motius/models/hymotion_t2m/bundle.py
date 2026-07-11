"""HyMotion-T2M Bundle: text-to-motion generation via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - predict_flow()          -- single forward through the transformer
  - decode_motion_from_latent() -- denormalize + FK to 3D keypoints
  - mask_text_cond()        -- classifier-free guidance null masking
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
        self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None
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
        from motius.models.hymotion_t2m.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**self._losses_cfg)

        # ---- SMPL body model (optional for FK losses / decode) ----
        self._body_model_path = body_model_path
        self._body_model: Optional[nn.Module] = None  # lazy

        # ---- store vtxt/ctxt dims for later ----
        self._vtxt_input_dim = vtxt_input_dim
        self._ctxt_input_dim = ctxt_input_dim

        # ---- infer params ----
        self.validation_steps = self._infer_noise_scheduler_cfg.get(
            'validation_steps', 50
        )

        if motion_weights_path is not None:
            self._load_artifact_weights(motion_weights_path)

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

    def _load_artifact_weights(self, weights_path: str) -> None:
        p = str(weights_path)
        if p.endswith('.safetensors'):
            from safetensors.torch import load_file

            state = load_file(p)
        else:
            state = torch.load(p, map_location='cpu')
        mt_prefix = 'motion_transformer.'
        mt_state = {
            k[len(mt_prefix):]: v
            for k, v in state.items()
            if k.startswith(mt_prefix)
        }
        if not mt_state:
            raise ValueError(f'No motion_transformer.* weights found in {weights_path}')
        self.motion_transformer.load_state_dict(mt_state, strict=True)
        for name in ('null_vtxt_feat', 'null_ctxt_input'):
            if name not in state:
                raise ValueError(f'{weights_path} is missing {name}')
            getattr(self, name).data.copy_(state[name].to(getattr(self, name).device))
        for name in ('special_game_vtxt_feat', 'special_game_ctxt_feat'):
            if name in state:
                getattr(self, name).data.copy_(state[name].to(getattr(self, name).device))

    @property
    def body_model(self):
        """Lazy-load SmplxLiteJ24 body model."""
        if self._body_model is None:
            from motius.models.hymotion_t2m.network.smpl_lite import SmplxLiteJ24
            kwargs = {}
            if self._body_model_path is not None:
                kwargs['model_path'] = self._body_model_path
            try:
                self._body_model = SmplxLiteJ24(**kwargs)
                self._body_model.to(_get_module_device(self))
                self._body_model.eval()
            except Exception:
                return None
        return self._body_model

    # ------------------------------------------------------------------
    # Atomic forward functions (shared by Trainer and Pipeline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
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
            if self._text_dtype is not None:
                cfg['torch_dtype'] = self._text_dtype
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

    def _copy_text_encoder_artifacts(
        self,
        save_dir: Path,
        text_encoder_cfg: dict,
        text_encoder_subdir: str,
    ) -> dict:
        from motius.models.hymotion_t2m.network.text_encoder import (
            LLM_ENCODER_LAYOUT,
            SENTENCE_EMB_LAYOUT,
        )

        cfg = deepcopy(text_encoder_cfg)
        llm_type = cfg.get('llm_type', _DEFAULT_TEXT_ENCODER_CFG['llm_type'])
        sentence_type = cfg.get(
            'sentence_emb_type',
            _DEFAULT_TEXT_ENCODER_CFG['sentence_emb_type'],
        )
        if llm_type not in LLM_ENCODER_LAYOUT:
            raise ValueError(f'Unsupported HYMotion LLM type: {llm_type}')
        if sentence_type not in SENTENCE_EMB_LAYOUT:
            raise ValueError(f'Unsupported HYMotion sentence encoder type: {sentence_type}')

        llm_src = _resolve_repo_path(
            cfg.get('llm_model_path') or LLM_ENCODER_LAYOUT[llm_type]['module_path']
        )
        sentence_src = _resolve_repo_path(
            cfg.get('sentence_emb_model_path')
            or SENTENCE_EMB_LAYOUT[sentence_type]['module_path']
        )
        llm_rel = f'{text_encoder_subdir}/llm'
        sentence_rel = f'{text_encoder_subdir}/sentence'
        _copy_pretrained_tree(llm_src, save_dir / llm_rel, ignore_patterns=('.cache',))
        sentence_ignore = ('.cache',)
        if (sentence_src / 'model.safetensors').exists():
            # CLIP-L snapshots often keep equivalent PyTorch/TF/Flax exports
            # next to the safetensors file. Transformers loads model.safetensors
            # directly, so omit duplicate framework weights from artifacts.
            sentence_ignore = (
                '.cache',
                'pytorch_model.bin',
                'tf_model.h5',
                'flax_model.msgpack',
            )
        _copy_pretrained_tree(sentence_src, save_dir / sentence_rel, sentence_ignore)
        cfg['llm_model_path'] = llm_rel
        cfg['llm_tokenizer_path'] = llm_rel
        cfg['sentence_emb_model_path'] = sentence_rel
        cfg['sentence_emb_tokenizer_path'] = sentence_rel
        return cfg

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

        state = {
            f'motion_transformer.{k}': v.detach().cpu().contiguous()
            for k, v in self.motion_transformer.state_dict().items()
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
        text_dtype: Optional[Any] = 'bf16',
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
        weights = path / 'motion_transformer.safetensors'
        if not weights.exists():
            weights = path / 'motion_transformer.pt'

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
            mean_path=str(path / 'Mean.npy'),
            std_path=str(path / 'Std.npy'),
            device=device,
            **cfg,
        )

    def mask_text_cond(
        self,
        vtxt: Tensor,
        ctxt: Tensor,
        force_mask: bool = False,
        cond_mask_prob: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """Apply classifier-free guidance masking to text conditions."""
        bs = vtxt.shape[0]
        if force_mask:
            return (
                self.null_vtxt_feat.expand(*vtxt.shape),
                self.null_ctxt_input.expand(*ctxt.shape),
            )
        if self.training and cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=vtxt.device) * cond_mask_prob
            ).view(bs, 1).bool()
            mask_vtxt = mask
            while mask_vtxt.ndim < vtxt.ndim:
                mask_vtxt = mask_vtxt.unsqueeze(-1)
            vtxt = torch.where(
                mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt
            )
            mask_ctxt = mask
            while mask_ctxt.ndim < ctxt.ndim:
                mask_ctxt = mask_ctxt.unsqueeze(-1)
            ctxt = torch.where(
                mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt
            )
        return vtxt, ctxt

    def maybe_inject_source_token(
        self,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        ctxt_mask_temporal: Tensor,
        sources: Optional[List[str]],
        trigger_sources: Optional[Set[str]] = None,
        prob: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Inject the official special source token for selected data sources."""
        if (sources is None or trigger_sources is None) or not self.enable_special_game_feat:
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        B, Lc, Dc = ctxt_input.shape
        if not isinstance(sources, (list, tuple)) or len(sources) != B:
            raise ValueError(f'sources length must equal batch size: {len(sources)} vs {B}')

        trig = {str(s).lower() for s in trigger_sources}
        src_mask = torch.tensor(
            [str(s).lower() in trig for s in sources],
            dtype=torch.bool,
            device=ctxt_input.device,
        )
        if not src_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        if self.training:
            rand_mask = torch.rand(B, device=ctxt_input.device) < prob
        else:
            rand_mask = torch.ones(B, dtype=torch.bool, device=ctxt_input.device)
        apply_mask = src_mask & rand_mask
        if not apply_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        vtxt_token = self.special_game_vtxt_feat.to(vtxt_input).expand(B, 1, -1)
        vtxt_input = vtxt_input + vtxt_token * apply_mask.view(B, 1, 1).to(vtxt_input.dtype)

        if ctxt_mask_temporal.dtype == torch.bool:
            cur_len = ctxt_mask_temporal.sum(dim=1).long()
        else:
            cur_len = (ctxt_mask_temporal > 0).sum(dim=1).long()

        can_inplace = apply_mask & (cur_len < Lc)
        b_inplace = torch.nonzero(can_inplace, as_tuple=False).squeeze(1)
        if b_inplace.numel() > 0:
            pos = cur_len[b_inplace]
            token = self.special_game_ctxt_feat.squeeze(0).squeeze(0).to(ctxt_input)
            ctxt_input = ctxt_input.clone()
            ctxt_mask_temporal = ctxt_mask_temporal.clone()
            ctxt_input[b_inplace, pos, :] = token.unsqueeze(0).expand(b_inplace.numel(), Dc)
            if ctxt_mask_temporal.dtype == torch.bool:
                ctxt_mask_temporal[b_inplace, pos] = True
            else:
                ctxt_mask_temporal[b_inplace, pos] = 1

        need_expand = (apply_mask & (cur_len >= Lc)).any()
        if need_expand:
            suffix = torch.zeros((B, 1, Dc), dtype=ctxt_input.dtype, device=ctxt_input.device)
            full_hit = apply_mask & (cur_len >= Lc)
            b_full = torch.nonzero(full_hit, as_tuple=False).squeeze(1)
            if b_full.numel() > 0:
                suffix[b_full, 0, :] = (
                    self.special_game_ctxt_feat.expand(b_full.numel(), 1, -1)
                    .to(ctxt_input)
                    .squeeze(1)
                )
            ctxt_input = torch.cat([ctxt_input, suffix], dim=1)

            if ctxt_mask_temporal.dtype == torch.bool:
                suffix_mask = torch.zeros((B, 1), dtype=torch.bool, device=ctxt_input.device)
                suffix_mask[b_full, 0] = True
            else:
                suffix_mask = torch.zeros(
                    (B, 1), dtype=ctxt_mask_temporal.dtype, device=ctxt_input.device)
                suffix_mask[b_full, 0] = 1
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal, suffix_mask], dim=1)

        return vtxt_input, ctxt_input, ctxt_mask_temporal

    def predict_flow(
        self,
        x_input: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
        sources: Optional[List[str]] = None,
        trigger_sources: Optional[Set[str]] = None,
        special_game_prob: float = 0.5,
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
        if ctxt_mask_temporal is not None:
            vtxt_input, ctxt_input, ctxt_mask_temporal = self.maybe_inject_source_token(
                vtxt_input=vtxt_input,
                ctxt_input=ctxt_input,
                ctxt_mask_temporal=ctxt_mask_temporal,
                sources=sources,
                trigger_sources=trigger_sources,
                prob=special_game_prob,
            )

        return self.motion_transformer(
            x=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

    def decode_motion_from_latent(
        self,
        latent: Tensor,
        should_apply_smoothing: bool = True,
    ) -> Dict[str, Tensor]:
        """Denormalize latent and run FK to get 3D keypoints.

        When ``should_apply_smoothing`` (the official HY-Motion-1.0 inference
        default) the raw flow-matching output is temporally filtered before FK:
        rot6d gets quaternion-Gaussian SLERP smoothing (sigma=1.0) and the root
        translation gets Savitzky-Golay smoothing (window=11, polyorder=5),
        matching ``MotionFlowMatching.decode_motion_from_latent`` /
        ``_decode_o6dp``. Skipping this (the previous behaviour) leaves severe
        high-frequency jitter in the generated motion.

        Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.
        """
        from motius.models.hymotion_t2m.network.geometry import rot6d_to_rotation_matrix

        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()
        body6d = latent_denorm[..., 9:135].reshape(B, L, 21, 6).clone()
        rot6d = torch.cat([root_rot6d, body6d], dim=2)

        if should_apply_smoothing:
            from motius.models.hymotion_t2m._smoothing import (
                smooth_with_savgol,
                smooth_with_slerp,
            )
            # SLERP smoothing on the 22 body joints' rot6d (sigma=1.0); transl
            # Savitzky-Golay (window=11, polyorder=5) — official inference path.
            rot6d = smooth_with_slerp(rot6d, sigma=1.0)
            transl = smooth_with_savgol(transl, window_length=11, polyorder=5)
            root_rot6d = rot6d[:, :, 0:1, :]
            body6d = rot6d[:, :, 1:, :]

        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, :, 0, :])

        k3d = None
        if self.body_model is not None:
            try:
                device = latent.device
                betas = torch.zeros(1, 16, device=device)
                k3d_list = []
                for b in range(B):
                    out = self.body_model(
                        body6d[b].to(device),
                        betas,
                        root_rot6d[b].to(device),
                        transl[b].to(device),
                    )
                    k3d_list.append(out)
                k3d = torch.stack(k3d_list, dim=0)
            except Exception:
                k3d = None

        # Ground alignment: offset translation so lowest joint touches Y=0.
        # Matches official HY-Motion-1.0 post-FK processing.
        if k3d is not None:
            # min Y across all joints and frames per batch sample
            min_y = k3d[:, :, :, 1].min(dim=2)[0].min(dim=1)[0]  # (B,)
            transl[:, :, 1] = transl[:, :, 1] - min_y.unsqueeze(1)
            k3d[:, :, :, 1] = k3d[:, :, :, 1] - min_y.unsqueeze(1).unsqueeze(1)

        return {
            'latent_denorm': latent_denorm,
            'keypoints3d': k3d,
            'rot6d': rot6d,
            'transl': transl,
            'root_rotations_mat': root_rotmat,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers.

        Dims with near-zero std (constant in training data) produce 0 after normalization.
        This matches official HY-Motion-1.0 behavior.
        """
        # Safe division: where std==0, output 0 (those dims are constant)
        safe_std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        result = (motion - self.mean) / safe_std
        # Zero out dims where std was near-zero
        result = torch.where(self.std.unsqueeze(0) < 1e-3, torch.zeros_like(result), result)
        return result

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion (matching official HY-Motion-1.0: zeros for near-zero std)."""
        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        return motion * std + self.mean
