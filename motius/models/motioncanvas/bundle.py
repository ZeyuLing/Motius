"""MotionCanvas Bundle: motion-to-motion editing via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - prepare_padding()       -- align src/tgt motions + build masks
  - prepare_condition_context() -- build edit context and target mask
  - predict_flow()          -- single forward through the transformer
  - decode_motion_from_latent() -- denormalize + FK to 3D keypoints
  - mask_text_cond()        -- classifier-free guidance null masking
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_ARTIFACT_FORMAT = 'motius-motioncanvas-v1'
_CONFIG_NAME = 'motioncanvas_config.json'
_DEFAULT_TEXT_ENCODER_CFG = {
    'type': 'HYTextModel',
    'llm_type': 'qwen3',
    'max_length_llm': 128,
    'sentence_emb_type': 'clipl',
    'max_length_sentence_emb': 77,
    'enable_llm_padding': True,
}
_DTYPE_ALIASES = {
    'fp32': torch.float32,
    'float32': torch.float32,
    'fp16': torch.float16,
    'float16': torch.float16,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
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


def _resolve_dtype(dtype: Optional[Any]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and dtype in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[dtype]
    raise ValueError(f'Unsupported dtype: {dtype!r}')


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
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    return value


def _resolve_artifact_path(value: Optional[str], artifact_dir: Path) -> Optional[str]:
    if not value:
        return value
    value_path = Path(value)
    if value_path.is_absolute() or '://' in value:
        return value
    return str(artifact_dir / value_path)


def _resolve_source_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute():
        return path
    repo_path = Path(__file__).resolve().parents[3] / path
    return repo_path if repo_path.exists() else path


def _copy_pretrained_tree(src: Path, dst: Path, ignore_patterns=()) -> None:
    if not src.exists():
        raise FileNotFoundError(f'pretrained component not found: {src}')
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*ignore_patterns) if ignore_patterns else None
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@MODEL_BUNDLES.register_module()
class MotionCanvasBundle(ModelBundle):
    """ModelBundle for MotionCanvas motion-to-motion editing.

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer``  (the HunyuanMotionMMDiT).  Auxiliary objects
    (M2MLoss, SmplxLiteJ24, null embeddings, mean/std buffers) are created
    directly in ``__init__`` as regular attributes.
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
        uncondition_mode: bool = True,
        losses_cfg: Optional[dict] = None,
        noise_scheduler_cfg: Optional[dict] = None,
        infer_noise_scheduler_cfg: Optional[dict] = None,
        cond_mask_prob: float = 0.0,
        motion_cond_mask_prob: float = 0.0,
        enable_special_game_feat: bool = False,
        train_null_embeddings: bool = True,
        train_special_game_embeddings: bool = True,
        enable_ctxt_null_feat: bool = False,
        vtxt_input_dim: int = 768,
        ctxt_input_dim: int = 4096,
        # ----- self-contained artifact loading -----
        motion_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        bone_offsets_path: Optional[str] = None,
        text_dtype: Optional[Any] = None,
        device: Optional[str] = None,
        # ----- SMPL body model path (optional; skipped if None) -----
        body_model_path: Optional[str] = None,
        # ----- rotation space -----
        rotation_space: str = 'local',
        # ----- T2M pretrained checkpoint loading (optional) -----
        t2m_pretrained_path: Optional[str] = None,
        t2m_freeze_strategy: str = 'none',
        # ----- caption-specific freezing (optional) -----
        caption_freeze_strategy: str = 'none',
    ):
        super().__init__()

        # ---- build trainable module via _build_modules ----
        import motius.models.motioncanvas.network  # noqa: F401

        self._build_modules({'motion_transformer': motion_transformer})

        # ---- hyper-params ----
        self.motion_type = motion_type
        self.pred_type = pred_type
        self.uncondition_mode = uncondition_mode
        self.cond_mask_prob = cond_mask_prob
        self.motion_cond_mask_prob = motion_cond_mask_prob
        self._losses_cfg = deepcopy(losses_cfg or {})
        self.enable_ctxt_null_feat = bool(enable_ctxt_null_feat)
        self.rotation_space = rotation_space
        assert rotation_space in ('local', 'global'), (
            f"rotation_space must be 'local' or 'global', got {rotation_space!r}"
        )
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg or {'method': 'euler'})
        self._infer_noise_scheduler_cfg = deepcopy(
            infer_noise_scheduler_cfg or {'validation_steps': 50}
        )

        # ---- text encoder config (lazy-loaded) ----
        self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None
        self._text_dtype = _resolve_dtype(text_dtype)

        # ---- null embeddings for classifier-free guidance ----
        # Zero defaults match HYMotion T2M; warm-start checkpoints carry the
        # learned null and special-source embeddings.
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

        # ---- M2M loss ----
        from motius.models.motioncanvas.network.m2m_loss import M2MLoss
        self.m2m_loss = M2MLoss(**self._losses_cfg)

        # ---- SMPL body model (optional for FK losses / decode) ----
        self._body_model_path = body_model_path
        self._body_model: Optional[nn.Module] = None  # lazy
        # Inference adapters may need the exact rest-skeleton asset used to
        # construct the model's 198-D training representation.  Keep this
        # override non-persistent so it is an explicit runtime contract rather
        # than an accidental checkpoint parameter.
        self.register_buffer('_bone_offsets_override', None, persistent=False)
        if bone_offsets_path is not None:
            offsets = torch.load(
                bone_offsets_path,
                map_location='cpu',
                weights_only=True,
            )
            self.set_bone_offsets_override(offsets)

        # ---- store vtxt/ctxt dims for later ----
        self._vtxt_input_dim = vtxt_input_dim
        self._ctxt_input_dim = ctxt_input_dim

        # ---- infer params ----
        self.validation_steps = self._infer_noise_scheduler_cfg.get(
            'validation_steps', 50
        )

        # ---- Load T2M pretrained backbone (optional) ----
        self._t2m_pretrained_path = t2m_pretrained_path
        self._t2m_freeze_strategy = t2m_freeze_strategy
        if t2m_pretrained_path:
            self.load_t2m_backbone(t2m_pretrained_path, t2m_freeze_strategy)

        # ---- Apply caption-specific freezing (after T2M loading) ----
        self._caption_freeze_strategy = caption_freeze_strategy
        if caption_freeze_strategy != 'none':
            self.apply_caption_freeze_strategy(caption_freeze_strategy)

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
        if mean_path is not None or std_path is not None:
            if mean_path is None or std_path is None:
                raise ValueError('mean_path and std_path must be provided together')
            mean_np = np.load(mean_path)
            std_np = np.load(std_path)
            if mean_np.shape != std_np.shape or mean_np.ndim != 1:
                raise ValueError(
                    f'Mean/Std must be same-shape vectors, got '
                    f'{mean_np.shape} and {std_np.shape}'
                )
            self.register_buffer('mean', torch.from_numpy(mean_np).float())
            self.register_buffer('std', torch.from_numpy(std_np).float())
            return
        if mean_std_dir is None:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('std', torch.ones(1))
            return

        resolved_dir = osp.abspath(osp.expanduser(mean_std_dir))
        if not osp.isdir(resolved_dir):
            raise FileNotFoundError(
                'mean_std_dir was configured but is not a directory: '
                f'{resolved_dir}'
            )
        mean_path = osp.join(resolved_dir, 'Mean.npy')
        std_path = osp.join(resolved_dir, 'Std.npy')
        missing = [
            path for path in (mean_path, std_path) if not osp.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                'mean_std_dir lacks required normalization files: '
                + ', '.join(missing)
            )
        mean_np = np.load(mean_path)
        std_np = np.load(std_path)
        if mean_np.ndim != 1 or std_np.shape != mean_np.shape:
            raise ValueError(
                'Mean.npy/Std.npy must be same-shape vectors, got '
                f'{mean_np.shape} and {std_np.shape}'
            )
        if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all():
            raise ValueError('Mean.npy/Std.npy contain non-finite values')
        mean = torch.from_numpy(mean_np).float()
        std = torch.from_numpy(std_np).float()
        # Clamp std to avoid div-by-zero
        std = torch.where(std < 1e-3, torch.ones_like(std), std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _load_artifact_weights(self, weights_path: str) -> None:
        if str(weights_path).endswith('.safetensors'):
            from safetensors.torch import load_file

            state = load_file(str(weights_path))
        else:
            state = torch.load(weights_path, map_location='cpu')
        prefix = 'motion_transformer.'
        transformer_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not transformer_state:
            # Raw source checkpoints store the transformer state without a prefix.
            expected = set(self.motion_transformer.state_dict())
            if set(state) == expected:
                transformer_state = state
        self.motion_transformer.load_state_dict(transformer_state, strict=True)
        for name in (
            'null_vtxt_feat',
            'null_ctxt_input',
            'special_game_vtxt_feat',
            'special_game_ctxt_feat',
        ):
            if name not in state:
                raise ValueError(f'{weights_path} is missing required tensor {name}')
            target = getattr(self, name)
            target.data.copy_(state[name].to(device=target.device, dtype=target.dtype))

    @property
    def body_model(self):
        """Lazy-load SmplxLiteJ24 body model."""
        if self._body_model is None:
            from motius.models.motioncanvas.network.smpl_lite import SmplxLiteJ24
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
            from motius.models.motioncanvas.network.text_encoder import (
                HYTextModel,
            )
            cfg = deepcopy(self._text_encoder_cfg)
            cfg.pop('type', None)
            if self._text_dtype is not None:
                cfg['torch_dtype'] = self._text_dtype
            elif 'torch_dtype' in cfg:
                cfg['torch_dtype'] = _resolve_dtype(cfg['torch_dtype'])
            self._text_encoder = HYTextModel(**cfg)
            # Keep text encoder on CPU — it is inference-only and not part of
            # the trainable graph.  Moving an 8B LLM to each rank's GPU would
            # exhaust memory.  encode() uses get_module_device(self) internally
            # so inputs/outputs stay on CPU; we move the returned tensors to
            # the training device below.
        vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
        return {
            'text_vec_raw': vtxt.to(device),
            'text_ctxt_raw': ctxt.to(device),
            'text_ctxt_raw_length': ctxt_len.to(device),
        }

    @torch.no_grad()
    def encode_task_instruction(self, task_instructions: List[str]) -> Dict[str, Tensor]:
        """Encode task instructions to vtxt-like embeddings using CLIP.

        Task instructions (e.g., 'complete motion from sparse random cells') are
        CLIP-encoded to the same space as caption sentence embeddings (768-dim),
        then projected to 1024-dim via vtxt_encoder to match the adapter signal.

        Args:
            task_instructions: List of task instruction strings (e.g., from mask_strategy)

        Returns:
            Dict with key 'task_emb' containing (B, 1, 1024) embeddings ready to add to adapter.
        """
        device = _get_module_device(self)
        if not hasattr(self, '_text_encoder') or self._text_encoder is None:
            if self._text_encoder_cfg is None:
                raise RuntimeError(
                    'No text_encoder config provided; cannot encode task instructions.'
                )
            from motius.models.motioncanvas.network.text_encoder import (
                HYTextModel,
            )
            cfg = deepcopy(self._text_encoder_cfg)
            cfg.pop('type', None)
            self._text_encoder = HYTextModel(**cfg)

        # Encode task instructions using CLIP (sentence_emb branch)
        task_vtxt, _, _ = self._text_encoder.encode(task_instructions)

        # Project to 1024-dim to match vtxt_encoder output
        task_emb = self.vtxt_encoder(task_vtxt.float().to(device))  # (B, 1, 1024)

        return {'task_emb': task_emb}

    # ------------------------------------------------------------------
    # Self-contained Hugging Face artifact I/O
    # ------------------------------------------------------------------

    def _artifact_text_encoder_cfg(self) -> dict:
        cfg = (
            deepcopy(self._text_encoder_cfg)
            if self._text_encoder_cfg
            else deepcopy(_DEFAULT_TEXT_ENCODER_CFG)
        )
        if self._text_dtype is not None:
            cfg['torch_dtype'] = _dtype_name(self._text_dtype)
        return cfg

    def config_dict(self) -> dict:
        """Return the complete architecture and inference contract."""
        return {
            'format': _ARTIFACT_FORMAT,
            'motion_transformer': self.get_module_build_cfg('motion_transformer'),
            'text_encoder': self._artifact_text_encoder_cfg(),
            'motion_type': self.motion_type,
            'pred_type': self.pred_type,
            'uncondition_mode': self.uncondition_mode,
            'losses_cfg': deepcopy(self._losses_cfg),
            'noise_scheduler_cfg': deepcopy(self._noise_scheduler_cfg),
            'infer_noise_scheduler_cfg': deepcopy(self._infer_noise_scheduler_cfg),
            'cond_mask_prob': self.cond_mask_prob,
            'motion_cond_mask_prob': self.motion_cond_mask_prob,
            'enable_special_game_feat': self.enable_special_game_feat,
            'train_null_embeddings': self.train_null_embeddings,
            'train_special_game_embeddings': self.train_special_game_embeddings,
            'enable_ctxt_null_feat': self.enable_ctxt_null_feat,
            'vtxt_input_dim': self._vtxt_input_dim,
            'ctxt_input_dim': self._ctxt_input_dim,
            'body_model_path': None,
            'rotation_space': self.rotation_space,
            't2m_pretrained_path': None,
            't2m_freeze_strategy': self._t2m_freeze_strategy,
            'caption_freeze_strategy': self._caption_freeze_strategy,
        }

    @staticmethod
    def _resolve_text_encoder_paths(cfg: Optional[dict], artifact_dir: Path):
        if cfg is None:
            return None
        cfg = deepcopy(cfg)
        for key in (
            'llm_model_path',
            'llm_tokenizer_path',
            'sentence_emb_model_path',
            'sentence_emb_tokenizer_path',
        ):
            cfg[key] = _resolve_artifact_path(cfg.get(key), artifact_dir)
        return cfg

    def _package_text_encoder(self, save_dir: Path, cfg: dict) -> dict:
        from motius.models.motioncanvas.network.text_encoder import (
            LLM_ENCODER_LAYOUT,
            SENTENCE_EMB_LAYOUT,
        )

        cfg = deepcopy(cfg)
        llm_type = cfg.get('llm_type', 'qwen3')
        sentence_type = cfg.get('sentence_emb_type', 'clipl')
        llm_source = _resolve_source_path(
            cfg.get('llm_model_path')
            or LLM_ENCODER_LAYOUT[llm_type]['module_path']
        )
        sentence_source = _resolve_source_path(
            cfg.get('sentence_emb_model_path')
            or SENTENCE_EMB_LAYOUT[sentence_type]['module_path']
        )
        _copy_pretrained_tree(
            llm_source,
            save_dir / 'text_encoder' / 'llm',
            ignore_patterns=('.cache',),
        )
        sentence_ignores = ('.cache',)
        if (sentence_source / 'model.safetensors').exists():
            sentence_ignores += (
                'pytorch_model.bin',
                'tf_model.h5',
                'flax_model.msgpack',
            )
        _copy_pretrained_tree(
            sentence_source,
            save_dir / 'text_encoder' / 'sentence',
            ignore_patterns=sentence_ignores,
        )
        cfg.update(
            llm_model_path='text_encoder/llm',
            llm_tokenizer_path='text_encoder/llm',
            sentence_emb_model_path='text_encoder/sentence',
            sentence_emb_tokenizer_path='text_encoder/sentence',
        )
        return cfg

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        include_text_encoder: bool = True,
        variant: Optional[str] = None,
    ):
        """Write a complete MotionCanvas artifact loadable by ``Pipeline``."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = self.config_dict()
        if include_text_encoder:
            config['text_encoder'] = self._package_text_encoder(
                save_dir,
                config['text_encoder'],
            )

        weights_name = (
            'motion_transformer.safetensors'
            if safe_serialization
            else 'motion_transformer.pt'
        )
        state = {
            f'motion_transformer.{key}': value.detach().cpu().contiguous()
            for key, value in self.motion_transformer.state_dict().items()
        }
        for name in (
            'null_vtxt_feat',
            'null_ctxt_input',
            'special_game_vtxt_feat',
            'special_game_ctxt_feat',
        ):
            state[name] = getattr(self, name).detach().cpu().contiguous()
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state, str(save_dir / weights_name))
        else:
            torch.save(state, save_dir / weights_name)
        np.save(save_dir / 'Mean.npy', self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(save_dir / 'Std.npy', self.std.detach().cpu().numpy().astype(np.float32))
        torch.save(self.get_bone_offsets().detach().cpu(), save_dir / 'bone_offsets_22.pt')

        meta = {
            'model_type': 'motioncanvas',
            'format': _ARTIFACT_FORMAT,
            'variant': variant,
            'config': _jsonable(config),
            'pipeline_class': (
                'motius.pipelines.motioncanvas.motioncanvas_pipeline.'
                'MotionCanvasPipeline'
            ),
            'bundle_class': 'motius.models.motioncanvas.bundle.MotionCanvasBundle',
        }
        (save_dir / _CONFIG_NAME).write_text(
            json.dumps(meta, indent=2),
            encoding='utf-8',
        )
        required_files = [
            _CONFIG_NAME,
            weights_name,
            'Mean.npy',
            'Std.npy',
            'bone_offsets_22.pt',
        ]
        if include_text_encoder:
            required_files += [
                'text_encoder/llm/config.json',
                'text_encoder/llm/tokenizer_config.json',
                'text_encoder/sentence/config.json',
            ]
        model_index = {
            '_class_name': 'MotionCanvasPipeline',
            '_library_name': 'motius',
            'model_type': 'motioncanvas',
            'format': _ARTIFACT_FORMAT,
            'bundle_class': meta['bundle_class'],
            'pipeline_class': meta['pipeline_class'],
            'tasks': [
                'motion-to-motion',
                'temporal-motion-completion',
                'keyframe-motion-control',
                'motion-editing',
            ],
            'required_files': required_files,
            'artifacts': {
                'motion_transformer': weights_name,
                'mean': 'Mean.npy',
                'std': 'Std.npy',
                'bone_offsets': 'bone_offsets_22.pt',
                'text_encoder': 'text_encoder',
            },
            'api': {
                'load': 'motius.Pipeline.from_pretrained',
                'inference': 'MotionCanvasPipeline.infer_m2m',
            },
        }
        (save_dir / 'model_index.json').write_text(
            json.dumps(model_index, indent=2),
            encoding='utf-8',
        )
        return str(save_dir)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        if isinstance(cfg, (str, Path)):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / _CONFIG_NAME
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        config = cls._to_plain_dict(cfg)
        if config.get('model_type') == 'motioncanvas' and 'config' in config:
            config = deepcopy(config['config'])
        config.pop('format', None)
        return super().from_config(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        device: Optional[str] = None,
        text_dtype: Optional[Any] = 'bf16',
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        path = Path(pretrained_model_name_or_path).expanduser()
        if not (path / _CONFIG_NAME).exists():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        meta = json.loads((path / _CONFIG_NAME).read_text(encoding='utf-8'))
        config = deepcopy(meta['config'])
        config.pop('format', None)
        config['text_encoder'] = cls._resolve_text_encoder_paths(
            config.get('text_encoder'),
            path,
        )
        config.update(kwargs)
        weights = path / 'motion_transformer.safetensors'
        if not weights.exists():
            weights = path / 'motion_transformer.pt'
        return cls(
            motion_weights_path=str(weights),
            mean_path=str(path / 'Mean.npy'),
            std_path=str(path / 'Std.npy'),
            bone_offsets_path=str(path / 'bone_offsets_22.pt'),
            text_dtype=text_dtype,
            device=device,
            **config,
        )

    def mask_text_cond(
        self,
        vtxt: Tensor,
        ctxt: Tensor,
        force_mask: bool = False,
        cond_mask_prob: float = 0.0,
        return_text_available: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Apply classifier-free guidance masking to text conditions.

        Args:
            vtxt: Sentence-level text embeddings, shape (B, 1, D_v).
            ctxt: Token-level text embeddings, shape (B, L_c, D_c).
            force_mask: If True, return null embeddings for all samples.
            cond_mask_prob: Probability of masking text (CFG dropout rate).
            return_text_available: If True, also return boolean mask indicating
                which samples have real text (not masked). Shape (B,).

        Returns:
            - If return_text_available=False: (vtxt_masked, ctxt_masked)
            - If return_text_available=True: (vtxt_masked, ctxt_masked, text_available)
              where text_available[b]=True means sample b has real text,
              text_available[b]=False means sample b was masked to null.
        """
        bs = vtxt.shape[0]
        # Track which samples have real (non-masked) text
        text_available = torch.ones(bs, dtype=torch.bool, device=vtxt.device)

        if force_mask:
            text_available.fill_(False)
            result = (
                self.null_vtxt_feat.expand(*vtxt.shape),
                self.null_ctxt_input.expand(*ctxt.shape),
            )
            if return_text_available:
                return result + (text_available,)
            return result

        if self.training and cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=vtxt.device) * cond_mask_prob
            ).view(bs, 1).bool()
            # Invert: mask=1 (drop text) -> text_available=0 (no real text)
            text_available = ~mask.squeeze(-1)

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

        result = (vtxt, ctxt)
        if return_text_available:
            return result + (text_available,)
        return result

    def prepare_padding(
        self,
        src_motion: Tensor,
        tgt_motion: Optional[Tensor],
        tgt_length: List[int],
        src_mask: Optional[Tensor] = None,
        src_length: Optional[List[int]] = None,
        ref_pose: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[int], List[int], Tensor]:
        """Pad src/tgt motions to the same length and build tgt_padding_mask.

        Returns:
            (src_motion, src_mask, tgt_motion, src_length, tgt_length,
             tgt_padding_mask)
        """
        device = src_motion.device
        B, L_s, D = src_motion.shape
        L_t = tgt_motion.shape[1] if tgt_motion is not None else L_s
        L_r = ref_pose.shape[1] if ref_pose is not None else 0

        if src_length is None:
            src_length = tgt_length

        max_len = max(L_s, L_t)
        if src_mask is None:
            src_mask = torch.ones_like(src_motion)

        # Pad src
        if L_s < max_len:
            pad = max_len - L_s
            src_motion = F.pad(src_motion, (0, 0, 0, pad))
            src_mask = F.pad(src_mask, (0, 0, 0, pad))

        # Pad tgt
        if tgt_motion is not None and L_t < max_len:
            pad = max_len - L_t
            tgt_motion = F.pad(tgt_motion, (0, 0, 0, pad))
        elif tgt_motion is None:
            tgt_motion = torch.zeros(B, max_len, D, dtype=src_motion.dtype, device=device)

        # Build tgt_padding_mask
        if L_r > 0:
            ref_mask = torch.ones(B, L_r, dtype=torch.bool, device=device)
        else:
            ref_mask = torch.empty(B, 0, dtype=torch.bool, device=device)

        tgt_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), max_len
        )
        tgt_padding_mask = torch.cat([ref_mask, tgt_mask], dim=1)

        return src_motion, src_mask, tgt_motion, src_length, tgt_length, tgt_padding_mask

    def prepare_condition_context(
        self,
        src_motion: Tensor,
        ref_pose: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build the fixed motion-conditioning context.

        The transformer always receives ``[x_t, edit_context, target_mask]``.
        Hard motion observations are already imputed into ``x_t`` and are
        re-imputed after every inference step, so they are not duplicated in
        this context. ``edit_context`` contains the source motion only where
        the model must generate (``target_mask == 1``); it is therefore zero
        for completion tasks and populated for motion editing.

        Returns a tensor of shape ``(B, L, 2 * D)``. Concatenating it with
        ``x_t`` gives the transformer's fixed ``3 * D`` motion input.
        """
        B, L_src, D = src_motion.shape
        if src_mask is None:
            src_mask = torch.ones_like(src_motion)

        edit_context = src_motion * src_mask

        if ref_pose is not None:
            _, L_ref, _ = ref_pose.shape
            src_mask = torch.cat(
                [torch.zeros(B, L_ref, D, dtype=src_mask.dtype, device=src_mask.device), src_mask],
                dim=1,
            )
            edit_context = torch.cat([ref_pose, edit_context], dim=1)

        return torch.cat([edit_context, src_mask], dim=-1)

    def predict_flow(
        self,
        x_input: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
        mask_density: Optional[Tensor] = None,
        task_emb: Optional[Tensor] = None,
        sources: Optional[List[str]] = None,
        trigger_sources: Optional[Set[str]] = None,
        special_game_prob: float = 0.5,
    ) -> Tensor:
        """Single forward pass through the MMDiT transformer.

        Args:
            x_input: concatenated ``[x_t, edit_context, target_mask]``,
                shape ``(B, L, 3 * D_motion)``.
            ctxt_input: token-level text embeddings, (B, Lc, Dc).
            vtxt_input: sentence-level text embeddings, (B, 1, Dv).
            timesteps: diffusion timesteps, (B,).
            x_mask_temporal: (B, L) boolean mask for motion sequence.
            ctxt_mask_temporal: (B, Lc) boolean mask for text tokens.
            mask_density: (B,) optional mask density for CDE (CRFM v3).
            task_emb: (B, 1, 1024) optional task instruction embeddings to add to adapter.
            sources: Optional data source names for official special-source token injection.
            trigger_sources: Source names that should receive the learned special token.

        Returns:
            Model prediction, shape (B, L, D_motion).
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
            mask_density=mask_density,
            task_emb=task_emb,
        )

    def maybe_inject_source_token(
        self,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        ctxt_mask_temporal: Tensor,
        sources: Optional[List[str]],
        trigger_sources: Optional[Set[str]] = None,
        prob: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Inject the official HYMotion special source token for selected sources."""
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

    def decode_motion_from_latent(
        self,
        latent: Tensor,
    ) -> Dict[str, Tensor]:
        """Denormalize latent and run FK to get 3D keypoints.

        Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.

        When ``rotation_space == 'global'``, the denormalized rot6d is in
        world-frame global rotation.  We convert it back to local (SMPL)
        rotation before FK so that the output NPZ is always SMPL-compatible.
        """
        from motius.models.motioncanvas.network.geometry import rot6d_to_rotation_matrix

        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()

        # Extract rot6d: (B, L, 22, 6)
        rot6d_all = latent_denorm[..., 3:135].reshape(B, L, 22, 6).clone()

        # If trained in global rotation space, convert back to local for SMPL output
        if self.rotation_space == 'global':
            from motius.motion.skeleton.fk import global_to_local_rot6d
            rot6d_all = global_to_local_rot6d(rot6d_all)

        root_rot6d = rot6d_all[:, :, 0:1, :]   # (B, L, 1, 6)
        body6d = rot6d_all[:, :, 1:, :]         # (B, L, 21, 6)
        rot6d = rot6d_all
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

        return {
            'latent_denorm': latent_denorm,
            'keypoints3d': k3d,
            'rot6d': rot6d,
            'transl': transl,
            'root_rotations_mat': root_rotmat,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers."""
        return (motion - self.mean) / self.std

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion."""
        std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        return motion * std + self.mean


    def load_t2m_backbone(
        self,
        checkpoint_path: str,
        freeze_strategy: str = 'none',
    ) -> Dict[str, Any]:
        """Load T2M pretrained weights selectively into this M2M bundle.

        Handles architecture differences (conditioning input expansion, output dimension).
        See checkpoint_loading.load_t2m_pretrained_selective() for details.

        Args:
            checkpoint_path: Path to T2M pretrained checkpoint (.ckpt or .pt)
            freeze_strategy: 'none', 'encoders', 'text_refiner', 'blocks', 'full'

        Returns:
            Dict with loading statistics (modules_loaded, modules_skipped, etc.)
        """
        from motius.models.motioncanvas.checkpoint_loading import (
            load_t2m_pretrained_selective,
        )
        return load_t2m_pretrained_selective(
            bundle=self,
            t2m_checkpoint_path=checkpoint_path,
            freeze_strategy=freeze_strategy,
        )


    def apply_caption_freeze_strategy(self, strategy: str) -> None:
        """Freeze modules per strategy to preserve T2M text understanding.

        Called in __init__ when caption_freeze_strategy != 'none'. The freeze
        is applied early (before checkpoint loading in the runner) so that:
        - _build_optimizers() correctly excludes frozen params
        - accelerator.prepare() sees the right requires_grad flags
        - load_state_dict_selective() later overwrites weights without
          resetting requires_grad (it uses param.data.copy_)

        Strategies:
          'none'         — no freezing (default)
          'encoders'     — freeze vtxt_encoder + ctxt_encoder
          'text_refiner' — above + text_refiner
          'blocks'       — above + double_blocks + single_blocks
          'full'         — all reusable T2M modules
        """
        from .checkpoint_loading import _apply_freeze_strategy

        frozen = _apply_freeze_strategy(self, strategy)

        if frozen:
            total = sum(p.numel() for p in self.parameters())
            frozen_n = sum(
                p.numel() for p in self.parameters() if not p.requires_grad
            )
            logger.info(
                f"Caption freeze ({strategy}): {frozen_n:,}/{total:,} params "
                f"frozen ({frozen_n / total * 100:.1f}%)"
            )

    def train(self, mode: bool = True):
        """Override train() to re-enforce caption freeze strategy.

        Safety net: if anything upstream (e.g. accelerator.prepare, DDP wrapper)
        calls module.requires_grad_(True), the freeze is re-applied on the next
        train() call.
        """
        super().train(mode)
        strategy = getattr(self, '_caption_freeze_strategy', 'none')
        if strategy != 'none':
            from .checkpoint_loading import _apply_freeze_strategy
            _apply_freeze_strategy(self, strategy)
        return self

    def set_bone_offsets_override(self, bone_offsets: Tensor) -> None:
        """Install an explicit, non-checkpointed FK-offset authority.

        ``SmplxLiteJ24.J_template`` and the precomputed SMPL-H training asset
        can differ by floating-point reconstruction even when they originate
        from the same rest pose.  Formal inference must not let optional body
        model availability choose which representation is fed to the network.
        """
        offsets = torch.as_tensor(bone_offsets).detach().clone().float()
        if tuple(offsets.shape) != (22, 3):
            raise ValueError(
                'bone-offset override must have shape (22,3), got '
                f'{tuple(offsets.shape)}'
            )
        if not torch.isfinite(offsets).all():
            raise ValueError('bone-offset override contains non-finite values')
        first_parameter = next(self.parameters(), None)
        target_device = (
            first_parameter.device
            if first_parameter is not None
            else offsets.device
        )
        self._bone_offsets_override = offsets.to(target_device)

    def get_bone_offsets(self) -> Tensor:
        """Get bone offsets for FK/IK.

        Attempts to compute from body model first; falls back to pre-computed
        file at ``data/hymotion_m2m_data/bone_offsets_22.pt``.

        Returns:
            bone_offsets: (22, 3) tensor of bone offsets.
        """
        from motius.motion.skeleton.names import SMPL22_PARENTS

        # Formal inference can pin the training-representation authority.  The
        # override must win regardless of whether the optional body model is
        # present on a particular runtime host.
        if self._bone_offsets_override is not None:
            return self._bone_offsets_override

        # Try computing from body model
        if self.body_model is not None:
            try:
                J_template = self.body_model.J_template[:22].clone()
                offsets = torch.zeros(22, 3, device=J_template.device, dtype=J_template.dtype)
                offsets[0] = J_template[0]
                for j in range(1, 22):
                    parent = SMPL22_PARENTS[j]
                    offsets[j] = J_template[j] - J_template[parent]
                return offsets
            except Exception:
                pass

        # Fallback: load pre-computed file
        fallback_path = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))),
            'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
        )
        if osp.isfile(fallback_path):
            offsets = torch.load(fallback_path, map_location='cpu')
            return offsets.to(_get_module_device(self))

        raise RuntimeError(
            'Cannot compute bone offsets: body model unavailable and '
            f'fallback file not found at {fallback_path}. '
            'Run `python tools/precompute_bone_offsets.py` first.'
        )
