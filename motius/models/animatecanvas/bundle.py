"""AnimateCanvas Bundle: motion-to-motion editing via flow matching.

This bundle holds a HunyuanMotionMMDiT transformer and provides atomic
forward functions shared between Trainer and Pipeline:

  - predict_flow()          -- single forward through the transformer
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
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_ARTIFACT_FORMAT = 'motius-animatecanvas-v1'
_CONFIG_NAME = 'animatecanvas_config.json'
_LEGACY_CONFIG_NAME = 'motioncanvas_config.json'


def _artifact_config_path(directory: Path) -> Path:
    """Prefer the canonical name while reading previously released artifacts."""
    canonical = directory / _CONFIG_NAME
    return canonical if canonical.exists() else directory / _LEGACY_CONFIG_NAME


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
class AnimateCanvasBundle(ModelBundle):
    """ModelBundle for AnimateCanvas motion-to-motion editing.

    The only sub-module managed via ``_build_modules`` is
    ``motion_transformer``  (the HunyuanMotionMMDiT).  Auxiliary objects
    (null embeddings and mean/std buffers) are created
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
        import motius.models.animatecanvas.network  # noqa: F401

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
        if self._text_encoder_cfg is not None:
            from motius.configuration.hymotion_text import MOTIONCANVAS_PROMPT_TEMPLATE
            self._text_encoder_cfg.setdefault("prompt_template", MOTIONCANVAS_PROMPT_TEMPLATE)
            self._text_encoder_cfg.setdefault("sentence_padding", "max_length")
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


        # ---- SMPL body model (optional for FK losses / decode) ----
        self._body_model_path = body_model_path
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
            from motius.motion.representation.hymotion_runtime.codecs import MotionCanvasCodec
            MotionCanvasCodec(self).set_bone_offsets_override(offsets)

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
            elif 'torch_dtype' in cfg:
                cfg['torch_dtype'] = _resolve_dtype(cfg['torch_dtype'])
            else:
                cfg['torch_dtype'] = torch.float32
            self._text_encoder = HYTextModel(**cfg).eval()
        vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
        return {
            'text_vec_raw': vtxt.to(device),
            'text_ctxt_raw': ctxt.to(device),
            'text_ctxt_raw_length': ctxt_len.to(device),
        }

    @torch.no_grad()
    def encode_task_instruction(self, task_instructions: dict) -> Dict[str, Tensor]:
        """Encode task instructions to vtxt-like embeddings using CLIP.

        Task instructions (e.g., 'complete motion from sparse random cells') are
        CLIP-encoded to the same space as caption sentence embeddings (768-dim),
        then projected to 1024-dim via vtxt_encoder to match the adapter signal.

        Args:
            task_instructions: Tensor features prepared by the external text processor.

        Returns:
            Dict with key 'task_emb' containing (B, 1, 1024) embeddings ready to add to adapter.
        """
        device = _get_module_device(self)
        task_vtxt = self.encode_text(task_instructions)['text_vec_raw']

        # Project to 1024-dim to match vtxt_encoder output
        network = self.motion_transformer
        while hasattr(network, 'module'):
            network = network.module
        task_emb = network.vtxt_encoder(task_vtxt.float().to(device))

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
        from motius.configuration.hymotion_text import export_text_networks
        return export_text_networks(self, save_dir, cfg, fallback_dtype=torch.float32)

    def save_pretrained(
        self,
        save_directory: str,
        *,
        safe_serialization: bool = True,
        include_text_encoder: bool = True,
        variant: Optional[str] = None,
    ):
        """Write a complete AnimateCanvas artifact loadable by ``Pipeline``."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        from motius.configuration.hymotion_text import copy_hymotion_notices
        copy_hymotion_notices(save_dir)
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
        network = self.motion_transformer
        while hasattr(network, 'module'):
            network = network.module
        state = {
            f'motion_transformer.{key}': value.detach().cpu().contiguous()
            for key, value in network.state_dict().items()
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
        from motius.motion.representation.hymotion_runtime.codecs import MotionCanvasCodec
        torch.save(MotionCanvasCodec(self).get_bone_offsets().detach().cpu(), save_dir / 'bone_offsets_22.pt')

        meta = {
            'model_type': 'animatecanvas',
            'format': _ARTIFACT_FORMAT,
            'variant': variant,
            'config': _jsonable(config),
            'pipeline_class': (
                'motius.pipelines.animatecanvas.animatecanvas_pipeline.'
                'AnimateCanvasPipeline'
            ),
            'bundle_class': 'motius.models.animatecanvas.bundle.AnimateCanvasBundle',
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
            '_class_name': 'AnimateCanvasPipeline',
            '_library_name': 'motius',
            'model_type': 'animatecanvas',
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
                'inference': 'AnimateCanvasPipeline.infer_m2m',
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
                cfg_path = _artifact_config_path(cfg_path)
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        config = cls._to_plain_dict(cfg)
        if config.get('model_type') in {'animatecanvas', 'motioncanvas'} and 'config' in config:
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
        if not _artifact_config_path(path).exists():
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
        meta = json.loads(_artifact_config_path(path).read_text(encoding='utf-8'))
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

        Returns:
            Model prediction, shape (B, L, D_motion).
        """
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
        from motius.configuration.animatecanvas_checkpoint import (
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
        from motius.configuration.animatecanvas_checkpoint import _apply_freeze_strategy

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
            from motius.configuration.animatecanvas_checkpoint import _apply_freeze_strategy
            _apply_freeze_strategy(self, strategy)
        return self
