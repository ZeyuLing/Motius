# coding=utf-8
"""MotionCLIPBundle.

Wraps :class:`MotionCLIPModel` with:
  * a CLIP tokenizer for text input,
  * a frozen :class:`SMPLPoseProcessor` for motion normalization (mean/std).

The bundle exposes atomic forward APIs shared between the trainer and the
inference pipeline:

  - :meth:`tokenize` (text -> input_ids/attention_mask)
  - :meth:`encode_text` (text features after projection)
  - :meth:`encode_motion` (motion features after projection)
  - :meth:`forward` (full contrastive pass with CLIP loss)

It provides Motius artifact I/O around ``bundle_config.json`` plus
``motionclip_model.safetensors`` for inference and evaluation.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.models.motion_clip.configuration_motionclip import (
    MotionCLIPConfig,
    MotionCLIPMotionConfig,
    MotionCLIPTextConfig,
)
from motius.models.motion_clip.modeling_motionclip import MotionCLIPModel
from motius.registry import MODEL_BUNDLES


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


def _load_state_dict(path: Path) -> Dict[str, Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu")


def _build_tokenizer(cfg: Dict[str, Any]):
    cfg = deepcopy(cfg)
    cfg_type = cfg.pop('type', 'CLIPTokenizer')
    from_pretrained = cfg.pop('from_pretrained', None)
    if cfg_type == 'CLIPTokenizer':
        from transformers import CLIPTokenizer as _Cls
    elif cfg_type == 'AutoTokenizer':
        from transformers import AutoTokenizer as _Cls
    else:
        from transformers import AutoTokenizer as _Cls
    if from_pretrained is None:
        raise ValueError("Tokenizer config must include 'from_pretrained'.")
    return _Cls.from_pretrained(**from_pretrained)


@MODEL_BUNDLES.register_module()
class MotionCLIPBundle(ModelBundle):
    """ModelBundle for MotionCLIP contrastive evaluator.

    Args:
        text_config: dict for :class:`MotionCLIPTextConfig` (or ``MotionCLIPTextConfig`` instance).
        motion_config: dict for :class:`MotionCLIPMotionConfig`.
        projection_dim: shared embedding dim, default 512.
        logit_scale_init_value: CLIP temperature init, default 2.6592.
        tokenizer: dict ``{'type': 'CLIPTokenizer', 'from_pretrained': {...}}``.
        smpl_pose_processor: dict for :class:`SMPLPoseProcessor` (registered in HF_MODELS).
        clip_pretrained: optional path to ``openai/clip-vit-base-patch32`` (or local copy)
            used to initialize the text encoder and text projection at construction time.
            Set to ``None`` to skip CLIP-init (e.g. when restoring from a MotionCLIP checkpoint).
        freeze_text_encoder: whether to freeze the text encoder + text projection.
            Default ``True`` to preserve pretrained CLIP representations.
    """

    def __init__(
        self,
        text_config: Dict[str, Any],
        motion_config: Dict[str, Any],
        projection_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        tokenizer: Optional[Dict[str, Any]] = None,
        smpl_pose_processor: Optional[Dict[str, Any]] = None,
        clip_pretrained: Optional[str] = None,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()

        # Resolve sub-configs (allow PretrainedConfig instances or plain dicts).
        if isinstance(text_config, dict):
            text_config = MotionCLIPTextConfig(**text_config)
        if isinstance(motion_config, dict):
            motion_config = MotionCLIPMotionConfig(**motion_config)
        full_cfg = MotionCLIPConfig(
            text_config=text_config,
            motion_config=motion_config,
            projection_dim=projection_dim,
            logit_scale_init_value=logit_scale_init_value,
        )
        self.motionclip_model = MotionCLIPModel(full_cfg)

        # Bookkeeping for ModelBundle save/load.
        self._save_ckpt_modules.append('motionclip_model')
        self._trainable_modules.append('motionclip_model')
        self._module_build_configs['motionclip_model'] = {
            'text_config': text_config.to_dict(),
            'motion_config': motion_config.to_dict(),
            'projection_dim': projection_dim,
            'logit_scale_init_value': logit_scale_init_value,
        }

        # Tokenizer (not an nn.Module).
        if tokenizer is None:
            tokenizer = {
                'type': 'CLIPTokenizer',
                'from_pretrained': {
                    'pretrained_model_name_or_path': clip_pretrained
                    or 'openai/clip-vit-base-patch32',
                },
            }
        self.tokenizer = _build_tokenizer(tokenizer)
        self._extra_attributes['tokenizer'] = self.tokenizer

        # SMPL motion stats (mean/std). For evaluation we only need normalize.
        if smpl_pose_processor is not None:
            from motius.registry import HF_MODELS
            self.smpl_pose_processor = HF_MODELS.build(smpl_pose_processor)
            self.smpl_pose_processor.eval()
            for p in self.smpl_pose_processor.parameters():
                p.requires_grad = False
        else:
            self.smpl_pose_processor = None

        if clip_pretrained is not None:
            self.load_clip_text_weights(clip_pretrained)

        self.freeze_text_encoder = bool(freeze_text_encoder)
        if self.freeze_text_encoder:
            self._freeze_text_encoder()

    # ------------------------------------------------------------------
    # Module mode helpers
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.smpl_pose_processor is not None:
                self.smpl_pose_processor.eval()
                for p in self.smpl_pose_processor.parameters():
                    p.requires_grad = False
            if self.freeze_text_encoder:
                self._freeze_text_encoder()
        return self

    def _freeze_text_encoder(self):
        for p in self.motionclip_model.text_model.parameters():
            p.requires_grad = False
        for p in self.motionclip_model.text_projection.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # CLIP weight initialization (mirrors versatilemotion logic)
    # ------------------------------------------------------------------

    def load_clip_text_weights(self, clip_model_name_or_path: str):
        """Initialize text encoder + text projection from a pretrained CLIP.

        Handles position_embedding extension (CLIP has 77 positions, we may
        use 256). The first 77 positions are copied from CLIP, the rest keep
        their random init.
        """
        from transformers import CLIPModel

        try:
            clip_model = CLIPModel.from_pretrained(
                clip_model_name_or_path, use_safetensors=True
            )
        except (TypeError, OSError):
            clip_model = CLIPModel.from_pretrained(clip_model_name_or_path)
        clip_text_sd = clip_model.text_model.state_dict()
        clip_proj_sd = {
            'text_projection.weight': clip_model.text_projection.weight.data
        }

        target_sd = self.motionclip_model.state_dict()
        new_sd: Dict[str, Tensor] = {}
        pos_embed_key = 'text_model.embeddings.position_embedding.weight'

        for key, value in clip_text_sd.items():
            target_key = (
                f'text_model.{key}'
                if not key.startswith('text_model.')
                else key
            )
            if target_key not in target_sd:
                continue
            target_shape = target_sd[target_key].shape
            if value.shape == target_shape:
                new_sd[target_key] = value
            elif (
                target_key == pos_embed_key
                and value.shape[1] == target_shape[1]
            ):
                n_clip = value.shape[0]
                n_target = target_shape[0]
                new_w = target_sd[target_key].clone()
                copy_n = min(n_clip, n_target)
                new_w[:copy_n] = value[:copy_n]
                new_sd[target_key] = new_w

        for key, value in clip_proj_sd.items():
            if (
                key in target_sd
                and value.shape == target_sd[key].shape
            ):
                new_sd[key] = value

        self.motionclip_model.load_state_dict(new_sd, strict=False)

    # ------------------------------------------------------------------
    # Tokenization + encoding
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if max_length is None:
            max_length = self.motionclip_model.config.text_config.max_position_embeddings
        if isinstance(texts, str):
            texts = [texts]
        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        return encoding

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.motionclip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )

    def encode_motion(
        self,
        motion_values: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.motionclip_model.get_motion_features(
            motion_values=motion_values, attention_mask=attention_mask,
        )

    def normalize_motion(self, motion: Tensor) -> Tensor:
        if self.smpl_pose_processor is None:
            return motion
        return self.smpl_pose_processor.normalize(motion)

    # ------------------------------------------------------------------
    # Bundle-level forward (used by trainer)
    # ------------------------------------------------------------------

    def forward(
        self,
        motion: Tensor,
        captions: List[str],
        num_frames: List[int],
        return_loss: bool = True,
    ):
        """Full contrastive forward pass.

        Args:
            motion: (B, T_max, D) raw motion (will be normalized internally).
            captions: list of B caption strings.
            num_frames: list of B valid frame counts (T_i <= T_max).
            return_loss: whether to compute CLIP loss.

        Returns:
            ``MotionCLIPOutput`` from :class:`MotionCLIPModel.forward`.
        """
        device = motion.device

        # Normalize motion in eval mode (no grad through processor).
        if self.smpl_pose_processor is not None:
            with torch.no_grad():
                motion = self.smpl_pose_processor.normalize(motion)

        # Build motion attention mask (1 = attend, 0 = pad).
        max_len = max(int(nf) for nf in num_frames)
        motion = motion[:, :max_len].contiguous()
        B = motion.shape[0]
        motion_attn = torch.zeros(
            B, max_len, device=device, dtype=motion.dtype,
        )
        for i, nf in enumerate(num_frames):
            motion_attn[i, : int(nf)] = 1.0

        # Tokenize text.
        enc = self.tokenize(captions)
        input_ids = enc['input_ids'].to(device)
        text_attn = enc['attention_mask'].to(device)

        outputs = self.motionclip_model(
            input_ids=input_ids,
            motion_values=motion,
            attention_mask=text_attn,
            motion_attention_mask=motion_attn,
            return_loss=return_loss,
        )
        return outputs

    # ------------------------------------------------------------------
    # Motius artifact I/O
    # ------------------------------------------------------------------

    def config_dict(self) -> Dict[str, Any]:
        return {
            "text_config": self.motionclip_model.config.text_config.to_dict(),
            "motion_config": self.motionclip_model.config.motion_config.to_dict(),
            "projection_dim": self.motionclip_model.config.projection_dim,
            "logit_scale_init_value": float(self.motionclip_model.config.logit_scale_init_value),
        }

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """Export a Motius MotionCLIP artifact.

        Layout::

            <dir>/bundle_config.json
            <dir>/motionclip_model.safetensors

        Tokenizer files are intentionally not duplicated; by default the bundle
        reloads ``openai/clip-vit-base-patch32``.
        """
        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)
        (save_dir / "bundle_config.json").write_text(
            json.dumps(self.config_dict(), indent=2),
            encoding="utf-8",
        )

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(
                {k: v.detach().cpu().contiguous() for k, v in self.motionclip_model.state_dict().items()},
                str(save_dir / "motionclip_model.safetensors"),
            )
        else:
            torch.save(
                {k: v.detach().cpu() for k, v in self.motionclip_model.state_dict().items()},
                str(save_dir / "motionclip_model.pt"),
            )

        (save_dir / "README.md").write_text(
            "# MotionCLIP Evaluator\n\n"
            "Load with `MotionCLIPPipeline.from_pretrained(...)` from Motius.\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        tokenizer: Optional[Dict[str, Any]] = None,
        smpl_pose_processor: Optional[Dict[str, Any]] = None,
        clip_pretrained: Optional[str] = None,
        freeze_text_encoder: bool = True,
        strict: bool = True,
        **kwargs,
    ):
        path = Path(str(pretrained_model_name_or_path))
        path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if not path.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        cfg_path = path / "bundle_config.json"
        if not cfg_path.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.update(kwargs)
        cfg.setdefault("tokenizer", tokenizer)
        cfg.setdefault("smpl_pose_processor", smpl_pose_processor)
        cfg.setdefault("clip_pretrained", clip_pretrained)
        cfg.setdefault("freeze_text_encoder", freeze_text_encoder)

        bundle = cls.from_config(cfg)
        weights = path / "motionclip_model.safetensors"
        if not weights.exists():
            weights = path / "motionclip_model.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"MotionCLIP artifact at {path} is missing motionclip_model.safetensors"
            )
        state = _load_state_dict(weights)
        bundle.motionclip_model.load_state_dict(state, strict=strict)
        bundle.eval()
        return bundle
