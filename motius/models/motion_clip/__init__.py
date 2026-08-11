# coding=utf-8
"""MotionCLIP - Contrastive Learning for Motion and Text.

Originally implemented in versatilemotion/mmotion/models/transformers/motion_clip.
Ported to motius with:
  * mmotion-specific Registry / BaseTrainerModel deps stripped (model code is
    pure transformers + torch).
  * HF_MODELS registration for use in configs.
  * MotionCLIPBundle (ModelBundle pattern) added in bundle.py.
"""

from motius.models.motion_clip.configuration_motionclip import (
    MotionCLIPConfig,
    MotionCLIPMotionConfig,
    MotionCLIPTextConfig,
)
from motius.models.motion_clip.modeling_motionclip import (
    MotionCLIPModel,
    MotionCLIPOutput,
    MotionCLIPPreTrainedModel,
)
from motius.models.motion_clip.modeling_motionclip_base import (
    MotionCLIPAttention,
    MotionCLIPEncoder,
    MotionCLIPEncoderLayer,
    MotionCLIPMLP,
    clip_loss,
    contrastive_loss,
    gather_with_grad,
    get_vector_norm,
)
from motius.models.motion_clip.modeling_motionclip_motion import (
    MotionCLIPMotionEmbeddings,
    MotionCLIPMotionModel,
    MotionCLIPMotionModelOutput,
    MotionCLIPMotionModelWithProjection,
    MotionCLIPMotionPreTrainedModel,
    MotionCLIPMotionTransformer,
)
from motius.models.motion_clip.modeling_motionclip_text import (
    MotionCLIPTextEmbeddings,
    MotionCLIPTextModel,
    MotionCLIPTextModelOutput,
    MotionCLIPTextModelWithProjection,
    MotionCLIPTextPreTrainedModel,
    MotionCLIPTextTransformer,
)
from motius.registry import HF_MODELS

# Register all MotionCLIP classes in HF_MODELS so they can be referenced from
# Config files via {'type': 'MotionCLIPModel', ...}.
for _cls in (
    MotionCLIPModel,
    MotionCLIPMotionModel,
    MotionCLIPMotionModelWithProjection,
    MotionCLIPTextModel,
    MotionCLIPTextModelWithProjection,
    MotionCLIPConfig,
    MotionCLIPTextConfig,
    MotionCLIPMotionConfig,
):
    if _cls.__name__ not in HF_MODELS._module_dict:
        HF_MODELS.register_module(name=_cls.__name__, module=_cls)

# Bundle is registered via decorator inside bundle.py
from motius.models.motion_clip.bundle import MotionCLIPBundle  # noqa: E402

__all__ = [
    'MotionCLIPConfig',
    'MotionCLIPMotionConfig',
    'MotionCLIPTextConfig',
    'MotionCLIPModel',
    'MotionCLIPOutput',
    'MotionCLIPPreTrainedModel',
    'MotionCLIPMotionModel',
    'MotionCLIPMotionModelWithProjection',
    'MotionCLIPMotionModelOutput',
    'MotionCLIPMotionEmbeddings',
    'MotionCLIPMotionTransformer',
    'MotionCLIPMotionPreTrainedModel',
    'MotionCLIPTextModel',
    'MotionCLIPTextModelWithProjection',
    'MotionCLIPTextModelOutput',
    'MotionCLIPTextEmbeddings',
    'MotionCLIPTextTransformer',
    'MotionCLIPTextPreTrainedModel',
    'MotionCLIPAttention',
    'MotionCLIPMLP',
    'MotionCLIPEncoderLayer',
    'MotionCLIPEncoder',
    'MotionCLIPBundle',
    'clip_loss',
    'contrastive_loss',
    'gather_with_grad',
    'get_vector_norm',
]
