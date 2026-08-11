# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MotionCLIP Model - Combined text-motion contrastive learning model."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import init

from transformers import PreTrainedModel, logging
from transformers.modeling_outputs import BaseModelOutputWithPooling

try:
    from transformers.file_utils import ModelOutput
except ImportError:
    from transformers.utils import ModelOutput

from .configuration_motionclip import (
    MotionCLIPConfig,
    MotionCLIPTextConfig,
    MotionCLIPMotionConfig,
)

# Import base components
from .modeling_motionclip_base import (
    MotionCLIPAttention,
    MotionCLIPMLP,
    MotionCLIPEncoderLayer,
    MotionCLIPEncoder,
    clip_loss,
    contrastive_loss,
    gather_with_grad,
    get_vector_norm,
)

# Import text and motion models
from .modeling_motionclip_text import (
    MotionCLIPTextModel,
    MotionCLIPTextModelWithProjection,
    MotionCLIPTextModelOutput,
    MotionCLIPTextEmbeddings,
    MotionCLIPTextTransformer,
)
from .modeling_motionclip_motion import (
    MotionCLIPMotionModel,
    MotionCLIPMotionModelWithProjection,
    MotionCLIPMotionModelOutput,
    MotionCLIPMotionEmbeddings,
    MotionCLIPMotionTransformer,
)

logger = logging.get_logger(__name__)


@dataclass
class MotionCLIPOutput(ModelOutput):
    """
    Output class for MotionCLIP model.

    Args:
        loss: Contrastive loss for motion-text similarity.
        logits_per_motion: The scaled dot product scores between motion_embeds and text_embeds.
        logits_per_text: The scaled dot product scores between text_embeds and motion_embeds.
        text_embeds: The text embeddings obtained by applying the projection layer.
        motion_embeds: The motion embeddings obtained by applying the projection layer.
        text_model_output: The output of the MotionCLIPTextModel.
        motion_model_output: The output of the MotionCLIPMotionModel.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_motion: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    motion_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    motion_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "motion_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )


class MotionCLIPPreTrainedModel(PreTrainedModel):
    """Base class for all MotionCLIP models."""

    config_class = MotionCLIPConfig
    base_model_prefix = "motion_clip"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MotionCLIPTextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=factor * 0.02)
            init.normal_(module.position_embedding.weight, mean=0.0, std=factor * 0.02)
        elif isinstance(module, MotionCLIPMotionEmbeddings):
            factor = self.config.initializer_factor
            init.normal_(
                module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor
            )
            init.normal_(
                module.motion_projection.weight,
                std=module.config.initializer_range * factor,
            )
            init.normal_(
                module.position_embedding.weight,
                std=module.config.initializer_range * factor,
            )
        elif isinstance(module, MotionCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.embed_dim**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MotionCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MotionCLIPModel):
            init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            init.normal_(
                module.motion_projection.weight,
                std=module.motion_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MotionCLIPMotionModelWithProjection):
            init.normal_(
                module.motion_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MotionCLIPTextModelWithProjection):
            init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            init.zeros_(module.bias)


class MotionCLIPModel(MotionCLIPPreTrainedModel):
    """
    MotionCLIP Model for contrastive learning between motion sequences and text descriptions.

    This model combines a motion encoder and a text encoder to learn aligned representations
    in a shared embedding space, similar to CLIP but for motion-text pairs.
    """

    config_class = MotionCLIPConfig
    _no_split_modules = [
        "MotionCLIPTextEmbeddings",
        "MotionCLIPEncoderLayer",
        "MotionCLIPMotionEmbeddings",
    ]

    def __init__(self, config: MotionCLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, MotionCLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type MotionCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.motion_config, MotionCLIPMotionConfig):
            raise TypeError(
                "config.motion_config is expected to be of type MotionCLIPMotionConfig but is of type"
                f" {type(config.motion_config)}."
            )

        text_config = config.text_config
        motion_config = config.motion_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.motion_embed_dim = motion_config.hidden_size

        text_model = MotionCLIPTextModel._from_config(text_config)
        self.text_model = text_model.text_model

        motion_model = MotionCLIPMotionModel._from_config(motion_config)
        self.motion_model = motion_model.motion_model

        self.motion_projection = nn.Linear(
            self.motion_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        # Use 1D tensor instead of scalar for FSDP compatibility
        self.logit_scale = nn.Parameter(
            torch.tensor([self.config.logit_scale_init_value])
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Get text features from the text encoder.

        Args:
            input_ids: Text input token ids
            attention_mask: Attention mask for text
            position_ids: Position ids for text

        Returns:
            text_features: The text embeddings after projection
        """
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_motion_features(
        self,
        motion_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Get motion features from the motion encoder.

        Args:
            motion_values: Motion sequence tensor of shape (batch_size, seq_len, motion_dim)
            attention_mask: Attention mask for motion

        Returns:
            motion_features: The motion embeddings after projection
        """
        motion_outputs: BaseModelOutputWithPooling = self.motion_model(
            motion_values=motion_values,
            attention_mask=attention_mask,
        )
        pooled_output = motion_outputs.pooler_output
        motion_features = self.motion_projection(pooled_output)

        return motion_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        motion_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        motion_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        **kwargs,
    ) -> MotionCLIPOutput:
        """
        Forward pass for MotionCLIP model.

        Args:
            input_ids: Text input token ids of shape (batch_size, seq_len)
            motion_values: Motion sequence tensor of shape (batch_size, motion_seq_len, motion_dim)
            attention_mask: Attention mask for text of shape (batch_size, seq_len)
            motion_attention_mask: Attention mask for motion of shape (batch_size, motion_seq_len)
            position_ids: Position ids for text
            return_loss: Whether or not to return the contrastive loss.

        Returns:
            MotionCLIPOutput with loss, logits, and embeddings
        """
        motion_outputs: BaseModelOutputWithPooling = self.motion_model(
            motion_values=motion_values,
            attention_mask=motion_attention_mask,
            **kwargs,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        motion_embeds = motion_outputs.pooler_output
        motion_embeds = self.motion_projection(motion_embeds)

        text_embeds = text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        motion_embeds = motion_embeds / get_vector_norm(motion_embeds)
        text_embeds = text_embeds / get_vector_norm(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100.0)

        loss = None
        if return_loss and self.training:
            # During training, gather embeddings across GPUs for richer negatives
            all_motion_embeds = gather_with_grad(motion_embeds)
            all_text_embeds = gather_with_grad(text_embeds)

            global_logits = torch.matmul(
                all_text_embeds, all_motion_embeds.t()
            ) * logit_scale
            loss = clip_loss(global_logits)
            if dist.is_initialized() and dist.get_world_size() > 1:
                loss = loss * dist.get_world_size()

        # Local logits for output (consistent with local embed batch size)
        logits_per_text = torch.matmul(
            text_embeds, motion_embeds.t()
        ) * logit_scale
        logits_per_motion = logits_per_text.t()

        if return_loss and not self.training:
            loss = clip_loss(logits_per_text)

        return MotionCLIPOutput(
            loss=loss,
            logits_per_motion=logits_per_motion,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            motion_embeds=motion_embeds,
            text_model_output=text_outputs,
            motion_model_output=motion_outputs,
        )


__all__ = [
    # Main model
    "MotionCLIPModel",
    "MotionCLIPPreTrainedModel",
    "MotionCLIPOutput",
    # Text models (re-exported)
    "MotionCLIPTextModel",
    "MotionCLIPTextModelWithProjection",
    "MotionCLIPTextModelOutput",
    # Motion models (re-exported)
    "MotionCLIPMotionModel",
    "MotionCLIPMotionModelWithProjection",
    "MotionCLIPMotionModelOutput",
    # Base components (re-exported for convenience)
    "MotionCLIPAttention",
    "MotionCLIPMLP",
    "MotionCLIPEncoderLayer",
    "MotionCLIPEncoder",
]
