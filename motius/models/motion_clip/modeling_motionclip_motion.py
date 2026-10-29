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
"""MotionCLIP Motion Model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import init

from transformers import PreTrainedModel, logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)

try:
    from transformers.file_utils import ModelOutput
except ImportError:
    from transformers.utils import ModelOutput

from .configuration_motionclip import MotionCLIPMotionConfig

# Import shared components from base module
from .modeling_motionclip_base import (
    MotionCLIPAttention,
    MotionCLIPMLP,
    MotionCLIPEncoderLayer,
    MotionCLIPEncoder,
)

logger = logging.get_logger(__name__)


@dataclass
class MotionCLIPMotionModelOutput(ModelOutput):
    """
    Base class for motion model's outputs that also contains motion embeddings of the pooling of the last hidden states.

    Args:
        motion_embeds: The motion embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state: Last hidden state of the model.
        hidden_states: Tuple of hidden states at each layer.
        attentions: Tuple of attention weights at each layer.
    """

    motion_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MotionCLIPMotionEmbeddings(nn.Module):
    """
    Embeddings for 1D motion sequences (b, t, c) -> (b, t+1, hidden_size).
    Includes a CLS token prepended to the sequence.
    """

    def __init__(self, config: MotionCLIPMotionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.motion_dim = config.motion_dim
        self.max_position_embeddings = config.max_position_embeddings

        # CLS token embedding
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # Linear projection from motion_dim to hidden_size
        self.motion_projection = nn.Linear(self.motion_dim, self.embed_dim, bias=False)

        # Position embeddings: +1 for CLS token
        self.num_positions = self.max_position_embeddings + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        motion_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            motion_values: Motion sequence tensor of shape (batch_size, seq_len, motion_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len + 1, hidden_size)
        """
        batch_size, seq_len, _ = motion_values.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Motion sequence length ({seq_len}) exceeds max_position_embeddings ({self.max_position_embeddings})."
            )

        # Project motion features to hidden dimension
        target_dtype = self.motion_projection.weight.dtype
        motion_embeds = self.motion_projection(motion_values.to(dtype=target_dtype))

        # Prepend CLS token
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, motion_embeds], dim=1)

        # Add position embeddings (seq_len + 1 for CLS token)
        position_ids = self.position_ids[:, : seq_len + 1]
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


class MotionCLIPMotionTransformer(nn.Module):
    def __init__(self, config: MotionCLIPMotionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = MotionCLIPMotionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = MotionCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        motion_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Args:
            motion_values: Motion sequence tensor of shape (batch_size, seq_len, motion_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
        """
        if motion_values is None:
            raise ValueError("You have to specify motion_values")

        hidden_states = self.embeddings(motion_values, attention_mask=attention_mask)
        hidden_states = self.pre_layernorm(hidden_states)

        # Create attention mask for transformer if provided
        # Need to account for CLS token (prepended)
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            # Add 1 for CLS token (always attended)
            cls_mask = torch.ones(
                batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype
            )
            extended_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
            # Convert to attention mask format (0 for attend, -inf for ignore)
            extended_attention_mask = (
                1.0 - extended_attention_mask[:, None, None, :]
            ) * torch.finfo(hidden_states.dtype).min
        else:
            extended_attention_mask = None

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=extended_attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        # Use CLS token (first position) as pooled output
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class MotionCLIPMotionPreTrainedModel(PreTrainedModel):
    """PreTrainedModel base class for MotionCLIP motion models."""

    config_class = MotionCLIPMotionConfig
    base_model_prefix = "motion_clip_motion"
    main_input_name = "motion_values"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MotionCLIPMotionEmbeddings):
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
        elif isinstance(module, MotionCLIPMotionModelWithProjection):
            init.normal_(
                module.motion_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            init.zeros_(module.bias)


class MotionCLIPMotionModel(MotionCLIPMotionPreTrainedModel):
    """The motion model from MotionCLIP without any head or projection on top."""

    config_class = MotionCLIPMotionConfig
    main_input_name = "motion_values"
    _no_split_modules = ["MotionCLIPEncoderLayer"]

    def __init__(self, config: MotionCLIPMotionConfig):
        super().__init__(config)
        self.motion_model = MotionCLIPMotionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.motion_model.embeddings.motion_projection

    def forward(
        self,
        motion_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Args:
            motion_values: Motion sequence tensor of shape (batch_size, seq_len, motion_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and pooler_output
        """
        return self.motion_model(
            motion_values=motion_values,
            attention_mask=attention_mask,
            **kwargs,
        )


class MotionCLIPMotionModelWithProjection(MotionCLIPMotionPreTrainedModel):
    """MotionCLIP Motion Model with a projection layer on top."""

    config_class = MotionCLIPMotionConfig
    main_input_name = "motion_values"

    def __init__(self, config: MotionCLIPMotionConfig):
        super().__init__(config)

        motion_model = MotionCLIPMotionModel._from_config(config)
        self.motion_model = motion_model.motion_model

        self.motion_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.motion_model.embeddings.motion_projection

    def forward(
        self,
        motion_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MotionCLIPMotionModelOutput:
        """
        Args:
            motion_values: Motion sequence tensor of shape (batch_size, seq_len, motion_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            MotionCLIPMotionModelOutput with motion_embeds and last_hidden_state
        """
        motion_outputs: BaseModelOutputWithPooling = self.motion_model(
            motion_values=motion_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled_output = motion_outputs.pooler_output
        motion_embeds = self.motion_projection(pooled_output)

        return MotionCLIPMotionModelOutput(
            motion_embeds=motion_embeds,
            last_hidden_state=motion_outputs.last_hidden_state,
        )


__all__ = [
    "MotionCLIPMotionModel",
    "MotionCLIPMotionModelWithProjection",
    "MotionCLIPMotionModelOutput",
    "MotionCLIPMotionEmbeddings",
    "MotionCLIPMotionTransformer",
    "MotionCLIPMotionPreTrainedModel",
]
