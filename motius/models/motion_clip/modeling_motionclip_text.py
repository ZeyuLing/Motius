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
"""MotionCLIP Text Model."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import init

from transformers import PreTrainedModel, logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
# `_create_4d_causal_attention_mask` lived in `transformers.models.clip.modeling_clip`
# in older transformers, then moved to `transformers.modeling_attn_mask_utils`, and was
# removed entirely in newer releases. Import it robustly with a local fallback so the
# MotionCLIP evaluator works across transformers versions.
try:
    from transformers.models.clip.modeling_clip import _create_4d_causal_attention_mask
except ImportError:
    try:
        from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    except ImportError:
        def _create_4d_causal_attention_mask(
            input_shape, dtype, device, past_key_values_length=0, sliding_window=None
        ):
            bsz, tgt_len = input_shape
            src_len = tgt_len + past_key_values_length
            min_val = torch.finfo(dtype).min
            mask = torch.full((tgt_len, src_len), min_val, dtype=dtype, device=device)
            cond = torch.arange(src_len, device=device) <= (
                torch.arange(tgt_len, device=device).view(-1, 1) + past_key_values_length
            )
            mask = mask.masked_fill(cond, 0.0)
            if sliding_window is not None:
                rows = torch.arange(tgt_len, device=device).view(-1, 1) + past_key_values_length
                cols = torch.arange(src_len, device=device).view(1, -1)
                window = (rows - cols) >= sliding_window
                mask = mask.masked_fill(window, min_val)
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)

try:
    from transformers.file_utils import ModelOutput
except ImportError:
    from transformers.utils import ModelOutput

from .configuration_motionclip import MotionCLIPTextConfig

# Import shared components from base module
from .modeling_motionclip_base import (
    MotionCLIPAttention,
    MotionCLIPMLP,
    MotionCLIPEncoderLayer,
    MotionCLIPEncoder,
)

logger = logging.get_logger(__name__)


@dataclass
class MotionCLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds: The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state: Last hidden state of the model.
        hidden_states: Tuple of hidden states at each layer.
        attentions: Tuple of attention weights at each layer.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MotionCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: MotionCLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class MotionCLIPTextTransformer(nn.Module):
    def __init__(self, config: MotionCLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = MotionCLIPTextEmbeddings(config)
        self.encoder = MotionCLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # Build causal attention mask
        bsz, seq_len = input_shape
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # Expand attention_mask if provided
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = (
                attention_mask[:, None, None, :]
                .expand(bsz, 1, seq_len, seq_len)
                .to(hidden_states.dtype)
            )
            inverted_mask = 1.0 - expanded_attn_mask
            expanded_attn_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(hidden_states.dtype).min
            )
            causal_attention_mask = causal_attention_mask + expanded_attn_mask

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=causal_attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                    dim=-1
                ),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of extra new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                    == self.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class MotionCLIPTextPreTrainedModel(PreTrainedModel):
    """PreTrainedModel base class for MotionCLIP text models."""

    config_class = MotionCLIPTextConfig
    base_model_prefix = "motion_clip_text"
    supports_gradient_checkpointing = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MotionCLIPTextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=factor * 0.02)
            init.normal_(module.position_embedding.weight, mean=0.0, std=factor * 0.02)
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


class MotionCLIPTextModel(MotionCLIPTextPreTrainedModel):
    """The text model from MotionCLIP without any head or projection on top."""

    config_class = MotionCLIPTextConfig

    _no_split_modules = ["MotionCLIPTextEmbeddings", "MotionCLIPEncoderLayer"]

    def __init__(self, config: MotionCLIPTextConfig):
        super().__init__(config)
        self.text_model = MotionCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            position_ids: Position ids

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and pooler_output
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class MotionCLIPTextModelWithProjection(MotionCLIPTextPreTrainedModel):
    """MotionCLIP Text Model with a projection layer on top."""

    config_class = MotionCLIPTextConfig

    _no_split_modules = ["MotionCLIPTextEmbeddings", "MotionCLIPEncoderLayer"]

    def __init__(self, config: MotionCLIPTextConfig):
        super().__init__(config)

        text_model = MotionCLIPTextModel._from_config(config)
        self.text_model = text_model.text_model

        self.text_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MotionCLIPTextModelOutput:
        """
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            position_ids: Position ids

        Returns:
            MotionCLIPTextModelOutput with text_embeds and last_hidden_state
        """
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        pooled_output = text_outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)

        return MotionCLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
        )


__all__ = [
    "MotionCLIPTextModel",
    "MotionCLIPTextModelWithProjection",
    "MotionCLIPTextModelOutput",
    "MotionCLIPTextEmbeddings",
    "MotionCLIPTextTransformer",
    "MotionCLIPTextPreTrainedModel",
]
