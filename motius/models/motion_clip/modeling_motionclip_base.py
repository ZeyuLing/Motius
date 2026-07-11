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
"""MotionCLIP Base Components - Shared modules for text and motion encoders."""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers import logging

from .configuration_motionclip import (
    MotionCLIPConfig,
    MotionCLIPTextConfig,
    MotionCLIPMotionConfig,
)

logger = logging.get_logger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Standard eager attention implementation."""
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MotionCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[MotionCLIPMotionConfig, MotionCLIPTextConfig]):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, -1, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MotionCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MotionCLIPEncoderLayer(nn.Module):
    def __init__(self, config: Union[MotionCLIPMotionConfig, MotionCLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MotionCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MotionCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MotionCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MotionCLIPEncoderLayer`].

    Args:
        config: MotionCLIPConfig
    """

    def __init__(self, config: MotionCLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MotionCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    **kwargs,
                )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


# Contrastive loss functions
def contrastive_loss(logits: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Contrastive loss. offset shifts the label for the global-batch case."""
    labels = torch.arange(logits.shape[0], device=logits.device) + offset
    return nn.functional.cross_entropy(logits, labels)


class GatherWithGrad(torch.autograd.Function):
    """all_gather that preserves gradients on the local shard."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ctx.rank = dist.get_rank()
        ctx.world_size = world_size
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk_size = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * chunk_size : (ctx.rank + 1) * chunk_size]


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return GatherWithGrad.apply(tensor)


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """Symmetric contrastive loss for CLIP-style training."""
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


__all__ = [
    "MotionCLIPAttention",
    "MotionCLIPMLP",
    "MotionCLIPEncoderLayer",
    "MotionCLIPEncoder",
    "eager_attention_forward",
    "contrastive_loss",
    "clip_loss",
    "gather_with_grad",
    "get_vector_norm",
]
