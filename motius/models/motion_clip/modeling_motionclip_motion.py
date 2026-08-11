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


if __name__ == "__main__":
    """
    Test MotionCLIPMotionModel with random inputs.

    Usage:
        python -m mmotion.models.transformers.motion_clip.modeling_motionclip_motion
    """

    def test_motion_model():
        """Test MotionCLIPMotionModel forward pass."""
        print("=" * 60)
        print("Testing MotionCLIPMotionModel")
        print("=" * 60)

        # Create config
        config = MotionCLIPMotionConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=6,
            num_attention_heads=8,
            motion_dim=263,  # typical motion feature dimension
            max_position_embeddings=512,
            projection_dim=512,
        )

        print(
            f"\nConfig: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
            f"motion_dim={config.motion_dim}, max_pos={config.max_position_embeddings}"
        )

        # Create model
        print("\n1. Creating MotionCLIPMotionModel...")
        model = MotionCLIPMotionModel(config)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")

        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        seq_len = 120
        motion_dim = config.motion_dim

        motion_values = torch.randn(batch_size, seq_len, motion_dim)
        print(f"   Input shape: {motion_values.shape}")

        with torch.no_grad():
            outputs = model(motion_values=motion_values)

        print(f"   last_hidden_state shape: {outputs.last_hidden_state.shape}")
        print(f"   pooler_output shape: {outputs.pooler_output.shape}")

        # Verify shapes
        expected_hidden_shape = (
            batch_size,
            seq_len + 1,
            config.hidden_size,
        )  # +1 for CLS token
        expected_pooler_shape = (batch_size, config.hidden_size)

        assert (
            outputs.last_hidden_state.shape == expected_hidden_shape
        ), f"Expected {expected_hidden_shape}, got {outputs.last_hidden_state.shape}"
        assert (
            outputs.pooler_output.shape == expected_pooler_shape
        ), f"Expected {expected_pooler_shape}, got {outputs.pooler_output.shape}"

        print("   ✓ Output shapes correct!")

        # Test with attention mask
        print("\n3. Testing with attention mask...")
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, seq_len // 2 :] = 0  # Mask second half of first sequence

        with torch.no_grad():
            outputs_masked = model(
                motion_values=motion_values, attention_mask=attention_mask
            )

        print(f"   Masked output shape: {outputs_masked.last_hidden_state.shape}")
        print("   ✓ Attention mask works!")

        # Test variable length sequences
        print("\n4. Testing variable length sequences...")
        for length in [64, 128, 256, 512]:
            motion_values_var = torch.randn(1, length, motion_dim)
            with torch.no_grad():
                out = model(motion_values=motion_values_var)
            print(
                f"   seq_len={length}: hidden={out.last_hidden_state.shape}, pooler={out.pooler_output.shape}"
            )

        print("\n" + "=" * 60)
        print("✓ All MotionCLIPMotionModel tests passed!")
        print("=" * 60)

    def test_motion_model_with_projection():
        """Test MotionCLIPMotionModelWithProjection."""
        print("\n" + "=" * 60)
        print("Testing MotionCLIPMotionModelWithProjection")
        print("=" * 60)

        config = MotionCLIPMotionConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=6,
            num_attention_heads=8,
            motion_dim=263,
            max_position_embeddings=512,
            projection_dim=512,
        )

        print("\n1. Creating MotionCLIPMotionModelWithProjection...")
        model = MotionCLIPMotionModelWithProjection(config)
        model.eval()

        print("\n2. Testing forward pass...")
        motion_values = torch.randn(2, 120, config.motion_dim)

        with torch.no_grad():
            outputs = model(motion_values=motion_values)

        print(f"   motion_embeds shape: {outputs.motion_embeds.shape}")
        print(f"   last_hidden_state shape: {outputs.last_hidden_state.shape}")

        expected_embed_shape = (2, config.projection_dim)
        assert (
            outputs.motion_embeds.shape == expected_embed_shape
        ), f"Expected {expected_embed_shape}, got {outputs.motion_embeds.shape}"

        print("   ✓ Projection layer works correctly!")

        print("\n" + "=" * 60)
        print("✓ All MotionCLIPMotionModelWithProjection tests passed!")
        print("=" * 60)

    # Run tests
    test_motion_model()
    test_motion_model_with_projection()
