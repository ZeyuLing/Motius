# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

"""
Wan Transformer for Motion Generation with Attention Mask Support.

This module implements a motion-adapted Wan Transformer architecture that extends
the original Wan video generation model for human motion synthesis. Key adaptations:

1. **2D Input Structure**: Processes motion data as [B, C, T, J] tensors where T is
   the number of frames and J is the number of joints (vs. 3D video [B, C, T, H, W]).

2. **Attention Masking**: Supports variable-length motion sequences via attention masks,
   enabling efficient batched training with sequences of different durations.

3. **Patch Embedding**: Uses 2D convolution to patchify motion along temporal dimension,
   with joint dimension typically kept intact (patch_size[-1] = 1).

Architecture Overview:
    Input Motion [B, C, T, J]
           ↓
    Patch Embedding (Conv2d) → [B, inner_dim, T//p_t, J//p_j]
           ↓
    Flatten & Transpose → [B, N, inner_dim] where N = (T//p_t) * (J//p_j)
           ↓
    Transformer Blocks (Self-Attn + Cross-Attn + FFN) × num_layers
           ↓
    Output Projection & Unpatchify → [B, C, T, J]

Example:
    >>> model = PrismTransformerMotionModel(
    ...     patch_size=(2, 1),
    ...     num_attention_heads=12,
    ...     attention_head_dim=128,
    ...     in_channels=16,
    ...     num_layers=30,
    ... )
    >>> motion = torch.randn(2, 16, 64, 24)  # [B, C, T, J]
    >>> timestep = torch.tensor([0, 1])
    >>> text_emb = torch.randn(2, 512, 4096)  # [B, N_ctx, text_dim]
    >>> output = model(motion, timestep, text_emb)  # [B, C, T, J]
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.utils import (
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel,
)
from diffusers.configuration_utils import register_to_config
from diffusers.models.normalization import FP32LayerNorm

from motius.registry import HF_MODELS
from einops import rearrange

from .block_with_mask import WanTransformerBlockWithMask
from .motion_rope import MotionWanRotaryPosEmbed
from .embedding import WanTimeTextEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@HF_MODELS.register_module(force=True)
class PrismTransformerMotionModel(WanTransformer3DModel):
    """
    A Transformer model adapted from Wan for motion generation with attention mask support.

    This model extends the Wan video generation architecture for human motion synthesis,
    processing motion sequences represented as [B, C, T, J] tensors. It supports:

    - Variable-length motion sequences via attention masking
    - Text-conditioned generation via cross-attention
    - Temporal patchification for computational efficiency
    - Rotary position embeddings (RoPE) for positional encoding

    The model follows a DiT-like architecture with adaptive layer normalization
    controlled by timestep embeddings.

    Args:
        patch_size (Tuple[int, int], optional): Patch size for temporal and joint dimensions
            (p_t, p_j). Currently only p_j=1 is supported. Defaults to (1, 1).
        num_attention_heads (int, optional): Number of attention heads. Defaults to 40.
        attention_head_dim (int, optional): Dimension per attention head. Defaults to 128.
        in_channels (int, optional): Number of input channels (motion latent dim). Defaults to 16.
        out_channels (int, optional): Number of output channels. Defaults to 16.
        text_dim (int, optional): Dimension of text encoder hidden states. Defaults to 4096.
        freq_dim (int, optional): Dimension of timestep frequency embedding. Defaults to 256.
        ffn_dim (int, optional): Hidden dimension of feed-forward network. Defaults to 13824.
        num_layers (int, optional): Number of transformer blocks. Defaults to 40.
        cross_attn_norm (bool, optional): Whether to apply LayerNorm before cross-attention.
            Defaults to True.
        qk_norm (str, optional): Type of query-key normalization. Defaults to "rms_norm_across_heads".
        eps (float, optional): Epsilon for layer normalization. Defaults to 1e-6.
        added_kv_proj_dim (int, optional): Dimension for additional KV projections (for I2V).
            Defaults to None.
        rope_max_seq_len (int, optional): Maximum sequence length for RoPE. Defaults to 1024.
        pos_embed_seq_len (int, optional): Sequence length for text positional embeddings.
            Defaults to None (no positional embedding).
        use_fp32_upcast_attention (bool, optional): Whether to use FP32 upcast for attention
            computation to prevent softmax overflow in fp16 training. Defaults to True.

    Attributes:
        rope: Rotary position embedding module for motion sequences.
        patch_embedding: 2D convolution for patchifying input motion.
        condition_embedder: Module for processing timestep and text conditions.
        blocks: Stack of transformer blocks with mask support.
        norm_out: Output layer normalization.
        proj_out: Output projection layer.
        scale_shift_table: Learnable parameters for adaptive layer norm.
    """

    _keep_in_fp32_modules = None

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 1),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        joint_pos_mode: str = "sequential",
        num_spectral_modes: int = 4,
        spectral_scale: Optional[float] = None,
        use_fp32_upcast_attention: bool = True,
    ) -> None:
        # Skip WanTransformer3DModel.__init__ and call ModelMixin.__init__ directly
        # to avoid initializing 3D-specific components
        super(WanTransformer3DModel, self).__init__()

        # Validate patch size: joint patchification is not supported
        assert patch_size[-1] == 1, "we don't support joints patchify"

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # ==========================================================
        # 1. Patch Embedding & Rotary Position Embedding
        # ==========================================================
        # RoPE for encoding temporal and joint positions
        self.rope = MotionWanRotaryPosEmbed(
            attention_head_dim, patch_size, rope_max_seq_len,
            joint_pos_mode=joint_pos_mode,
            num_spectral_modes=num_spectral_modes,
            spectral_scale=spectral_scale,
        )
        # Conv2d patchifies motion: [B, C, T, J] -> [B, inner_dim, T//p_t, J//p_j]
        self.patch_embedding = nn.Conv2d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # ==========================================================
        # 2. Condition Embeddings (Timestep + Text)
        # ==========================================================
        self.condition_embedder = WanTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,  # 6 modulation params per block
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # ==========================================================
        # 3. Transformer Blocks with Mask Support
        # ==========================================================
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlockWithMask(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    use_fp32_upcast_attention,
                )
                for _ in range(num_layers)
            ]
        )

        # ==========================================================
        # 4. Output Normalization & Projection
        # ==========================================================
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        # Scale-shift table for adaptive output normalization
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # Enable gradient checkpointing by default for memory efficiency
        self.enable_gradient_checkpointing(True)

    def enable_gradient_checkpointing(self, value: bool = True) -> None:
        """
        Enable or disable gradient checkpointing for memory-efficient training.

        When enabled, intermediate activations are recomputed during backward pass
        instead of being stored, reducing memory usage at the cost of computation.

        Args:
            value (bool, optional): Whether to enable gradient checkpointing. Defaults to True.
        """
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the motion transformer.

        Args:
            hidden_states (torch.Tensor): Motion latent features from VAE encoder.
                Shape: [B, C, T, J] where B=batch, C=channels, T=frames, J=joints.
            timestep (torch.LongTensor): Diffusion timestep.
                Shape: [B] for standard diffusion, or [B, N] for per-token timesteps
                (Wan 2.2 TI2V mode), where N is the token sequence length.
            encoder_hidden_states (torch.Tensor): Text encoder hidden states.
                Shape: [B, N_ctx, text_dim] where N_ctx is typically 512.
            hidden_states_mask (torch.Tensor, optional): Attention mask for motion tokens.
                Shape: [B, T, J]. Values: 1 = visible/valid, 0 = masked/padding.
                Used to handle variable-length motion sequences in a batch.
                Defaults to None (no masking, all tokens are valid).
            encoder_hidden_states_mask (torch.Tensor, optional): Attention mask for text tokens.
                Shape: [B, N_ctx]. Values: 1 = visible/valid, 0 = masked/padding.
                Used to mask padding in text sequences for cross-attention.
                Defaults to None (no masking, following original Wan behavior).
            attention_kwargs (Dict[str, Any], optional): Additional kwargs for attention.
                Supports 'scale' for LoRA scaling when using PEFT backend.
                Defaults to None.

        Returns:
            torch.Tensor: Predicted noise or velocity with same shape as input.
                Shape: [B, C, T, J].

        Note:
            - The hidden_states_mask is patchified to match the token sequence length
              after patch embedding. If any position within a patch is masked, the
              entire patch token is masked.
            - Both masks are converted to attention bias format: 0 for valid positions,
              -inf (dtype min) for masked positions.
        """
        # ==========================================================
        # Handle LoRA scaling for PEFT backend
        # ==========================================================
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # Weight the LoRA layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ==========================================================
        # 1. Extract dimensions and compute patch parameters
        # ==========================================================
        batch_size, num_channels, num_frames, num_joints = hidden_states.shape
        p_t, p_j = self.config.patch_size
        post_patch_num_frames = (
            num_frames // p_t
        )  # Sequence length after temporal patchification
        post_patch_num_joints = (
            num_joints // p_j
        )  # Joint dimension after patchification (typically unchanged)

        # ==========================================================
        # 2. Compute rotary position embeddings
        # ==========================================================
        # RoPE encodes temporal and joint positions for attention
        rotary_emb = self.rope(hidden_states)

        # ==========================================================
        # 3. Patch embedding: [B, C, T, J] -> [B, N, inner_dim]
        # ==========================================================
        hidden_states = self.patch_embedding(hidden_states)
        # Flatten spatial dims and transpose: [B, inner_dim, T', J'] -> [B, N, inner_dim]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # ==========================================================
        # 4. Process hidden_states_mask (motion attention mask)
        # ==========================================================
        # Patchify the mask to match token sequence length.
        # Original shape: [B, T, J] with 1=visible, 0=masked
        # Target shape: [B, 1, 1, N] with 0=valid, -inf=masked (for attention bias)
        if hidden_states_mask is not None:
            # Step 1: Reshape to separate patch dimensions
            # [B, T, J] -> [B, T//p_t, p_t, J//p_j, p_j]
            hidden_states_mask = hidden_states_mask.reshape(
                batch_size,
                post_patch_num_frames,
                p_t,
                post_patch_num_joints,
                p_j,
            )
            # Step 2: Min pooling across patch dimensions
            # If ANY position in a patch is masked (0), the entire patch is masked
            # [B, T//p_t, p_t, J//p_j, p_j] -> [B, T//p_t, J//p_j]
            hidden_states_mask = hidden_states_mask.amin(dim=(2, 4))

            # Step 3: Flatten to token sequence
            # [B, T//p_t, J//p_j] -> [B, N]
            hidden_states_mask = hidden_states_mask.flatten(1)

            # Step 4: Convert to attention bias format
            # 1 (visible) -> 0.0, 0 (masked) -> -inf (dtype min)
            # Final shape: [B, 1, 1, N] for broadcasting in attention
            hidden_states_mask = (
                (
                    (1.0 - hidden_states_mask.float())
                    * torch.finfo(hidden_states.dtype).min
                )
                .unsqueeze(1)
                .unsqueeze(2)
            )

        # ==========================================================
        # 5. Process encoder_hidden_states_mask (text attention mask)
        # ==========================================================
        # Convert text mask to attention bias format for cross-attention.
        # Original shape: [B, N_ctx] with 1=visible, 0=masked
        # Target shape: [B, 1, 1, N_ctx] with 0=valid, -inf=masked
        # Note: This is optional - if None, follows original Wan behavior (no text masking)
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = (
                (
                    (1.0 - encoder_hidden_states_mask.float())
                    * torch.finfo(hidden_states.dtype).min
                )
                .unsqueeze(1)
                .unsqueeze(2)
            )

        # ==========================================================
        # 5b. Build causal attention mask if is_causal=True
        # ==========================================================
        # Frame-level (block) causal mask: all joint tokens within the same
        # frame can attend to each other, but tokens can only attend to the
        # current frame and earlier frames — never future frames.
        #
        # Token sequence is flattened from [T', J'] in row-major order, so
        # token i lives at frame (i // J') and joint (i % J').
        #
        # Shape: [1, 1, N, N] with 0=attend, -inf=masked
        causal_mask = None
        if is_causal:
            seq_len = hidden_states.shape[1]
            # Frame index for each token position
            frame_idx = torch.arange(
                seq_len, device=hidden_states.device
            ) // post_patch_num_joints
            # mask[i, j] = -inf if frame_idx[j] > frame_idx[i] (future frame)
            #            = 0    otherwise (same or past frame)
            causal_mask = (
                (frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1))
                .to(hidden_states.dtype)
                * torch.finfo(hidden_states.dtype).min
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        # ==========================================================
        # 6. Process timestep and text conditions
        # ==========================================================
        # Handle per-token timesteps for Wan 2.2 TI2V mode
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # [B, N] -> [B*N]
        else:
            ts_seq_len = None

        # Get timestep embedding (temb), timestep projection, and processed text embeddings
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            timestep_seq_len=ts_seq_len,
        )

        # Reshape timestep projection for block modulation
        if ts_seq_len is not None:
            # Wan 2.2 TI2V: per-token modulation [B, N, 6*inner_dim] -> [B, N, 6, inner_dim]
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # Standard: global modulation [B, 6*inner_dim] -> [B, 6, inner_dim]
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # ==========================================================
        # 7. Transformer blocks (self-attention + cross-attention + FFN)
        # ==========================================================
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Gradient checkpointing: recompute activations during backward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    hidden_states_mask,
                    encoder_hidden_states_mask,
                    causal_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    hidden_states_mask=hidden_states_mask,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    causal_mask=causal_mask,
                )

        # ==========================================================
        # 8. Output normalization with adaptive scale-shift
        # ==========================================================
        if temb.ndim == 3:
            # Wan 2.2 TI2V: per-token scale-shift
            # temb: [B, N, inner_dim] -> scale_shift: [B, N, 2, inner_dim]
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # Standard: global scale-shift
            # temb: [B, inner_dim] -> scale_shift: [B, 2, inner_dim]
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        # Handle multi-GPU case: ensure shift/scale are on same device as hidden_states
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer norm: out = norm(x) * (1 + scale) + shift
        # hidden_states is already fp32 (fp32 residual stream), so .float() is a no-op
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        )

        # ==========================================================
        # 9. Output projection
        # ==========================================================
        hidden_states = hidden_states.to(dtype=self.proj_out.weight.dtype)
        hidden_states = self.proj_out(hidden_states)

        # ==========================================================
        # 10. Unpatchify: [B, N, C*p_t*p_j] -> [B, C, T, J]
        # ==========================================================
        # Reshape to separate patch dimensions
        # [B, N, C*p_t*p_j] -> [B, T', J', p_t, p_j, C]
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_num_joints,
            p_t,
            p_j,
            -1,
        )
        # Permute to interleave patches: [B, T', J', p_t, p_j, C] -> [B, C, T', p_t, J', p_j]
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        # Flatten to original spatial layout: [B, C, T'*p_t, J'*p_j] = [B, C, T, J]
        output = hidden_states.flatten(4, 5).flatten(2, 3)

        # ==========================================================
        # Cleanup: remove LoRA scaling
        # ==========================================================
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return output


if __name__ == "__main__":
    from mmengine.device import get_device

    device = get_device()
    dtype = torch.bfloat16

    # Test configuration
    batch_size = 2
    num_channels = 16
    num_frames = 16  # Must be divisible by patch_size[0]
    num_joints = 24
    text_seq_len = 15
    text_dim = 4096

    print("=" * 60)
    print("Testing PrismTransformerMotionModel with Mask Support")
    print("=" * 60)

    # ==================== Test 1: Basic forward (no mask) ====================
    print("\nTest 1: Basic forward pass (no masking)")
    model = PrismTransformerMotionModel(
        patch_size=(1, 1),
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=num_channels,
        num_attention_heads=12,
        num_layers=4,  # Small for testing
        out_channels=num_channels,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        text_dim=text_dim,
    ).to(device=device, dtype=dtype)
    model.eval()

    with torch.no_grad():
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, num_joints
        ).to(device=device, dtype=dtype)
        timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
            device=device, dtype=dtype
        )
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        print(f"Input shape: {hidden_states.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 1 passed!")

    # ==================== Test 2: With hidden_states_mask ====================
    print("\n" + "-" * 40)
    print("Test 2: Forward with hidden_states_mask (variable motion lengths)")
    with torch.no_grad():
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, num_joints
        ).to(device=device, dtype=dtype)
        timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
            device=device, dtype=dtype
        )

        # Create hidden_states_mask: [B, T, J], 1 = visible, 0 = masked
        hidden_states_mask = torch.zeros(
            batch_size, num_frames, num_joints, device=device
        )
        hidden_states_mask[0, :12, :] = 1  # First sample: 12 valid frames
        hidden_states_mask[1, :16, :] = 1  # Second sample: 16 valid frames (all)

        print(f"hidden_states_mask shape: {hidden_states_mask.shape}")
        print(f"Valid frames: sample 0 = 12, sample 1 = 16")

        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_mask=hidden_states_mask,
        )
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 2 passed!")

    # ==================== Test 3: With encoder_hidden_states_mask ====================
    print("\n" + "-" * 40)
    print("Test 3: Forward with encoder_hidden_states_mask (variable text lengths)")
    with torch.no_grad():
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, num_joints
        ).to(device=device, dtype=dtype)
        timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
            device=device, dtype=dtype
        )

        # Create encoder_hidden_states_mask: [B, N_ctx], 1 = visible, 0 = masked
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, device=device)
        encoder_hidden_states_mask[0, 10:] = 0  # First sample: mask after 10 tokens

        print(f"encoder_hidden_states_mask shape: {encoder_hidden_states_mask.shape}")
        print(f"Valid text tokens: sample 0 = 10, sample 1 = 15")

        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 3 passed!")

    # ==================== Test 4: With both masks ====================
    print("\n" + "-" * 40)
    print("Test 4: Forward with both masks")
    with torch.no_grad():
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, num_joints
        ).to(device=device, dtype=dtype)
        timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
            device=device, dtype=dtype
        )

        # hidden_states_mask: [B, T, J]
        hidden_states_mask = torch.zeros(
            batch_size, num_frames, num_joints, device=device
        )
        hidden_states_mask[0, :12, :] = 1
        hidden_states_mask[1, :16, :] = 1

        # encoder_hidden_states_mask: [B, N_ctx]
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, device=device)
        encoder_hidden_states_mask[0, 10:] = 0

        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 4 passed!")

    # ==================== Test 5: With patch_size > 1 ====================
    print("\n" + "-" * 40)
    print("Test 5: With patch_size=(2, 1) and hidden_states_mask")

    model_patched = PrismTransformerMotionModel(
        patch_size=(2, 1),  # Temporal patch size = 2
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=num_channels,
        num_attention_heads=12,
        num_layers=4,
        out_channels=num_channels,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        text_dim=text_dim,
    ).to(device=device, dtype=dtype)
    model_patched.eval()

    with torch.no_grad():
        hidden_states = torch.randn(
            batch_size, num_channels, num_frames, num_joints
        ).to(device=device, dtype=dtype)
        timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
            device=device, dtype=dtype
        )

        # hidden_states_mask: [B, T, J]
        hidden_states_mask = torch.zeros(
            batch_size, num_frames, num_joints, device=device
        )
        hidden_states_mask[0, :10, :] = 1  # First sample: 10 valid frames -> 5 patches
        hidden_states_mask[1, :16, :] = 1  # Second sample: 16 valid frames -> 8 patches

        print(f"Input shape: {hidden_states.shape}")
        print(f"patch_size: (2, 1)")
        print(f"hidden_states_mask shape: {hidden_states_mask.shape}")
        print(f"Valid frames: sample 0 = 10 (5 patches), sample 1 = 16 (8 patches)")

        output = model_patched(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_mask=hidden_states_mask,
        )
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 5 passed!")

    # ==================== Test 6: Verify mask patchify logic ====================
    print("\n" + "-" * 40)
    print("Test 6: Verify mask patchify logic (partial patch masking)")

    with torch.no_grad():
        # Create mask where one frame in a patch is masked
        hidden_states_mask = torch.ones(
            batch_size, num_frames, num_joints, device=device
        )
        # Mask frame 9 (second frame in patch 4 when patch_size=2)
        # This should cause the entire patch 4 to be masked
        hidden_states_mask[0, 9, :] = 0

        print(f"Testing partial patch masking: frame 9 is masked")
        print(
            f"With patch_size=(2,1), patch 4 covers frames [8,9], should be fully masked"
        )

        output = model_patched(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            hidden_states_mask=hidden_states_mask,
        )
        print(f"Output shape: {output.shape}")
        assert output.shape == hidden_states.shape, "Output shape mismatch!"
        print("✓ Test 6 passed!")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
