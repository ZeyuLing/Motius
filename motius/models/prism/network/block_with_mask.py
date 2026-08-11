"""
Wan Transformer Block with Mask Support.

This module extends the standard Wan Transformer Block to support attention masking
for both self-attention and cross-attention operations. This is essential for handling
variable-length sequences (e.g., motion sequences with different durations) in a batched
setting.

Key Features:
    - Self-attention with rotary position embeddings and optional masking
    - Cross-attention between motion features and conditioning (e.g., text) with masking
    - Adaptive layer normalization with shift and scale modulation
    - Feed-forward network with gated residual connections
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
try:
    from diffusers.models.transformers.transformer_wan import (
        FeedForward,
        WanAttention,
        WanAttnProcessor,
    )
    from .attention_fp32_upcast import WanAttnProcessorFP32Upcast

    _WAN_ATTENTION_API = "wan"
except ImportError:
    from diffusers.models.transformers.transformer_wan import (
        Attention as WanAttention,
        FeedForward,
        WanAttnProcessor2_0 as WanAttnProcessor,
    )
    from .attention_fp32_upcast import WanAttnProcessorFP32Upcast

    _WAN_ATTENTION_API = "attention"
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


def _make_wan_attention(
    *,
    dim: int,
    num_heads: int,
    eps: float,
    qk_norm: str,
    added_kv_proj_dim: Optional[int],
    cross_attention_dim_head: Optional[int],
    use_fp32_upcast_attention: bool,
) -> nn.Module:
    processor = (
        WanAttnProcessorFP32Upcast()
        if use_fp32_upcast_attention and WanAttnProcessorFP32Upcast is not None
        else WanAttnProcessor()
    )
    dim_head = dim // num_heads
    if _WAN_ATTENTION_API == "wan":
        return WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=cross_attention_dim_head,
            processor=processor,
        )

    return WanAttention(
        query_dim=dim,
        heads=num_heads,
        kv_heads=num_heads,
        dim_head=dim_head,
        qk_norm=qk_norm,
        eps=eps,
        bias=True,
        cross_attention_dim=None,
        out_bias=True,
        added_kv_proj_dim=added_kv_proj_dim,
        added_proj_bias=True,
        processor=processor,
    )


def _rotary_emb_for_attention_api(
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    """Adapt PRISM RoPE to the older diffusers ``Attention`` API when needed."""

    if _WAN_ATTENTION_API != "attention" or rotary_emb is None:
        return rotary_emb

    freqs_cos, freqs_sin = rotary_emb
    # PRISM/new WanAttention uses tuple cos/sin shaped like (1, seq, 1, dim).
    # Older WanAttnProcessor2_0 expects complex freqs broadcastable to
    # (batch, heads, seq, dim / 2).
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    if cos.ndim == 4 and cos.shape[2] == 1:
        cos = cos.permute(0, 2, 1, 3).contiguous()
        sin = sin.permute(0, 2, 1, 3).contiguous()
    return torch.complex(cos.to(torch.float64), sin.to(torch.float64))


@maybe_allow_in_graph
class WanTransformerBlockWithMask(nn.Module):
    """
    A Transformer block based on the Wan architecture with support for attention masking.

    This block follows the standard transformer architecture with:
        1. Self-attention layer (with rotary position embeddings)
        2. Cross-attention layer (for conditioning on external features like text)
        3. Feed-forward network

    Each sub-layer uses adaptive layer normalization with shift/scale modulation
    controlled by the timestep embedding, and gated residual connections.

    Attention Masking:
        - Self-attention uses `hidden_states_mask` to mask out padding positions
          in the motion sequence itself.
        - Cross-attention uses `encoder_hidden_states_mask` to mask out padding
          positions in the conditioning sequence (e.g., text tokens).

    Args:
        dim (int): The hidden dimension of the transformer.
        ffn_dim (int): The intermediate dimension of the feed-forward network.
        num_heads (int): Number of attention heads.
        qk_norm (str, optional): Type of query-key normalization. Defaults to "rms_norm_across_heads".
        cross_attn_norm (bool, optional): Whether to apply layer normalization before
            cross-attention. If False, uses identity. Defaults to False.
        eps (float, optional): Epsilon for layer normalization. Defaults to 1e-6.
        added_kv_proj_dim (int, optional): Dimension for additional key-value projections
            (used in image-to-video models). Defaults to None.
        use_fp32_upcast_attention (bool, optional): Whether to use FP32 upcast for attention
            computation to prevent softmax overflow in fp16 training. Defaults to True.

    Example:
        >>> block = WanTransformerBlockWithMask(dim=1024, ffn_dim=4096, num_heads=16)
        >>> hidden_states = torch.randn(2, 100, 1024)  # (batch, seq_len, dim)
        >>> encoder_hidden_states = torch.randn(2, 77, 1024)  # (batch, text_len, dim)
        >>> temb = torch.randn(2, 6, 1024)  # timestep embedding
        >>> rotary_emb = (cos_emb, sin_emb)  # rotary position embeddings
        >>> # Mask: True = valid, False = padding (or use 0/-inf attention bias)
        >>> hidden_mask = torch.ones(2, 1, 1, 100)  # self-attn mask
        >>> encoder_mask = torch.ones(2, 1, 1, 77)  # cross-attn mask
        >>> output = block(hidden_states, encoder_hidden_states, temb, rotary_emb,
        ...                hidden_mask, encoder_mask)
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        use_fp32_upcast_attention: bool = True,
    ):
        super().__init__()

        # ========== 1. Self-attention ==========
        # Layer normalization before self-attention (without learnable affine parameters,
        # as shift/scale are provided by the timestep embedding)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # Self-attention: Query, Key, Value all come from hidden_states
        # cross_attention_dim_head=None indicates this is self-attention
        # Use FP32 upcast processor if enabled to prevent softmax overflow in fp16
        self.attn1 = _make_wan_attention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            qk_norm=qk_norm,
            added_kv_proj_dim=None,
            cross_attention_dim_head=None,
            use_fp32_upcast_attention=use_fp32_upcast_attention,
        )

        # ========== 2. Cross-attention ==========
        # Cross-attention: Query from hidden_states, Key/Value from encoder_hidden_states
        # cross_attention_dim_head is set to enable cross-attention mode
        # Use FP32 upcast processor if enabled to prevent softmax overflow in fp16
        self.attn2 = _make_wan_attention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            qk_norm=qk_norm,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            use_fp32_upcast_attention=use_fp32_upcast_attention,
        )

        # Optional normalization before cross-attention
        # Some configurations use learnable LayerNorm, others use Identity
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # ========== 3. Feed-forward network ==========
        # Layer normalization before FFN (without learnable affine, modulated by timestep)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # Feed-forward network with GELU activation
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        # ========== Adaptive modulation parameters ==========
        # Learnable table for shift, scale, and gate parameters
        # Shape: (1, 6, dim) -> 6 parameters: shift, scale, gate for self-attn and FFN
        # - shift_msa, scale_msa, gate_msa: for self-attention
        # - c_shift_msa, c_scale_msa, c_gate_msa: for feed-forward (c = "context/condition")
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        hidden_states_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Wan Transformer Block with mask support.

        Args:
            hidden_states (torch.Tensor): Input hidden states (motion features).
                Shape: (batch_size, seq_len, dim)
            encoder_hidden_states (torch.Tensor): Conditioning hidden states (e.g., text embeddings).
                Shape: (batch_size, encoder_seq_len, dim)
            temb (torch.Tensor): Timestep embedding for adaptive modulation.
                Shape: (batch_size, 6, dim) for Wan 2.1/2.2 14B
                   or (batch_size, seq_len, 6, dim) for Wan 2.2 TI2V
            rotary_emb (Tuple[torch.Tensor, torch.Tensor], optional): Rotary position embeddings
                as (freqs_cos, freqs_sin). Applied only to self-attention.
            hidden_states_mask (torch.Tensor, optional): Attention mask for self-attention.
                Used to mask out padding positions in the motion sequence.
                Shape: (batch_size, 1, 1, seq_len) or broadcastable shape.
                Values: 0 for positions to attend, -inf (or large negative) for masked positions.
            encoder_hidden_states_mask (torch.Tensor, optional): Attention mask for cross-attention.
                Used to mask out padding positions in the encoder sequence (e.g., text).
                Shape: (batch_size, 1, 1, encoder_seq_len) or broadcastable shape.
                Values: 0 for positions to attend, -inf (or large negative) for masked positions.

        Returns:
            torch.Tensor: Output hidden states with same shape as input hidden_states.
                Shape: (batch_size, seq_len, dim)

        Note:
            - Self-attention mask (`hidden_states_mask`): Masks the Key positions that
              Query should NOT attend to. Since Q, K, V all come from hidden_states,
              this masks padding in the motion sequence.
            - Cross-attention mask (`encoder_hidden_states_mask`): Masks the Key positions
              (from encoder_hidden_states) that Query (from hidden_states) should NOT
              attend to. This masks padding in the conditioning sequence.
        """
        # ========== Compute adaptive modulation parameters ==========
        # The timestep embedding modulates the layer normalization via shift/scale,
        # and controls the residual connection strength via gates.
        if temb.ndim == 4:
            # Wan 2.2 TI2V: temb has per-position modulation
            # temb shape: (batch_size, seq_len, 6, dim)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # Squeeze to shape: (batch_size, seq_len, dim)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # Wan 2.1 / Wan 2.2 14B: global modulation
            # temb shape: (batch_size, 6, dim)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # ========== 1. Self-attention ==========
        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        # NOTE: hidden_states stays in fp32 throughout all sub-layers (fp32 residual stream).
        # Under fp16 autocast, Linear layers inside attn/ffn use fp16 tensor cores,
        # but residual additions remain fp32 to prevent overflow after 30 layers.
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        )
        attn1_dtype = self.attn1.to_q.weight.dtype
        norm_hidden_states = norm_hidden_states.to(dtype=attn1_dtype)

        # Self-attention with rotary embeddings and optional masking
        # - encoder_hidden_states=None indicates self-attention (Q, K, V from hidden_states)
        # - attention_mask masks padding in the sequence (and optionally causal)
        # - rotary_emb applies rotary position embeddings to Q and K
        # Combine padding mask with causal mask if provided
        combined_self_attn_mask = hidden_states_mask
        if causal_mask is not None:
            if combined_self_attn_mask is not None:
                # Both masks: add them (both use 0/-inf convention, adding preserves -inf)
                combined_self_attn_mask = combined_self_attn_mask + causal_mask
            else:
                combined_self_attn_mask = causal_mask

        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,  # Self-attention: Q, K, V from hidden_states
            attention_mask=combined_self_attn_mask,  # Mask padding + causal in motion sequence
            rotary_emb=_rotary_emb_for_attention_api(rotary_emb),  # Apply rotary position embeddings
        )

        # Gated residual connection: x = x + gate * attn_output
        hidden_states = hidden_states.float() + attn_output * gate_msa

        # ========== 2. Cross-attention ==========
        # Apply layer normalization (may be identity if cross_attn_norm=False)
        norm_hidden_states = self.norm2(hidden_states.float())
        attn2_dtype = self.attn2.to_q.weight.dtype
        norm_hidden_states = norm_hidden_states.to(dtype=attn2_dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=attn2_dtype)

        # Cross-attention: Query from hidden_states, Key/Value from encoder_hidden_states
        # - attention_mask=encoder_hidden_states_mask masks padding in conditioning sequence
        # - rotary_emb=None: no positional encoding for cross-attention (text has its own)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,  # Query source
            encoder_hidden_states=encoder_hidden_states,  # Key/Value source
            attention_mask=encoder_hidden_states_mask,  # Mask padding in encoder sequence
            rotary_emb=None,  # No rotary embeddings for cross-attention
        )

        # Simple residual connection (no gate for cross-attention in this architecture)
        hidden_states = hidden_states + attn_output

        # ========== 3. Feed-forward network ==========
        # Apply adaptive layer norm with FFN-specific shift/scale
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        )
        norm_hidden_states = norm_hidden_states.to(dtype=attn2_dtype)

        # Feed-forward network — under fp16 autocast, Linear layers use fp16 tensor
        # cores (fp16×fp16 with fp32 accumulation on V100). GELU-tanh computes x^3
        # which could overflow fp16 if |x| > 40.3, but with LayerNorm-normalized
        # inputs and pretrained weights, Linear outputs are typically magnitude 1-5,
        # well below overflow threshold. The fp32 residual stream ensures no
        # accumulation overflow across the 30 transformer blocks.
        ff_output = self.ffn(norm_hidden_states)

        # Gated residual connection for FFN
        hidden_states = hidden_states.float() + ff_output.float() * c_gate_msa

        return hidden_states


def _create_rotary_embeddings(
    seq_len: int,
    dim: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create simple rotary position embeddings for testing.

    Args:
        seq_len: Sequence length
        dim: Hidden dimension
        num_heads: Number of attention heads
        device: Target device
        dtype: Data type

    Returns:
        Tuple of (freqs_cos, freqs_sin) with shape (1, seq_len, 1, head_dim)
    """
    head_dim = dim // num_heads
    # Simple frequency schedule
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim // 2)

    # Interleave cos and sin
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)

    # Reshape for attention: (1, seq_len, 1, head_dim)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2).to(dtype)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2).to(dtype)

    return freqs_cos, freqs_sin


def _create_attention_mask(
    batch_size: int,
    seq_len: int,
    valid_lengths: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Create attention mask from valid lengths.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        valid_lengths: Tensor of valid lengths for each sample, shape (batch_size,)
                      If None, returns None (no masking)
        device: Target device
        dtype: Data type

    Returns:
        Attention mask with shape (batch_size, 1, 1, seq_len)
        Values: 0 for valid positions, -inf for masked (padding) positions
    """
    if valid_lengths is None:
        return None

    # Create position indices: (1, seq_len)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    # Create mask: True where position < valid_length
    # valid_lengths shape: (batch_size, 1)
    mask = positions < valid_lengths.unsqueeze(1)  # (batch_size, seq_len)

    # Convert to attention bias: 0 for valid, -inf for masked
    attention_mask = torch.zeros(batch_size, 1, 1, seq_len, device=device, dtype=dtype)
    attention_mask.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    return attention_mask


if __name__ == "__main__":
    """
    Test script for WanTransformerBlockWithMask.

    This script tests:
        1. Basic forward pass without masking
        2. Forward pass with self-attention masking (variable motion lengths)
        3. Forward pass with cross-attention masking (variable text lengths)
        4. Forward pass with both masks
        5. Wan 2.2 TI2V mode (4D timestep embedding)
        6. Gradient flow verification
    """
    print("=" * 60)
    print("Testing WanTransformerBlockWithMask")
    print("=" * 60)

    # ==================== Configuration ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Model configuration
    dim = 512  # Hidden dimension
    ffn_dim = 2048  # FFN intermediate dimension
    num_heads = 8  # Number of attention heads

    # Input configuration
    batch_size = 2
    motion_seq_len = 64  # Motion sequence length
    text_seq_len = 32  # Text/encoder sequence length

    print(f"\nDevice: {device}")
    print(f"Model config: dim={dim}, ffn_dim={ffn_dim}, num_heads={num_heads}")
    print(
        f"Input config: batch={batch_size}, motion_len={motion_seq_len}, text_len={text_seq_len}"
    )

    # ==================== Initialize Model ====================
    print("\n" + "-" * 40)
    print("Initializing model...")

    block = WanTransformerBlockWithMask(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        cross_attn_norm=True,  # Use learnable norm before cross-attention
    ).to(device=device, dtype=dtype)

    num_params = sum(p.numel() for p in block.parameters())
    print(f"Model initialized. Parameters: {num_params:,}")

    # ==================== Create Test Inputs ====================
    print("\n" + "-" * 40)
    print("Creating test inputs...")

    # Hidden states (motion features)
    hidden_states = torch.randn(
        batch_size, motion_seq_len, dim, device=device, dtype=dtype
    )

    # Encoder hidden states (text embeddings)
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, dim, device=device, dtype=dtype
    )

    # Timestep embedding (Wan 2.1/2.2 14B style: global modulation)
    temb = torch.randn(batch_size, 6, dim, device=device, dtype=dtype)

    # Rotary position embeddings
    rotary_emb = _create_rotary_embeddings(
        motion_seq_len, dim, num_heads, device, dtype
    )

    print(f"hidden_states: {hidden_states.shape}")
    print(f"encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"temb: {temb.shape}")
    print(f"rotary_emb: cos={rotary_emb[0].shape}, sin={rotary_emb[1].shape}")

    # ==================== Test 1: Basic forward (no mask) ====================
    print("\n" + "-" * 40)
    print("Test 1: Basic forward pass (no masking)")

    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        rotary_emb=rotary_emb,
        hidden_states_mask=None,
        encoder_hidden_states_mask=None,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 1 passed!")

    # ==================== Test 2: Self-attention mask ====================
    print("\n" + "-" * 40)
    print("Test 2: Forward with self-attention mask (variable motion lengths)")

    # Variable motion lengths: [48, 64] for batch of 2
    motion_valid_lengths = torch.tensor([48, 64], device=device)
    hidden_states_mask = _create_attention_mask(
        batch_size, motion_seq_len, motion_valid_lengths, device, dtype
    )

    print(f"Motion valid lengths: {motion_valid_lengths.tolist()}")
    print(f"hidden_states_mask: {hidden_states_mask.shape}")

    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        rotary_emb=rotary_emb,
        hidden_states_mask=hidden_states_mask,
        encoder_hidden_states_mask=None,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 2 passed!")

    # ==================== Test 3: Cross-attention mask ====================
    print("\n" + "-" * 40)
    print("Test 3: Forward with cross-attention mask (variable text lengths)")

    # Variable text lengths: [20, 32] for batch of 2
    text_valid_lengths = torch.tensor([20, 32], device=device)
    encoder_hidden_states_mask = _create_attention_mask(
        batch_size, text_seq_len, text_valid_lengths, device, dtype
    )

    print(f"Text valid lengths: {text_valid_lengths.tolist()}")
    print(f"encoder_hidden_states_mask: {encoder_hidden_states_mask.shape}")

    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        rotary_emb=rotary_emb,
        hidden_states_mask=None,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 3 passed!")

    # ==================== Test 4: Both masks ====================
    print("\n" + "-" * 40)
    print("Test 4: Forward with both self-attention and cross-attention masks")

    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        rotary_emb=rotary_emb,
        hidden_states_mask=hidden_states_mask,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 4 passed!")

    # ==================== Test 5: Wan 2.2 TI2V mode (4D temb) ====================
    print("\n" + "-" * 40)
    print("Test 5: Wan 2.2 TI2V mode (4D timestep embedding)")

    # 4D timestep embedding: per-position modulation
    temb_4d = torch.randn(
        batch_size, motion_seq_len, 6, dim, device=device, dtype=dtype
    )

    print(f"temb_4d: {temb_4d.shape}")

    output = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb_4d,
        rotary_emb=rotary_emb,
        hidden_states_mask=hidden_states_mask,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 5 passed!")

    # ==================== Test 6: Gradient flow ====================
    print("\n" + "-" * 40)
    print("Test 6: Gradient flow verification")

    # Enable gradient computation
    hidden_states_grad = hidden_states.clone().requires_grad_(True)
    encoder_hidden_states_grad = encoder_hidden_states.clone().requires_grad_(True)

    output = block(
        hidden_states=hidden_states_grad,
        encoder_hidden_states=encoder_hidden_states_grad,
        temb=temb,
        rotary_emb=rotary_emb,
        hidden_states_mask=hidden_states_mask,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
    )

    # Backward pass
    loss = output.sum()
    loss.backward()

    print(f"hidden_states gradient: {hidden_states_grad.grad is not None}")
    print(
        f"encoder_hidden_states gradient: {encoder_hidden_states_grad.grad is not None}"
    )

    assert hidden_states_grad.grad is not None, "No gradient for hidden_states!"
    assert (
        encoder_hidden_states_grad.grad is not None
    ), "No gradient for encoder_hidden_states!"

    # Check gradient values are reasonable (not zero, not NaN)
    assert not torch.isnan(
        hidden_states_grad.grad
    ).any(), "NaN in hidden_states gradient!"
    assert not torch.isnan(
        encoder_hidden_states_grad.grad
    ).any(), "NaN in encoder_hidden_states gradient!"
    assert hidden_states_grad.grad.abs().sum() > 0, "Zero hidden_states gradient!"
    assert (
        encoder_hidden_states_grad.grad.abs().sum() > 0
    ), "Zero encoder_hidden_states gradient!"

    print("✓ Test 6 passed!")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
