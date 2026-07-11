"""Motion Condition Encoder for DSCF v3.

Encodes known-region motion into a compact set of condition tokens via
cross-attention from learnable queries to frame-level features.

Architecture:
    1. Input projection: Linear(motion_dim * 2, feat_dim) for [known_motion, mask] concat
    2. Temporal PE: sinusoidal positional encoding for frame positions
    3. Density embedding: MLP(1 → feat_dim) for global mask density context
    4. N transformer encoder layers where learnable queries cross-attend to frame features
    5. Output: (B, num_queries, feat_dim) — compact condition tokens

Design choices:
    - 128 learnable queries compress arbitrary-length frame features into fixed-size tokens
    - Cross-attention (queries attend to frames) instead of self-attention over frames
      allows O(num_queries * num_frames) instead of O(num_frames^2) complexity
    - Density embedding injects global "how much is known" context, helping the model
      adapt its encoding strategy based on condition sparsity
    - Frame features include [known_motion_values, binary_mask] concatenated — the mask
      channel tells the encoder which dimensions are truly observed vs. zero-padded

The output tokens are then used as KV for cross-attention in DualCondMMDiTBlocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import attention
from .bricks import RMSNorm, get_activation_layer


class MotionCondEncoder(nn.Module):
    """Compress known-region motion into condition tokens via cross-attention.

    Args:
        motion_dim: Raw motion dimension (e.g. 198 for v2 representation).
        feat_dim: Internal feature dimension (should match transformer feat_dim, e.g. 1024).
        num_queries: Number of learnable query tokens (default 128).
        num_layers: Number of cross-attention encoder layers (default 4).
        num_heads: Number of attention heads (default 16).
        max_seq_len: Maximum temporal length for positional encoding (default 512).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        motion_dim: int = 198,
        feat_dim: int = 1024,
        num_queries: int = 128,
        num_layers: int = 4,
        num_heads: int = 16,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.feat_dim = feat_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0, (
            f"feat_dim={feat_dim} must be divisible by num_heads={num_heads}"
        )

        # Input projection: [known_motion(D), mask(D)] -> feat_dim
        self.input_proj = nn.Linear(motion_dim * 2, feat_dim)

        # Temporal positional encoding (sinusoidal, registered as buffer)
        pe = self._build_sinusoidal_pe(max_seq_len, feat_dim)
        self.register_buffer('temporal_pe', pe)  # (1, max_seq_len, feat_dim)

        # Density embedding: scalar density -> feat_dim
        self.density_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, feat_dim),
        )

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, feat_dim) * 0.02)

        # Cross-attention encoder layers
        self.layers = nn.ModuleList([
            MotionCondEncoderLayer(
                feat_dim=feat_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final norm on output queries
        self.output_norm = nn.LayerNorm(feat_dim, eps=1e-6)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> Tensor:
        """Build sinusoidal positional encoding buffer."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, dim)

    def forward(
        self,
        known_motion: Tensor,
        mask: Tensor,
        frame_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode known-region motion into condition tokens.

        Args:
            known_motion: (B, L, motion_dim) — motion values; zero where mask=1.
            mask: (B, L, motion_dim) — binary mask (1=generate, 0=known).
            frame_mask: (B, L) — optional boolean mask for valid frames (True=valid).
                Used to prevent attention to padding frames.

        Returns:
            cond_tokens: (B, num_queries, feat_dim) — condition tokens for
                cross-attention in DualCondMMDiTBlocks.
        """
        B, L, D = known_motion.shape

        # 1. Build frame features: [known_motion, mask] -> project
        frame_input = torch.cat([known_motion, mask], dim=-1)  # (B, L, 2*D)
        frame_feat = self.input_proj(frame_input)  # (B, L, feat_dim)

        # 2. Add temporal positional encoding
        frame_feat = frame_feat + self.temporal_pe[:, :L, :]

        # 3. Compute and add density embedding (global context)
        # density = fraction of known dimensions (mask=0 means known)
        density = (1.0 - mask).mean(dim=(-1, -2), keepdim=False)  # (B,)
        density_emb = self.density_embed(density.unsqueeze(-1))  # (B, feat_dim)
        frame_feat = frame_feat + density_emb.unsqueeze(1)  # broadcast over L

        # 4. Expand learnable queries
        queries = self.queries.expand(B, -1, -1)  # (B, num_queries, feat_dim)

        # 5. Cross-attention layers: queries attend to frame features
        for layer in self.layers:
            queries = layer(queries, frame_feat, kv_mask=frame_mask)

        # 6. Final norm
        cond_tokens = self.output_norm(queries)
        return cond_tokens


class MotionCondEncoderLayer(nn.Module):
    """Single layer of the Motion Condition Encoder.

    Architecture:
        1. Cross-attention: queries attend to frame features (with RMSNorm on q/k)
        2. FFN: standard MLP with SiLU activation

    Both have pre-norm (LayerNorm) and residual connections.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.0,
        ffn_ratio: float = 4.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        ffn_dim = int(feat_dim * ffn_ratio)

        # Cross-attention components
        self.norm_q = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)

        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=True)

        # QK norm (RMSNorm, matching existing MMDiT convention)
        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)

        self.dropout = dropout

        # FFN components
        self.norm_ffn = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, feat_dim),
        )

    def forward(
        self,
        queries: Tensor,
        kv_feat: Tensor,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            queries: (B, Q, feat_dim) — learnable query tokens.
            kv_feat: (B, L, feat_dim) — frame features (keys/values).
            kv_mask: (B, L) — optional boolean mask. True=valid, False=padding.

        Returns:
            Updated queries: (B, Q, feat_dim).
        """
        B, Q, _ = queries.shape
        _, L, _ = kv_feat.shape
        H = self.num_heads
        D = self.head_dim

        # --- Cross-attention ---
        residual = queries
        q_in = self.norm_q(queries)
        kv_in = self.norm_kv(kv_feat)

        q = self.q_proj(q_in).reshape(B, Q, H, D)
        k = self.k_proj(kv_in).reshape(B, L, H, D)
        v = self.v_proj(kv_in).reshape(B, L, H, D)

        # Apply QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Build attention mask from kv_mask if provided
        attn_mask = None
        if kv_mask is not None:
            # kv_mask: (B, L) bool, True=valid → we need to mask out False positions
            # For torch mode, attn_mask is additive: 0 for valid, -inf for masked
            attn_mask = torch.zeros(B, 1, Q, L, dtype=q.dtype, device=q.device)
            # Expand kv_mask: (B, L) -> (B, 1, 1, L)
            padding_mask = ~kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_mask.masked_fill_(padding_mask, float('-inf'))

        # Attention: q(B,Q,H,D), k(B,L,H,D), v(B,L,H,D) -> out(B,Q,H*D)
        out = attention(
            q, k, v,
            mode='torch',
            drop_rate=self.dropout if self.training else 0.0,
            attn_mask=attn_mask,
        )  # (B, Q, H*D)

        queries = residual + self.out_proj(out)

        # --- FFN ---
        residual = queries
        queries = residual + self.ffn(self.norm_ffn(queries))

        return queries
