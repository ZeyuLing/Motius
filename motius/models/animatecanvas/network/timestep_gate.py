"""Timestep-Adaptive Fusion Gate for DSCF v3.

Produces per-block, per-sample scalar gates to control the relative contribution
of text cross-attention vs motion-condition cross-attention. The gates are
conditioned on the adapter embedding (timestep + vtxt), so the model can learn
to rely more on text at high noise (structure/semantics) and more on motion
condition at low noise (fine detail preservation).

Architecture:
    gate_mlp = Linear(feat_dim, feat_dim) → SiLU → Linear(feat_dim, 2) → Sigmoid
    Output: (text_gate, motion_gate), each a scalar in [0, 1] per sample.

Design choices:
    - Sigmoid output (not softmax): text and motion gates are independent.
      At t≈1 (high noise), both might be needed; at t≈0 (low noise), motion
      condition dominates while text fades. Independence is crucial.
    - Zero-init on final linear: at initialization, sigmoid(0)=0.5, so both
      cross-attention branches contribute equally. This preserves pretrained
      backbone behavior when first added.
    - Adapter input (timestep_feat + vtxt_feat, shape B×1×feat_dim): same
      signal that drives ModulateDiT in existing blocks, so gates co-adapt
      with the existing modulation without extra conditioning cost.
    - Per-block instantiation: each DualCondMMDiTBlock has its own gate MLP,
      allowing different layers to learn different fusion strategies (early
      layers may favor text for global structure, later layers may favor
      motion condition for local detail).

Usage in DualCondMMDiTBlock:
    text_gate, motion_gate = self.fusion_gate(adapter)
    fused = text_gate * text_cross_attn_out + motion_gate * motion_cross_attn_out
    x = x + fused
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .bricks import get_activation_layer


class TimestepAdaptiveFusionGate(nn.Module):
    """Per-block adaptive gate for text/motion-condition fusion.

    Args:
        feat_dim: Input feature dimension (must match adapter dim, e.g. 1024).
        hidden_dim: Hidden dimension of the gate MLP. Default: same as feat_dim.
        act_type: Activation type for hidden layer (default 'silu').
        init_bias: Initial bias for output linear. Default 0.0 → sigmoid(0)=0.5,
            meaning equal contribution from both branches at initialization.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_dim: int | None = None,
        act_type: str = 'silu',
        init_bias: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=True),
            get_activation_layer(act_type)(),
            nn.Linear(hidden_dim, 2, bias=True),
        )

        # Zero-init: at start, output = 0 → sigmoid(0) = 0.5
        # Both branches contribute equally, preserving pretrained behavior.
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, init_bias)

    def forward(self, adapter: Tensor) -> tuple[Tensor, Tensor]:
        """Compute text and motion-condition gates from adapter embedding.

        Args:
            adapter: (B, 1, feat_dim) — adapter embedding (timestep + vtxt).

        Returns:
            text_gate: (B, 1, 1) — scalar gate for text cross-attention output.
            motion_gate: (B, 1, 1) — scalar gate for motion-cond cross-attention output.
        """
        # adapter: (B, 1, feat_dim) → squeeze → MLP → (B, 2)
        x = adapter.squeeze(1)  # (B, feat_dim)
        gates = torch.sigmoid(self.gate_mlp(x))  # (B, 2)

        text_gate = gates[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
        motion_gate = gates[:, 1:2].unsqueeze(-1)  # (B, 1, 1)

        return text_gate, motion_gate


class DensityAwareFusionGate(TimestepAdaptiveFusionGate):
    """Extended gate that also conditions on mask density.

    When mask density is high (most of the motion is masked, i.e. more
    generation needed), the model should rely more on text condition.
    When density is low (mostly known motion), motion condition dominates.

    This variant adds a density projection that is added to the adapter
    before the gate MLP, providing an explicit density signal.

    Args:
        feat_dim: Input feature dimension.
        hidden_dim: Hidden dimension of gate MLP.
        act_type: Activation type.
        init_bias: Initial output bias.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_dim: int | None = None,
        act_type: str = 'silu',
        init_bias: float = 0.0,
    ):
        super().__init__(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            act_type=act_type,
            init_bias=init_bias,
        )
        # Density projection: scalar density → feat_dim, added to adapter
        self.density_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, feat_dim),
        )
        # Zero-init so density has no effect at start
        nn.init.zeros_(self.density_proj[2].weight)
        nn.init.zeros_(self.density_proj[2].bias)

    def forward(
        self,
        adapter: Tensor,
        mask_density: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute gates with optional density conditioning.

        Args:
            adapter: (B, 1, feat_dim) — adapter embedding.
            mask_density: (B,) — fraction of masked dims, in [0, 1].
                None → falls back to base behavior (no density conditioning).

        Returns:
            text_gate: (B, 1, 1)
            motion_gate: (B, 1, 1)
        """
        x = adapter.squeeze(1)  # (B, feat_dim)

        if mask_density is not None:
            density_emb = self.density_proj(mask_density.unsqueeze(-1))  # (B, feat_dim)
            x = x + density_emb

        gates = torch.sigmoid(self.gate_mlp(x))  # (B, 2)

        text_gate = gates[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
        motion_gate = gates[:, 1:2].unsqueeze(-1)  # (B, 1, 1)

        return text_gate, motion_gate
