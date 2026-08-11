"""
Hunyuan Motion Diffusion Transformer (DiT) — Text-Free Implementation.

A simplified variant of HunyuanMotionMMDiT that removes ALL text-related modules:
  - ctxt_encoder (Linear 4096->feat_dim)
  - vtxt_encoder (MLPEncoder 768->feat_dim)
  - text_refiner (SingleTokenRefiner, self-attention over text tokens)
  - Text branch in double-stream blocks (text_mod, text_qkv, text_norms, text_mlp)
  - Text tokens in single-stream blocks (split_len / concat logic)
  - T->M blocking in attention masks

For unconditioned motion-to-motion editing, text modules are never used
(the trainer passes null embeddings throughout). Removing them:
  - Saves ~180M params (460M -> ~280M for same feat_dim=1024)
  - Reduces FLOPs by ~40% (no text branch computation)
  - Simplifies attention masks (no cross-modal blocking)

Architecture:
  Motion Input -> Input Encoder -> Transformer Blocks (x N) -> Final Layer -> Output
  Timestep -> Timestep Encoder -> Adapter (for adaLN modulation)

Each block uses the efficient fused single-stream design:
  - Modulate(factor=3): shift, scale, gate
  - Fused Linear1: QKV + MLP_hidden
  - Self-Attention with RoPE
  - Fused Output: [attn_out || mlp_act(mlp_hidden)] -> Linear2
  - Residual + Gate

The network accepts the same forward interface as HunyuanMotionMMDiT
(ctxt_input, vtxt_input args) for trainer/pipeline compatibility, but
ignores text inputs entirely.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .attention import attention
from .bricks import get_activation_layer, get_norm_layer
from .encoders import MLP, FinalLayer, TimestepEmbeddingEncoder
from .modulate import ModulateDiT, apply_gate, modulate
from .positional_encoding import RotaryEmbedding


def get_module_device(module):
    return next(module.parameters()).device


class DiTBlock(nn.Module):
    """Single-stream transformer block for motion-only DiT.

    Efficient fused design (matches MMSingleStreamBlock minus text logic):
      1. Modulation (shift, scale, gate) from adapter
      2. Fused Linear1 -> QKV + MLP_hidden
      3. Self-Attention with RoPE + optional narrowband mask
      4. [attn_out || mlp_act(mlp_hidden)] -> Linear2
      5. Residual + Gate

    Parameters
    ----------
    feat_dim : int
        Hidden feature dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dim = feat_dim * mlp_ratio.
    dropout : float
        Attention dropout (training only).
    mlp_act_type : str
        MLP activation type.
    qk_norm_type : str or None
        QK normalization type ('rms', 'layer', None).
    qkv_bias : bool
        Bias in linear projections.
    positional_encoding_cfg : dict
        Config for RotaryEmbedding.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        assert feat_dim % num_heads == 0, (
            f"feat_dim {feat_dim} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = feat_dim // num_heads
        self.mlp_hidden_dim = int(feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(
            num_feats=self.head_dim, **self._positional_encoding_cfg
        )

        # Modulation: 3 factors (shift, scale, gate)
        self.modulation = ModulateDiT(feat_dim, factor=3, act_type="silu")
        self.norm = get_norm_layer(norm_type="layer")(
            feat_dim, elementwise_affine=False, eps=1e-6
        )

        # Fused QKV + MLP_hidden projection
        self.linear1 = nn.Linear(
            feat_dim, feat_dim * 3 + self.mlp_hidden_dim, bias=qkv_bias
        )
        # Fused attn_out + MLP_out projection
        self.linear2 = nn.Linear(
            feat_dim + self.mlp_hidden_dim, feat_dim, bias=qkv_bias
        )

        self.q_norm = get_norm_layer(qk_norm_type)(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )
        self.k_norm = get_norm_layer(qk_norm_type)(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
        self,
        x: Tensor,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: (B, L, D) motion features.
            adapter: (B, 1, D) timestep conditioning.
            attn_mask: (B, 1, L, L) attention mask.

        Returns:
            (B, L, D) updated motion features.
        """
        shift_msa, scale_msa, gate_msa = self.modulation(adapter).chunk(3, dim=-1)

        x_mod = modulate(self.norm(x), shift_msa, scale_msa)

        # Fused QKV + MLP
        qkv, mlp_hidden = torch.split(
            self.linear1(x_mod),
            [3 * self.feat_dim, self.mlp_hidden_dim],
            dim=-1,
        )
        q, k, v = rearrange(
            qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
        )

        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE to all positions (no text split needed)
        q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz = x.shape[0]
        dropout_p = 0.0 if not self.training else self.dropout

        attn_output = attention(
            q, k, v,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
        )

        output = self.linear2(
            torch.cat((attn_output, self.mlp_act(mlp_hidden)), dim=2)
        )
        return x + apply_gate(output, gate=gate_msa)


class HunyuanMotionDiT(nn.Module):
    """Text-free Hunyuan Motion Diffusion Transformer.

    A simplified DiT architecture for unconditioned motion-to-motion editing.
    All text-related modules are removed. The model uses a uniform stack of
    DiTBlock layers (efficient fused single-stream design).

    Parameters
    ----------
    input_dim : int
        Input dimension (motion + VACE context, e.g. 540).
    feat_dim : int
        Hidden dimension throughout the network.
    output_dim : int or None
        Output dimension (default: input_dim).
    num_layers : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension ratio.
    mlp_act_type : str
        MLP activation type.
    qk_norm_type : str
        QK normalization type.
    qkv_bias : bool
        Whether to use bias in linear projections.
    dropout : float
        Attention dropout rate.
    final_layer_cfg : dict
        Config for the FinalLayer.
    mask_mode : str or None
        Attention mask mode ('narrowband', 'causal', None).
    time_factor : float
        Timestep scaling factor.
    narrowband_length : float
        Narrowband window in seconds (converted to frames at 30fps).
    with_long_skip_connection : bool
        Add residual from input to output.
    """

    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        output_dim: Optional[int] = None,
        # Text-related args accepted but IGNORED for config compatibility
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_module: str = "",
        text_refiner_cfg: dict = {},
        # Core architecture
        num_layers: int = 18,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_module: str = "",
        final_layer_cfg: dict = {"act_type": "silu"},
        mask_mode: Optional[str] = None,
        apply_rope_to_single_branch: bool = True,  # ignored, kept for compat
        insert_start_token: bool = False,
        with_long_skip_connection: bool = False,
        time_factor: float = 1.0,
        narrowband_length: float = 2.0,
        elementwise_attn_output_gate: bool = False,  # ignored (not needed w/o text)
        **kwargs,
    ):
        super().__init__()
        self.motion_input_dim = input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or input_dim
        self.mask_mode = mask_mode
        self.insert_start_token = insert_start_token
        self.time_factor = time_factor
        self.narrowband_length = narrowband_length * 30.0
        self.num_layers = num_layers

        if self.insert_start_token:
            self.start_token = nn.Parameter(torch.randn(1, feat_dim))

        self.with_long_skip_connection = with_long_skip_connection
        if self.with_long_skip_connection:
            self.long_skip_net = FinalLayer(
                feat_dim=feat_dim, out_dim=feat_dim, act_type="silu"
            )

        # ============ Input Encoder ============
        self.input_encoder = nn.Linear(
            in_features=input_dim, out_features=feat_dim
        )

        # ============ Timestep Encoder ============
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim,
            feat_dim=feat_dim,
            time_factor=time_factor,
        )

        # ============ Transformer Blocks ============
        self.blocks = nn.ModuleList([
            DiTBlock(
                feat_dim=feat_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                mlp_act_type=mlp_act_type,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
            )
            for _ in range(num_layers)
        ])

        # ============ Final Layer ============
        _final_cfg = final_layer_cfg.copy()
        _final_cfg.update(feat_dim=feat_dim, out_dim=self.output_dim)
        self.final_layer = FinalLayer(**_final_cfg)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        # Text args accepted but IGNORED (for trainer/pipeline compatibility)
        ctxt_input: Optional[Tensor] = None,
        vtxt_input: Optional[Tensor] = None,
        ctxt_mask_temporal: Optional[Tensor] = None,
        pre_encoded_motion: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Accepts the same interface as HunyuanMotionMMDiT for compatibility
        with the existing trainer and pipeline. Text inputs (ctxt_input,
        vtxt_input, ctxt_mask_temporal) are accepted but completely ignored.

        Args:
            x: Noisy motion input (B, L, input_dim).
            timesteps: Diffusion timesteps (B,).
            x_mask_temporal: Boolean mask (B, L), True=valid.
            ctxt_input: IGNORED. Kept for interface compatibility.
            vtxt_input: IGNORED. Kept for interface compatibility.
            ctxt_mask_temporal: IGNORED. Kept for interface compatibility.
            pre_encoded_motion: Pre-encoded motion features (B, L, feat_dim).

        Returns:
            Predicted output (B, L, output_dim).
        """
        device = get_module_device(self)

        # Encode motion
        if pre_encoded_motion is not None:
            motion_feat = pre_encoded_motion
        else:
            motion_feat = self.input_encoder(x)

        if self.with_long_skip_connection:
            origin_feat = motion_feat

        if self.insert_start_token:
            start_token = self.start_token[None].repeat(
                motion_feat.shape[0], 1, 1
            )
            motion_feat = torch.cat((start_token, motion_feat), dim=1)
            x_mask_temporal = torch.cat(
                [
                    torch.ones_like(
                        x_mask_temporal[:, :1], dtype=torch.bool
                    ),
                    x_mask_temporal,
                ],
                dim=1,
            )

        # Encode timestep -> adapter (no text component)
        adapter = self.timestep_encoder(timesteps)

        # Build attention mask (motion-only, no cross-modal complexity)
        motion_len = motion_feat.shape[1]
        bsz = motion_feat.shape[0]
        mask_dtype = motion_feat.dtype

        if self.mask_mode is None:
            seq_mask = None
        elif self.mask_mode == "causal":
            seq_mask = torch.triu(
                torch.full(
                    (motion_len, motion_len), float("-inf"), device=device
                ),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            idx = torch.arange(motion_len, device=device)
            dist_mat = (idx[None, :] - idx[:, None]).abs()
            band = dist_mat <= window
            seq_mask = torch.full(
                (motion_len, motion_len), float("-inf"), device=device
            )
            seq_mask = seq_mask.masked_fill(band, 0.0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        # Build final attention mask: (B, 1, L, L)
        attn_mask = self._build_attn_mask(
            bsz=bsz,
            seq_len=motion_len,
            dtype=mask_dtype,
            x_mask_temporal=x_mask_temporal,
            seq_mask=seq_mask,
            device=device,
        )

        # Run through transformer blocks
        for block in self.blocks:
            motion_feat = block(
                x=motion_feat,
                adapter=adapter,
                attn_mask=attn_mask,
            )

        # Remove start token if inserted
        if self.insert_start_token:
            motion_feat = motion_feat[:, 1:, ...]

        # Long skip connection
        if self.with_long_skip_connection:
            timestep_feat = adapter  # adapter == timestep_feat here
            motion_feat = self.long_skip_net(origin_feat, timestep_feat) + motion_feat

        # Final layer
        return self.final_layer(motion_feat, adapter)

    def _build_attn_mask(
        self,
        bsz: int,
        seq_len: int,
        dtype: torch.dtype,
        x_mask_temporal: Tensor,
        seq_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """Build attention mask for motion-only self-attention.

        Returns (B, 1, L, L) mask with 0=attend, -inf=block.
        """
        base = torch.zeros(
            (bsz, 1, seq_len, seq_len), dtype=dtype, device=device
        )

        # Apply sequence mask (narrowband/causal)
        if seq_mask is not None:
            base = base + seq_mask.view(1, 1, seq_len, seq_len)

        # Apply padding mask
        key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        base = base + key_padding_mask.view(bsz, 1, 1, seq_len)

        return base

    @staticmethod
    def _canonical_mask(input_mask: Tensor) -> Tensor:
        """Convert boolean mask (True=valid) to additive mask (0=valid, -inf=masked)."""
        if input_mask.ndim == 1:
            input_mask = input_mask.unsqueeze(1)
        return torch.where(
            input_mask,
            torch.zeros_like(input_mask, dtype=torch.float),
            torch.full_like(input_mask, float("-inf"), dtype=torch.float),
        )

    def params_count(self):
        """Count and print model parameters breakdown."""
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            block_params = sum(
                sum(p.numel() for p in block.linear1.parameters())
                + sum(p.numel() for p in block.linear2.parameters())
                for block in self.blocks
            )
            mod_params = sum(
                sum(p.numel() for p in block.modulation.parameters())
                for block in self.blocks
            )
            enc_params = (
                sum(p.numel() for p in self.input_encoder.parameters())
                + sum(p.numel() for p in self.timestep_encoder.parameters())
            )
            final_params = sum(
                p.numel() for p in self.final_layer.parameters()
            )
            total_params = sum(p.numel() for p in self.parameters())

            counts = {
                "blocks_attn_mlp": block_params,
                "modulations": mod_params,
                "encoders": enc_params,
                "final": final_params,
                "total": total_params,
            }
            print(f"[HunyuanMotionDiT] Encoders: {enc_params/1e6:.1f}M")
            print(f"[HunyuanMotionDiT] Blocks (attn+mlp): {block_params/1e6:.1f}M")
            print(f"[HunyuanMotionDiT] Modulations: {mod_params/1e6:.1f}M")
            print(f"[HunyuanMotionDiT] Final layer: {final_params/1e6:.1f}M")
            print(f"[HunyuanMotionDiT] Total: {total_params/1e6:.1f}M ({total_params/1e9:.3f}B)")
            return counts
