"""
Hunyuan Motion Multi-Modal Diffusion Transformer (MMDiT) Implementation.

This module implements a multi-modal diffusion transformer architecture specifically
designed for motion generation conditioned on text inputs. The architecture follows
a hybrid design with both double-stream and single-stream transformer blocks.

Key Terminology:
    - ctxt (Context Text): Token-level text embeddings from a language model (e.g., T5, CLIP).
      These are fine-grained, per-token features with high dimensionality (default 4096D).
      Each text token has its own feature vector, enabling detailed semantic understanding.

    - vtxt (Vector Text): Sentence-level or global text embeddings representing the entire
      text description as a single vector. Lower dimensionality (default 256D) and used
      as a conditioning signal combined with timestep embeddings to form the 'adapter'.

Architecture Overview:
    1. Input encoders project motion, ctxt, and vtxt to a common feature dimension
    2. Double-stream blocks: Process motion and text in parallel streams with joint attention
    3. Single-stream blocks: Concatenate motion and text, process jointly
    4. Final layer: Projects features back to motion output dimension

The model supports various attention masking modes (causal, narrowband) and optional
features like rotary position embeddings, element-wise attention gating, and long
skip connections.
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
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder, FinalLayer
from .modulate import ModulateDiT, apply_gate, modulate
from .positional_encoding import RotaryEmbedding
from .token_refiner import SingleTokenRefiner


def get_module_device(module):
    return next(module.parameters()).device


class MMDoubleStreamBlock(nn.Module):
    """
    Multi-Modal Double Stream Transformer Block.

    This block processes motion and text features in two parallel streams, each with
    its own normalization, QKV projection, and MLP layers. The two streams share
    attention computation by concatenating their Q, K, V tensors and performing
    joint attention, allowing cross-modal interaction.

    Architecture:
        Motion Stream:
            - Layer Norm -> Modulation -> QKV Projection -> Joint Attention -> Output Projection
            - Layer Norm -> Modulation -> MLP

        Text Stream:
            - Layer Norm -> Modulation -> QKV Projection -> Joint Attention -> Output Projection
            - Layer Norm -> Modulation -> MLP

    The modulation is conditioned on the 'adapter' signal (timestep + vtxt embeddings),
    following the DiT (Diffusion Transformer) paradigm with shift, scale, and gate parameters.

    Attributes:
        feat_dim (int): Hidden feature dimension for both streams.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension per attention head (feat_dim // num_heads).
        mlp_hidden_dim (int): Hidden dimension of MLP layers.
        rotary_emb (RotaryEmbedding): Rotary position embedding module.
        motion_mod (ModulateDiT): Modulation layer for motion stream.
        text_mod (ModulateDiT): Modulation layer for text stream.
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
        apply_rope_to_single_branch: bool = True,
        elementwise_attn_output_gate: bool = False,
    ):
        """
        Initialize the MMDoubleStreamBlock.

        Args:
            feat_dim (int): The hidden feature dimension. Must be divisible by num_heads.
            num_heads (int): Number of attention heads for multi-head attention.
            mlp_ratio (float): Ratio to compute MLP hidden dimension (mlp_hidden = feat_dim * mlp_ratio).
            dropout (float): Dropout probability applied during attention (only during training).
            mlp_act_type (str): Activation function type for MLP (e.g., 'gelu', 'gelu_tanh', 'silu').
            qk_norm_type (Optional[str]): Normalization type for Q and K projections (e.g., 'rms', 'layer').
                If None, identity normalization is used.
            qkv_bias (bool): Whether to include bias in QKV and output projection layers.
            positional_encoding_cfg (dict): Configuration for rotary position embeddings.
                - max_seq_len (int): Maximum sequence length for position encoding.
                - use_real (bool): Whether to use real-valued rotary embeddings.
            apply_rope_to_single_branch (bool): If True, apply RoPE only to motion branch.
                If False, apply RoPE to the concatenated motion+text sequence.
            elementwise_attn_output_gate (bool): If True, use element-wise gating on attention output
                by computing an additional gate vector alongside Q, K, V.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads

        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch

        self.motion_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.motion_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        # Adjust output dimension based on gate flag
        motion_qkv_out_dim = self.feat_dim * (4 if self.elementwise_attn_output_gate else 3)
        self.motion_qkv = nn.Linear(self.feat_dim, motion_qkv_out_dim, bias=qkv_bias)

        self.motion_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.motion_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.motion_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

        self.text_mod = ModulateDiT(
            self.feat_dim,
            factor=6,
            act_type="silu",
        )
        self.text_norm1 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        # Adjust output dimension based on gate flag
        text_qkv_out_dim = self.feat_dim * (4 if self.elementwise_attn_output_gate else 3)
        self.text_qkv = nn.Linear(self.feat_dim, text_qkv_out_dim, bias=qkv_bias)

        self.text_q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_out_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=qkv_bias)
        self.text_norm2 = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = MLP(
            self.feat_dim,
            self.mlp_hidden_dim,
            act_type=mlp_act_type,
            bias=True,
        )

    def forward(
        self,
        motion_feat: Tensor,
        text_feat: Tensor,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: bool = False,
        attn_collector: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the double stream block.

        The forward pass performs the following steps:
        1. Generate modulation parameters (shift, scale, gate) for both streams from adapter
        2. Apply layer norm and modulation to motion features, compute Q, K, V (and optional gate)
        3. Apply layer norm and modulation to text features, compute Q, K, V (and optional gate)
        4. Concatenate motion and text Q, K, V and perform joint attention
        5. Split attention output back into motion and text portions
        6. Apply residual connections with gating for both attention and MLP outputs

        Args:
            motion_feat (Tensor): Motion sequence features of shape (B, L_motion, D).
            text_feat (Tensor): Text sequence features of shape (B, L_text, D).
            adapter (Tensor): Conditioning signal (timestep + vtxt) of shape (B, 1, D).
            attn_mask (Optional[Tensor]): Attention mask of shape (B, 1, L_total, L_total)
                where L_total = L_motion + L_text. Negative infinity values mask out positions.
            return_attn (bool): If True, return attention weights for visualization/analysis.
            attn_collector (Optional[List[Tensor]]): If provided, append attention weights to this list.

        Returns:
            Tuple[Tensor, Tensor]: Updated (motion_feat, text_feat) both with shape (B, L, D).
        """
        # Generate 6 modulation parameters for motion stream:
        # shift_msa, scale_msa, gate_msa for attention; shift_mlp, scale_mlp, gate_mlp for MLP
        (
            motion_shift_msa,
            motion_scale_msa,
            motion_gate_msa,
            motion_shift_mlp,
            motion_scale_mlp,
            motion_gate_mlp,
        ) = self.motion_mod(adapter).chunk(6, dim=-1)
        # Generate 6 modulation parameters for text stream
        (
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = self.text_mod(
            adapter
        ).chunk(6, dim=-1)

        # ============ Motion Stream Processing ============
        # Apply layer normalization followed by adaptive modulation (shift and scale)
        motion_modulated = self.motion_norm1(motion_feat)
        motion_modulated = modulate(motion_modulated, shift=motion_shift_msa, scale=motion_scale_msa)
        # Project to Q, K, V (and optionally gate G) for attention
        motion_qkv = self.motion_qkv(motion_modulated)

        # Reshape QKV tensor: (B, L, K*H*D) -> (K, B, L, H, D) where K=3 or 4
        if self.elementwise_attn_output_gate:
            # When using element-wise gating, we have Q, K, V, G (4 components)
            motion_q, motion_k, motion_v, motion_g = rearrange(
                motion_qkv, "B L (K H D) -> K B L H D", K=4, H=self.num_heads
            )
        else:
            # Standard Q, K, V (3 components)
            motion_q, motion_k, motion_v = rearrange(motion_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            motion_g = None

        # Apply Q/K normalization (e.g., RMSNorm) for training stability
        motion_q = self.motion_q_norm(motion_q).to(motion_v)
        motion_k = self.motion_k_norm(motion_k).to(motion_v)

        if self.apply_rope_to_single_branch:
            # Apply Rotary Position Embedding (RoPE) only to motion branch
            # Text branch doesn't get RoPE as it has different positional semantics
            motion_q, motion_k = self.rotary_emb.apply_rotary_emb(motion_q, motion_k)

        # ============ Text Stream Processing ============
        # Apply layer normalization followed by adaptive modulation (shift and scale)
        text_modulated = self.text_norm1(text_feat)
        text_modulated = modulate(text_modulated, shift=text_shift_msa, scale=text_scale_msa)
        # Project to Q, K, V (and optionally gate G) for attention
        text_qkv = self.text_qkv(text_modulated)

        # Reshape text QKV tensor
        if self.elementwise_attn_output_gate:
            text_q, text_k, text_v, text_g = rearrange(
                text_qkv,
                "B L (K H D) -> K B L H D",
                K=4,
                H=self.num_heads,
            )
        else:
            text_q, text_k, text_v = rearrange(
                text_qkv,
                "B L (K H D) -> K B L H D",
                K=3,
                H=self.num_heads,
            )
            text_g = None

        # Apply Q/K normalization for text stream
        text_q = self.text_q_norm(text_q).to(text_v)
        text_k = self.text_k_norm(text_k).to(text_v)

        # ============ Joint Attention Computation ============
        # Concatenate motion and text Q, K, V along sequence dimension for joint attention
        # This allows motion tokens to attend to text tokens and vice versa
        q = torch.cat((motion_q, text_q), dim=1)  # (B, L_motion + L_text, H, D)
        k = torch.cat((motion_k, text_k), dim=1)
        v = torch.cat((motion_v, text_v), dim=1)

        # Concatenate gates if using element-wise attention gating
        if self.elementwise_attn_output_gate:
            assert motion_g is not None and text_g is not None, "motion_g and text_g must not be None"
            g = torch.cat((motion_g, text_g), dim=1)
        else:
            g = None

        # Alternative: Apply RoPE to concatenated sequence (if not applied to single branch)
        if not self.apply_rope_to_single_branch:
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len, _, _ = q.shape
        motion_len = motion_feat.shape[1]
        text_len = text_feat.shape[1]
        # Disable dropout during inference
        dropout_p = 0.0 if not self.training else self.dropout

        # Compute scaled dot-product attention over concatenated motion+text sequence
        ret = attention(
            q,
            k,
            v,
            mode="torch",  # TODO: support flash mode later
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,  # Non-causal attention (bidirectional)
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
            return_attn=return_attn,
            gate=g,  # Element-wise gate for attention output
        )

        # Handle return value: attention may return (output, weights) or just output
        if isinstance(ret, tuple):
            attn_output, attn_w = ret
            if attn_collector is not None:
                attn_collector.append(attn_w.detach())
        else:
            attn_output = ret

        # Split attention output back into motion and text portions
        motion_attn_output, text_attn_output = (
            attn_output[:, :motion_len, ...],
            attn_output[:, motion_len:, ...],
        )

        # ============ Motion Stream Residual Updates ============
        # Residual connection for attention with gating: x = x + gate * proj(attn_out)
        motion_feat = motion_feat + apply_gate(self.motion_out_proj(motion_attn_output), gate=motion_gate_msa)
        # Residual connection for MLP with modulation and gating
        motion_feat = motion_feat + apply_gate(
            self.motion_mlp(
                modulate(
                    self.motion_norm2(motion_feat),
                    shift=motion_shift_mlp,
                    scale=motion_scale_mlp,
                )
            ),
            gate=motion_gate_mlp,
        )

        # ============ Text Stream Residual Updates ============
        # Residual connection for attention with gating
        text_feat = text_feat + apply_gate(self.text_out_proj(text_attn_output), gate=text_gate_msa)
        # Residual connection for MLP with modulation and gating
        text_feat = text_feat + apply_gate(
            self.text_mlp(
                modulate(
                    self.text_norm2(text_feat),
                    shift=text_shift_mlp,
                    scale=text_scale_mlp,
                )
            ),
            gate=text_gate_mlp,
        )

        return motion_feat, text_feat


class MMSingleStreamBlock(nn.Module):
    """
    Multi-Modal Single Stream Transformer Block.

    Unlike the double stream block, this block processes the concatenated motion+text
    sequence as a single unified stream. It uses a fused linear layer that computes
    QKV and MLP input simultaneously for efficiency.

    Architecture:
        - Layer Norm -> Modulation
        - Fused Linear1: Compute QKV (for attention) + MLP hidden (for feedforward)
        - Parallel paths:
            a) QKV -> Q/K norm -> RoPE -> Attention
            b) MLP hidden -> Activation
        - Concatenate attention output and MLP activation
        - Linear2: Project back to feature dimension
        - Residual connection with gating

    This design is more parameter-efficient than double stream as it shares all layers
    between motion and text, but may have less capacity for modality-specific processing.

    Attributes:
        feat_dim (int): Hidden feature dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension per attention head.
        mlp_hidden_dim (int): Hidden dimension of MLP (feat_dim * mlp_ratio).
        linear1 (nn.Linear): Fused projection for QKV + MLP input.
        linear2 (nn.Linear): Fused projection for attention output + MLP output.
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
        apply_rope_to_single_branch: bool = True,
        elementwise_attn_output_gate: bool = False,
    ):
        """
        Initialize the MMSingleStreamBlock.

        Args:
            feat_dim (int): The hidden feature dimension. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio to compute MLP hidden dimension.
            dropout (float): Dropout probability for attention (training only).
            mlp_act_type (str): Activation type for MLP (e.g., 'gelu', 'gelu_tanh', 'silu').
            qk_norm_type (Optional[str]): Normalization type for Q and K ('rms', 'layer', None).
            qkv_bias (bool): Whether to use bias in linear projections.
            positional_encoding_cfg (dict): Configuration for rotary embeddings.
            apply_rope_to_single_branch (bool): If True, apply RoPE only to motion portion.
            elementwise_attn_output_gate (bool): If True, compute additional gate vector for attention.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        assert self.feat_dim % num_heads == 0, f"feat_dim {self.feat_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = self.feat_dim // num_heads

        self.mlp_hidden_dim = int(self.feat_dim * mlp_ratio)

        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(num_feats=self.head_dim, **self._positional_encoding_cfg)
        self.apply_rope_to_single_branch = apply_rope_to_single_branch

        self.modulation = ModulateDiT(self.feat_dim, factor=3, act_type="silu")
        self.norm = get_norm_layer(norm_type="layer")(self.feat_dim, elementwise_affine=False, eps=1e-6)

        # qkv and mlp_in
        qkv_factor = 4 if self.elementwise_attn_output_gate else 3
        self.linear1 = nn.Linear(self.feat_dim, self.feat_dim * qkv_factor + self.mlp_hidden_dim, bias=qkv_bias)
        # proj and mlp_out
        self.linear2 = nn.Linear(self.feat_dim + self.mlp_hidden_dim, self.feat_dim, bias=qkv_bias)

        self.q_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.k_norm = get_norm_layer(qk_norm_type)(self.head_dim, elementwise_affine=True, eps=1e-6)

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
        self,
        x: Tensor,
        split_len: int,
        adapter: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: bool = False,
        attn_collector: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Forward pass of the single stream block.

        Args:
            x (Tensor): Concatenated motion+text features of shape (B, L_motion + L_text, D).
            split_len (int): Length of motion sequence. Used to split x into motion/text portions
                for applying RoPE only to the motion part when apply_rope_to_single_branch=True.
            adapter (Tensor): Conditioning signal (timestep + vtxt) of shape (B, 1, D).
            attn_mask (Optional[Tensor]): Attention mask of shape (B, 1, L, L).
            return_attn (bool): If True, collect attention weights.
            attn_collector (Optional[List[Tensor]]): List to append attention weights.

        Returns:
            Tensor: Updated concatenated motion+text features of shape (B, L_total, D).
        """
        # Generate modulation parameters: shift, scale, gate for the combined stream
        (
            shift_msa,
            scale_msa,
            gate_msa,
        ) = self.modulation(
            adapter
        ).chunk(3, dim=-1)

        # Apply layer norm followed by adaptive modulation
        x_modulated = modulate(self.norm(x), shift_msa, scale_msa)

        # Fused linear layer computes QKV for attention AND MLP hidden state simultaneously
        # This is more efficient than separate projections
        if self.elementwise_attn_output_gate:
            # Split into QKV+Gate and MLP hidden
            qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [4 * self.feat_dim, self.mlp_hidden_dim], dim=-1)
            q, k, v, g = rearrange(qkv, "B L (K H D) -> K B L H D", K=4, H=self.num_heads)
        else:
            # Split into QKV and MLP hidden
            qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [3 * self.feat_dim, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            g = None

        # Apply Q/K normalization for stable training
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Split Q and K into motion (q1, k1) and text (q2, k2) portions
        q1, q2 = q[:, :split_len, ...], q[:, split_len:, ...]
        k1, k2 = k[:, :split_len, ...], k[:, split_len:, ...]

        # Apply Rotary Position Embedding (RoPE)
        if self.apply_rope_to_single_branch:
            # Apply RoPE only to motion portion, text portion keeps original position info
            q1, k1 = self.rotary_emb.apply_rotary_emb(q1, k1)
        q = torch.cat((q1, q2), dim=1)
        k = torch.cat((k1, k2), dim=1)
        if not self.apply_rope_to_single_branch:
            # Alternative: Apply RoPE to entire concatenated sequence
            q, k = self.rotary_emb.apply_rotary_emb(q, k)

        bsz, total_len = x_modulated.shape[:2]
        dropout_p = 0.0 if not self.training else self.dropout

        # Compute scaled dot-product attention
        ret = attention(
            q,
            k,
            v,
            mode="torch",  # TODO: support flash mode later
            drop_rate=dropout_p,
            attn_mask=attn_mask,
            causal=False,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            batch_size=bsz,
            training=self.training,
            return_attn=return_attn,
            gate=g,
        )

        # Handle optional attention weight return
        if isinstance(ret, tuple):
            attn_output, attn_w = ret
            if attn_collector is not None:
                attn_collector.append(attn_w.detach())
        else:
            attn_output = ret

        # Concatenate attention output with activated MLP hidden, then project back
        # This fused design: [attn_out || mlp_act(mlp_hidden)] -> linear2 -> output
        output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp_hidden)), 2))

        # Residual connection with gating: x = x + gate * output
        return x + apply_gate(output, gate=gate_msa)


class HunyuanMotionMMDiT(nn.Module):
    """
    Hunyuan Motion Multi-Modal Diffusion Transformer (MMDiT).

    This is the main model class for text-conditioned motion generation using a diffusion
    transformer architecture. It combines:

    1. **Input Encoders**: Project motion, context text (ctxt), and vector text (vtxt) to
       a common hidden dimension.
    2. **Double Stream Blocks**: Process motion and text in parallel streams with joint attention,
       allowing cross-modal interaction while maintaining modality-specific parameters.
    3. **Single Stream Blocks**: Concatenate motion and text into a single sequence and process
       with shared parameters for efficiency.
    4. **Final Layer**: Project the output features back to motion dimension.

    Text Conditioning:
        - ctxt (Context Text): Token-level embeddings from language model (e.g., T5-XXL).
          Shape: (B, L_text, ctxt_input_dim). Rich semantic features per token.
        - vtxt (Vector Text): Global sentence embedding (e.g., CLIP embedding).
          Shape: (B, 1, vtxt_input_dim). Compact representation of entire description.
          Combined with timestep embedding to form the 'adapter' for modulation.

    Architecture Diagram:
        Motion Input ─→ Input Encoder ─┐
                                       ├─→ Double Stream Blocks (×N/3) ─→ Single Stream Blocks (×2N/3) ─→ Final Layer ─→ Output
        Text Input ────→ Text Encoder ─┤
                                       │
        vtxt + Timestep ─→ Adapter ────┘

    Attributes:
        motion_input_dim (int): Input dimension of motion features.
        ctxt_input_dim (int): Input dimension of context text features.
        vtxt_input_dim (int): Input dimension of vector text features.
        feat_dim (int): Hidden feature dimension throughout the network.
        output_dim (int): Output dimension (default same as input_dim).
        double_blocks (nn.ModuleList): List of MMDoubleStreamBlock modules.
        single_blocks (nn.ModuleList): List of MMSingleStreamBlock modules.
    """

    def __init__(
        self,
        input_dim: int,
        feat_dim: int,
        output_dim: Optional[int] = None,
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_module: str = "hymotion/network/token_refiner.SingleTokenRefiner",
        text_refiner_cfg: dict = {
            "num_layers": 2,
        },
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_module: str = "hymotion/network/encoders.FinalLayer",
        final_layer_cfg: dict = {
            "act_type": "silu",
        },
        mask_mode: Optional[str] = None,
        apply_rope_to_single_branch: bool = True,
        insert_start_token: bool = False,
        with_long_skip_connection: bool = False,
        time_factor: float = 1.0,
        narrowband_length: float = 2.0,
        elementwise_attn_output_gate: bool = False,
        **kwargs,
    ):
        """
        Initialize the HunyuanMotionMMDiT model.

        Args:
            input_dim (int): Dimension of input motion features (e.g., joint rotations, positions).
            feat_dim (int): Hidden dimension for all transformer blocks.
            output_dim (Optional[int]): Output dimension. Defaults to input_dim if not specified.
            ctxt_input_dim (int): Dimension of context text embeddings (default 4096 for T5-XXL).
            vtxt_input_dim (int): Dimension of vector text embeddings (default 256).
            text_refiner_module (str): Module path for text refiner (processes ctxt before attention).
            text_refiner_cfg (dict): Configuration for text refiner module.
            num_layers (int): Total number of transformer layers. Must be divisible by 3.
                The first 1/3 are double stream blocks, remaining 2/3 are single stream blocks.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): MLP hidden dimension ratio (mlp_hidden = feat_dim * mlp_ratio).
            mlp_act_type (str): Activation function for MLP layers.
            norm_type (str): Normalization type for layers (e.g., 'layer', 'rms').
            qk_norm_type (str): Normalization type for Q/K projections.
            qkv_bias (bool): Whether to use bias in QKV projections.
            dropout (float): Dropout rate for attention (training only).
            final_layer_module (str): Module path for final output layer.
            final_layer_cfg (dict): Configuration for final layer.
            mask_mode (Optional[str]): Attention mask mode:
                - None: No additional masking (full attention)
                - 'causal': Causal/autoregressive masking
                - 'narrowband': Local attention with window size = narrowband_length
            apply_rope_to_single_branch (bool): Apply RoPE only to motion branch (not text).
            insert_start_token (bool): Prepend a learnable start token to motion sequence.
            with_long_skip_connection (bool): Add skip connection from input to output.
            time_factor (float): Scaling factor for timestep embeddings.
            narrowband_length (float): Window size in seconds for narrowband attention.
                Converted to frames: window = narrowband_length * 30 (assuming 30fps).
            elementwise_attn_output_gate (bool): Use element-wise gating on attention outputs.
            **kwargs: Additional unused arguments for compatibility.
        """
        super().__init__()
        # Store input/output dimensions
        self.motion_input_dim = input_dim
        self.ctxt_input_dim = ctxt_input_dim
        self.vtxt_input_dim = vtxt_input_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or input_dim

        # Attention mask configuration
        self.mask_mode = mask_mode
        self.insert_start_token = insert_start_token
        self.time_factor = time_factor
        # Convert narrowband_length from seconds to frames (assuming 30fps)
        self.narrowband_length = narrowband_length * 30.0
        self.elementwise_attn_output_gate = elementwise_attn_output_gate

        # Learnable start token for autoregressive-style generation
        if self.insert_start_token:
            self.start_token = nn.Parameter(torch.randn(1, feat_dim))

        # Long skip connection: adds residual from input directly to output
        self.with_long_skip_connection = with_long_skip_connection
        if self.with_long_skip_connection:
            self.long_skip_net = FinalLayer(feat_dim=feat_dim, out_dim=feat_dim, act_type="silu")

        # ============ Input Encoders ============
        # Project motion features to hidden dimension
        self.input_encoder = nn.Linear(in_features=input_dim, out_features=feat_dim)
        # Project context text (ctxt) token embeddings to hidden dimension
        self.ctxt_encoder = nn.Linear(in_features=ctxt_input_dim, out_features=feat_dim)
        # Project vector text (vtxt) global embedding to hidden dimension using MLP
        self.vtxt_encoder = MLPEncoder(in_dim=vtxt_input_dim, feat_dim=feat_dim, num_layers=2, act_type="silu")
        # Encode diffusion timestep using sinusoidal embeddings + MLP
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim,
            feat_dim=feat_dim,
            time_factor=time_factor,
        )

        # ============ Optional Text Refiner ============
        # Refines context text embeddings before cross-attention (e.g., self-attention over text tokens)
        if text_refiner_module != "" and text_refiner_module is not None:
            text_refiner_cfg.update(input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads)
            self._text_refiner_cfg = text_refiner_cfg.copy()
            self.text_refiner = SingleTokenRefiner(**text_refiner_cfg)

        # ============ Transformer Block Configuration ============
        self.num_layers = num_layers
        # Split layers: 1/3 double stream blocks, 2/3 single stream blocks
        assert num_layers % 3 == 0, f"num_layers must be divisible by 3, but got {num_layers}"
        self.mm_double_blocks_layers = int(num_layers // 3)
        self.mm_single_blocks_layers = int(num_layers - num_layers // 3)

        # ============ Double Stream Blocks ============
        # Process motion and text in parallel streams with joint attention
        # Each modality has separate normalization, QKV, and MLP layers
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                    elementwise_attn_output_gate=elementwise_attn_output_gate,
                )
                for _ in range(self.mm_double_blocks_layers)
            ]
        )

        # ============ Single Stream Blocks ============
        # Process concatenated motion+text sequence with shared parameters
        # More parameter-efficient but less modality-specific capacity
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    feat_dim=feat_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    apply_rope_to_single_branch=apply_rope_to_single_branch,
                    elementwise_attn_output_gate=elementwise_attn_output_gate,
                )
                for _ in range(self.mm_single_blocks_layers)
            ]
        )

        # ============ Final Layer ============
        # Project hidden features back to motion output dimension
        final_layer_cfg.update(feat_dim=feat_dim, out_dim=self.output_dim)
        self._final_layer_cfg = final_layer_cfg.copy()
        self.final_layer = FinalLayer(**final_layer_cfg)


    def forward(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
        pre_encoded_motion: Optional[Tensor] = None,
        task_emb: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of the HunyuanMotionMMDiT model.

        This method performs the full diffusion transformer forward pass for motion denoising:
        1. Encode all inputs (motion, ctxt, vtxt, timestep) to hidden dimension
        2. Create adapter signal (timestep + vtxt + task instruction) for modulation
        3. Optionally refine context text embeddings
        4. Build attention masks based on mask_mode
        5. Process through double stream blocks (motion + text parallel)
        6. Process through single stream blocks (motion + text concatenated)
        7. Apply optional long skip connection
        8. Project to output dimension via final layer

        Args:
            x (Tensor): Noisy motion input of shape (B, L_motion, input_dim).
                Can be None when pre_encoded_motion is provided.
            ctxt_input (Tensor): Context text embeddings (token-level) of shape (B, L_text, ctxt_input_dim).
                These are typically from a language model like T5-XXL.
            vtxt_input (Tensor): Vector text embeddings (sentence-level) of shape (B, 1, vtxt_input_dim).
                These are typically from CLIP or similar models.
            timesteps (Tensor): Diffusion timesteps of shape (B,). Integer values representing noise level.
            x_mask_temporal (Tensor): Boolean mask for motion sequence of shape (B, L_motion).
                True indicates valid positions, False indicates padding.
            ctxt_mask_temporal (Tensor): Boolean mask for text sequence of shape (B, L_text).
                True indicates valid tokens, False indicates padding.
            pre_encoded_motion (Optional[Tensor]): Pre-encoded motion features of shape
                (B, L_motion, feat_dim). When provided, bypasses input_encoder and uses
                these features directly. Used by UMO-style temporal fusion where the
                caller fuses E_in(x_t) + E_ctx(source) externally. Default: None.
            task_emb (Optional[Tensor]): Task instruction embeddings for M2M task awareness of shape (B, 1, 1024).
                These are CLIP-encoded natural language task descriptions (e.g. "complete motion from sparse random cells")
                that are added to the adapter signal to provide explicit task modulation.
                When provided, injected into adapter = timestep_feat + vtxt_feat + task_emb.
                Default: None.
            **kwargs: Additional unused arguments for compatibility.

        Returns:
            Tensor: Predicted noise/residual of shape (B, L_motion, output_dim).
        """
        device = get_module_device(self)

        # ============ Encode Motion Input ============
        if pre_encoded_motion is not None:
            # UMO-style: caller already fused E_in(x_t) + E_ctx(source)
            motion_feat = pre_encoded_motion
        else:
            motion_feat = self.input_encoder(x)
        # Store original features for long skip connection if enabled
        if self.with_long_skip_connection:
            origin_feat = motion_feat

        # Optionally prepend learnable start token for autoregressive-style generation
        if self.insert_start_token:
            # (B, 1, D) + (B, L, D) -> (B, L+1, D)
            start_token = self.start_token[None].repeat(motion_feat.shape[0], 1, 1)
            motion_feat = torch.cat((start_token, motion_feat), dim=1)
            # Update mask to include start token (always valid)
            x_mask_temporal = torch.cat(
                [
                    torch.ones_like(x_mask_temporal[:, :1], dtype=torch.bool),
                    x_mask_temporal,
                ],
                dim=1,
            )

        # ============ Encode Conditioning Signals ============
        # Encode diffusion timestep using sinusoidal embeddings
        timestep_feat = self.timestep_encoder(timesteps)
        # Encode global text vector (vtxt) using MLP
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())
        # Combine timestep and vtxt to form the adapter signal for modulation
        # adapter is broadcast to all sequence positions via ModulateDiT layers
        adapter = timestep_feat + vtxt_feat
        if task_emb is not None:
            adapter = adapter + task_emb

        # ============ Build Attention Masks ============
        # Convert boolean masks to attention-compatible format (0 for valid, -inf for masked)
        motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
        seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)

        # Build sequence mask based on mask_mode
        if self.mask_mode is None:
            # Full attention: all tokens can attend to all other tokens
            seq_mask = None
        elif self.mask_mode == "causal":
            # Causal/autoregressive mask: tokens can only attend to previous positions
            motion_len = motion_feat.shape[1]
            seq_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            # Narrowband/local attention: tokens attend within a fixed window
            window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window  # True if within window
            seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            seq_mask = seq_mask.masked_fill(band, 0.0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        # ============ Encode Context Text ============
        ctxt_feat = self.ctxt_encoder(ctxt_input.float())
        # Optionally refine text embeddings with self-attention layers
        if hasattr(self, "text_refiner"):
            ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))

        # ============ Precompute Attention Masks ============
        # Build attention masks once and reuse across all blocks for efficiency
        bsz = motion_feat.shape[0]
        motion_len = motion_feat.shape[1]
        text_len = ctxt_feat.shape[1]
        total_len = motion_len + text_len
        mask_dtype = motion_feat.dtype

        # Build attention mask for double stream blocks
        # Shape: (B, 1, total_len, total_len), broadcastable over heads
        attn_mask_double = self._build_dmm_attn_mask_shared(
            bsz=bsz,
            motion_len=motion_len,
            text_len=text_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )

        # ============ Double Stream Blocks ============
        # Process motion and text in parallel with joint attention
        for i_layer, mod in enumerate(self.double_blocks):
            motion_feat, ctxt_feat = mod(
                motion_feat=motion_feat,
                text_feat=ctxt_feat,
                adapter=adapter,
                attn_mask=attn_mask_double,
            )

        # ============ Single Stream Blocks ============
        # Concatenate motion and text for joint processing
        split_len = motion_feat.shape[1]
        x = torch.cat((motion_feat, ctxt_feat), 1)

        # Build attention mask for single stream blocks
        attn_mask_single = self._build_smm_attn_mask_shared(
            bsz=bsz,
            split_len=split_len,
            total_len=total_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )

        # Process through single stream blocks
        for i_layer, mod in enumerate(self.single_blocks):
            x = mod(
                x=x,
                split_len=split_len,
                adapter=adapter,
                attn_mask=attn_mask_single,
            )

        # ============ Extract Motion Output ============
        # Only keep motion portion, discard text portion
        x = x[:, :split_len, ...]
        # Remove start token if it was inserted
        if self.insert_start_token:
            x = x[:, 1:, ...]

        # ============ Long Skip Connection ============
        # Add residual from input encoder output, modulated by timestep only
        if self.with_long_skip_connection:
            x = self.long_skip_net(origin_feat, timestep_feat) + x

        # ============ Final Layer ============
        # Project to output dimension with adapter modulation
        predicted_res = self.final_layer(x, adapter)
        return predicted_res

    def forward_with_attn(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
    ):
        """
        Forward pass with attention weight collection for visualization/analysis.

        Similar to the regular forward pass, but also returns attention weights
        from all transformer blocks. Useful for debugging, interpretability analysis,
        and visualizing how the model attends to different parts of the input.

        Note: This method does not support insert_start_token or long_skip_connection
        for simplicity. Use the regular forward method for full functionality.

        Args:
            x (Tensor): Noisy motion input of shape (B, L_motion, input_dim).
            ctxt_input (Tensor): Context text embeddings of shape (B, L_text, ctxt_input_dim).
            vtxt_input (Tensor): Vector text embeddings of shape (B, 1, vtxt_input_dim).
            timesteps (Tensor): Diffusion timesteps of shape (B,).
            x_mask_temporal (Tensor): Motion sequence mask of shape (B, L_motion).
            ctxt_mask_temporal (Tensor): Text sequence mask of shape (B, L_text).

        Returns:
            Tuple containing:
                - predicted_res (Tensor): Predicted output of shape (B, L_motion, output_dim).
                - attn_collector (List[Tensor]): List of attention weight tensors from each block.
                - motion_len (int): Length of motion sequence.
                - text_len (int): Length of text sequence.
        """
        device = get_module_device(self)
        motion_feat = self.input_encoder(x)
        timestep_feat = self.timestep_encoder(timesteps)
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())
        adapter = timestep_feat + vtxt_feat

        motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
        seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)

        if self.mask_mode is None:
            seq_mask = None
        elif self.mask_mode == "causal":
            motion_len = motion_feat.shape[1]
            seq_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device),
                diagonal=1,
            )
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            motion_len = motion_feat.shape[1]
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window
            seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
            seq_mask = seq_mask.masked_fill(band, 0.0)
        else:
            raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

        ctxt_feat = self.ctxt_encoder(ctxt_input.float())
        if hasattr(self, "text_refiner"):
            ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))

        bsz = x.shape[0]
        motion_len = motion_feat.shape[1]
        text_len = ctxt_feat.shape[1]
        total_len = motion_len + text_len
        mask_dtype = motion_feat.dtype

        attn_collector = []

        # Double blocks
        attn_mask_double = self._build_dmm_attn_mask_shared(
            bsz=bsz,
            motion_len=motion_len,
            text_len=text_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for mod in self.double_blocks:
            motion_feat, ctxt_feat = mod(
                motion_feat=motion_feat,
                text_feat=ctxt_feat,
                adapter=adapter,
                attn_mask=attn_mask_double,
                return_attn=True,
                attn_collector=attn_collector,
            )

        # Single blocks
        split_len = motion_feat.shape[1]
        x_all = torch.cat((motion_feat, ctxt_feat), 1)
        attn_mask_single = self._build_smm_attn_mask_shared(
            bsz=bsz,
            split_len=split_len,
            total_len=total_len,
            dtype=mask_dtype,
            key_padding_mask=seq_key_padding_mask,
            attn_mask=seq_mask,
            device=device,
        )
        for mod in self.single_blocks:
            x_all = mod(
                x=x_all,
                split_len=split_len,
                adapter=adapter,
                attn_mask=attn_mask_single,
                return_attn=True,
                attn_collector=attn_collector,
            )

        x_out = x_all[:, :split_len, ...]
        predicted_res = self.final_layer(x_out, adapter)
        return predicted_res, attn_collector, motion_len, text_len

    @staticmethod
    def _canonical_mask(input_mask: Tensor) -> Tensor:
        """
        Convert a boolean padding mask to attention-compatible format.

        Transforms a boolean mask (True=valid, False=padding) into an additive
        attention mask (0=valid, -inf=masked). This format is compatible with
        scaled dot-product attention where masked positions become -inf after
        adding to attention logits, resulting in zero attention weights after softmax.

        Args:
            input_mask (Tensor): Boolean mask of shape (B, L) or (L,).
                True indicates valid positions, False indicates padding.

        Returns:
            Tensor: Attention mask of shape (B, L) with values 0 (valid) or -inf (masked).
        """
        if input_mask.ndim == 1:
            input_mask = input_mask.unsqueeze(1)
        # Convert: True -> 0.0 (valid), False -> -inf (masked)
        key_padding_mask = torch.where(
            input_mask,
            torch.zeros_like(input_mask, dtype=torch.float),
            torch.full_like(input_mask, float("-inf"), dtype=torch.float),
        )
        return key_padding_mask

    @staticmethod
    def _visible_from_mask(mask: Tensor) -> Tensor:
        """Return a boolean visibility mask from bool or additive masks."""
        if mask.dtype == torch.bool:
            return mask
        return torch.isfinite(mask)

    @staticmethod
    def _ensure_nonempty_attention_rows(mask: Tensor) -> Tensor:
        """Make fully masked query rows attend to themselves to avoid NaNs."""
        all_false = ~mask.any(dim=-1)
        total_len = mask.size(-1)
        diag = torch.eye(
            total_len,
            dtype=torch.bool,
            device=mask.device,
        ).view(1, 1, total_len, total_len)
        return torch.where(
            all_false.unsqueeze(-1).expand_as(mask),
            diag.expand_as(mask),
            mask,
        )

    def _build_dmm_attn_mask_shared(
        self,
        bsz: int,
        motion_len: int,
        text_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """
        Build attention mask for Double-stream Multi-Modal (DMM) blocks.

        Constructs a combined attention mask for joint motion-text attention with
        specific cross-modal attention constraints:

        Attention Matrix Layout:
                        motion_k    text_k
            motion_q   [M→M]       [M→T]
            text_q     [T→M]       [T→T]

        Attention Patterns:
            - M→M (motion to motion): Applies the sequence mask (causal/narrowband)
            - M→T (motion to text): Motion can attend to all valid text tokens
            - T→M (text to motion): BLOCKED (-inf) - text cannot attend to motion
            - T→T (text to text): Full attention between text tokens

        The T→M blocking prevents text representations from being influenced by
        noisy motion during the diffusion process, maintaining clean text features.

        Args:
            bsz (int): Batch size.
            motion_len (int): Length of motion sequence.
            text_len (int): Length of text sequence.
            dtype (torch.dtype): Data type for the mask tensor.
            key_padding_mask (Optional[Tensor]): Key padding mask of shape (B, L).
            attn_mask (Optional[Tensor]): Sequence attention mask of shape (motion_len, motion_len).
            device (torch.device): Device to create tensor on.

        Returns:
            Tensor: Boolean combined attention mask of shape
            (B, 1, total_len, total_len), where True means visible.
        """
        total_len = motion_len + text_len
        base = torch.ones((bsz, 1, total_len, total_len), dtype=torch.bool, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.shape != (motion_len, motion_len):
                raise RuntimeError(
                    f"attn_mask should be 2D with shape {(motion_len, motion_len)}, got {attn_mask.shape}"
                )
            motion_visible = self._visible_from_mask(attn_mask).to(device)
            base[:, :, :motion_len, :motion_len] &= motion_visible.view(1, 1, motion_len, motion_len)
        if key_padding_mask is not None:
            key_visible = self._visible_from_mask(key_padding_mask).to(device)
            mask_total_len = key_visible.shape[1]
            if mask_total_len == motion_len:
                pad = torch.ones((bsz, text_len), dtype=torch.bool, device=device)
                key_visible = torch.cat((key_visible, pad), dim=-1)
            base &= key_visible.view(bsz, 1, 1, total_len)
        # disable T→M
        base[:, :, motion_len:, :motion_len] = False
        return self._ensure_nonempty_attention_rows(base)

    def _build_smm_attn_mask_shared(
        self,
        bsz: int,
        split_len: int,
        total_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """
        Build attention mask for Single-stream Multi-Modal (SMM) blocks.

        Similar to _build_dmm_attn_mask_shared, but for single stream blocks where
        motion and text are concatenated into a single sequence. Uses split_len
        to determine the boundary between motion and text portions.

        Attention Matrix Layout:
                        motion_k    text_k
            motion_q   [M→M]       [M→T]
            text_q     [T→M]       [T→T]

        Attention Patterns (same as DMM):
            - M→M: Applies sequence mask (causal/narrowband)
            - M→T: Motion attends to all valid text tokens
            - T→M: BLOCKED - text cannot attend to motion
            - T→T: Full attention between text tokens

        Args:
            bsz (int): Batch size.
            split_len (int): Position where motion ends and text begins in concatenated sequence.
            total_len (int): Total sequence length (motion_len + text_len).
            dtype (torch.dtype): Data type for the mask tensor.
            key_padding_mask (Optional[Tensor]): Key padding mask of shape (B, L).
            attn_mask (Optional[Tensor]): Sequence attention mask of shape (split_len, split_len).
            device (torch.device): Device to create tensor on.

        Returns:
            Tensor: Boolean combined attention mask of shape
            (B, 1, total_len, total_len), where True means visible.
        """
        # Initialize base mask with True (all positions can attend)
        base = torch.ones((bsz, 1, total_len, total_len), dtype=torch.bool, device=device)
        if attn_mask is not None:
            if attn_mask.dim() != 2 or attn_mask.shape != (split_len, split_len):
                raise RuntimeError(f"attn_mask should be 2D with shape {(split_len, split_len)}, got {attn_mask.shape}")
            motion_visible = self._visible_from_mask(attn_mask).to(device)
            base[:, :, :split_len, :split_len] &= motion_visible.view(1, 1, split_len, split_len)
        if key_padding_mask is not None:
            key_visible = self._visible_from_mask(key_padding_mask).to(device)
            mask_total_len = key_visible.shape[1]
            if mask_total_len == split_len:
                pad = torch.zeros(
                    (bsz, total_len - split_len),
                    dtype=torch.bool,
                    device=device,
                )
                pad.fill_(True)
                key_visible = torch.cat((key_visible, pad), dim=-1)
            base &= key_visible.view(bsz, 1, 1, total_len)
        # disable T→M
        base[:, :, split_len:, :split_len] = False
        return self._ensure_nonempty_attention_rows(base)

    def params_count(self):
        """
        Count and print model parameters breakdown by component.

        Computes parameter counts for:
            - Text refiner (if present)
            - Double stream blocks (separate motion/text branches)
            - Single stream blocks (shared parameters)
            - Final layer
            - Extra components (encoders, modulations)

        Only prints on rank 0 in distributed settings to avoid duplicate output.

        Returns:
            dict: Dictionary with parameter counts:
                - 'refiner': Text refiner parameters
                - 'double': Double stream block parameters
                - 'single': Single stream block parameters
                - 'final': Final layer parameters
                - 'total': Total model parameters
                - 'attn+mlp': Combined attention and MLP parameters
        """
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            counts = {
                "refiner": (
                    sum(p.numel() for p in self.text_refiner.parameters()) if hasattr(self, "text_refiner") else 0
                ),
                "double": sum(
                    [
                        sum(p.numel() for p in block.motion_qkv.parameters())
                        + sum(p.numel() for p in block.motion_out_proj.parameters())
                        + sum(p.numel() for p in block.motion_mlp.parameters())
                        + sum(p.numel() for p in block.text_qkv.parameters())
                        + sum(p.numel() for p in block.text_out_proj.parameters())
                        + sum(p.numel() for p in block.text_mlp.parameters())
                        for block in self.double_blocks
                    ]
                ),
                "single": sum(
                    [
                        sum(p.numel() for p in block.linear1.parameters())
                        + sum(p.numel() for p in block.linear2.parameters())
                        for block in self.single_blocks
                    ]
                ),
                "final": sum(p.numel() for p in self.final_layer.parameters()),
                "total": sum(p.numel() for p in self.parameters()),
            }
            extra_mod = sum(
                [
                    sum(p.numel() for p in block.motion_mod.parameters())
                    + sum(p.numel() for p in block.text_mod.parameters())
                    for block in self.double_blocks
                ]
            ) + sum([sum(p.numel() for p in block.modulation.parameters()) for block in self.single_blocks])
            extra_enc = (
                sum(p.numel() for p in self.input_encoder.parameters())
                + sum(p.numel() for p in self.ctxt_encoder.parameters())
                + sum(p.numel() for p in self.vtxt_encoder.parameters())
                + sum(p.numel() for p in self.timestep_encoder.parameters())
            )
            counts["attn+mlp"] = counts["double"] + counts["single"] + counts["refiner"]
            print(f"Extra encoders parameters: {extra_enc/1e9:.2f}B")
            print(f"Attn+mlp parameters: {counts['attn+mlp'] / 1e9:.2f}B")
            print(f"Modulations parameters: {extra_mod/1e9:.2f}B")
            print(f"Final layer parameters: {counts['final'] / 1e9:.2f}B")
            print(f"Total parameters: {counts['total'] / 1e9:.2f}B")
            return counts

    def flops_count(self, motion_seq_len: int = 200, text_seq_len: int = 256, batch_size: int = 1):
        """
        Estimate and print FLOPs (Floating Point Operations) breakdown by component.

        Provides approximate FLOPs calculation for a forward pass, useful for:
            - Comparing model efficiency across configurations
            - Understanding computational bottlenecks
            - Hardware resource planning

        FLOPs are computed for:
            - Input encoders (linear projections)
            - Text refiner (self-attention + MLP)
            - Double stream blocks (per-modality attention + MLP)
            - Single stream blocks (joint attention + MLP)
            - Final layer (output projection)

        Note: This is an approximate calculation and may not account for all
        operations (e.g., activation functions, layer norms, softmax).

        Args:
            motion_seq_len (int): Length of motion sequence for estimation.
            text_seq_len (int): Length of text sequence for estimation.
            batch_size (int): Batch size multiplier for total FLOPs.

        Returns:
            dict: Dictionary with GFLOPs (10^9 FLOPs) for each component:
                - 'encoders': Input encoder FLOPs
                - 'refiner': Text refiner FLOPs
                - 'double': Double stream block FLOPs
                - 'single': Single stream block FLOPs
                - 'final': Final layer FLOPs
                - 'total': Total model FLOPs
        """
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            flops = {}

            # 编码器FLOPs
            # input_encoder: motion_seq_len * input_dim * feat_dim
            input_encoder_flops = motion_seq_len * self.motion_input_dim * self.feat_dim
            # ctxt_encoder: text_seq_len * ctxt_input_dim * feat_dim
            ctxt_encoder_flops = text_seq_len * self.ctxt_input_dim * self.feat_dim

            # vtxt_encoder: MLP编码器，假设2层
            vtxt_encoder_flops = self.vtxt_input_dim * self.feat_dim + self.feat_dim * self.feat_dim

            # timestep_encoder: embedding + MLP
            timestep_encoder_flops = self.feat_dim * self.feat_dim

            encoder_flops = input_encoder_flops + ctxt_encoder_flops + vtxt_encoder_flops + timestep_encoder_flops
            flops["encoders"] = encoder_flops

            # Text Refiner FLOPs
            if hasattr(self, "text_refiner"):
                refiner_flops = 0
                refiner_layers = getattr(self, "_text_refiner_cfg", {}).get("num_layers", 0)
                for _ in range(refiner_layers):
                    # Self-attention: Q,K,V计算 + attention计算 + output projection
                    qkv_flops = text_seq_len * self.feat_dim * (self.feat_dim * 3)
                    attn_flops = text_seq_len * text_seq_len * self.feat_dim  # attention计算
                    proj_flops = text_seq_len * self.feat_dim * self.feat_dim
                    mlp_flops = (
                        text_seq_len * self.feat_dim * (self.feat_dim * 4)
                        + text_seq_len * (self.feat_dim * 4) * self.feat_dim
                    )
                    refiner_flops += qkv_flops + attn_flops + proj_flops + mlp_flops
                flops["refiner"] = refiner_flops
            else:
                flops["refiner"] = 0

            # Double Stream Blocks FLOPs
            double_flops = 0
            total_seq_len = motion_seq_len + text_seq_len

            # Determine QKV factor
            qkv_factor = 4 if self.elementwise_attn_output_gate else 3

            for _ in range(self.mm_double_blocks_layers):
                # Motion branch
                motion_qkv_flops = motion_seq_len * self.feat_dim * (self.feat_dim * qkv_factor)
                motion_attn_flops = motion_seq_len * total_seq_len * self.feat_dim
                motion_proj_flops = motion_seq_len * self.feat_dim * self.feat_dim
                motion_mlp_flops = (
                    motion_seq_len * self.feat_dim * int(self.feat_dim * 4.0)
                    + motion_seq_len * int(self.feat_dim * 4.0) * self.feat_dim
                )

                # Text branch
                text_qkv_flops = text_seq_len * self.feat_dim * (self.feat_dim * qkv_factor)
                text_attn_flops = text_seq_len * total_seq_len * self.feat_dim
                text_proj_flops = text_seq_len * self.feat_dim * self.feat_dim
                text_mlp_flops = (
                    text_seq_len * self.feat_dim * int(self.feat_dim * 4.0)
                    + text_seq_len * int(self.feat_dim * 4.0) * self.feat_dim
                )

                # Modulation FLOPs
                motion_mod_flops = self.feat_dim * (self.feat_dim * 6)  # ModulateDiT
                text_mod_flops = self.feat_dim * (self.feat_dim * 6)

                block_flops = (
                    motion_qkv_flops
                    + motion_attn_flops
                    + motion_proj_flops
                    + motion_mlp_flops
                    + text_qkv_flops
                    + text_attn_flops
                    + text_proj_flops
                    + text_mlp_flops
                    + motion_mod_flops
                    + text_mod_flops
                )
                double_flops += block_flops

            flops["double"] = double_flops

            # Single Stream Blocks FLOPs
            single_flops = 0
            for _ in range(self.mm_single_blocks_layers):
                # QKV + MLP input计算
                linear1_flops = total_seq_len * self.feat_dim * (self.feat_dim * qkv_factor + int(self.feat_dim * 4.0))

                # Attention计算
                attn_flops = total_seq_len * total_seq_len * self.feat_dim

                # Output projection + MLP output
                linear2_flops = total_seq_len * (self.feat_dim + int(self.feat_dim * 4.0)) * self.feat_dim

                # Modulation
                mod_flops = self.feat_dim * (self.feat_dim * 3)

                block_flops = linear1_flops + attn_flops + linear2_flops + mod_flops
                single_flops += block_flops

            flops["single"] = single_flops

            # Final Layer FLOPs
            final_flops = motion_seq_len * self.feat_dim * self.output_dim
            flops["final"] = final_flops

            flops["total"] = sum(flops.values())
            for key in flops:
                flops[key] *= batch_size

            gflops = {k: v / 1e9 for k, v in flops.items()}
            print(f"Encoders GFLOPs: {gflops['encoders']:.2f}G")
            print(f"Text Refiner GFLOPs: {gflops['refiner']:.2f}G")
            print(f"Double Blocks GFLOPs: {gflops['double']:.2f}G")
            print(f"Single Blocks GFLOPs: {gflops['single']:.2f}G")
            print(f"Final Layer GFLOPs: {gflops['final']:.2f}G")
            print(f"Total GFLOPs: {gflops['total']:.2f}G")

            return gflops


def visualize_mask(mask, motion_len, title=""):
    """
    Visualize attention mask as a heatmap image.

    Creates a visualization of the attention mask matrix, showing which positions
    can attend to which. Useful for debugging and understanding mask patterns
    (causal, narrowband, cross-modal blocking, etc.).

    The visualization includes:
        - Heatmap of mask values (0 = can attend, -inf shown as NaN/white)
        - White lines indicating boundary between motion and text regions
        - Colorbar showing mask value scale

    Args:
        mask (Tensor): Attention mask of shape (B, H, L, L) or (B, 1, L, L).
            Takes the first batch and first head for visualization.
        motion_len (int): Length of motion sequence, used to draw boundary lines.
        title (str): Title for the plot.

    Side Effects:
        Saves the visualization to 'output/test/test.png'.
    """
    import matplotlib.pyplot as plt

    # Extract first batch, first head; convert -inf to NaN for visualization
    m = mask[0, 0].detach().float()
    m = torch.where(torch.isinf(m), torch.full_like(m, float("nan")), m)

    # Create heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(m.cpu(), cmap="viridis", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Draw boundary lines between motion and text regions
    plt.axvline(motion_len - 0.5, color="w", lw=1)
    plt.axhline(motion_len - 0.5, color="w", lw=1)

    plt.xlabel("key index")
    plt.ylabel("query index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output/test/test.png")


if __name__ == "__main__":
    # python -m hymotion.network.hymotion_mmdit

    from configs._base_.model_network_base import MOTION_MODEL_CONFIG  # pyright: ignore

    network_module_cfg = MOTION_MODEL_CONFIG["1.04B_narrowband"]["network_module_args"]
    network_module_cfg = dict(network_module_cfg)  # 转普通dict

    bsz, seq_len, text_seq_len, input_dim = 1, 360, 128, 201
    network_module_cfg["input_dim"] = input_dim
    MMDiT = HunyuanMotionMMDiT(**network_module_cfg)
    print("=== 参数统计 ===")
    MMDiT.params_count()
    print("\n=== FLOPs统计 ===")
    MMDiT.flops_count(motion_seq_len=seq_len, text_seq_len=text_seq_len, batch_size=bsz)

    x = torch.randn(bsz, seq_len, input_dim)
    ctxt_condition = torch.randn(bsz, text_seq_len, 4096)
    vtxt_condition = torch.randn(bsz, 1, 768)
    timesteps = torch.randint(0, 1000, (bsz,))
    length = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1)
    ctxt_length = torch.arange(text_seq_len).unsqueeze(0).repeat(bsz, 1)
    x_mask_temporal = length < 100
    ctxt_mask_temporal = ctxt_length < 50
    x = MMDiT(
        x=x,
        ctxt_input=ctxt_condition,
        vtxt_input=vtxt_condition,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
    assert x.shape == (
        bsz,
        seq_len,
        input_dim,
    ), f"unexpected output shape: {x.shape}, which should be ({bsz}, {seq_len}, {input_dim})"
    print(x.shape)
