from typing import List, Optional
from torch import nn
import torch
from .wan_resnet import WanResidualBlock1D, WanResidualBlock2DTK, CACHE_T
from .wan_resample import (
    WanResample1D,
    AvgDown1D,
    DupUp1D,
    WanResample2DTK,
    AvgDown2DTK,
    DupUp2DTK,
)
from .wan_causalconv import WanCausalConv1d, WanCausalConv2dTK
from .wan_norm import WanRMSNorm
from .wan_attention import (
    WanKWiseAttention,
    WanChannelLinearAttention,
    WanJointTokenAttention,
    WanTemporalAttention,
)
from diffusers.models.activations import get_activation


class WanUpBlock1D(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        dropout=0.0,
        upsample_mode: Optional[str] = None,
        upsample_out_dim: Optional[int] = None,
        non_linearity="silu",
    ):
        super().__init__()
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock1D(cur, out_dim, dropout, non_linearity))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            # For "upsample_channel", pass upsample_out_dim so next block gets expected channels (avoid dim//2)
            resample_kw = {}
            if upsample_mode == "upsample_channel":
                resample_kw["upsample_out_dim"] = (
                    upsample_out_dim if upsample_out_dim is not None else out_dim
                )
            self.upsamplers = nn.ModuleList(
                [WanResample1D(out_dim, mode=upsample_mode, **resample_kw)]
            )

    def forward(
        self, x: torch.Tensor, feat_cache=None, feat_idx=[0], first_chunk: bool = False
    ):
        for res in self.resnets:
            if feat_cache is not None:
                x = res(x, feat_cache, feat_idx)
            else:
                x = res(x)
        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class WanUpBlock2DTK(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        dropout=0.0,
        upsample_mode: Optional[str] = None,
        non_linearity="silu",
    ):
        super().__init__()
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock2DTK(cur, out_dim, dropout, non_linearity))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [WanResample2DTK(out_dim, mode=upsample_mode)]
            )

    def forward(
        self, x: torch.Tensor, feat_cache=None, feat_idx=[0], first_chunk: bool = False
    ):
        for res in self.resnets:
            if feat_cache is not None:
                x = res(x, feat_cache, feat_idx)
            else:
                x = res(x)
        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class WanMidBlockNoAttn1D(nn.Module):
    r"""
    Mid block without attention.

    This module stacks pure residual blocks only, mirroring the structure of WAN mid block
    but with all attention operations removed. It keeps the same cache protocol as
    `WanResidualBlock1D` (each causal conv consumes and updates the shared feat cache).

    Args:
        dim (int):     Channel width.
        dropout (float): Dropout probability inside residual blocks.
        non_linearity (str): Activation name (e.g., "silu").
        num_layers (int): Number of *extra* residual blocks after the first one.
                          When set to 1 (default), the block becomes: ResBlock -> ResBlock.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        # First residual block
        resnets = [WanResidualBlock1D(dim, dim, dropout, non_linearity)]
        # Additional residual blocks (no attention in-between)
        for _ in range(num_layers):
            resnets.append(WanResidualBlock1D(dim, dim, dropout, non_linearity))
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:

        x = self.resnets[0](x, feat_cache, feat_idx)

        for res in self.resnets[1:]:
            x = res(x, feat_cache, feat_idx)
        return x


# ---------------------------------------------------------------------------
# Mid blocks with attention (1D), same forward signature as WanMidBlockNoAttn1D
# ---------------------------------------------------------------------------
class WanMidBlockChannelLinearAttn1D(nn.Module):
    """Mid: channel-wise linear attention, O(C) cost."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        channel_proj_dim: int = 64,
    ):
        super().__init__()
        self.resnet0 = WanResidualBlock1D(dim, dim, dropout, non_linearity)
        self.attn = WanChannelLinearAttention(
            dim, proj_dim=channel_proj_dim, dropout=dropout
        )
        self.resnet1 = WanResidualBlock1D(dim, dim, dropout, non_linearity)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        x = self.resnet0(x, feat_cache, feat_idx)
        x = self.attn(x)
        x = self.resnet1(x, feat_cache, feat_idx)
        return x


class WanMidBlockJointTokenAttn1D(nn.Module):
    """Mid: aggregate by joint K into tokens, then K×K attention."""

    def __init__(
        self,
        dim: int,
        num_joints: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        token_dim: int = 64,
    ):
        super().__init__()
        assert dim % num_joints == 0
        self.resnet0 = WanResidualBlock1D(dim, dim, dropout, non_linearity)
        self.attn = WanJointTokenAttention(
            dim, num_joints=num_joints, token_dim=token_dim, dropout=dropout
        )
        self.resnet1 = WanResidualBlock1D(dim, dim, dropout, non_linearity)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        x = self.resnet0(x, feat_cache, feat_idx)
        x = self.attn(x)
        x = self.resnet1(x, feat_cache, feat_idx)
        return x


class WanMidBlockTemporalAttn1D(nn.Module):
    """Mid: temporal causal attention."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_heads: int = 1,
        temporal_window_size: Optional[int] = None,
    ):
        super().__init__()
        self.resnet0 = WanResidualBlock1D(dim, dim, dropout, non_linearity)
        self.attn = WanTemporalAttention(
            dim, num_heads=num_heads, dropout=dropout, window_size=temporal_window_size
        )
        self.resnet1 = WanResidualBlock1D(dim, dim, dropout, non_linearity)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        x = self.resnet0(x, feat_cache, feat_idx)
        x = self.attn(x)
        x = self.resnet1(x, feat_cache, feat_idx)
        return x


class WanMidBlock1DTK(nn.Module):
    """Mid: split channels into K parts, K-wise attention."""

    def __init__(
        self,
        dim: int,
        num_joints: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        assert dim % num_joints == 0
        self.channel_per_joint = dim // num_joints
        self.num_joints = num_joints
        self.resnet0 = WanResidualBlock1D(dim, dim, dropout, non_linearity)
        self.attn = WanKWiseAttention(self.channel_per_joint)
        self.resnet1 = WanResidualBlock1D(dim, dim, dropout, non_linearity)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        x = self.resnet0(x, feat_cache, feat_idx)
        B, dim, T = x.shape
        K, C = self.num_joints, self.channel_per_joint
        x = x.view(B, C, T, K)
        x = self.attn(x)
        x = x.view(B, dim, T)
        x = self.resnet1(x, feat_cache, feat_idx)
        return x


def _build_mid_block_1d(
    out_dim: int,
    mid_attention: str,
    dropout: float,
    non_linearity: str,
    num_joints: Optional[int] = None,
    channel_proj_dim: int = 128,
    joint_token_dim: Optional[int] = None,
    temporal_window_size: Optional[int] = 64,
) -> nn.Module:
    """Build encoder/decoder mid block: none -> WanMidBlockNoAttn1D; else one of the attention mid blocks."""
    if mid_attention == "linear_channel":
        return WanMidBlockChannelLinearAttn1D(
            out_dim, dropout, non_linearity, channel_proj_dim=channel_proj_dim
        )
    if mid_attention == "joint_tokens" and num_joints is not None and out_dim % num_joints == 0:
        _jt = joint_token_dim if joint_token_dim is not None else 64
        return WanMidBlockJointTokenAttn1D(
            out_dim, num_joints, dropout, non_linearity, token_dim=_jt
        )
    if mid_attention == "kwise" and num_joints is not None and out_dim % num_joints == 0:
        return WanMidBlock1DTK(out_dim, num_joints, dropout, non_linearity)
    if mid_attention == "temporal":
        return WanMidBlockTemporalAttn1D(
            out_dim, dropout, non_linearity, temporal_window_size=temporal_window_size
        )
    return WanMidBlockNoAttn1D(out_dim, dropout, non_linearity, num_layers=1)


class WanMidBlock2DTK(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock2DTK(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanKWiseAttention(dim))
            resnets.append(WanResidualBlock2DTK(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class WanResidualDownBlock1D(nn.Module):
    r"""
    WAN-style residual down block for 1D temporal sequences.

    This mirrors the 3D `WanResidualDownBlock` but removes spatial logic:
      - Shortcut branch: `AvgDown1D(in_dim -> out_dim, factor_t=2 if temporal_downsample else 1)`
      - Main branch:     `num_res_blocks` × `WanResidualBlock1D`, then optional time downsample
                         via `WanResample1D(mode="downsample1d")` when `temporal_downsample=True`.

    Causality & cross-chunk caching:
      - The main residual blocks and the optional downsampler take `(feat_cache, feat_idx)`,
        consuming cache slots in the same order as the 3D WAN implementation. The shortcut
        path is purely algebraic (rearrange + mean), so it does not use cache.

    Shape:
      Input:  x ∈ R[B, C_in, T]
      Output: y ∈ R[B, C_out, T']  where
              T' = T        if temporal_downsample == False
              T' = ⌊T/2⌋   if temporal_downsample == True
              (exact length matches the main branch output; shortcut matches via AvgDown1D)

    Args:
        in_dim (int):         input channels (C_in)
        out_dim (int):        output channels (C_out)
        dropout (float):      dropout prob used inside WanResidualBlock1D
        num_res_blocks (int): number of residual blocks in the main branch
        temporal_downsample (bool): if True, perform 2× temporal downsample (WAN uses factor_t=2)

    Notes:
        - `AvgDown1D` asserts: (in_dim * factor_t) % out_dim == 0.
          This mirrors the 3D block's divisibility constraint. Please ensure your channel
          schedule (dim_mult) satisfies this across levels.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temporal_downsample: bool = False,
    ):
        super().__init__()

        # ---- Shortcut path (no cache; purely algebraic) ----
        # In 3D: AvgDown3D(in_dim -> out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        # In 1D: only temporal exists, so we keep factor_t only.
        self.avg_shortcut = AvgDown1D(
            in_channels=in_dim,
            out_channels=out_dim,
            factor_t=2 if temporal_downsample else 1,
        )

        # ---- Main path: stacked residual blocks (each block uses causal conv + cache) ----
        resnets: List[nn.Module] = []
        cur = in_dim
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock1D(cur, out_dim, dropout))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)

        # ---- Optional temporal downsample at the end of main branch (causal + cache) ----
        # 3D WAN picks 'downsample3d' or 'downsample2d' depending on flags; in 1D we only have temporal.
        self.downsampler = (
            WanResample1D(out_dim, mode="downsample1d") if temporal_downsample else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        """
        Forward pass mirrors the 3D block:
          1) Save x_copy for the shortcut path (clone to avoid aliasing issues)
          2) Apply stacked residual blocks (consume cache slots)
          3) (Optional) temporal downsample (consume one cache slot)
          4) Add the averaged shortcut
        """
        # 1) keep a copy for the AvgDown path; clone for parity with 3D code (safer wrt autograd aliasing)
        x_copy = x.clone()

        # 2) main residual tower
        for res in self.resnets:
            x = res(x, feat_cache, feat_idx)

        # 3) optional temporal downsample (stride-2 causal conv with first-call skip inside WanResample1D)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)

        # 4) residual add with the averaged shortcut (sizes match because both paths
        #    either downsample by 2 in time or keep original length)
        return x + self.avg_shortcut(x_copy)


class WanResidualDownBlock2DTK(nn.Module):
    r"""
    WAN-style residual down block for 1D temporal sequences.

    This mirrors the 3D `WanResidualDownBlock` but removes spatial logic:
      - Shortcut branch: `AvgDown1D(in_dim -> out_dim, factor_t=2 if temporal_downsample else 1)`
      - Main branch:     `num_res_blocks` × `WanResidualBlock1D`, then optional time downsample
                         via `WanResample1D(mode="downsample1d")` when `temporal_downsample=True`.

    Causality & cross-chunk caching:
      - The main residual blocks and the optional downsampler take `(feat_cache, feat_idx)`,
        consuming cache slots in the same order as the 3D WAN implementation. The shortcut
        path is purely algebraic (rearrange + mean), so it does not use cache.

    Shape:
      Input:  x ∈ R[B, C_in, T, K]
      Output: y ∈ R[B, C_out, T', K]  where
              T' = T        if temporal_downsample == False
              T' = ⌊T/2⌋   if temporal_downsample == True
              (exact length matches the main branch output; shortcut matches via AvgDown1D)

    Args:
        in_dim (int):         input channels (C_in)
        out_dim (int):        output channels (C_out)
        dropout (float):      dropout prob used inside WanResidualBlock1D
        num_res_blocks (int): number of residual blocks in the main branch
        temporal_downsample (bool): if True, perform 2× temporal downsample (WAN uses factor_t=2)

    Notes:
        - `AvgDown1D` asserts: (in_dim * factor_t) % out_dim == 0.
          This mirrors the 3D block's divisibility constraint. Please ensure your channel
          schedule (dim_mult) satisfies this across levels.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temporal_downsample: bool = False,
    ):
        super().__init__()

        # ---- Shortcut path (no cache; purely algebraic) ----
        # In 3D: AvgDown3D(in_dim -> out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        # In 1D: only temporal exists, so we keep factor_t only.
        self.avg_shortcut = AvgDown2DTK(
            in_channels=in_dim,
            out_channels=out_dim,
            factor_t=2 if temporal_downsample else 1,
        )

        # ---- Main path: stacked residual blocks (each block uses causal conv + cache) ----
        resnets: List[nn.Module] = []
        cur = in_dim
        for _ in range(num_res_blocks):
            resnets.append(WanResidualDownBlock2DTK(cur, out_dim, dropout))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)

        # ---- Optional temporal downsample at the end of main branch (causal + cache) ----
        # 3D WAN picks 'downsample3d' or 'downsample2d' depending on flags; in 1D we only have temporal.
        self.downsampler = (
            WanResample2DTK(out_dim, mode="downsample1d")
            if temporal_downsample
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        """
        Forward pass mirrors the 3D block:
          1) Save x_copy for the shortcut path (clone to avoid aliasing issues)
          2) Apply stacked residual blocks (consume cache slots)
          3) (Optional) temporal downsample (consume one cache slot)
          4) Add the averaged shortcut
        """
        # 1) keep a copy for the AvgDown path; clone for parity with 3D code (safer wrt autograd aliasing)
        x_copy = x.clone()

        # 2) main residual tower
        for res in self.resnets:
            x = res(x, feat_cache, feat_idx)

        # 3) optional temporal downsample (stride-2 causal conv with first-call skip inside WanResample1D)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)

        # 4) residual add with the averaged shortcut (sizes match because both paths
        #    either downsample by 2 in time or keep original length)
        return x + self.avg_shortcut(x_copy)


class WanResidualUpBlock1D(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temporal_upsample: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if temporal_upsample:
            self.avg_shortcut = DupUp1D(in_dim, out_dim, factor_t=2)
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock1D(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if temporal_upsample:
            upsample_mode = "upsample1d"
            self.upsampler = WanResample1D(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache, feat_idx)
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanResidualUpBlock2DTK(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temporal_upsample: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if temporal_upsample:
            self.avg_shortcut = DupUp2DTK(in_dim, out_dim, factor_t=2)
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock2DTK(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if temporal_upsample:
            upsample_mode = "upsample1d"
            self.upsampler = WanResample2DTK(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache, feat_idx)
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanEncoder1D(nn.Module):
    """1D encoder, aligned with 2D structure. mid_attention: none / linear_channel / joint_tokens / kwise / temporal."""

    def __init__(
        self,
        in_channels: int = 135,
        dim: int = 128,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_downsample: List[bool] = [True, True, False],
        dropout: float = 0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
        mid_attention: str = "none",
        num_joints: Optional[int] = None,
        channel_proj_dim: int = 128,
        joint_token_dim: Optional[int] = None,
        temporal_window_size: Optional[int] = 64,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # Channel schedule: matches 3D version
        dims = [dim * u for u in [1] + dim_mult]
        out_dim = dims[-1]

        self.conv_in = WanCausalConv1d(in_channels, dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            last_stage = i == len(dim_mult) - 1

            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock1D(
                        in_dim=in_dim,
                        out_dim=out_d,
                        dropout=dropout,
                        num_res_blocks=num_res_blocks,
                        temporal_downsample=(
                            self.temporal_downsample[i] if not last_stage else False
                        ),
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(
                        WanResidualBlock1D(in_dim, out_d, dropout)
                    )
                    in_dim = out_d

                if i != len(dim_mult) - 1:
                    mode = (
                        "downsample1d"
                        if temporal_downsample[i]
                        else "downsample_channel"
                    )
                    self.down_blocks.append(WanResample1D(out_d, mode=mode))

        self.mid_block = _build_mid_block_1d(
            out_dim,
            mid_attention,
            dropout,
            non_linearity,
            num_joints=num_joints,
            channel_proj_dim=channel_proj_dim,
            joint_token_dim=joint_token_dim,
            temporal_window_size=temporal_window_size,
        )

        self.norm_out = WanRMSNorm(out_dim, channel_dim=1)
        self.conv_out = WanCausalConv1d(out_dim, z_dim, kernel_size=3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        # --- conv_in (causal + cache), aligned with 3D version's first-layer cache ---
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            h = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            h = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                h = layer(h, feat_cache, feat_idx)
            else:
                h = layer(h)

        h = self.mid_block(h, feat_cache, feat_idx)

        # --- Head: norm -> act -> conv_out (causal + cache) ---
        h = self.norm_out(h)
        h = self.nonlinearity(h)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = h[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            h = self.conv_out(h, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            h = self.conv_out(h)

        return h


class WanEncoder2DTK(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv2dTK(in_channels, dims[0], (3, 1), padding=(1, 0))

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            last_stage = i == len(dim_mult) - 1

            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock2DTK(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        dropout=dropout,
                        num_res_blocks=num_res_blocks,
                        temporal_downsample=(
                            self.temporal_downsample[i] if not last_stage else False
                        ),
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(
                        WanResidualBlock2DTK(in_dim, out_dim, dropout)
                    )
                    in_dim = out_dim

                if i != len(dim_mult) - 1:
                    if i != len(dim_mult) - 1:
                        mode = (
                            "downsample1d"
                            if temporal_downsample[i]
                            else "downsample_channel"
                        )
                        self.down_blocks.append(WanResample2DTK(out_dim, mode=mode))

        # middle blocks
        self.mid_block = WanMidBlock2DTK(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMSNorm(out_dim)
        self.conv_out = WanCausalConv2dTK(out_dim, z_dim, (3, 1), padding=(1, 0))

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class WanDecoder1D(nn.Module):
    r"""
    Causal 1D decoder (WAN-style), aligned with the 3D `WanDecoder3d` used in Diffusers'
    `AutoencoderKLWan`.

    High-level layout:
        z --conv_in(causal)--> h
          --mid_block--> h
          --[up blocks x #stages]--> h
          --norm_out--> act --> conv_out(causal) --> sample

    Parity with the 3D version:
      • Channel schedule: `dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]`.
      • Non-residual (Wan 2.1) path: for i > 0, set `in_dim = in_dim // 2` before building the block.
      • Temporal upsampling only (there is no spatial axis in 1D). If `up_flag` and
        `temporal_upsample[i]` are True, the block receives `upsample_mode="upsample1d"`;
        otherwise, there is no upsample in that block (keeps the temporal length).
      • Residual up path may use a shortcut upsampler (e.g., `DupUp1D`) that trims the very
        first chunk when `first_chunk=True` to fix off-by-one alignment—mirrors 3D behavior.

    Causality & cross-chunk inference:
      • `WanCausalConv1d` performs manual left padding and accepts a temporal `cache_x`, so the
        decoder can be run on chunks without seeing future frames.
      • Each causal conv reads one cache slot from `feat_cache` and writes back the new tail.
      • `first_chunk` is forwarded to up blocks so residual shortcuts can adjust the first output.

    Args:
        dim (int):         Base channels of the first decoder stage.
        z_dim (int):       Latent channels coming in.
        dim_mult (List[int]): Multipliers per stage (same list as encoder, but reversed here).
        num_res_blocks (int): Residual blocks per stage.
        attn_scales (List[float]): Kept for API parity; not used in this 1D decoder (same as 3D code).
        temporal_upsample (List[bool]): Whether each (non-final) stage upsamples time by 2.
        dropout (float):   Dropout rate inside residual blocks.
        non_linearity (str): Activation name (e.g., "silu").
        out_channels (int): Output channels (e.g., 3 translation + J*D rotations).
        is_residual (bool): If True, use residual up blocks (Wan 2.2 style); else Wan 2.1 style.

    Input:
        x: [B, z_dim, T]  latent sequence

    Output:
        y: [B, out_channels, T']  where T' depends on cumulative temporal upsampling.

    Notes:
        • This mirrors the official 3D design (causal VAE for videos) but restricted to the temporal
          axis only. The design choices—causal padding, chunked caches, and stage wiring—follow
          the same intent as the 3D model in Diffusers’ `AutoencoderKLWan`.
    """

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 16,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 168,
        is_residual: bool = False,
        mid_attention: str = "none",
        num_joints: Optional[int] = None,
        channel_proj_dim: int = 128,
        joint_token_dim: Optional[int] = None,
        temporal_window_size: Optional[int] = 64,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_upsample = temporal_upsample
        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        d0 = dims[0]

        self.conv_in = WanCausalConv1d(z_dim, d0, kernel_size=3, padding=1)

        self.mid_block = _build_mid_block_1d(
            d0,
            mid_attention,
            dropout,
            non_linearity,
            num_joints=num_joints,
            channel_proj_dim=channel_proj_dim,
            joint_token_dim=joint_token_dim,
            temporal_window_size=temporal_window_size,
        )

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):

            # determine upsampling mode
            upsample_mode = None
            if i != len(dim_mult) - 1:
                if temporal_upsample[i]:
                    upsample_mode = "upsample1d"
                else:
                    upsample_mode = "upsample_channel"
            # Create and add the upsampling block

            if is_residual:
                up_block = WanResidualUpBlock1D(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temporal_upsample=(
                        temporal_upsample[i] if i != len(dim_mult) - 1 else False
                    ),
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock1D(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMSNorm(out_dim, channel_dim=1)
        self.conv_out = WanCausalConv1d(out_dim, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
        first_chunk: bool = False,
    ) -> torch.Tensor:
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class WanDecoder2DTK(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv2dTK(z_dim, dims[0], (3, 1), padding=(1, 0))

        # middle blocks
        self.mid_block = WanMidBlock2DTK(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if i != len(dim_mult) - 1:
                if temporal_upsample[i]:
                    upsample_mode = "upsample1d"
                else:
                    upsample_mode = "upsample_channel"
            # Create and add the upsampling block

            if is_residual:
                up_block = WanResidualUpBlock2DTK(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temporal_upsample=(
                        temporal_upsample[i] if i != len(dim_mult) - 1 else False
                    ),
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock2DTK(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMSNorm(out_dim)
        self.conv_out = WanCausalConv2dTK(out_dim, out_channels, (3, 1), padding=(1, 0))

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1:].to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x
