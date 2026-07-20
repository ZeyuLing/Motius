from typing import List, Optional
from torch import nn
import torch
from diffusers.models.activations import get_activation

from .wan_causalconv import WanCausalConv1d, WanCausalConv2dTK
from .wan_norm import WanRMSNorm

CACHE_T = 2


# ====== WanResidualBlock1D: 1x1 shortcut conv padding=0 ======
class WanResidualBlock1D(nn.Module):
    r"""
    Causal 1D residual block (WAN-style), mirroring the 3D `WanResidualBlock`.

    Structure:
        input x ---------------> (+) ------------------> output
           |                     ^
           |                     |
        [shortcut]           [conv2]
           |                 /  ^
           v                /   |
         norm1 -> act -> conv1  |
                         norm2 -> act -> dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # main path
        self.norm1 = WanRMSNorm(in_dim, channel_dim=1)
        self.conv1 = WanCausalConv1d(in_dim, out_dim, kernel_size=3, padding=1)

        self.norm2 = WanRMSNorm(out_dim, channel_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv1d(out_dim, out_dim, kernel_size=3, padding=1)

        # ***** IMPORTANT: 1x1 causal conv MUST use padding=0 (otherwise T increases by +2) *****
        self.shortcut = (
            WanCausalConv1d(in_dim, out_dim, kernel_size=1, padding=0)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        # shortcut (no cache)
        h = self.shortcut(x)

        # norm -> act -> conv1 (causal + cache)
        y = self.norm1(x)
        y = self.nonlinearity(y)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = y[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            y = self.conv1(y, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            y = self.conv1(y)

        # norm -> act -> dropout -> conv2 (causal + cache)
        y = self.norm2(y)
        y = self.nonlinearity(y)
        y = self.dropout(y)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = y[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            y = self.conv2(y, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            y = self.conv2(y)

        return y + h


class WanResidualBlock2DTK(nn.Module):
    r"""
    Causal 2D residual block (WAN-style), mirroring the 3D `WanResidualBlock`.

    Structure:
        input x ---------------> (+) ------------------> output
           |                     ^
           |                     |
        [shortcut]           [conv2]
           |                 /  ^
           v                /   |
         norm1 -> act -> conv1  |
                         norm2 -> act -> dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # main path
        self.norm1 = WanRMSNorm(in_dim, channel_dim=1)
        self.conv1 = WanCausalConv2dTK(
            in_dim, out_dim, kernel_size=(3, 1), padding=(1, 0)
        )

        self.norm2 = WanRMSNorm(out_dim, channel_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv2dTK(
            out_dim, out_dim, kernel_size=(3, 1), padding=(1, 0)
        )

        # ***** IMPORTANT: 1x1 causal conv MUST use padding=0 (otherwise T increases by +2) *****
        self.shortcut = (
            WanCausalConv2dTK(in_dim, out_dim, kernel_size=1, padding=0)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Optional[torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        """
        Forward pass of the WanResidualBlockTK.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, K).
            feat_cache (Optional[List[Optional[torch.Tensor]]]): Feature cache for causal convolution.
            feat_idx (List[int]): Index for feature cache.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T, K).
        """
        # shortcut (no cache)
        h = self.shortcut(x)

        # norm -> act -> conv1 (causal + cache)
        y = self.norm1(x)
        y = self.nonlinearity(y)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = y[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            y = self.conv1(y, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            y = self.conv1(y)

        # norm -> act -> dropout -> conv2 (causal + cache)
        y = self.norm2(y)
        y = self.nonlinearity(y)
        y = self.dropout(y)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = y[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2
                )
            y = self.conv2(y, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            y = self.conv2(y)

        return y + h


if __name__ == "__main__":
    motion = torch.randn(2, 32, 17, 22)
    conv = WanResidualBlockTK(32, 32)
    out = conv(motion)
    print(out.shape)
