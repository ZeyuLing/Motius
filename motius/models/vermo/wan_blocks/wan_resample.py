from typing import List, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F

from .wan_resnet import CACHE_T
from .wan_causalconv import WanCausalConv1d, WanCausalConv2dTK


class AvgDown1D(nn.Module):
    r"""
    Average downsampling + channel remapping on the temporal axis (1D), mirroring WAN's AvgDown3D.

    Input shape:
        x: [B, C_in, T]

    Behavior (aligns with WAN AvgDown3D):
    1) Compute left padding on time axis so that T is divisible by factor_t:
           pad_t = (factor_t - (T % factor_t)) % factor_t
       Then pad on the **left** of T (causal-friendly; matches AvgDown3D's (pad_t, 0) on time dim).
    2) Reshape to expose the temporal factor and group channels:
           [B, C_in, T'] -> [B, C_in, T'//f_t, f_t]
    3) Permute so the factor axis is next to channels (same ordering idea as 3D):
           -> [B, C_in, f_t, T'//f_t]
    4) Merge (C_in * f_t) and split into (out_channels, group_size):
           group_size = (C_in * f_t) // out_channels
    5) Mean over the group dimension (reduce grouped channels):
           -> [B, out_channels, T'//f_t]

    Args:
        in_channels (int): C_in
        out_channels (int): C_out
        factor_t (int): temporal downsample factor (f_t)

    Constraints:
        in_channels * factor_t % out_channels == 0
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor = factor_t  # keep naming parity with 3D version

        assert (
            in_channels * self.factor % out_channels == 0
        ), "in_channels * factor_t must be divisible by out_channels"
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T] -> [B, C_out, T//factor_t]
        """
        if x.dim() != 3:
            raise ValueError(f"AvgDown1D expects [B, C, T], got {tuple(x.shape)}")

        # --- 1) Left-pad time dim to multiple of factor_t (match AvgDown3D) ---
        # pad = (pad_left, pad_right) for the last dimension (T)
        T = x.shape[2]
        pad_t = (self.factor_t - (T % self.factor_t)) % self.factor_t
        if pad_t:
            x = F.pad(x, (pad_t, 0))  # left pad on time axis
            T = T + pad_t

        B, C, T = x.shape
        # --- 2) Expose temporal factor: [B, C, T//f_t, f_t] ---
        x = x.view(B, C, T // self.factor_t, self.factor_t)

        # --- 3) Permute to put factor next to channels (same spirit as 3D's (1,3,5,7,2,4,6)) ---
        x = x.permute(0, 1, 3, 2).contiguous()  # [B, C, f_t, T']

        # --- 4) Merge C and f_t, then split into (out_channels, group_size) ---
        x = x.reshape(B, C * self.factor_t, T // self.factor_t)  # [B, C*f_t, T']
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t)

        # --- 5) Group-wise average over channels ---
        x = x.mean(dim=2)  # [B, out_channels, T']
        return x


class AvgDown2DTK(nn.Module):
    r"""
    Average downsampling + channel remapping on the temporal axis (1D), mirroring WAN's AvgDown3D.

    Input shape:
        x: [B, C_in, T, K]

    Behavior (aligns with WAN AvgDown3D):
    1) Compute left padding on time axis so that T is divisible by factor_t:
           pad_t = (factor_t - (T % factor_t)) % factor_t
       Then pad on the **left** of T (causal-friendly; matches AvgDown3D's (pad_t, 0) on time dim).
    2) Reshape to expose the temporal factor and group channels:
           [B, C_in, T', K] -> [B, C_in, T'//f_t, f_t, K]
    3) Permute so the factor axis is next to channels (same ordering idea as 3D):
           -> [B, C_in, f_t, T'//f_t, K]
    4) Merge (C_in * f_t) and split into (out_channels, group_size):
           group_size = (C_in * f_t) // out_channels
    5) Mean over the group dimension (reduce grouped channels):
           -> [B, out_channels, T'//f_t, K]

    Args:
        in_channels (int): C_in
        out_channels (int): C_out
        factor_t (int): temporal downsample factor (f_t)

    Constraints:
        in_channels * factor_t % out_channels == 0
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor = factor_t  # keep naming parity with 3D version

        assert (
            in_channels * self.factor % out_channels == 0
        ), "in_channels * factor_t must be divisible by out_channels"
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T, K] -> [B, C_out, T//factor_t, K]
        """
        if x.dim() != 4:
            raise ValueError(f"AvgDown2DTK expects [B, C, T, K], got {tuple(x.shape)}")

        # --- 1) Left-pad time dim to multiple of factor_t (match AvgDown3D) ---
        # pad = (pad_left, pad_right) for the last dimension (T)
        T, K = x.shape[2:]
        pad_t = (self.factor_t - (T % self.factor_t)) % self.factor_t
        if pad_t:
            x = F.pad(x, (pad_t, 0))  # left pad on time axis
            T = T + pad_t

        B, C, T, K = x.shape
        # --- 2) Expose temporal factor: [B, C, T//f_t, f_t] ---
        x = x.view(B, C, T // self.factor_t, self.factor_t, K)

        # --- 3) Permute to put factor next to channels (same spirit as 3D's (1,3,5,7,2,4,6)) ---
        x = x.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, f_t, T', K]

        # --- 4) Merge C and f_t, then split into (out_channels, group_size) ---
        x = x.reshape(B, C * self.factor_t, T // self.factor_t, K)  # [B, C*f_t, T', K]
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, K)

        # --- 5) Group-wise average over channels ---
        x = x.mean(dim=2)  # [B, out_channels, T'//f_t, K]
        return x


class DupUp1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        assert out_channels * factor_t % in_channels == 0
        self.repeats = out_channels * factor_t // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        B, _, T = x.shape
        x = x.view(B, self.out_channels, self.factor_t, T)  # [B, Cout, f, T]
        x = x.permute(0, 1, 3, 2).contiguous()  # [B, Cout, T, f]
        x = x.reshape(B, self.out_channels, T * self.factor_t)
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :]
        return x


class DupUp2DTK(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        assert out_channels * factor_t % in_channels == 0
        self.repeats = out_channels * factor_t // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        B, _, T, K = x.shape
        x = x.view(B, self.out_channels, self.factor_t, T, K)  # [B, Cout, f, T, K]
        x = x.permute(0, 1, 3, 2, 4).contiguous()  # [B, Cout, T, f]
        x = x.reshape(B, self.out_channels, T * self.factor_t, K)
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :]
        return x


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample1D(nn.Module):
    r"""
    A custom resampling module for 1D temporal data (Causal).

    Args:
        dim (int): Number of input/output channels.
        mode (str): One of:
            - 'none': Identity.
            - 'upsample1d': Temporal upsampling via causal 1D conv -> [B, 2*C, T] reshape -> [B, C, 2T],
                            then a 1D conv to adjust channels (mirrors WAN upsample3d).
            - 'downsample1d': Temporal downsampling via causal 1D conv with stride=2 (mirrors WAN downsample3d).
        upsample_out_dim (int, optional): Output channels after upsampling. Defaults to dim // 2 (as in WAN).
    """

    def __init__(
        self,
        dim: int,
        mode: Optional[str] = None,
        upsample_out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample1d":
            # time upsample branch: causal conv to 2*C, then channel proj conv1d (like 2D conv in WAN)
            self.time_conv = WanCausalConv1d(dim, dim * 2, kernel_size=3, padding=1)
            # time upsample is implemented by channel upsample

            self.resample = nn.Sequential(
                nn.Conv1d(dim, upsample_out_dim, kernel_size=1, padding=0)
            )
        elif mode == "downsample1d":
            # time downsample branch: causal conv stride=2; first-call is skipped per WAN semantics
            self.time_conv = WanCausalConv1d(
                dim, dim, kernel_size=3, stride=2, padding=0
            )
            self.resample = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        elif mode == "upsample_channel":
            self.resample = nn.Conv1d(dim, upsample_out_dim, 1, padding=0)
        elif mode == "downsample_channel":
            self.resample = nn.Conv1d(dim, dim, 1, padding=0)
        else:
            assert mode is None
            self.time_conv = None
            self.resample = nn.Identity()

    def upsample1d(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        # ----- temporal upsample (mirrors WAN upsample3d) -----
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first time: register 'Rep' sentinel, do NOT time-upsample yet
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()  # keep tail for next call
                # if previous cache is real (not 'Rep') but we only have < CACHE_T frames, prepend the last frame
                if (
                    cache_x.shape[2] < CACHE_T
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x],
                        dim=2,
                    )
                # if previous was 'Rep', fabricate zeros to maintain 2-frame cache semantics (matches WAN logic)
                elif cache_x.shape[2] < CACHE_T and feat_cache[idx] == "Rep":
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x, device=cache_x.device), cache_x],
                        dim=2,
                    )

                # causal time conv with/without external cache
                if feat_cache[idx] == "Rep":
                    y = self.time_conv(x)  # no external cache on first real conv
                else:
                    y = self.time_conv(x, feat_cache[idx])

                # update cache slot + cursor
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                # channel-split -> interleave on time axis: [B, 2C, T] -> [B, C, 2T]
                B, C2, T = y.shape
                y = y.view(B, 2, C2 // 2, T)  # [B, 2, C, T]
                y = torch.stack((y[:, 0], y[:, 1]), dim=3)  # [B, C, T, 2]
                x = y.reshape(B, C2 // 2, T * 2)  # [B, C, 2T]
        # channel adjust (always apply, even on the very first call like WAN's 2D conv)
        x = self.resample(x)
        return x

    def downsample1d(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ):

        x = self.resample(x)
        # ----- temporal downsample (mirrors WAN downsample3d) -----
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first time: only cache the tail frame, DO NOT stride-2 yet
                feat_cache[idx] = x[:, :, -1:].clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ):
        """
        x: [B, C, T]
        feat_cache: list-like cache shared across the network (WAN-style). We consume one slot per resample module.
        feat_idx: single-element list as a mutable cursor into feat_cache.
        """
        if self.mode == "upsample1d":
            return self.upsample1d(x, feat_cache, feat_idx)

        elif self.mode == "downsample1d":
            return self.downsample1d(x, feat_cache, feat_idx)

        else:
            return self.resample(x)


class WanResample2DTK(nn.Module):
    r"""
    A custom resampling module for 1D temporal data (Causal).

    Args:
        dim (int): Number of input/output channels.
        mode (str): One of:
            - 'none': Identity.
            - 'upsample1d': Temporal upsampling via causal 1D conv -> [B, 2*C, T] reshape -> [B, C, 2T],
                            then a 1D conv to adjust channels (mirrors WAN upsample3d).
            - 'downsample1d': Temporal downsampling via causal 1D conv with stride=2 (mirrors WAN downsample3d).
        upsample_out_dim (int, optional): Output channels after upsampling. Defaults to dim // 2 (as in WAN).
    """

    def __init__(
        self,
        dim: int,
        mode: Optional[str] = None,
        upsample_out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample1d":
            # time upsample branch: causal conv to 2*C, then channel proj conv1d (like 2D conv in WAN)
            self.time_conv = WanCausalConv2dTK(
                dim, dim * 2, kernel_size=(3, 1), padding=(1, 0)
            )
            # time upsample is implemented by channel upsample

            self.resample = nn.Sequential(
                WanCausalConv2dTK(
                    dim, upsample_out_dim, kernel_size=(1, 1), padding=(0, 0)
                )
            )
        elif mode == "downsample1d":
            # time downsample branch: causal conv stride=2; first-call is skipped per WAN semantics
            self.time_conv = WanCausalConv2dTK(
                dim, dim, kernel_size=(3, 1), stride=(2, 1), padding=(0, 0)
            )
            self.resample = WanCausalConv2dTK(
                dim, dim, kernel_size=(1, 1), padding=(0, 0)
            )
        elif mode == "upsample_channel":
            self.resample = WanCausalConv2dTK(
                dim, upsample_out_dim, kernel_size=(1, 1), padding=(0, 0)
            )
        elif mode == "downsample_channel":
            self.resample = WanCausalConv2dTK(
                dim, dim, kernel_size=(1, 1), padding=(0, 0)
            )
        else:
            assert mode is None
            self.time_conv = None
            self.resample = nn.Identity()

    def upsample1d(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ) -> torch.Tensor:
        # ----- temporal upsample (mirrors WAN upsample3d) -----
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first time: register 'Rep' sentinel, do NOT time-upsample yet
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()  # keep tail for next call
                # if previous cache is real (not 'Rep') but we only have < CACHE_T frames, prepend the last frame
                if (
                    cache_x.shape[2] < CACHE_T
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x],
                        dim=2,
                    )
                # if previous was 'Rep', fabricate zeros to maintain 2-frame cache semantics (matches WAN logic)
                elif cache_x.shape[2] < CACHE_T and feat_cache[idx] == "Rep":
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x, device=cache_x.device), cache_x],
                        dim=2,
                    )

                # causal time conv with/without external cache
                if feat_cache[idx] == "Rep":
                    y = self.time_conv(x)  # no external cache on first real conv
                else:
                    y = self.time_conv(x, feat_cache[idx])

                # update cache slot + cursor
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                # channel-split -> interleave on time axis: [B, 2C, T] -> [B, C, 2T]
                B, C2, T, K = y.shape
                y = y.view(B, 2, C2 // 2, T, K)  # [B, 2, C, T, K]
                y = torch.stack((y[:, 0], y[:, 1]), dim=3)  # [B, C, T, 2]
                x = y.reshape(B, C2 // 2, T * 2, K)  # [B, C, 2T, K]
        # channel adjust (always apply, even on the very first call like WAN's 2D conv)
        x = self.resample(x)
        return x

    def downsample1d(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ):

        x = self.resample(x)
        # ----- temporal downsample (mirrors WAN downsample3d) -----
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first time: only cache the tail frame, DO NOT stride-2 yet
                feat_cache[idx] = x[:, :, -1:].clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: Optional[List[Union[str, torch.Tensor]]] = None,
        feat_idx: List[int] = [0],
    ):
        """
        x: [B, C, T]
        feat_cache: list-like cache shared across the network (WAN-style). We consume one slot per resample module.
        feat_idx: single-element list as a mutable cursor into feat_cache.
        """
        if self.mode == "upsample1d":
            return self.upsample1d(x, feat_cache, feat_idx)

        elif self.mode == "downsample1d":
            return self.downsample1d(x, feat_cache, feat_idx)

        else:
            return self.resample(x)


if __name__ == "__main__":
    x = torch.randn(2, 32, 16, 22)
    down = AvgDown2DTK(32, 64, 2)
    up = DupUp2DTK(64, 32, 2)
    y = down(x)
    z = up(y, first_chunk=True)
    z_2 = up(y)
    print(y.shape, z.shape, z_2.shape)

    x = torch.randn(2, 32, 17, 22)
    resample_down1d = WanResample2DTK(32, mode="downsample1d")
    y_down = resample_down1d(x)
    print(y_down.shape)

    resample_up1d = WanResample2DTK(32, mode="upsample1d")
    z_up = resample_up1d(x)
    print(z_up.shape)
