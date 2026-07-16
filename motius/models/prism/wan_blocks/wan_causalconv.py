import os
from typing import Optional, Tuple, Union
from torch import nn
import torch
from torch.nn import functional as F


class WanCausalConv1d(nn.Conv1d):
    r"""
    Causal 1D convolution with feature caching (WAN-style), mirroring `WanCausalConv3d`.

    Design parity with WAN 3D version:
      1) Initialize `nn.Conv1d(..., padding=padding)` so that kernel/weight shapes are consistent with config,
         then **override** `self.padding = (0,)` and do **manual left padding** in `forward` via `F.pad`.
      2) Temporal causality: we **only pad on the left** (time axis), amount = `2 * padding`.
      3) Cross-chunk inference: if `cache_x` (previous tail) is given, we **prepend** it and reduce the left-pad
         by `cache_x.size(-1)` (clamped at 0). This exactly matches the 3D code path in WAN.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        kernel_size (int): kernel length (commonly 3)
        stride (int): temporal stride (e.g., 1 or 2)
        padding (int): the *base* padding used to compute the causal left-pad (we will manual-pad left by 2*padding)
        dilation (int): dilation
        bias (bool): bias flag

    Shapes:
        Input:  [B, C_in, T]
        Cache:  [B, C_in, T_cache] or None
        Output: [B, C_out, T_out] (as per Conv1d with given stride/dilation, but causally padded)

    Notes:
        - We avoid any right padding to prevent "seeing the future".
        - Manual padding order in `F.pad` for 1D is (pad_left, pad_right). We use (L, 0).
        - This mirrors the 3D implementation that sets `_padding = (..., 2*pad_t, 0)` and then `F.pad` before conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        # Initialize with the requested padding to keep config parity,
        # then we'll disable it and handle padding manually (like the 3D variant).
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  # temporary: we will zero it out below
            dilation=dilation,
            bias=bias,
        )

        # Normalize padding to int
        pad_val = self.padding if isinstance(self.padding, int) else self.padding[0]
        # Store manual left/right padding tuple -> (left, right) = (2*pad, 0)
        self._padding = (2 * pad_val, 0)

        # Disable Conv1d's implicit symmetric padding; we will use F.pad explicitly.
        self.padding = (0,)

    def forward(
        self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, C, T] current chunk
            cache_x: [B, C, T_cache], optional previous tail to prepend
        """
        # Determine how much manual left padding we still need after prepending cache
        pad_left, pad_right = self._padding
        if cache_x is not None and pad_left > 0:
            # prepend cached tail (already from the past -> preserves causality)
            x = torch.cat([cache_x.to(x.device, dtype=x.dtype), x], dim=2)
            pad_left = max(0, pad_left - cache_x.size(2))  # reduce required left pad

        # Manual causal left padding on time axis
        if pad_left > 0 or pad_right > 0:
            x = F.pad(x, (pad_left, pad_right))  # only-left pad (L, R) -> (2*pad, 0)

        # Now run the actual 1D conv (with internal padding disabled)
        return super().forward(x)


class WanCausalConv2dTK(nn.Conv2d):
    r"""
    Causal 2D conv over [B, C, T, K] with caching on time axis (WAN-style).

    Mapping: input [B, C, T, K] -> Conv2d expects [N, C, H, W] with H=T (time), W=K (joints).
    We do *left-only* padding on time (H) to enforce causality; joints (W) padding is user-controlled.
    If `cache_x` is provided, we prepend it along T and reduce the required left-pad accordingly.

    Args:
        in_channels:    C_in
        out_channels:   C_out
        kernel_size:    (k_t, k_k) or int (applied to both dims)
        stride:         (s_t, s_k) or int
        padding:        base padding (p_t, p_k) or int.
                        We will manual-pad time by (2*p_t, 0) like WAN 3D/1D variants,
                        and pad joints symmetrically by (p_k, p_k).
        dilation:       (d_t, d_k) or int
        bias:           bool

    Shapes:
        x:       [B, C_in, T, K]
        cache_x: [B, C_in, T_cache, K] or None
        out:     [B, C_out, T_out, K_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 1),
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = (1, 0),
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:

        # Normalize tuples
        def _to_2t(x):
            return (x, x) if isinstance(x, int) else x

        k_t, k_k = _to_2t(kernel_size)
        s_t, s_k = _to_2t(stride)
        p_t, p_k = _to_2t(padding)
        d_t, d_k = _to_2t(dilation)

        assert (
            k_k == s_k == d_k == 1
        ), "WAN causal conv over joints (K) is not supported"
        assert p_k == 0, "WAN causal conv over joints (K) is not supported"

        # Initialize with "config" padding (parity with your 1D), then disable it.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(k_t, k_k),
            stride=(s_t, s_k),
            padding=(p_t, p_k),  # temporary; we will override to (0,0)
            dilation=(d_t, d_k),
            bias=bias,
        )

        # Store manual pad plan: time gets left-only (2*p_t, 0); joints get symmetric (p_k, p_k).
        self._t_pad_left_base = 2 * p_t  # WAN-style: double base pad on time, left-only
        self._k_pad = p_k  # symmetric on joints (left==right==p_k)

        # Disable Conv2d's internal symmetric padding.
        self.padding = (0, 0)

    def _use_temporal_conv1d(self) -> bool:
        return (
            os.environ.get("VERMO_WAN_TK_CONV1D", "1") != "0"
            and self._k_pad == 0
            and self.kernel_size[1] == 1
            and self.stride[1] == 1
            and self.dilation[1] == 1
        )

    def _forward_temporal_conv1d(self, x: torch.Tensor, pad_t_left: int = 0) -> torch.Tensor:
        if os.environ.get("VERMO_WAN_TK_EXPLICIT_CONV", "1") != "0":
            return self._forward_temporal_explicit(x, pad_t_left=pad_t_left)

        if pad_t_left > 0:
            x = F.pad(x, (0, 0, pad_t_left, 0))
        b, c, _, k = x.shape
        x_1d = x.permute(0, 3, 1, 2).reshape(b * k, c, x.shape[2]).contiguous()
        out = F.conv1d(
            x_1d,
            self.weight.squeeze(-1),
            self.bias,
            stride=self.stride[0],
            padding=0,
            dilation=self.dilation[0],
            groups=self.groups,
        )
        return out.reshape(b, k, self.out_channels, out.shape[-1]).permute(0, 2, 3, 1).contiguous()

    def _forward_temporal_explicit(self, x: torch.Tensor, pad_t_left: int = 0) -> torch.Tensor:
        """Small-kernel temporal conv path that avoids cuDNN/cublas conv backends."""
        b, c, t, k = x.shape
        kernel_t = self.kernel_size[0]
        stride_t = self.stride[0]
        dilation_t = self.dilation[0]
        logical_t = t + pad_t_left
        out_t = (logical_t - dilation_t * (kernel_t - 1) - 1) // stride_t + 1
        if out_t <= 0:
            raise RuntimeError(
                f"Invalid temporal conv output length: input_t={t}, pad_left={pad_t_left}, kernel={kernel_t}, "
                f"stride={stride_t}, dilation={dilation_t}"
            )

        groups = self.groups
        in_per_group = c // groups
        out_per_group = self.out_channels // groups
        weight = self.weight.squeeze(-1)
        out_groups = []
        out_positions = None
        if pad_t_left > 0:
            out_positions = torch.arange(out_t, device=x.device, dtype=torch.long) * stride_t

        for group_idx in range(groups):
            in_start = group_idx * in_per_group
            out_start = group_idx * out_per_group
            x_group = x[:, in_start : in_start + in_per_group]
            w_group = weight[out_start : out_start + out_per_group]
            y_group = None

            for kernel_idx in range(kernel_t):
                if pad_t_left > 0:
                    input_idx = out_positions + kernel_idx * dilation_t - pad_t_left
                    valid = ((input_idx >= 0) & (input_idx < t)).to(dtype=x.dtype).view(1, 1, out_t, 1)
                    safe_idx = input_idx.clamp(0, max(t - 1, 0))
                    x_slice = x_group.index_select(2, safe_idx) * valid
                else:
                    t_start = kernel_idx * dilation_t
                    x_slice = x_group[:, :, t_start : t_start + out_t * stride_t : stride_t, :]
                w_slice = w_group[:, :, kernel_idx].view(1, out_per_group, in_per_group, 1, 1)
                term = (x_slice.unsqueeze(1) * w_slice).sum(dim=2)
                y_group = term if y_group is None else y_group + term

            if self.bias is not None:
                y_group = y_group + self.bias[out_start : out_start + out_per_group].view(1, -1, 1, 1)
            out_groups.append(y_group)

        if len(out_groups) == 1:
            out = out_groups[0]
        else:
            out = torch.cat(out_groups, dim=1)
        return out.contiguous()

    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, K]
        cache_x: Optional[torch.Tensor] = None,  # [B, C, T_cache, K]
    ) -> torch.Tensor:
        assert x.dim() == 4, f"Expect [B,C,T,K], got {tuple(x.shape)}"
        pad_t_left = self._t_pad_left_base

        if cache_x is not None and pad_t_left > 0:
            # prepend cached tail along time axis (dim=2)
            if cache_x.numel() > 0:
                x = torch.cat([cache_x.to(x.device, dtype=x.dtype), x], dim=2)
                pad_t_left = max(0, pad_t_left - cache_x.size(2))

        # F.pad order for NCHW is (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
        pad_w_left = self._k_pad
        pad_w_right = self._k_pad
        pad_h_top = pad_t_left
        pad_h_bot = 0

        if self._use_temporal_conv1d():
            assert pad_w_left == pad_w_right == pad_h_bot == 0
            return self._forward_temporal_conv1d(x, pad_t_left=pad_h_top)

        if pad_w_left or pad_w_right or pad_h_top or pad_h_bot:
            x = F.pad(x, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot))

        return super().forward(x)


if __name__ == "__main__":
    motion = torch.randn(2, 6, 17, 32)
    conv = WanCausalConv2dTK(6, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
    out = conv(motion)
    print(out.shape)
