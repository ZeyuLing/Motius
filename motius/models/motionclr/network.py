"""MotionCLR denoising network.

Adapted from ``models/unet.py`` in IDEA-Research/MotionCLR. The module and
parameter names intentionally follow the official implementation so released
``EvanTHU/MotionCLR`` state dictionaries load strictly. MotionCLR is licensed
under the IDEA License 1.0, Copyright (c) IDEA. All Rights Reserved.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero a module in place, matching the official initialization."""
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class FFN(nn.Module):
    def __init__(self, latent_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(self.dropout(self.activation(self.linear1(x))))


class Conv1dAdaGNBlock(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Conv1d(
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        # Keep the upstream attribute spelling; it does not affect state keys.
        self.avtication = nn.Mish()

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        x = self.block(x)
        batch_size, channels, horizon = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * horizon, channels)
        x = self.group_norm(x)
        x = x.reshape(batch_size, horizon, channels).permute(0, 2, 1)
        return self.avtication(ada_shift_scale(x, shift, scale))


class SelfAttention(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        text_latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del text_latent_dim, kwargs
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = x.shape
        heads = self.num_head
        normed = self.norm(x)
        query = self.query(normed).view(batch, frames, heads, -1)
        key = self.key(normed).view(batch, frames, heads, -1)
        attention = torch.einsum("bnhd,bmhd->bnmh", query, key)
        attention = attention / math.sqrt(dim // heads)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(normed).view(batch, frames, heads, -1)
        return torch.einsum("bnmh,bmhd->bnhd", weight, value).reshape(
            batch, frames, dim
        )


class TimestepEmbedder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.pe[timesteps]


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim_in, dim_out or dim_in, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 4,
        zero: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Conv1d(
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()
        if zero:
            nn.init.zeros_(self.block.weight)
            nn.init.zeros_(self.block.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        batch_size, channels, horizon = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * horizon, channels)
        x = self.norm(x)
        x = x.reshape(batch_size, horizon, channels).permute(0, 2, 1)
        return self.activation(x)


def ada_shift_scale(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale) + shift


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        zero: bool = True,
        n_groups: int = 8,
        dropout: float = 0.1,
        adagn: bool = True,
    ):
        super().__init__()
        self.adagn = adagn
        first_block: nn.Module
        if adagn:
            first_block = Conv1dAdaGNBlock(
                inp_channels, out_channels, kernel_size, n_groups
            )
        else:
            first_block = Conv1dBlock(
                inp_channels, out_channels, kernel_size, n_groups
            )
        self.blocks = nn.ModuleList(
            [
                first_block,
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups,
                    zero=zero,
                ),
            ]
        )
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            _UnsqueezeLast(),
        )
        self.dropout = nn.Dropout(dropout)
        if zero:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.adagn:
            scale, shift = self.time_mlp(time_embeds).chunk(2, dim=1)
            output = self.blocks[0](x, scale, shift)
        else:
            output = self.blocks[0](x) + self.time_mlp(time_embeds)
        output = self.dropout(self.blocks[1](output))
        return output + self.residual_conv(x)


class _UnsqueezeLast(nn.Module):
    """Parameter-free replacement for upstream einops Rearrange."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


class CrossAttention(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        text_latent_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = x.shape
        text_tokens = text.shape[1]
        heads = self.num_head
        query = self.query(self.norm(x)).view(batch, frames, heads, -1)
        text_normed = self.text_norm(text)
        key = self.key(text_normed).view(batch, text_tokens, heads, -1)
        attention = torch.einsum("bnhd,bmhd->bnmh", query, key)
        attention = attention / math.sqrt(dim // heads)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(text_normed).view(batch, text_tokens, heads, -1)
        return torch.einsum("bnmh,bmhd->bnhd", weight, value).reshape(
            batch, frames, dim
        )


class ResidualCLRAttentionLayer(nn.Module):
    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        no_eff: bool = False,
        self_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not no_eff:
            raise NotImplementedError(
                "The released MotionCLR source does not define its "
                "LinearCrossAttention path. Use no_eff=True, as in the official "
                "EvanTHU/MotionCLR checkpoint."
            )
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        self.cross_attention = CrossAttention(
            latent_dim=dim1,
            text_latent_dim=dim2,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.self_attn_use = bool(self_attention)
        if self.self_attn_use:
            self.self_attention = SelfAttention(
                latent_dim=dim1,
                text_latent_dim=dim2,
                num_heads=num_heads,
                dropout=dropout,
                **kwargs,
            )

    def forward(
        self,
        input_tensor: torch.Tensor,
        condition_tensor: torch.Tensor,
        cond_indices: torch.Tensor,
    ) -> torch.Tensor:
        if cond_indices.numel() == 0:
            return input_tensor
        if self.self_attn_use:
            attended = self.self_attention(input_tensor.permute(0, 2, 1))
            input_tensor = input_tensor + attended.permute(0, 2, 1)
        selected = input_tensor[cond_indices].permute(0, 2, 1)
        selected = self.cross_attention(selected, condition_tensor[cond_indices])
        # Avoid mutating a view needed by autograd while retaining upstream math.
        output = input_tensor.clone()
        output[cond_indices] = output[cond_indices] + selected.permute(0, 2, 1)
        return output


class CLRBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        cond_dim: int,
        time_dim: int,
        adagn: bool = True,
        zero: bool = True,
        no_eff: bool = False,
        self_attention: bool = False,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.conv1d = ResidualTemporalBlock(
            dim_in,
            dim_out,
            embed_dim=time_dim,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
        )
        self.clr_attn = ResidualCLRAttentionLayer(
            dim1=dim_out,
            dim2=cond_dim,
            no_eff=no_eff,
            dropout=dropout,
            self_attention=self_attention,
            **kwargs,
        )
        self.ffn = FFN(dim_out, dim_out * 4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cond_indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1d(x, timestep)
        x = self.clr_attn(x, condition, cond_indices)
        return self.ffn(x.permute(0, 2, 1)).permute(0, 2, 1)


class CondUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int = 128,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        dims: Optional[Sequence[int]] = None,
        time_dim: int = 512,
        adagn: bool = True,
        zero: bool = True,
        dropout: float = 0.1,
        no_eff: bool = False,
        self_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        dims = list(dims) if dims else [input_dim, *[int(dim * m) for m in dim_mults]]
        if len(dims) < 2:
            raise ValueError("MotionCLR UNet needs at least one down/up stage")
        in_out = list(zip(dims[:-1], dims[1:]))
        self.time_mlp = nn.Sequential(
            TimestepEmbedder(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.ModuleList(
                    [
                        CLRBlock(
                            dim_in,
                            dim_out,
                            cond_dim,
                            time_dim,
                            adagn=adagn,
                            zero=zero,
                            no_eff=no_eff,
                            dropout=dropout,
                            self_attention=self_attention,
                            **kwargs,
                        ),
                        CLRBlock(
                            dim_out,
                            dim_out,
                            cond_dim,
                            time_dim,
                            adagn=adagn,
                            zero=zero,
                            no_eff=no_eff,
                            dropout=dropout,
                            self_attention=self_attention,
                            **kwargs,
                        ),
                        Downsample1d(dim_out),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = CLRBlock(
            mid_dim,
            mid_dim,
            cond_dim,
            time_dim,
            adagn=adagn,
            zero=zero,
            no_eff=no_eff,
            dropout=dropout,
            self_attention=self_attention,
            **kwargs,
        )
        self.mid_block2 = CLRBlock(
            mid_dim,
            mid_dim,
            cond_dim,
            time_dim,
            adagn=adagn,
            zero=zero,
            no_eff=no_eff,
            dropout=dropout,
            self_attention=self_attention,
            **kwargs,
        )
        last_dim = mid_dim
        for dim_out in reversed(dims[1:]):
            self.ups.append(
                nn.ModuleList(
                    [
                        Upsample1d(last_dim, dim_out),
                        CLRBlock(
                            dim_out * 2,
                            dim_out,
                            cond_dim,
                            time_dim,
                            adagn=adagn,
                            zero=zero,
                            no_eff=no_eff,
                            dropout=dropout,
                            self_attention=self_attention,
                            **kwargs,
                        ),
                        CLRBlock(
                            dim_out,
                            dim_out,
                            cond_dim,
                            time_dim,
                            adagn=adagn,
                            zero=zero,
                            no_eff=no_eff,
                            dropout=dropout,
                            self_attention=self_attention,
                            **kwargs,
                        ),
                    ]
                )
            )
            last_dim = dim_out
        self.final_conv = nn.Conv1d(last_dim, input_dim, 1)
        if zero:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cond_indices: torch.Tensor,
    ) -> torch.Tensor:
        timestep_embedding = self.time_mlp(timestep)
        skips = []
        for block1, block2, downsample in self.downs:
            x = block1(x, timestep_embedding, condition, cond_indices)
            x = block2(x, timestep_embedding, condition, cond_indices)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, timestep_embedding, condition, cond_indices)
        x = self.mid_block2(x, timestep_embedding, condition, cond_indices)
        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., : skip.shape[-1]]
            x = torch.cat((x, skip), dim=1)
            x = block1(x, timestep_embedding, condition, cond_indices)
            x = block2(x, timestep_embedding, condition, cond_indices)
        return self.final_conv(x)


class MotionCLR(nn.Module):
    """Official-checkpoint-compatible MotionCLR denoiser."""

    def __init__(
        self,
        input_feats: int,
        base_dim: int = 128,
        dim_mults: Sequence[int] = (1, 2, 2, 2),
        dims: Optional[Sequence[int]] = None,
        adagn: bool = True,
        zero: bool = True,
        dropout: float = 0.1,
        no_eff: bool = False,
        time_dim: int = 512,
        latent_dim: int = 256,
        cond_mask_prob: float = 0.1,
        clip_dim: int = 512,
        clip_version: str = "ViT-B/32",
        text_latent_dim: int = 256,
        text_ff_size: int = 2048,
        text_num_heads: int = 4,
        activation: str = "gelu",
        num_text_layers: int = 4,
        self_attention: bool = False,
        vis_attn: bool = False,
        edit_config=None,
        out_path: Optional[str] = None,
        load_clip: bool = True,
        clip_path: Optional[str] = None,
        clip_model: Optional[nn.Module] = None,
        tokenizer: Optional[Callable] = None,
    ):
        super().__init__()
        del edit_config, out_path
        self.input_feats = int(input_feats)
        self.dim_mults = tuple(int(value) for value in dim_mults)
        self.base_dim = int(base_dim)
        self.latent_dim = int(latent_dim)
        self.cond_mask_prob = float(cond_mask_prob)
        self.vis_attn = bool(vis_attn)
        self.counting_map = []
        self.embed_text = nn.Linear(clip_dim, text_latent_dim)
        self.clip_version = clip_version
        self._tokenizer = tokenizer
        if clip_model is not None:
            self.clip_model = clip_model
        elif load_clip:
            self.clip_model = self.load_and_freeze_clip(clip_path or clip_version)
        else:
            self.clip_model = None
        text_layer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.textTransEncoder = nn.TransformerEncoder(
            text_layer, num_layers=num_text_layers
        )
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.unet = CondUnet1D(
            input_dim=self.input_feats,
            cond_dim=text_latent_dim,
            dim=self.base_dim,
            dim_mults=self.dim_mults,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
            no_eff=no_eff,
            dims=dims,
            time_dim=time_dim,
            self_attention=self_attention,
            log_attn=self.vis_attn,
        )

    def load_and_freeze_clip(self, clip_source: str) -> nn.Module:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "MotionCLR requires OpenAI CLIP. Install with "
                "`pip install -e '.[motionclr]'`."
            ) from exc
        clip_model, _ = clip.load(clip_source, device="cpu", jit=False)
        self._tokenizer = clip.tokenize
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False
        return clip_model

    def encode_text(
        self,
        raw_text: Sequence[str],
        device: torch.device | str,
    ) -> torch.Tensor:
        if self.clip_model is None or self._tokenizer is None:
            raise RuntimeError(
                "MotionCLR was created without OpenAI CLIP; encoded text was not supplied."
            )
        texts = self._tokenizer(list(raw_text), truncate=True).to(device)
        with torch.no_grad():
            x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)
            x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            x = self.clip_model.transformer(x.permute(1, 0, 2))
            x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x.to(dtype=self.embed_text.weight.dtype)
        x = self.embed_text(x)
        x = self.textTransEncoder(x)
        return self.text_ln(x).permute(1, 0, 2)

    def mask_cond(
        self,
        batch_size: int,
        force_mask: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if force_mask:
            return torch.empty(0, dtype=torch.long, device=device)
        if self.training and self.cond_mask_prob > 0.0:
            keep = 1.0 - torch.bernoulli(
                torch.full((batch_size,), self.cond_mask_prob, device=device)
            )
            return torch.nonzero(keep, as_tuple=False).squeeze(-1)
        return torch.arange(batch_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text: Optional[Sequence[str]] = None,
        uncond: bool = False,
        enc_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, frames, _ = x.shape
        if enc_text is None:
            if text is None:
                raise ValueError("text or enc_text is required")
            enc_text = self.encode_text(text, x.device)
        cond_indices = self.mask_cond(batch, force_mask=uncond, device=x.device)
        x = x.transpose(1, 2)
        padding = (16 - (frames % 16)) % 16
        x = F.pad(x, (0, padding), value=0)
        x = self.unet(x, timesteps, enc_text, cond_indices)
        return x[:, :, :frames].transpose(1, 2)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text: Optional[Sequence[str]] = None,
        enc_text: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.5,
    ) -> torch.Tensor:
        batch, frames, _ = x.shape
        if enc_text is None:
            if text is None:
                raise ValueError("text or enc_text is required")
            enc_text = self.encode_text(text, x.device)
        cond_indices = self.mask_cond(batch, device=x.device)
        x = x.transpose(1, 2)
        padding = (16 - (frames % 16)) % 16
        x = F.pad(x, (0, padding), value=0)
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([timesteps, timesteps], dim=0)
        output = self.unet(combined_x, combined_t, enc_text, cond_indices)
        output = output[:, :, :frames].transpose(1, 2)
        output_cond, output_uncond = torch.chunk(output, 2, dim=0)
        return output_uncond + cfg_scale * (output_cond - output_uncond)


__all__ = ["CondUnet1D", "MotionCLR"]
