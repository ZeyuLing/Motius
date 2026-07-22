"""Motion tokenizer used by UniMuMo's shared Encodec codebooks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ResidualConv1dBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        activation: str = "relu",
        norm: str | None = None,
    ):
        super().__init__()
        if norm == "LN":
            norm_factory = lambda: nn.LayerNorm(channels)
        elif norm == "GN":
            norm_factory = lambda: nn.GroupNorm(32, channels, eps=1e-6)
        elif norm == "BN":
            norm_factory = lambda: nn.BatchNorm1d(channels, eps=1e-6)
        elif norm is None:
            norm_factory = nn.Identity
        else:
            raise ValueError(f"Unsupported motion codec norm: {norm!r}")
        activations = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported motion codec activation: {activation!r}")
        self.norm_type = norm
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        self.activation1 = activations[activation]()
        self.activation2 = activations[activation]()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def _normalize(self, value: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        if self.norm_type == "LN":
            return norm(value.transpose(1, 2)).transpose(1, 2)
        return norm(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.activation1(self._normalize(value, self.norm1))
        value = self.conv1(value)
        value = self.activation2(self._normalize(value, self.norm2))
        return residual + self.conv2(value)


class ResidualStack1d(nn.Sequential):
    def __init__(
        self,
        channels: int,
        depth: int,
        *,
        dilation_growth_rate: int,
        reverse_dilation: bool = True,
        activation: str = "relu",
        norm: str | None = None,
    ):
        dilations = [dilation_growth_rate**index for index in range(depth)]
        if reverse_dilation:
            dilations.reverse()
        super().__init__(
            *[
                ResidualConv1dBlock(
                    channels,
                    dilation=dilation,
                    activation=activation,
                    norm=norm,
                )
                for dilation in dilations
            ]
        )


class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 263,
        output_dim: int = 128,
        channels: Sequence[int] = (256, 224, 192, 144, 128),
        input_fps: float = 60.0,
        code_fps: float = 50.0,
        dilation_growth_rate: int = 2,
        depth_per_block: int = 6,
        activation: str = "relu",
        norm: str | None = None,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        self.input_fps = float(input_fps)
        self.code_fps = float(code_fps)
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        blocks = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    ResidualStack1d(
                        output_channels,
                        depth_per_block,
                        dilation_growth_rate=dilation_growth_rate,
                        activation=activation,
                        norm=norm,
                    ),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.post_conv = nn.Conv1d(channels[-1], output_dim, kernel_size=3, padding=1)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        target_length = round(motion.shape[-1] / self.input_fps * self.code_fps)
        motion = F.interpolate(
            motion,
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        return self.post_conv(self.blocks(self.init_conv(motion)))


class MotionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 263,
        latent_dim: int = 128,
        channels: Sequence[int] = (128, 144, 192, 224, 256),
        output_fps: float = 60.0,
        code_fps: float = 50.0,
        dilation_growth_rate: int = 2,
        depth_per_block: int = 6,
        activation: str = "relu",
        norm: str | None = None,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        self.output_fps = float(output_fps)
        self.code_fps = float(code_fps)
        self.init_conv = nn.Sequential(
            nn.Conv1d(latent_dim, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        blocks = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks.append(
                nn.Sequential(
                    ResidualStack1d(
                        input_channels,
                        depth_per_block,
                        dilation_growth_rate=dilation_growth_rate,
                        activation=activation,
                        norm=norm,
                    ),
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.post_conv = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels[-1], input_dim, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        motion = self.post_conv(self.blocks(self.init_conv(latent)))
        target_length = round(motion.shape[-1] / self.code_fps * self.output_fps)
        return F.interpolate(
            motion,
            size=target_length,
            mode="linear",
            align_corners=False,
        )


def _residual_projection(channels: int, multiplier: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(channels * 2, channels * 2, kernel_size=1),
        nn.ELU(),
        nn.Conv1d(
            channels * 2,
            channels * 2 * multiplier,
            kernel_size=3,
            padding=1,
        ),
        nn.ELU(),
        nn.Conv1d(
            channels * 2 * multiplier,
            channels * 2,
            kernel_size=3,
            padding=1,
        ),
        nn.ELU(),
        nn.Conv1d(channels * 2, channels, kernel_size=1),
    )


class UniMuMoMotionCodec(nn.Module):
    """Encode HML263 motion with Encodec's frozen residual codebooks."""

    def __init__(self, config: dict):
        super().__init__()
        latent_dim = int(config.get("latent_dim", 128))
        self.encoder = MotionEncoder(
            input_dim=int(config.get("motion_dim", 263)),
            output_dim=latent_dim,
            channels=config.get("encoder_channels", (256, 224, 192, 144, 128)),
            input_fps=float(config.get("motion_fps", 60.0)),
            code_fps=float(config.get("code_fps", 50.0)),
            dilation_growth_rate=int(config.get("dilation_growth_rate", 2)),
            depth_per_block=int(config.get("depth_per_block", 6)),
            activation=str(config.get("activation", "relu")),
            norm=config.get("norm"),
        )
        self.decoder = MotionDecoder(
            input_dim=int(config.get("motion_dim", 263)),
            latent_dim=latent_dim,
            channels=config.get("decoder_channels", (128, 144, 192, 224, 256)),
            output_fps=float(config.get("motion_fps", 60.0)),
            code_fps=float(config.get("code_fps", 50.0)),
            dilation_growth_rate=int(config.get("dilation_growth_rate", 2)),
            depth_per_block=int(config.get("depth_per_block", 6)),
            activation=str(config.get("activation", "relu")),
            norm=config.get("norm"),
        )
        self.pre_quantize = _residual_projection(
            latent_dim,
            int(config.get("pre_quant_multiplier", 4)),
        )
        self.post_quantize = _residual_projection(
            latent_dim,
            int(config.get("post_quant_multiplier", 4)),
        )

    def encode_embeddings(
        self,
        normalized_motion: torch.Tensor,
        music_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_motion.ndim != 3 or normalized_motion.shape[-1] != 263:
            raise ValueError("normalized_motion must have shape (B,T,263)")
        motion_embeddings = self.encoder(normalized_motion.transpose(1, 2))
        if music_embeddings.shape != motion_embeddings.shape:
            raise ValueError(
                "music and motion embeddings must share shape, got "
                f"{tuple(music_embeddings.shape)} and {tuple(motion_embeddings.shape)}"
            )
        residual = self.pre_quantize(
            torch.cat((music_embeddings, motion_embeddings), dim=1)
        )
        return motion_embeddings + residual

    def decode_embeddings(
        self,
        music_embeddings: torch.Tensor,
        motion_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if music_embeddings.shape != motion_embeddings.shape:
            raise ValueError("music and motion embeddings must share shape")
        residual = self.post_quantize(
            torch.cat((music_embeddings, motion_embeddings), dim=1)
        )
        motion = self.decoder(motion_embeddings + residual)
        return motion.transpose(1, 2)


__all__ = [
    "MotionDecoder",
    "MotionEncoder",
    "ResidualConv1dBlock",
    "ResidualStack1d",
    "UniMuMoMotionCodec",
]
