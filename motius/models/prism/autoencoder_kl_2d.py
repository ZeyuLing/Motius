# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, Union, List

import logging
import os
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
)

from diffusers.utils.accelerate_utils import apply_forward_hook  # diffusers utility

from diffusers.loaders import FromOriginalModelMixin

from motius.registry import MODELS, HF_MODELS
from .wan_blocks.wan_causalconv import WanCausalConv2dTK
from .wan_blocks.wan_encdec import WanEncoder2DTK, WanDecoder2DTK
from .gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)


logger = logging.getLogger(__name__)


@HF_MODELS.register_module(force=True)
class AutoencoderKLPrism2DTK(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Causal VAE for SMPL/SMPL-X sequences `[B, T, C]` (temporal-only), aligned with Diffusers'
    `AutoencoderKLWan` (video 3D) design:

      • `encode`: returns `AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)`
      • `decode`: returns `DecoderOutput(sample=...)`
      • `forward`: encode → sample/mode → decode

    Major implementation notes:
      • Uses causal 1D convolutions with cross-chunk feature caches (no future peeking).
      • Chunking mirrors WAN: first chunk has 1 frame, then stride-4 groups during encode; decode iterates per-frame.
      • 1×1 causal convs **must set `padding=0`** to avoid temporal shift (parity with 3D WAN).
      • Slicing (batch-wise) is supported; tiling is not applicable to 1D.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: Optional[int] = None,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[int] = [],
        temporal_downsample: Tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        is_residual: bool = False,
        in_channels: int = 6,  # e.g., 3 (translation) + J*D (rotation)
        out_channels: int = 6,  # reconstruct same-dim SMPL params
        scale_factor_temporal: Optional[int] = 4,
        # following params are not used in encode, decode, forward, but will be useful in latent diffusion model training and post-processing
        latents_mean=[],
        latents_std=[],
        use_static: bool = False,  # add a static joint label to post-process the motion
        use_rollout_trans: bool = True,  # use rollout translation to get absolute translation
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temporal_downsample = temporal_downsample
        self.temporal_upsample = temporal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.z_dim = z_dim

        # ---------------- Encoder / Decoder (1D causal) ----------------
        self.encoder = WanEncoder2DTK(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,  # mean+logvar
            dim_mult=list(dim_mult),
            attn_scales=attn_scales,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        # IMPORTANT: 1x1 causal convs must use padding=0 (parity with WAN-3D)
        self.quant_conv = WanCausalConv2dTK(
            z_dim * 2, z_dim * 2, kernel_size=1, padding=0
        )
        self.post_quant_conv = WanCausalConv2dTK(z_dim, z_dim, kernel_size=1, padding=0)

        self.decoder = WanDecoder2DTK(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_upsample=self.temporal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )

        # Pre-compute causal conv counts to size the cache arrays quickly
        self._cached_conv_counts = {
            "decoder": (
                sum(isinstance(m, WanCausalConv2dTK) for m in self.decoder.modules())
                if self.decoder is not None
                else 0
            ),
            "encoder": (
                sum(isinstance(m, WanCausalConv2dTK) for m in self.encoder.modules())
                if self.encoder is not None
                else 0
            ),
        }

    # ---------------- Cache management (WAN-style) ----------------
    def clear_cache(self):
        # decoder
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # encoder
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _chunked_encode_enabled(self) -> bool:
        return os.environ.get(
            "PRISM_VAE_2D_CHUNKED_ENCODE",
            os.environ.get("VERMO_VQVAE_2D_CHUNKED_ENCODE", "1"),
        ) != "0"

    def _encode_chunk_frames(self) -> int:
        raw = os.environ.get(
            "PRISM_VAE_2D_ENCODE_CHUNK_FRAMES",
            os.environ.get("VERMO_VQVAE_2D_ENCODE_CHUNK_FRAMES", "4"),
        )
        try:
            chunk_frames = int(raw)
        except ValueError:
            chunk_frames = 4
        chunk_frames = max(4, chunk_frames)
        return chunk_frames - (chunk_frames % 4)

    def _encode_with_cache_chunks(self, x: torch.Tensor, T: int) -> torch.Tensor:
        self._enc_conv_idx = [0]
        outputs = [
            self.encoder(
                x[:, :, :1],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
        ]

        chunk_frames = self._encode_chunk_frames()
        usable_T = 1 + max(0, (T - 1) // 4) * 4
        for start in range(1, usable_T, chunk_frames):
            self._enc_conv_idx = [0]
            outputs.append(
                self.encoder(
                    x[:, :, start : min(start + chunk_frames, usable_T)],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            )
        return torch.cat(outputs, 2)

    # ---------------- Internal encode/decode on [B, C, T] ----------------
    def _encode(self, x: torch.Tensor):
        # x_bct: [B, C_in, T]
        _, _, T, _ = x.shape
        self.clear_cache()

        if self._chunked_encode_enabled():
            out = self._encode_with_cache_chunks(x, T)
        else:
            out = self.encoder(x)

        # 1x1 causal conv (no padding, no cache needed)
        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    @apply_forward_hook
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, K, C] → latent_dist over [B, 2*z_dim, T, K] (mean & logvar packed along channel dim).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input as [B, T, K, C], got {tuple(x.shape)}")
        x_bct = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, K]

        # VAE must run in fp32 — override any global AMP autocast.
        device_type = x_bct.device.type
        with torch.autocast(device_type, enabled=False):
            h = self._encode(x_bct.float())

        return h

    def _decode(self, z: torch.Tensor):
        # z_bct: [B, z_dim, T]
        _, _, num_frames, _ = z.shape
        self.clear_cache()

        # 1x1 causal conv (no padding, no cache needed)
        x = self.post_quant_conv(z)

        outputs = []
        for i in range(num_frames):
            self._conv_idx = [0]
            if i == 0:
                outputs.append(
                    self.decoder(
                        x[:, :, i : i + 1],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                )
            else:
                outputs.append(
                    self.decoder(
                        x[:, :, i : i + 1],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )
                )

        self.clear_cache()
        return torch.cat(outputs, 2)

    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, z_dim, T, K] → sample [B, T, K, C_out]
        """
        if z.dim() != 4:
            raise ValueError(f"Expected latent as [B, z_dim, T], got {tuple(z.shape)}")

        # VAE must run in fp32 — override any global AMP autocast.
        device_type = z.device.type
        with torch.autocast(device_type, enabled=False):
            decoded = self._decode(z.float())  # [B, C_out, T, K]

        # Return [B, T, J, C] to match motion use-cases; no clamping for motion params
        decoded = decoded.permute(0, 2, 3, 1).contiguous()
        return decoded

    def forward(
        self,
        sample: torch.Tensor,  # [B, T, C_in]
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
        return_posterior: bool = False,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Encode → sample/mode → decode. Returns reconstructed motion `[B, T, C_out]`.
        """
        if sample.dim() != 4:
            raise ValueError(
                f"Expected sample as [B, T, K, C], got {tuple(sample.shape)}"
            )

        z = self.encode(sample)
        dist = DiagonalGaussianDistributionNd(z)
        z = dist.sample(generator=generator) if sample_posterior else dist.mode()
        dec = self.decode(z)
        if return_posterior:
            return dec, dist.parameters
        return dec


    # ----------------------------- Test code ----------------------------- #
if __name__ == "__main__":
    from mmengine import Config
    from mmengine.runner import load_checkpoint

    print(
        "run python -m mmotion.models.autoencoders.autoencoder_kl_wanmotion.autoencoder_kl_wanmotion_2d"
    )

    cfg = "configs/smpl_vae/smpl_vae2dtk_static_aug_hq.py"
    checkpoint = "work_dirs/smpl_vae2dtk_nostatic_aug_hq/iter_334000.pth"
    latents_mean = [
        -5.699e-03,
        5.415e-03,
        1.639e-03,
        2.7085e-02,
        2.068e-03,
        1.5188e-02,
        -6.291e-03,
        -7.814e-03,
        -6.0711e-02,
        -2.166e-03,
        1.1075e-02,
        -4.04e-04,
        1.592e-03,
        2.6383e-02,
        -4.833e-03,
        8.07e-04,
    ]

    latents_std = [
        0.993707,
        1.020968,
        0.996201,
        1.025335,
        0.997547,
        1.035847,
        1.008814,
        0.999811,
        0.980396,
        1.000318,
        1.033794,
        0.993485,
        0.998681,
        1.038657,
        1.001396,
        0.997597,
    ]

    cfg = Config.fromfile(cfg)["model"]
    cfg["vae"]["latents_mean"] = latents_mean
    cfg["vae"]["latents_std"] = latents_std
    cfg["vae"]["use_static"] = False
    cfg["vae"]["use_rollout_trans"] = True
    model = MODELS.build(cfg)
    load_checkpoint(model, checkpoint, strict=True, map_location="cpu")
    model: AutoencoderKLPrism2DTK = model.vae

    model.save_pretrained("checkpoints/wanmo_vae2d_aug")

    print("SMPL VAE 2D Configuration:")
    print(model.config)
    print(f"Successfully saved to checkpoints/wanmo_vae2d_aug")
