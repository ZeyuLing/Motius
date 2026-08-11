# -*- coding: utf-8 -*-
"""
VQ-VAE 1D for Motion: input/output [B, T, C], supports arbitrary-length encode/decode.
Attention is only on channel/spatial dims, independent of T, so it generalizes to arbitrary length.

mid_attention options:
- linear_channel (recommended): channel-wise linear attention, cost O(T*C*d), suitable for large C (e.g. 512).
- joint_tokens: aggregate by joint K into K tokens, then K×K attention, explicit channel-joint mapping, cost O(T*(C^2/K+K^2*d)).
- kwise: split channels into K parts, full attention within groups, cost O(T*C^2/K); no real joint semantics in 1D.
- temporal: temporal causal attention, less suitable for arbitrary-length generalization, kept as optional.
- none: no attention.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Optional, Tuple, Union, List, Any

from torch import nn
import logging
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.loaders import FromOriginalModelMixin

from motius.registry import HF_MODELS, MODELS

from .wan_blocks.wan_causalconv import WanCausalConv1d
from .wan_blocks.wan_encdec import WanEncoder1D, WanDecoder1D

logger = logging.getLogger(__name__)


@dataclass
class VQEncoderOutput(BaseOutput):
    quant: torch.Tensor
    indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.FloatTensor] = None


@dataclass
class VQVAEOutput(BaseOutput):
    quant: torch.Tensor
    sample: torch.Tensor
    indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# VQ-VAE 1D (reuses wan_encdec.WanEncoder1D / WanDecoder1D + mid_attention)
# ---------------------------------------------------------------------------
@HF_MODELS.register_module(force=True)
class VQVAEVermo1D(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    Motion VQ-VAE 1D: input/output [B, T, C], supports arbitrary length.
    mid_attention: "linear_channel" (recommended) / "joint_tokens" / "kwise" / "temporal" / "none".
    """

    _supports_gradient_checkpointing = False

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    @register_to_config
    def __init__(
        self,
        base_dim: int = 128,
        decoder_base_dim: Optional[int] = None,
        z_dim: int = 64,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[int] = [],
        temporal_downsample: Tuple[bool, ...] = (False, False, False),
        dropout: float = 0.0,
        is_residual: bool = True,
        in_channels: int = 144,  # (J+1)*6, e.g. 24*6
        out_channels: int = 144,
        num_joints: int = 24,
        mid_attention: str = "linear_channel",
        channel_proj_dim: int = 128,
        joint_token_dim: Optional[int] = None,
        temporal_window_size: Optional[int] = 64,
        quantizer_cfg: Any = None,
        scale_factor_temporal: Optional[int] = 4,
        use_static: bool = False,
        use_rollout_trans: bool = True,
    ) -> None:
        """
        Args:
            channel_proj_dim: used only when mid_attention=="linear_channel"; projection dim for linear attention,
                larger = more expressive but more compute, default 128, try 256/512 for better quality.
            joint_token_dim: used only when mid_attention=="joint_tokens"; if None, internal default 64.
                Other attention types can pass None, it will not be used.
        """
        super().__init__()
        self.z_dim = z_dim
        self.temporal_downsample = list(temporal_downsample)
        self.temporal_upsample = list(temporal_downsample)[::-1]
        self.num_joints = num_joints

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder1D(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_downsample=self.temporal_downsample,
            dropout=dropout,
            non_linearity="silu",
            is_residual=is_residual,
            mid_attention=mid_attention,
            num_joints=num_joints,
            channel_proj_dim=channel_proj_dim,
            joint_token_dim=joint_token_dim,
            temporal_window_size=temporal_window_size,
        )
        self.quant_conv = WanCausalConv1d(z_dim, z_dim, kernel_size=1, padding=0)
        self.post_quant_conv = WanCausalConv1d(z_dim, z_dim, kernel_size=1, padding=0)
        assert quantizer_cfg is not None, "quantizer_cfg is required"
        self.quantizer = MODELS.build(quantizer_cfg)
        self.decoder = WanDecoder1D(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            temporal_upsample=self.temporal_upsample,
            out_channels=out_channels,
            dropout=dropout,
            non_linearity="silu",
            is_residual=is_residual,
            mid_attention=mid_attention,
            num_joints=num_joints,
            channel_proj_dim=channel_proj_dim,
            joint_token_dim=joint_token_dim,
            temporal_window_size=temporal_window_size,
        )

        self._cached_conv_counts = {
            "decoder": sum(
                1 for m in self.decoder.modules() if isinstance(m, WanCausalConv1d)
            ),
            "encoder": sum(
                1 for m in self.encoder.modules() if isinstance(m, WanCausalConv1d)
            ),
        }

    def clear_cache(self):
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor) -> VQEncoderOutput:
        # x: [B, C, T]
        _, _, T = x.shape
        self.clear_cache()
        iters = 1 + max(0, (T - 1) // 4)
        out_list: List[torch.Tensor] = []
        for i in range(iters):
            self._enc_conv_idx = [0]
            if i == 0:
                out_list.append(
                    self.encoder(
                        x[:, :, :1],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                )
            else:
                out_list.append(
                    self.encoder(
                        x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                )
        out = torch.cat(out_list, 2)
        enc = self.quant_conv(out)
        x_quantized, indices, commit_loss, _ = self.quantizer(enc)
        self.clear_cache()
        return VQEncoderOutput(
            quant=x_quantized, indices=indices, commit_loss=commit_loss
        )

    @apply_forward_hook
    def encode(
        self,
        x: torch.Tensor,
        flatten: bool = False,
        return_dict: bool = True,
    ) -> Union[
        VQEncoderOutput,
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
        """x: [B, T, C] -> VQEncoderOutput(quant [B, z_dim, T])."""
        if x.dim() != 3:
            raise ValueError(f"Expected input [B, T, C], got {tuple(x.shape)}")
        x_bct = x.permute(0, 2, 1).contiguous()
        # VQ-VAE must run in fp32 — override any global AMP autocast.
        device_type = x_bct.device.type
        with torch.autocast(device_type, enabled=False):
            enc_out = self._encode(x_bct.float())
        h, h_idx, commit_loss = enc_out.quant, enc_out.indices, enc_out.commit_loss
        if not return_dict:
            return h, h_idx, commit_loss
        return enc_out

    def _indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        if hasattr(self.quantizer, "dequantize"):
            return self.quantizer.dequantize(indices)
        if hasattr(self.quantizer, "indices_to_codes"):
            return self.quantizer.indices_to_codes(indices)
        raise AttributeError("Quantizer does not provide indices->latents method.")

    def _decode(self, z: torch.Tensor, is_indices: bool = False) -> torch.Tensor:
        if is_indices:
            z = self._indices_to_latents(z)
        _, _, num_frames = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)
        out_list: List[torch.Tensor] = []
        for i in range(num_frames):
            self._conv_idx = [0]
            out_list.append(
                self.decoder(
                    x[:, :, i : i + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=(i == 0),
                )
            )
        out = torch.cat(out_list, 2)
        self.clear_cache()
        return out

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        flatten: bool = False,
        is_indices: bool = False,
    ) -> torch.Tensor:
        """z: [B, z_dim, T] or indices [B, T] -> [B, T, C]."""
        if is_indices:
            if z.dim() != 2:
                raise ValueError(f"Expected indices [B, T], got {tuple(z.shape)}")
        else:
            if z.dim() != 3:
                raise ValueError(f"Expected latent [B, z_dim, T], got {tuple(z.shape)}")
        # VQ-VAE must run in fp32 — override any global AMP autocast.
        device_type = z.device.type
        with torch.autocast(device_type, enabled=False):
            decoded = self._decode(z.float() if not is_indices else z, is_indices)
        return decoded.permute(0, 2, 1).contiguous()

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[VQVAEOutput, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """sample: [B, T, C] -> encode -> quantize -> decode -> [B, T, C]."""
        if sample.dim() != 3:
            raise ValueError(f"Expected sample [B, T, C], got {tuple(sample.shape)}")
        enc_out = self.encode(sample, return_dict=True)
        assert isinstance(enc_out, VQEncoderOutput)
        z_quant = enc_out.quant
        indices = enc_out.indices
        commit_loss = enc_out.commit_loss
        dec = self.decode(z_quant)
        out = VQVAEOutput(
            quant=z_quant, sample=dec, indices=indices, commit_loss=commit_loss
        )
        if not return_dict:
            return out.quant, out.sample, out.indices
        return out


def main(
    cfg: str = "configs/smpl_vae/smpl_vqvae1d_1x_nostatic_aug_rvq_1kx6_none_hq.py",
    checkpoint: str = "work_dirs/smpl_vqvae1d_1x_nostatic_aug_rvq_1kx6_none_hq/iter_77000.pth",
    save_dir: str = "checkpoints/vermo_vqvae1d_rvq_1kx6_none_hq_iter_77000",
):
    from mmengine.device import get_device
    from mmengine import Config
    from mmengine.runner import load_checkpoint

    device = get_device()
    dtype = torch.bfloat16

    cfg = Config.fromfile(cfg)["model"]
    model = MODELS.build(cfg)
    load_checkpoint(model, checkpoint, map_location="cpu", strict=True)
    model = model.to(device, dtype)

    model.vqvae.save_pretrained(save_dir)
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
