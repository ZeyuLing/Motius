# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from typing import Optional, Tuple, Union, List
from einops import rearrange
from torch import nn
import logging
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


from diffusers.utils.accelerate_utils import apply_forward_hook  # diffusers utility

from diffusers.loaders import FromOriginalModelMixin

from motius.registry import HF_MODELS, MODELS
from .wan_blocks.wan_causalconv import WanCausalConv2dTK
from .wan_blocks.wan_encdec import WanEncoder2DTK, WanDecoder2DTK

logger = logging.getLogger(__name__)


@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQEncoder.
    """

    quant: torch.Tensor
    indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.FloatTensor] = None


@dataclass
class VQVAEOutput(BaseOutput):
    """
    Output of VQ-VAE forward.
    """

    quant: torch.Tensor
    sample: torch.Tensor
    indices: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.FloatTensor] = None


@HF_MODELS.register_module(force=True)
class VQVAEVermo2DTK(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Motion VQ-VAE 2D: input [B,T,K,C] with (T,K) grid and C channels. Causal VAE for SMPL/SMPL-X sequences `[B, T, C]` (temporal-only), aligned with Diffusers'
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

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

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
        quantizer_cfg: nn.Module = None,
        scale_factor_temporal: Optional[int] = 4,
        # following params are not used in encode, decode, forward, but will be useful in latent diffusion model training and post-processing
        use_static: bool = False,  # add a static joint label to post-process the motion
        use_rollout_trans: bool = True,  # use rollout translation to g
        token_rescale: bool = True,  # learnable per-token scale+bias before quantization
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
            z_dim=z_dim,  # mean+logvar
            dim_mult=list(dim_mult),
            attn_scales=attn_scales,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        # IMPORTANT: 1x1 causal convs must use padding=0 (parity with WAN-3D)
        self.quant_conv = WanCausalConv2dTK(z_dim, z_dim, kernel_size=1, padding=0)
        self.post_quant_conv = WanCausalConv2dTK(z_dim, z_dim, kernel_size=1, padding=0)
        self.quantizer = MODELS.build(quantizer_cfg)
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

        # Learnable per-token-group rescaling in latent space (after quant_conv,
        # before quantizer).  Translation (token 0) and rotation (tokens 1+)
        # have very different dynamic ranges after encoding; the shared FSQ
        # codebook quantizes both uniformly, causing translation reconstruction
        # to suffer.  Two separate affine transforms let the model compress/shift
        # each group into a range that the quantizer can represent equally well.
        # Shape: [1, z_dim, 1, 1] — per-channel, broadcasts across T.
        self.token_rescale = token_rescale
        if token_rescale:
            self._transl_scale = nn.Parameter(torch.ones(1, z_dim, 1, 1))
            self._transl_bias = nn.Parameter(torch.zeros(1, z_dim, 1, 1))
            self._rot_scale = nn.Parameter(torch.ones(1, z_dim, 1, 1))
            self._rot_bias = nn.Parameter(torch.zeros(1, z_dim, 1, 1))

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
        return os.environ.get("VERMO_VQVAE_2D_CHUNKED_ENCODE", "1") != "0"

    def _encode_chunk_frames(self) -> int:
        raw = os.environ.get("VERMO_VQVAE_2D_ENCODE_CHUNK_FRAMES", "4")
        try:
            chunk_frames = int(raw)
        except ValueError:
            chunk_frames = 4
        chunk_frames = max(4, chunk_frames)
        return max(4, chunk_frames - (chunk_frames % 4))

    def _encode_with_cache_chunks(self, x: torch.Tensor, T: int) -> torch.Tensor:
        # WAN-style temporal chunking: first chunk 1 frame, then groups whose
        # length is a multiple of 4. The original path used 4-frame groups; using
        # larger groups keeps cache semantics and token alignment while avoiding
        # thousands of tiny conv launches for very long training clips.
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
        # b c t k
        enc = self.quant_conv(out)

        # Per-token-group rescaling before quantization
        if self.token_rescale:
            enc_transl = enc[:, :, :, :1] * self._transl_scale + self._transl_bias
            enc_rot = enc[:, :, :, 1:] * self._rot_scale + self._rot_bias
            enc = torch.cat([enc_transl, enc_rot], dim=3)

        K = enc.shape[-1]
        enc = rearrange(enc, "b c t k -> b c (t k)")
        x_quantized, indices, commit_loss, _ = self.quantizer(enc)
        x_quantized = rearrange(x_quantized, "b c (t k) -> b c t k", k=K)
        indices = rearrange(indices, "b (t k) -> b t k", k=K)
        self.clear_cache()
        return VQEncoderOutput(
            quant=x_quantized, indices=indices, commit_loss=commit_loss
        )

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, flatten: bool = False, return_dict: bool = True
    ) -> VQEncoderOutput:
        """
        x: [B, T, K, C] → latent_dist over [B, z_dim, T, K] (mean & logvar packed along channel dim).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input as [B, T, K, C], got {tuple(x.shape)}")
        x_bct = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, K]

        # VQ-VAE must run in fp32 — override any global AMP autocast.
        device_type = x_bct.device.type
        with torch.autocast(device_type, enabled=False):
            enc_out = self._encode(x_bct.float())

        h, h_idx, commit_loss = enc_out.quant, enc_out.indices, enc_out.commit_loss
        if flatten:
            h = rearrange(h, "b c t k -> b c (t k)")
            h_idx = rearrange(h_idx, "b t k -> b (t k)")

        if not return_dict:
            return h, h_idx, commit_loss
        return VQEncoderOutput(quant=h, indices=h_idx, commit_loss=commit_loss)

    def _indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        if hasattr(self.quantizer, "dequantize"):
            return self.quantizer.dequantize(indices)
        if hasattr(self.quantizer, "indices_to_codes"):
            return self.quantizer.indices_to_codes(indices)
        raise AttributeError("Quantizer does not provide an indices->latents method.")

    def _decode(self, z: torch.Tensor, is_indices: bool = False):
        # z_bct: [B, z_dim, T]
        if is_indices:
            z = self._indices_to_latents(z)

        # Inverse per-token-group rescaling (undo encode-side transform)
        if self.token_rescale:
            z_transl = (z[:, :, :, :1] - self._transl_bias) / self._transl_scale
            z_rot = (z[:, :, :, 1:] - self._rot_bias) / self._rot_scale
            z = torch.cat([z_transl, z_rot], dim=3)

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
    def decode(
        self,
        z: torch.Tensor,
        flatten: bool = False,
        is_indices: bool = False,
        K: int = 24,  # 1 for transl, 22 for each joints' rotation, 6 for static joints
    ) -> torch.Tensor:
        """
        z: [B, z_dim, T, K] → sample [B, T, K, C_out]
        """
        if not is_indices:
            if not flatten and z.dim() != 4:
                raise ValueError(
                    f"Expected latent as [B, z_dim, T, K], got {tuple(z.shape)}"
                )
            elif flatten and z.dim() != 3:
                raise ValueError(
                    f"Expected latent as [B, z_dim, T*K], got {tuple(z.shape)}"
                )
        elif is_indices:
            if not flatten and z.dim() != 3:
                raise ValueError(f"Expected indices as [B, T, K], got {tuple(z.shape)}")
            elif flatten and z.dim() != 2:
                raise ValueError(f"Expected indices as [B, T*K], got {tuple(z.shape)}")

        if flatten:
            z = rearrange(z, "... (t k) -> ... t k", k=K)

        # VQ-VAE must run in fp32 — override any global AMP autocast.
        device_type = z.device.type
        with torch.autocast(device_type, enabled=False):
            decoded = self._decode(z.float() if not is_indices else z, is_indices)  # [B, C_out, T, K]

        # Return [B, T, C_out] to match motion use-cases; no clamping for motion params
        decoded = decoded.permute(0, 2, 3, 1).contiguous()
        return decoded

    def forward(
        self,
        sample: torch.Tensor,  # [B, T, C_in]
        return_dict: bool = True,
    ) -> Union[VQVAEOutput, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encode -> Quantize -> Decode. Returns reconstructed sequence and quantization intermediates.
        """
        if sample.dim() != 4:
            raise ValueError(
                f"Expected sample as [B, T, K, C], got {tuple(sample.shape)}"
            )

        enc_out: VQEncoderOutput = self.encode(sample, return_dict=True)
        z_quant, indices, commit_loss = (
            enc_out.quant,
            enc_out.indices,
            enc_out.commit_loss,
        )
        dec = self.decode(z_quant)

        out = VQVAEOutput(
            quant=z_quant, sample=dec, indices=indices, commit_loss=commit_loss
        )
        if not return_dict:
            return out.quant, out.sample, out.indices
        return out


def main(
    cfg: str = "configs/smpl_vae/smpl_vqvae2dtk_nostatic_aug_4375.py",
    checkpoint: str = "work_dirs/smpl_vqvae2dtk_nostatic_aug_4375/iter_430000.pth",
    save_dir: str = "checkpoints/vermo_vqvae_nostatic_aug_hymotion_4375_iter_430000",
):
    # ----------------------------- Test code ----------------------------- #
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    os.chdir(_project_root)
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
