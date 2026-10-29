"""MotionCLR fused conditional/unconditional batch sampling."""

from typing import Optional
import torch
from torch.nn import functional as F


def predict_with_cfg(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    text: Optional[torch.Tensor] = None,
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
