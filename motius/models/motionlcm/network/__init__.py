"""MotionLCM model components used by motius.

The package contains the MLD motion VAE, latent consistency denoiser, text
encoder wrapper and sampling helper needed for text-to-motion inference.
Training-only upstream utilities are intentionally outside the runtime package.

MotionLCM (Dai et al., ECCV 2024) distills a latent consistency model from a
pretrained MLD diffusion model and samples motion latents in **1-4 steps**. The
``LCMScheduler`` (diffusers) drives the few-step sampling and the guidance scale
is folded into the timestep conditioning (distilled CFG). The text backbone is
reloaded by name and is **not** part of the artifact.
"""

from .inference import generate_motion, lcm_reverse_diffusion
from .mld_clip import MldTextEncoder
from .mld_denoiser import MldDenoiser
from .mld_vae import MldVae
from .utils import get_guidance_scale_embedding, lengths_to_mask, remove_padding

__all__ = [
    "MldVae",
    "MldDenoiser",
    "MldTextEncoder",
    "generate_motion",
    "lcm_reverse_diffusion",
    "get_guidance_scale_embedding",
    "lengths_to_mask",
    "remove_padding",
]
