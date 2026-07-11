"""MotionLCM (latent consistency model) bundle.

Open-source baseline integrated into the Motius zoo. The MLD motion VAE,
latent consistency denoiser and text encoder live in
``motius.models.motionlcm.network``. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from motius.models.motionlcm.bundle import MotionLCMBundle

__all__ = ["MotionLCMBundle"]
