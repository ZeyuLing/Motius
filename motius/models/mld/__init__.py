"""MLD (Motion Latent Diffusion) bundle.

Open-source T2M baseline integrated into the Motius zoo. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from motius.models.mld.bundle import MLDBundle

__all__ = ["MLDBundle"]
