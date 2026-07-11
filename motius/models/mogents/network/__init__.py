"""MoGenTS model components used by Motius.

The runtime keeps the official MIT MoGenTS architecture local to Motius:
the dual-codebook RVQ-VAE, 1D auxiliary token transformers and 2D
spatial-temporal transformers. Training-only utilities from the upstream
checkout are intentionally not imported here.
"""

from .inference import estimate_token_lengths, generate_motion
from .transformer import (
    MaskTransformer,
    MaskTransformer2D,
    ResidualTransformer,
    ResidualTransformer2D,
)
from .vq import LengthEstimator, RVQVAE

__all__ = [
    "RVQVAE",
    "LengthEstimator",
    "MaskTransformer",
    "MaskTransformer2D",
    "ResidualTransformer",
    "ResidualTransformer2D",
    "generate_motion",
    "estimate_token_lengths",
]
