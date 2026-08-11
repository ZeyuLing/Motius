"""MoMask (RVQ + Masked Transformer + Residual Transformer) bundle.

Open-source baseline integrated into the Motius zoo. The RVQ-VAE tokenizer,
masked / residual transformers and length estimator live in
``motius.models.momask.network``. Runtime loading is artifact-based;
raw upstream checkpoints are handled by converter/debug code.
"""

from motius.models.momask.bundle import MoMaskBundle

__all__ = ["MoMaskBundle"]
