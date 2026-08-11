"""MoGenTS (spatial-temporal T2M) bundle.

NeurIPS'24 open-source text-to-motion model integrated into the Motius Model
Zoo. Runtime components live in ``motius.models.mogents.network``;
raw upstream checkpoints are handled by converter/debug scripts.
"""

from motius.models.mogents.bundle import MoGenTSBundle

__all__ = ["MoGenTSBundle"]
