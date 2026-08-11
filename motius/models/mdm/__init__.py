"""MDM (Motion Diffusion Model) bundle.

Open-source baseline integrated into the Motius zoo. The neural network and
Gaussian diffusion live in ``motius.models.mdm.network``. Runtime
loading is artifact-based; raw upstream checkpoints are handled by converter
code.
"""

from motius.models.mdm.bundle import MDMBundle

__all__ = ["MDMBundle"]
