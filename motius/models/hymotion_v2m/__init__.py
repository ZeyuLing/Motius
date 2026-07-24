"""HYMotion-V2M video-feature-to-motion model bundle.

Stage 1 exposes pre-extracted-feature -> motion inference through the standard
Bundle / Pipeline / Registry surface, wrapping the vendored, self-contained
``MotionGenerationV2M`` source so the released checkpoints load numerically
unchanged without importing another repository.
"""

from .bundle import HyMotionV2MBundle

__all__ = ["HyMotionV2MBundle"]
