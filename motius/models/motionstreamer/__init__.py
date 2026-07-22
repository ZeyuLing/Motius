"""MotionStreamer (causal TAE + LLaMA AR + diffusion head) bundle.

Open-source text-to-motion baseline integrated into the Motius zoo. The TAE,
the LLaMA autoregressive transformer, the per-token diffusion head and the
Gaussian-diffusion sampler live in
``motius.models.motionstreamer.network``. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from motius.models.motionstreamer.bundle import MotionStreamerBundle
from motius.models.motionstreamer.evaluator import MotionStreamer272Evaluator

__all__ = ["MotionStreamer272Evaluator", "MotionStreamerBundle"]
