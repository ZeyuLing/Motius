"""MotionStreamer (causal TAE + LLaMA AR + diffusion head) bundle.

Open-source text-to-motion baseline integrated into the Motius zoo. The TAE, LLaMA transformer and per-token diffusion head live in
``motius.models.motionstreamer.network``; sampling is in the pipeline and
``motius.diffusion.motionstreamer``. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from motius.models.motionstreamer.bundle import MotionStreamerBundle
from motius.models.motionstreamer.evaluator_bundle import MotionStreamer272Evaluator

__all__ = ["MotionStreamer272Evaluator", "MotionStreamerBundle"]
