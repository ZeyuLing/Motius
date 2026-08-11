"""MotionCanvas network implementation and registry surface."""

from motius.models.motioncanvas.network.hymotion_mmdit import (
    HunyuanMotionMMDiT,
)
from motius.models.motioncanvas.network.hymotion_dit import (
    HunyuanMotionDiT,
)
from motius.registry import HF_MODELS


class MotionCanvasMMDiT(HunyuanMotionMMDiT):
    """Namespaced MotionCanvas backbone with source-identical parameters."""


HF_MODELS.register_module(
    name='MotionCanvasMMDiT',
    module=MotionCanvasMMDiT,
    force=True,
)

__all__ = ['MotionCanvasMMDiT', 'HunyuanMotionMMDiT', 'HunyuanMotionDiT']
