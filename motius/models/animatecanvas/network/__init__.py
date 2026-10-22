"""AnimateCanvas network implementation and registry surface."""

from motius.models.animatecanvas.network.hymotion_mmdit import (
    HunyuanMotionMMDiT,
)
from motius.models.animatecanvas.network.hymotion_dit import (
    HunyuanMotionDiT,
)
from motius.registry import HF_MODELS


class AnimateCanvasMMDiT(HunyuanMotionMMDiT):
    """Namespaced AnimateCanvas backbone with source-identical parameters."""


HF_MODELS.register_module(
    name='AnimateCanvasMMDiT',
    module=AnimateCanvasMMDiT,
    force=True,
)

# Old Hub artifacts keep their original backbone type; weights are identical.
HF_MODELS.register_module(name='MotionCanvasMMDiT', module=AnimateCanvasMMDiT, force=True)

__all__ = ['AnimateCanvasMMDiT', 'HunyuanMotionMMDiT', 'HunyuanMotionDiT']
