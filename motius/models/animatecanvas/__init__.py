"""AnimateCanvas: unified text-guided motion generation and editing."""

from motius.models.animatecanvas.bundle import AnimateCanvasBundle
from motius.registry import MODEL_BUNDLES

MODEL_BUNDLES.register_module(name='MotionCanvasBundle', module=AnimateCanvasBundle, force=True)

__all__ = ['AnimateCanvasBundle']
