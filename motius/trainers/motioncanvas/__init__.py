"""Compatibility imports for existing user training configurations."""
from motius.trainers.animatecanvas import (
    AnimateCanvasTrainer,
    AnimateCanvasSparseRolloutJoinTrainer,
)
from motius.registry import TRAINERS

MotionCanvasTrainer = AnimateCanvasTrainer
MotionCanvasSparseRolloutJoinTrainer = AnimateCanvasSparseRolloutJoinTrainer
TRAINERS.register_module(name='MotionCanvasTrainer', module=AnimateCanvasTrainer, force=True)
TRAINERS.register_module(name='MotionCanvasSparseRolloutJoinTrainer', module=AnimateCanvasSparseRolloutJoinTrainer, force=True)
__all__ = ['MotionCanvasTrainer', 'MotionCanvasSparseRolloutJoinTrainer']
