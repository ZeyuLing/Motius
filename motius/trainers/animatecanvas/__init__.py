"""AnimateCanvas training loops."""

from motius.trainers.animatecanvas.sparse_rollout_join_trainer import (
    AnimateCanvasSparseRolloutJoinTrainer,
)
from motius.trainers.animatecanvas.trainer import AnimateCanvasTrainer

__all__ = ['AnimateCanvasTrainer', 'AnimateCanvasSparseRolloutJoinTrainer']
