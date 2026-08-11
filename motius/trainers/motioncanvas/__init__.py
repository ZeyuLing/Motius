"""MotionCanvas training loops."""

from motius.trainers.motioncanvas.sparse_rollout_join_trainer import (
    MotionCanvasSparseRolloutJoinTrainer,
)
from motius.trainers.motioncanvas.trainer import MotionCanvasTrainer

__all__ = ['MotionCanvasTrainer', 'MotionCanvasSparseRolloutJoinTrainer']
