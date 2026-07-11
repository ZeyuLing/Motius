"""BaseVisualizer: abstract base class for visualizers."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.

    Visualizers consume the pure dict output from val_step and log
    images/videos/text to TensorBoard, W&B, or other backends.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval
        self._step_count = 0

    @abstractmethod
    def visualize(self, output: Dict[str, Any], step: int) -> None:
        """
        Visualize one batch of validation output.

        Args:
            output: val_step return dict
            step: global training step
        """

    def should_visualize(self, step: int) -> bool:
        return step % self.interval == 0 if self.interval > 0 else False
