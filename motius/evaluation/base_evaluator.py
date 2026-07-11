"""
BaseEvaluator: abstract base class for all evaluators.

Evaluators consume the pure dict output from val_step and compute metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.

    The evaluation flow:
      1. reset() — called at the start of each val epoch
      2. process(output) — called for each val batch (output is val_step's return dict)
      3. compute() — called at the end of val epoch, returns metrics dict
    """

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    def reset(self):
        """Clear accumulated results. Called at the start of each val epoch."""
        self._results = []

    def process(self, output: Dict[str, Any]) -> None:
        """
        Process one batch of validation output.
        Default implementation just appends to self._results.
        Override if you need custom per-batch processing.
        """
        self._results.append(output)

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated results.

        Returns:
            dict mapping metric name to value (e.g. {'top1_acc': 0.82})
        """

    def compute_from_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Convenience: reset, process all outputs, then compute."""
        self.reset()
        for output in outputs:
            self.process(output)
        return self.compute()
