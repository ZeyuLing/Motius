"""Backend-neutral interfaces for closed-loop motion tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class MotionTrackingEnvironment(ABC):
    """Minimal contract shared by MuJoCo and Isaac Lab tracking environments."""

    backend: str

    @abstractmethod
    def reset(self) -> Mapping[str, Any]:
        """Reset from the first reference frame and return reset metadata."""

    @abstractmethod
    def policy_inputs(self, pipeline: Any) -> Mapping[str, Any]:
        """Build one method-native policy input from simulator state."""

    @abstractmethod
    def step(self, pipeline: Any, policy_output: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply one policy output and advance the physical simulation."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the current rollout has terminated."""

    @abstractmethod
    def result(self) -> Any:
        """Return the finalized rollout artifact and metric summary."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

    def __enter__(self) -> "MotionTrackingEnvironment":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
