"""Pipeline mixin that exposes physical rollout without eager simulator imports."""

from __future__ import annotations

from typing import Any


class MotionTrackingPipelineMixin:
    def rollout_motion_tracking(self, reference_motion: Any, **kwargs: Any) -> Any:
        """Track one reference in MuJoCo or Isaac Lab and return trajectories and metrics."""

        from motius.simulators.rollout import run_motion_tracking_rollout

        return run_motion_tracking_rollout(
            self,
            reference_motion=reference_motion,
            **kwargs,
        )
