"""Task API for user-exported BeyondMimic G1 policies."""

from __future__ import annotations

from typing import Any

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class BeyondMimicPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.beyondmimic.BeyondMimicBundle"

    def infer_motion_tracking(
        self,
        observation: Any,
        time_step: int,
    ) -> dict[str, Any]:
        """Run one policy step and return action plus embedded reference state."""

        output = dict(self.bundle.forward(observation, time_step))
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1",
                "method": "BeyondMimic",
            }
        )
        return output

    __call__ = infer_motion_tracking


__all__ = ["BeyondMimicPipeline"]
