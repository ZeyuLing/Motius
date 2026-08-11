"""Task API for the OpenTrack Any2Track G1 generalist."""

from __future__ import annotations

from typing import Any

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class Any2TrackPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.any2track.Any2TrackBundle"

    def infer_motion_tracking(self, observation: Any) -> dict[str, Any]:
        """Run one 50 Hz Any2Track step from a vector or named components."""

        output = dict(self.bundle.forward(observation))
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1 29-DOF",
                "method": "Any2Track",
            }
        )
        return output

    __call__ = infer_motion_tracking


__all__ = ["Any2TrackPipeline"]
