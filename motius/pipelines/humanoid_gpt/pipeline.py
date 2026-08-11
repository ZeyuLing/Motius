"""Task API for the released Humanoid-GPT G1 motion tracker."""

from __future__ import annotations

from typing import Any, Mapping

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class HumanoidGPTPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.humanoid_gpt.HumanoidGPTBundle"

    def infer_motion_tracking(
        self,
        observation: Any | None = None,
        *,
        components: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one official 50 Hz Humanoid-GPT controller step."""

        output = dict(
            self.bundle.forward(observation=observation, components=components)
        )
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1-5010 29-DOF",
                "method": "HumanoidGPT",
            }
        )
        return output

    __call__ = infer_motion_tracking


__all__ = ["HumanoidGPTPipeline"]
