"""Task API for NVIDIA SONIC G1 motion tracking."""

from __future__ import annotations

from typing import Any

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class SONICPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.sonic.SONICBundle"

    def infer_motion_tracking(
        self,
        encoder_observation: Any,
        decoder_observation: Any,
    ) -> dict[str, Any]:
        """Run one 50 Hz SONIC policy step from deployment observations."""

        output = dict(
            self.bundle.forward(
                encoder_observation=encoder_observation,
                decoder_observation=decoder_observation,
            )
        )
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1 29-DOF",
                "method": "SONIC",
            }
        )
        return output

    __call__ = infer_motion_tracking


__all__ = ["SONICPipeline"]
