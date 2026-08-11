"""Task API for the ProtoMotions G1 deployment tracker."""

from __future__ import annotations

from typing import Any, Mapping

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class ProtoMotionsPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.protomotions.ProtoMotionsBundle"

    def infer_motion_tracking(
        self,
        observations: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one 50 Hz policy step using the unified ONNX input contract."""

        output = dict(self.bundle.forward(observations))
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1 29-DOF",
                "method": "ProtoMotions",
            }
        )
        return output

    __call__ = infer_motion_tracking


__all__ = ["ProtoMotionsPipeline"]
