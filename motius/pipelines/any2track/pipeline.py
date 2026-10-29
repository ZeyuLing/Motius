"""Task API for the OpenTrack Any2Track G1 generalist."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from motius.models.onnx_policy import as_numpy
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class Any2TrackPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.any2track.Any2TrackBundle"

    def assemble_observation(self, components: Mapping[str, Any]) -> np.ndarray:
        missing = [key for key in self.bundle.observation_keys if key not in components]
        extra = sorted(set(components) - set(self.bundle.observation_keys))
        if missing or extra:
            raise ValueError(
                f"Any2Track observation components mismatch: missing={missing}, "
                f"extra={extra}."
            )
        arrays = []
        batch_shape = None
        for key in self.bundle.observation_keys:
            value = as_numpy(components[key], np.dtype(np.float32))
            if value.ndim == 1:
                value = value[None]
            if batch_shape is None:
                batch_shape = value.shape[:-1]
            elif value.shape[:-1] != batch_shape:
                raise ValueError(
                    f"Any2Track component {key!r} has batch shape "
                    f"{value.shape[:-1]}, expected {batch_shape}."
                )
            arrays.append(value)
        observation = np.concatenate(arrays, axis=-1)
        if observation.shape[-1] != self.bundle.OBSERVATION_DIM:
            raise ValueError(
                f"Any2Track components produce {observation.shape[-1]} values, "
                f"expected {self.bundle.OBSERVATION_DIM}."
            )
        return observation

    def infer_motion_tracking(self, observation: Any) -> dict[str, Any]:
        """Run one 50 Hz Any2Track step from a vector or named components."""

        if isinstance(observation, Mapping):
            observation = self.assemble_observation(observation)
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
