"""Task API for the released Humanoid-GPT G1 motion tracker."""

from __future__ import annotations

from typing import Any, Mapping

from motius.configuration.humanoid_gpt import DEFAULT_JOINT_POSITION, ACTION_SCALE
import numpy as np
from motius.models.onnx_policy import as_numpy
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class HumanoidGPTPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.humanoid_gpt.HumanoidGPTBundle"

    def assemble_observation(self, components: Mapping[str, Any]) -> np.ndarray:
        """Assemble the official 136D observation without reordering fields."""

        expected_names = [name for name, _ in self.bundle.OBSERVATION_LAYOUT]
        missing = sorted(set(expected_names) - set(components))
        extra = sorted(set(components) - set(expected_names))
        if missing or extra:
            raise ValueError(
                f"HumanoidGPT observation components mismatch: missing={missing}, "
                f"extra={extra}."
            )
        arrays = []
        batch_size = None
        for name, dimension in self.bundle.OBSERVATION_LAYOUT:
            value = as_numpy(components[name], np.dtype(np.float32))
            if value.ndim == 1:
                value = value[None]
            if value.ndim != 2 or value.shape[1] != dimension:
                raise ValueError(
                    f"HumanoidGPT component {name!r} must have shape [B, {dimension}], "
                    f"got {value.shape}."
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"HumanoidGPT observation batch mismatch for {name!r}: "
                    f"expected {batch_size}, got {value.shape[0]}."
                )
            arrays.append(value)
        return np.ascontiguousarray(np.concatenate(arrays, axis=-1))

    def decode_motor_targets(self, action: Any) -> np.ndarray:
        action_array = as_numpy(action, np.dtype(np.float32))
        if action_array.ndim == 1:
            action_array = action_array[None]
        if action_array.ndim != 2 or action_array.shape[1] != self.bundle.ACTION_DIM:
            raise ValueError(
                f"HumanoidGPT action must have shape [B, {self.bundle.ACTION_DIM}], "
                f"got {action_array.shape}."
            )
        return (
            DEFAULT_JOINT_POSITION[None]
            + action_array * np.float32(0.25) * ACTION_SCALE[None]
        ).astype(np.float32, copy=False)

    def infer_motion_tracking(
        self,
        observation: Any | None = None,
        *,
        components: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one official 50 Hz Humanoid-GPT controller step."""

        if (observation is None) == (components is None):
            raise ValueError("Pass exactly one of observation or components.")
        obs = (
            self.assemble_observation(components)
            if components is not None
            else as_numpy(observation, np.dtype(np.float32))
        )
        if obs.ndim == 1:
            obs = obs[None]
        output = dict(self.bundle.forward(obs))
        output.update(
            motor_targets=self.decode_motor_targets(output['continuous_actions']),
            observation=obs,
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
