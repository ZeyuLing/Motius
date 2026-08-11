"""Motius runtime for user-exported BeyondMimic ONNX motion policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from motius.models.onnx_policy import OnnxTrackingBundle, as_numpy
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class BeyondMimicBundle(OnnxTrackingBundle):
    CONFIG_NAME = "beyondmimic_config.json"
    MODEL_TYPE = "beyondmimic_g1"
    METHOD_NAME = "BeyondMimic"
    PIPELINE_CLASS = "motius.pipelines.beyondmimic.BeyondMimicPipeline"
    BUNDLE_CLASS = "motius.models.beyondmimic.BeyondMimicBundle"
    DEFAULT_FILES = {"policy": "policy.onnx"}
    ONNX_ROLES = ("policy",)

    def artifact_contract(self) -> dict[str, Any]:
        return {
            "robot": "Unitree G1",
            "control_hz": 50,
            "inputs": ["obs", "time_step"],
            "outputs": [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            "policy_metadata": (
                "joint order, gains, defaults, action scale, observation terms, "
                "and body names are read from ONNX metadata"
            ),
        }

    def artifact_source(self) -> dict[str, Any]:
        return {
            "repo": "https://github.com/HybridRobotics/whole_body_tracking",
            "deployment_repo": "https://github.com/HybridRobotics/motion_tracking_controller",
            "paper": "https://arxiv.org/abs/2508.08241",
            "checkpoint_status": "upstream does not publish a named pretrained policy",
        }

    @property
    def policy_metadata(self) -> dict[str, str]:
        return self.session("policy").metadata

    def forward(self, observation: Any, time_step: int | np.ndarray) -> dict[str, np.ndarray]:
        obs = as_numpy(observation, np.dtype(np.float32))
        if obs.ndim == 1:
            obs = obs[None]
        step = as_numpy(time_step, np.dtype(np.float32))
        if step.ndim == 0:
            step = step.reshape(1, 1)
        elif step.ndim == 1:
            step = step[:, None]
        return self.session("policy").run({"obs": obs, "time_step": step})


__all__ = ["BeyondMimicBundle"]
