"""Motius bundle for the ProtoMotions G1 BONES-SEED deployment tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from motius.models.onnx_policy import OnnxTrackingBundle
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class ProtoMotionsBundle(OnnxTrackingBundle):
    CONFIG_NAME = "protomotions_config.json"
    MODEL_TYPE = "protomotions_g1_bones_deploy"
    METHOD_NAME = "ProtoMotions"
    PIPELINE_CLASS = "motius.pipelines.protomotions.ProtoMotionsPipeline"
    BUNDLE_CLASS = "motius.models.protomotions.ProtoMotionsBundle"
    DEFAULT_FILES = {
        "policy": "unified_pipeline.onnx",
        "deployment_config": "unified_pipeline.yaml",
    }
    ONNX_ROLES = ("policy",)
    INPUT_NAMES = (
        "current_anchor_rot",
        "current_dof_pos",
        "current_dof_vel",
        "current_root_local_ang_vel",
        "historical_processed_actions",
        "mimic_future_anchor_rot",
        "mimic_future_dof_pos",
        "mimic_future_dof_vel",
    )

    def _deployment_config(self) -> dict[str, Any]:
        config_path = self.file_paths.get("deployment_config", Path())
        try:
            import yaml
        except ImportError:
            return {}
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def artifact_contract(self) -> dict[str, Any]:
        deployment = self._deployment_config()
        policy_inputs = deployment.get("policy_inputs", [])
        policy_outputs = deployment.get("policy_outputs", [])
        runtime = deployment.get("_runtime", {})
        timing = deployment.get("timing", {})
        motion = deployment.get("motion", {})
        inputs = runtime.get("onnx_in_names") or [
            item.get("name")
            for item in policy_inputs
            if isinstance(item, Mapping) and item.get("name")
        ]
        outputs = runtime.get("onnx_out_names") or [
            item.get("name")
            for item in policy_outputs
            if isinstance(item, Mapping) and item.get("name")
        ]
        control_dt = timing.get("control_dt", deployment.get("dt", 0.02))
        future_steps = motion.get("future_step_indices", [1, 2, 4, 8])
        return {
            "robot": "Unitree G1 29-DOF",
            "control_hz": round(1.0 / float(control_dt)),
            "future_steps": list(future_steps),
            "future_seconds": [
                round(float(step) * float(control_dt), 6)
                for step in future_steps
            ],
            "inputs": list(inputs or self.INPUT_NAMES),
            "outputs": list(
                outputs
                or (
                    "actions",
                    "joint_pos_targets",
                    "stiffness_targets",
                    "damping_targets",
                )
            ),
            "policy_inputs": policy_inputs,
            "quaternion_convention": "xyzw",
        }

    def artifact_source(self) -> dict[str, Any]:
        return {
            "repo": "https://github.com/NVlabs/ProtoMotions",
            "checkpoint_path": "data/pretrained_models/motion_tracker/g1-bones-deploy",
            "paper": "https://arxiv.org/abs/2507.14320",
        }

    def forward(self, observations: Mapping[str, Any]) -> dict[str, Any]:
        return self.session("policy").run(observations)


__all__ = ["ProtoMotionsBundle"]
