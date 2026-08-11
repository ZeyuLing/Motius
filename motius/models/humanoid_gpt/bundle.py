"""Motius bundle for the released Humanoid-GPT G1 tracking policy."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from motius.models.onnx_policy import OnnxTrackingBundle, as_numpy
from motius.registry import MODEL_BUNDLES


DEFAULT_JOINT_POSITION = np.asarray(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
        0.2, -0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
TORQUE_LIMIT = np.asarray(
    [
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float32,
)
STIFFNESS = np.asarray(
    [
        40.17923737, 99.09842682, 40.17923737, 99.09842682,
        28.5012455, 28.5012455, 40.17923737, 99.09842682,
        40.17923737, 99.09842682, 28.5012455, 28.5012455,
        40.17923737, 28.5012455, 28.5012455,
        14.25062275, 14.25062275, 14.25062275, 14.25062275,
        14.25062275, 16.77832794, 16.77832794,
        14.25062275, 14.25062275, 14.25062275, 14.25062275,
        14.25062275, 16.77832794, 16.77832794,
    ],
    dtype=np.float32,
)
DAMPING = np.asarray(
    [
        2.5578897, 6.30880165, 2.5578897, 6.30880165,
        1.81444573, 1.81444573, 2.5578897, 6.30880165,
        2.5578897, 6.30880165, 1.81444573, 1.81444573,
        2.5578897, 1.81444573, 1.81444573,
        0.90722287, 0.90722287, 0.90722287, 0.90722287,
        0.90722287, 1.06814146, 1.06814146,
        0.90722287, 0.90722287, 0.90722287, 0.90722287,
        0.90722287, 1.06814146, 1.06814146,
    ],
    dtype=np.float32,
)
ACTION_SCALE = TORQUE_LIMIT / STIFFNESS


@MODEL_BUNDLES.register_module()
class HumanoidGPTBundle(OnnxTrackingBundle):
    """Official non-privileged Humanoid-GPT tracker and deployment assets."""

    CONFIG_NAME = "humanoid_gpt_config.json"
    MODEL_TYPE = "humanoid_gpt_g1_5010"
    METHOD_NAME = "HumanoidGPT"
    PIPELINE_CLASS = "motius.pipelines.humanoid_gpt.HumanoidGPTPipeline"
    BUNDLE_CLASS = "motius.models.humanoid_gpt.HumanoidGPTBundle"
    DEFAULT_FILES = {
        "policy": "model.onnx",
        "scene": "unitree_g1_5010/scene_mjx_track.xml",
        "robot_mjcf": "unitree_g1_5010/g1_mjx_track.xml",
        "robot_license": "unitree_g1_5010/LICENSE",
    }
    ONNX_ROLES = ("policy",)
    OBSERVATION_DIM = 136
    ACTION_DIM = 29
    OBSERVATION_LAYOUT = (
        ("gyro_pelvis", 3),
        ("gravity_pelvis", 3),
        ("joint_position_delta", 29),
        ("joint_velocity", 29),
        ("last_action", 29),
        ("reference_joint_position_delta", 29),
        ("reference_pelvis_height", 1),
        ("reference_gravity", 3),
        ("reference_pelvis_velocity", 6),
        ("heading_error_cos_sin", 2),
        ("position_error_heading_frame", 2),
    )

    def artifact_contract(self) -> dict[str, Any]:
        return {
            "robot": "Unitree G1-5010 29-DOF",
            "control_hz": 50,
            "simulation_timestep_seconds": 0.004,
            "observation": {
                "name": "obs",
                "shape": ["batch", self.OBSERVATION_DIM],
                "layout": [
                    {"name": name, "dimension": dimension}
                    for name, dimension in self.OBSERVATION_LAYOUT
                ],
            },
            "outputs": {
                "continuous_actions": ["batch", self.ACTION_DIM],
                "std_param": [self.ACTION_DIM],
            },
            "minimum_runtime": {
                "python": "3.10",
                "onnxruntime": "1.18",
            },
            "motor_target": "nominal + action * 0.25 * torque_limit / stiffness",
            "quaternion_convention": "wxyz",
        }

    def artifact_source(self) -> dict[str, Any]:
        return {
            "repo": "https://github.com/GalaxyGeneralRobotics/Humanoid-GPT",
            "paper": "https://arxiv.org/abs/2606.03985",
            "checkpoint_path": "storage/ckpts/pns_wo_priv216.onnx",
            "checkpoint_variant": "non-privileged 216",
        }

    def assemble_observation(self, components: Mapping[str, Any]) -> np.ndarray:
        """Assemble the official 136D observation without reordering fields."""

        expected_names = [name for name, _ in self.OBSERVATION_LAYOUT]
        missing = sorted(set(expected_names) - set(components))
        extra = sorted(set(components) - set(expected_names))
        if missing or extra:
            raise ValueError(
                f"HumanoidGPT observation components mismatch: missing={missing}, "
                f"extra={extra}."
            )
        arrays = []
        batch_size = None
        for name, dimension in self.OBSERVATION_LAYOUT:
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
        if action_array.ndim != 2 or action_array.shape[1] != self.ACTION_DIM:
            raise ValueError(
                f"HumanoidGPT action must have shape [B, {self.ACTION_DIM}], "
                f"got {action_array.shape}."
            )
        return (
            DEFAULT_JOINT_POSITION[None]
            + action_array * np.float32(0.25) * ACTION_SCALE[None]
        ).astype(np.float32, copy=False)

    def forward(
        self,
        observation: Any | None = None,
        *,
        components: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        if (observation is None) == (components is None):
            raise ValueError("Pass exactly one of observation or components.")
        if components is not None:
            obs = self.assemble_observation(components)
        else:
            obs = as_numpy(observation, np.dtype(np.float32))
            if obs.ndim == 1:
                obs = obs[None]
            if obs.ndim != 2 or obs.shape[1] != self.OBSERVATION_DIM:
                raise ValueError(
                    f"HumanoidGPT observation must have shape [B, "
                    f"{self.OBSERVATION_DIM}], got {obs.shape}."
                )
        outputs = self.session("policy").run({"obs": obs})
        action = outputs["continuous_actions"]
        return {
            **outputs,
            "motor_targets": self.decode_motor_targets(action),
            "observation": obs,
        }

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> str:
        """Save policy plus the complete G1-5010 MJCF dependency tree."""

        output = Path(super().save_pretrained(save_directory, **kwargs))
        source_assets = self.file_paths["scene"].parent
        target_assets = output / Path(self.artifact_files["scene"]).parent
        for source in source_assets.rglob("*"):
            if not source.is_file():
                continue
            destination = target_assets / source.relative_to(source_assets)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

        model_index_path = output / "model_index.json"
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        model_index["required_files"] = [
            self.CONFIG_NAME,
            "model.onnx",
            *[
                str(path.relative_to(output))
                for path in sorted(target_assets.rglob("*"))
                if path.is_file()
            ],
        ]
        model_index_path.write_text(
            json.dumps(model_index, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(output)


__all__ = [
    "ACTION_SCALE",
    "DAMPING",
    "DEFAULT_JOINT_POSITION",
    "HumanoidGPTBundle",
    "STIFFNESS",
    "TORQUE_LIMIT",
]
