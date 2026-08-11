"""Motius bundle for OpenTrack's Any2Track LAFAN1 v2 generalist."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from motius.models.onnx_policy import OnnxTrackingBundle, as_numpy
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class Any2TrackBundle(OnnxTrackingBundle):
    CONFIG_NAME = "any2track_config.json"
    MODEL_TYPE = "any2track_g1_lafan1_v2"
    METHOD_NAME = "Any2Track"
    PIPELINE_CLASS = "motius.pipelines.any2track.Any2TrackPipeline"
    BUNDLE_CLASS = "motius.models.any2track.Any2TrackBundle"
    DEFAULT_FILES = {
        "policy": "model.onnx",
        "training_config": "config.json",
    }
    ONNX_ROLES = ("policy",)
    OBSERVATION_DIM = 156
    ACTION_DIM = 29
    OFFICIAL_OBSERVATION_KEYS = (
        "dif_joint_pos",
        "dif_joint_vel",
        "gvec_pelvis",
        "gyro_pelvis",
        "joint_pos",
        "joint_vel",
        "last_motor_targets",
        "ref_feet_height",
        "ref_root_height",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_config = json.loads(
            self.file_paths["training_config"].read_text(encoding="utf-8")
        )
        self._validate_training_config()

    def _validate_training_config(self) -> None:
        if self.observation_keys != self.OFFICIAL_OBSERVATION_KEYS:
            raise ValueError(
                "Any2Track config must declare the official observation order, "
                f"got {list(self.observation_keys)}."
            )
        env_config = self.training_config.get("env_config", {})
        ctrl_dt = env_config.get("ctrl_dt")
        if ctrl_dt is not None and not np.isclose(float(ctrl_dt), 0.02):
            raise ValueError(f"Any2Track config expects ctrl_dt=0.02, got {ctrl_dt}.")
        policy_args = self.training_config.get("policy_config", {}).get(
            "policy_args",
            {},
        )
        declared = {
            "obs_dim": self.OBSERVATION_DIM,
            "act_dim": self.ACTION_DIM,
        }
        for key, expected in declared.items():
            value = policy_args.get(key)
            if value is not None and int(value) != expected:
                raise ValueError(
                    f"Any2Track config expects {key}={expected}, got {value}."
                )

    @property
    def observation_keys(self) -> tuple[str, ...]:
        keys = self.training_config.get("env_config", {}).get("obs_keys", ())
        return tuple(str(key) for key in keys)

    def artifact_contract(self) -> dict[str, Any]:
        return {
            "robot": "Unitree G1 29-DOF",
            "control_hz": 50,
            "input": {"name": "obs", "dimension": self.OBSERVATION_DIM},
            "output": {"name": "continuous_actions", "dimension": self.ACTION_DIM},
            "observation_keys": list(self.observation_keys),
            "reference_dataset": "LAFAN1, 40 motions",
        }

    def artifact_source(self) -> dict[str, Any]:
        return {
            "repo": "https://github.com/GalaxyGeneralRobotics/OpenTrack",
            "release": "LAFAN1 generalist v2, 2026-05-18",
            "paper": "https://arxiv.org/abs/2509.13833",
        }

    def assemble_observation(self, components: Mapping[str, Any]) -> np.ndarray:
        missing = [key for key in self.observation_keys if key not in components]
        extra = sorted(set(components) - set(self.observation_keys))
        if missing or extra:
            raise ValueError(
                f"Any2Track observation components mismatch: missing={missing}, "
                f"extra={extra}."
            )
        arrays = []
        batch_shape = None
        for key in self.observation_keys:
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
        if observation.shape[-1] != self.OBSERVATION_DIM:
            raise ValueError(
                f"Any2Track components produce {observation.shape[-1]} values, "
                f"expected {self.OBSERVATION_DIM}."
            )
        return observation

    def forward(self, observation: Any) -> dict[str, np.ndarray]:
        if isinstance(observation, Mapping):
            observation = self.assemble_observation(observation)
        observation = as_numpy(observation, np.dtype(np.float32))
        if observation.ndim == 1:
            observation = observation[None]
        return self.session("policy").run({"obs": observation})


__all__ = ["Any2TrackBundle"]
