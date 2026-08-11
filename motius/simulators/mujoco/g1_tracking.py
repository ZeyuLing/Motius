"""Self-contained Unitree G1 tracking environment for MuJoCo evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from motius.simulators.base import MotionTrackingEnvironment
from motius.simulators.g1 import G1_JOINT_NAMES
from motius.simulators.metrics import (
    TrackingMetricAccumulator,
    quaternion_geodesic_degrees,
)
from motius.simulators.reference import TrackingReference
from motius.simulators.rollout import TrackingRolloutResult
from motius.models.humanoid_gpt.bundle import (
    DAMPING as HUMANOID_GPT_DAMPING,
    DEFAULT_JOINT_POSITION as HUMANOID_GPT_DEFAULT_JOINT_POSITION,
    STIFFNESS as HUMANOID_GPT_STIFFNESS,
)


DEFAULT_DOF_POS = np.asarray(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
        0.2, -0.3, 0.0, 1.28, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
ANY2TRACK_KP = np.asarray(
    [
        100, 100, 100, 200, 80, 20,
        100, 100, 100, 200, 80, 20,
        300, 300, 300,
        90, 60, 20, 60, 20, 20, 20,
        90, 60, 20, 60, 20, 20, 20,
    ],
    dtype=np.float32,
)
ANY2TRACK_KD = np.asarray(
    [
        2, 2, 2, 4, 2, 1,
        2, 2, 2, 4, 2, 1,
        10, 10, 10,
        2, 2, 1, 1, 1, 1, 1,
        2, 2, 1, 1, 1, 1, 1,
    ],
    dtype=np.float32,
)
TORQUE_LIMITS = np.asarray(
    [
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float32,
)
BENCHMARK_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


class MujocoG1TrackingEnvironment(MotionTrackingEnvironment):
    """Native MuJoCo rollout for supported Unitree G1 tracking policies."""

    backend = "mujoco"
    protocol_id = "mujoco-g1-reference-tracking-50hz-v1"

    def __init__(
        self,
        *,
        reference: TrackingReference,
        render: bool = False,
        video_path: Optional[str | Path] = None,
        scene_path: Optional[str | Path] = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo tracking requires `pip install 'motius[motion-tracking-mujoco]'`."
            ) from exc

        self.mujoco = mujoco
        self.reference = reference
        if not np.isclose(reference.fps, 50.0):
            raise ValueError(f"MuJoCo tracking requires a 50 Hz reference, got {reference.fps}.")
        default_scene = (
            Path(__file__).resolve().parent
            / "assets"
            / "unitree_g1"
            / "scene_mjx_flat_terrain.xml"
        )
        self.scene_path = Path(scene_path).resolve() if scene_path else default_scene
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.ref_data = mujoco.MjData(self.model)
        self.render_enabled = bool(render)
        if render and video_path is None:
            raise ValueError("MuJoCo rendering requires a video_path.")
        self.renderer = mujoco.Renderer(self.model, height=720, width=960) if render else None
        self.render_camera = mujoco.MjvCamera() if render else None
        if self.render_camera is not None:
            self.render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.render_camera.distance = 3.2
            self.render_camera.azimuth = 140.0
            self.render_camera.elevation = -18.0
        self.video_path = Path(video_path).expanduser().resolve() if video_path else None
        self.video_writer = None

        self.control_dt = 0.02
        self.physics_dt = 0.002
        self.model.opt.timestep = self.physics_dt
        self.method: Optional[str] = None
        self._done = False
        self._termination_reason: Optional[str] = None
        self.frame_index = 1
        self.last_motor_targets = DEFAULT_DOF_POS.copy()
        self.last_processed_action = np.zeros(29, dtype=np.float32)
        self.metrics = TrackingMetricAccumulator()
        self.qpos_history: list[np.ndarray] = []
        self.reference_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []

        self.joint_ids = np.asarray([self.model.joint(name).id for name in G1_JOINT_NAMES])
        self.actuator_ids = np.asarray([self.model.actuator(name).id for name in G1_JOINT_NAMES])
        self.pelvis_site_id = self.model.site("imu_in_pelvis").id
        self.torso_body_id = self.model.body("torso_link").id
        self.benchmark_body_ids = np.asarray(
            [self.model.body(name).id for name in BENCHMARK_BODY_NAMES]
        )
        self.left_shoulder_body_id = self.model.body("left_shoulder_pitch_link").id
        self.foot_site_ids = np.asarray(
            [self.model.site("left_foot").id, self.model.site("right_foot").id]
        )
        foot_site_names = ["left_foot", "right_foot"]
        for optional_name in ("left_foot_top", "right_foot_top"):
            optional_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                optional_name,
            )
            if optional_id >= 0:
                foot_site_names.append(optional_name)
        self.all_foot_site_ids = np.asarray(
            [self.model.site(name).id for name in foot_site_names]
        )
        self.foot_velocity_sensors = (
            "left_foot_global_linvel",
            "right_foot_global_linvel",
        )
        self._cache_reference_fk()

    @property
    def done(self) -> bool:
        return self._done

    def _cache_reference_fk(self) -> None:
        frame_count = self.reference.num_frames
        self.ref_body_positions = np.empty(
            (frame_count, len(self.benchmark_body_ids), 3), dtype=np.float32
        )
        self.ref_torso_quat_xyzw = np.empty((frame_count, 4), dtype=np.float32)
        self.ref_feet_heights = np.empty(
            (frame_count, len(self.all_foot_site_ids)), dtype=np.float32
        )
        self.ref_shoulder_heights = np.empty(frame_count, dtype=np.float32)
        self.ref_gravity_pelvis = np.empty((frame_count, 3), dtype=np.float32)
        self.ref_pelvis_velocity_heading = np.empty(
            (frame_count, 6), dtype=np.float32
        )
        gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        for index in range(frame_count):
            self.ref_data.qpos[:] = self.reference.qpos[index]
            self.ref_data.qvel[:] = self.reference.qvel[index]
            self.mujoco.mj_forward(self.model, self.ref_data)
            self.ref_body_positions[index] = self.ref_data.xpos[self.benchmark_body_ids]
            torso_wxyz = self.ref_data.xquat[self.torso_body_id]
            self.ref_torso_quat_xyzw[index] = torso_wxyz[[1, 2, 3, 0]]
            self.ref_feet_heights[index] = self.ref_data.site_xpos[self.all_foot_site_ids, 2]
            self.ref_shoulder_heights[index] = self.ref_data.xpos[
                self.left_shoulder_body_id,
                2,
            ]
            pelvis_rot = self.ref_data.site_xmat[self.pelvis_site_id].reshape(3, 3)
            self.ref_gravity_pelvis[index] = pelvis_rot.T @ gravity
            yaw = self._yaw_wxyz(self.reference.qpos[index, 3:7])
            heading_world_to_local = np.asarray(
                [
                    [np.cos(yaw), np.sin(yaw), 0.0],
                    [-np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            pelvis_velocity = self.ref_data.cvel[
                self.model.body("pelvis").id
            ].reshape(2, 3)
            self.ref_pelvis_velocity_heading[index] = (
                pelvis_velocity @ heading_world_to_local.T
            ).reshape(6)

    def _bind_method(self, pipeline: Any) -> str:
        method = str(getattr(pipeline.bundle, "METHOD_NAME", "")).lower()
        if method == "any2track":
            physics_dt = 0.002
        elif method == "protomotions":
            physics_dt = 0.001
        elif method == "humanoidgpt":
            physics_dt = 0.004
        else:
            raise NotImplementedError(
                f"The MuJoCo backend does not yet implement the {method or 'unknown'} "
                "observation/action adapter. Supported adapters: Any2Track, "
                "ProtoMotions, HumanoidGPT."
            )
        if self.method is None:
            self.method = method
            self.physics_dt = physics_dt
            self.model.opt.timestep = physics_dt
        elif self.method != method:
            raise RuntimeError(f"Environment is already bound to {self.method}, not {method}.")
        return method

    def reset(self) -> Mapping[str, Any]:
        self._done = False
        self._termination_reason = None
        self.frame_index = 1
        self.metrics = TrackingMetricAccumulator()
        self.data.qpos[:] = self.reference.qpos[0]
        self.data.qvel[:] = self.reference.qvel[0]
        self.data.ctrl[:] = 0.0
        self.last_motor_targets = self.reference.qpos[0, 7:].copy()
        self.last_processed_action = np.zeros(29, dtype=np.float32)
        self.mujoco.mj_forward(self.model, self.data)
        self.qpos_history = [self.data.qpos.copy()]
        self.reference_history = [self.reference.qpos[0].copy()]
        self.action_history = []
        if self.renderer is not None:
            import imageio.v2 as imageio

            assert self.video_path is not None
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = imageio.get_writer(
                self.video_path,
                fps=self.reference.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=16,
            )
        self._render_frame()
        return {
            "backend": self.backend,
            "reference": self.reference.name,
            "num_frames": self.reference.num_frames,
            "control_hz": 50,
        }

    def _sensor(self, name: str) -> np.ndarray:
        sensor = self.model.sensor(name)
        start = self.model.sensor_adr[sensor.id]
        dimension = self.model.sensor_dim[sensor.id]
        return self.data.sensordata[start : start + dimension].copy()

    def _any2track_inputs(self) -> Mapping[str, Any]:
        index = self.frame_index
        pelvis_rot = self.data.site_xmat[self.pelvis_site_id].reshape(3, 3)
        components = {
            "dif_joint_pos": self.reference.qpos[index, 7:] - self.data.qpos[7:],
            "dif_joint_vel": (self.reference.qvel[index, 6:] - self.data.qvel[6:]) * 0.05,
            "gvec_pelvis": pelvis_rot.T @ np.asarray([0.0, 0.0, -1.0]),
            "gyro_pelvis": self._sensor("gyro_pelvis") * 0.05,
            "joint_pos": self.data.qpos[7:] - DEFAULT_DOF_POS,
            "joint_vel": self.data.qvel[6:] * 0.05,
            "last_motor_targets": self.last_motor_targets,
            "ref_feet_height": self.ref_feet_heights[index],
            "ref_root_height": self.reference.qpos[index, 2:3],
        }
        return {
            "observation": {
                key: np.asarray(value, dtype=np.float32)[None]
                for key, value in components.items()
            }
        }

    def _protomotions_inputs(self) -> Mapping[str, Any]:
        index = self.frame_index
        future_indices = np.minimum(
            index + np.asarray([1, 2, 4, 8], dtype=np.int64),
            self.reference.num_frames - 1,
        )
        current_torso_wxyz = self.data.xquat[self.torso_body_id]
        observations = {
            "current_anchor_rot": current_torso_wxyz[[1, 2, 3, 0]][None].astype(np.float32),
            "current_dof_pos": self.data.qpos[7:][None].astype(np.float32),
            "current_dof_vel": self.data.qvel[6:][None].astype(np.float32),
            "current_root_local_ang_vel": self.data.qvel[3:6][None].astype(np.float32),
            "historical_processed_actions": self.last_processed_action[None, None].astype(np.float32),
            "mimic_future_anchor_rot": self.ref_torso_quat_xyzw[future_indices][None],
            "mimic_future_dof_pos": self.reference.qpos[future_indices, 7:][None],
            "mimic_future_dof_vel": self.reference.qvel[future_indices, 6:][None],
        }
        return {"observations": observations}

    def _humanoid_gpt_inputs(self) -> Mapping[str, Any]:
        next_index = self.frame_index
        current_index = max(0, next_index - 1)
        current_yaw = self._yaw_wxyz(self.data.qpos[3:7])
        reference_yaw = self._yaw_wxyz(
            self.reference.qpos[current_index, 3:7]
        )
        yaw_error = np.arctan2(
            np.sin(reference_yaw - current_yaw),
            np.cos(reference_yaw - current_yaw),
        )
        xy_delta_world = (
            self.reference.qpos[current_index, :2] - self.data.qpos[:2]
        )
        c, s = np.cos(current_yaw), np.sin(current_yaw)
        xy_delta_heading = np.asarray(
            [
                c * xy_delta_world[0] + s * xy_delta_world[1],
                -s * xy_delta_world[0] + c * xy_delta_world[1],
            ],
            dtype=np.float32,
        )
        pelvis_rot = self.data.site_xmat[self.pelvis_site_id].reshape(3, 3)
        components = {
            "gyro_pelvis": self._sensor("gyro_pelvis"),
            "gravity_pelvis": pelvis_rot.T @ np.asarray([0.0, 0.0, -1.0]),
            "joint_position_delta": (
                self.data.qpos[7:] - HUMANOID_GPT_DEFAULT_JOINT_POSITION
            ),
            "joint_velocity": self.data.qvel[6:],
            "last_action": self.last_processed_action,
            "reference_joint_position_delta": (
                self.reference.qpos[next_index, 7:]
                - HUMANOID_GPT_DEFAULT_JOINT_POSITION
            ),
            "reference_pelvis_height": self.reference.qpos[next_index, 2:3],
            "reference_gravity": self.ref_gravity_pelvis[next_index],
            "reference_pelvis_velocity": self.ref_pelvis_velocity_heading[
                next_index
            ],
            "heading_error_cos_sin": np.asarray(
                [np.cos(yaw_error), np.sin(yaw_error)], dtype=np.float32
            ),
            "position_error_heading_frame": xy_delta_heading,
        }
        return {
            "components": {
                key: np.asarray(value, dtype=np.float32)[None]
                for key, value in components.items()
            }
        }

    def policy_inputs(self, pipeline: Any) -> Mapping[str, Any]:
        method = self._bind_method(pipeline)
        if method == "any2track":
            return self._any2track_inputs()
        if method == "protomotions":
            return self._protomotions_inputs()
        return self._humanoid_gpt_inputs()

    def _decode_control(
        self,
        method: str,
        policy_output: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if method == "any2track":
            action = np.asarray(policy_output["continuous_actions"], dtype=np.float32).reshape(-1)
            target = self.reference.qpos[self.frame_index, 7:] + action
            return action, target, ANY2TRACK_KP, ANY2TRACK_KD
        if method == "humanoidgpt":
            action = np.asarray(
                policy_output["continuous_actions"], dtype=np.float32
            ).reshape(-1)
            target = np.asarray(
                policy_output["motor_targets"], dtype=np.float32
            ).reshape(-1)
            return (
                action,
                target,
                HUMANOID_GPT_STIFFNESS,
                HUMANOID_GPT_DAMPING,
            )
        action = np.asarray(policy_output["actions"], dtype=np.float32).reshape(-1)
        target = np.asarray(policy_output["joint_pos_targets"], dtype=np.float32).reshape(-1)
        stiffness = np.asarray(policy_output["stiffness_targets"], dtype=np.float32).reshape(-1)
        damping = np.asarray(policy_output["damping_targets"], dtype=np.float32).reshape(-1)
        return action, target, stiffness, damping

    def step(self, pipeline: Any, policy_output: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._done:
            raise RuntimeError("Cannot step a terminated motion-tracking environment.")
        method = self._bind_method(pipeline)
        action, motor_targets, stiffness, damping = self._decode_control(method, policy_output)
        if action.shape != (29,) or motor_targets.shape != (29,):
            raise ValueError(
                f"{method} must produce 29D action and target, got "
                f"{action.shape} and {motor_targets.shape}."
            )
        self.last_motor_targets = motor_targets.astype(np.float32, copy=True)
        self.last_processed_action = action.astype(np.float32, copy=True)
        decimation = int(round(self.control_dt / self.physics_dt))
        last_torque = np.zeros(29, dtype=np.float32)
        for _ in range(decimation):
            torque = stiffness * (motor_targets - self.data.qpos[7:]) - damping * self.data.qvel[6:]
            last_torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS).astype(np.float32)
            self.data.ctrl[self.actuator_ids] = last_torque
            self.mujoco.mj_step(self.model, self.data)

        self._record_step(last_torque)
        self.action_history.append(action.copy())
        self.qpos_history.append(self.data.qpos.copy())
        self.reference_history.append(self.reference.qpos[self.frame_index].copy())
        self._render_frame()

        termination = self._termination()
        if termination is not None:
            self._done = True
            self._termination_reason = termination
        elif self.frame_index >= self.reference.num_frames - 1:
            self._done = True
        else:
            self.frame_index += 1
        return {
            "frame_index": self.frame_index,
            "done": self._done,
            "termination_reason": self._termination_reason,
        }

    def _record_step(self, torque: np.ndarray) -> None:
        index = self.frame_index
        current_body = self.data.xpos[self.benchmark_body_ids]
        reference_body = self.ref_body_positions[index]
        current_local = self._body_positions_in_root_frame(
            current_body,
            self.data.qpos[:3],
            self.data.qpos[3:7],
        )
        reference_local = self._body_positions_in_root_frame(
            reference_body,
            self.reference.qpos[index, :3],
            self.reference.qpos[index, 3:7],
        )
        foot_slips = []
        for site_id, sensor_name in zip(self.foot_site_ids, self.foot_velocity_sensors):
            if self.data.site_xpos[site_id, 2] < 0.05:
                foot_slips.append(float(np.linalg.norm(self._sensor(sensor_name)[:2])))
        self.metrics.update(
            {
                "local_body_mpjpe_m": np.linalg.norm(
                    current_local - reference_local,
                    axis=-1,
                ).mean(),
                "root_position_error_m": np.linalg.norm(
                    self.data.qpos[:3] - self.reference.qpos[index, :3]
                ),
                "root_orientation_error_deg": quaternion_geodesic_degrees(
                    self.data.qpos[3:7], self.reference.qpos[index, 3:7]
                ),
                "joint_position_mae_rad": np.abs(
                    self.data.qpos[7:] - self.reference.qpos[index, 7:]
                ).mean(),
                "joint_velocity_mae_rad_s": np.abs(
                    self.data.qvel[6:] - self.reference.qvel[index, 6:]
                ).mean(),
                "foot_slip_m_s": np.mean(foot_slips) if foot_slips else 0.0,
                "mechanical_power_w": np.abs(torque * self.data.qvel[6:]).sum(),
            }
        )

    def _termination(self) -> Optional[str]:
        index = self.frame_index
        if abs(float(self.data.qpos[2] - self.reference.qpos[index, 2])) > 0.3:
            return "root_height_error"
        reference_shoulder_height = self.ref_shoulder_heights[index]
        if abs(float(self.data.xpos[self.left_shoulder_body_id, 2] - reference_shoulder_height)) > 0.3:
            return "shoulder_height_error"
        current_body = self.data.xpos[self.benchmark_body_ids]
        reference_body = self.ref_body_positions[index]
        current_local = self._body_positions_in_root_frame(
            current_body,
            self.data.qpos[:3],
            self.data.qpos[3:7],
        )
        reference_local = self._body_positions_in_root_frame(
            reference_body,
            self.reference.qpos[index, :3],
            self.reference.qpos[index, 3:7],
        )
        if np.linalg.norm(current_local - reference_local, axis=-1).max() > 0.5:
            return "body_position_error"
        return None

    @staticmethod
    def _yaw_wxyz(quaternion: np.ndarray) -> float:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        return float(
            np.arctan2(
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (y * y + z * z),
            )
        )

    @staticmethod
    def _body_positions_in_root_frame(
        body_positions: np.ndarray,
        root_position: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        root_rotation = Rotation.from_quat(root_quaternion_wxyz[[1, 2, 3, 0]])
        return root_rotation.inv().apply(body_positions - root_position)

    def _render_frame(self) -> None:
        if self.renderer is None:
            return
        assert self.render_camera is not None
        self.render_camera.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data, camera=self.render_camera)
        assert self.video_writer is not None
        self.video_writer.append_data(self.renderer.render().copy())

    def result(self) -> TrackingRolloutResult:
        completed = len(self.action_history)
        total = self.reference.num_frames - 1
        success = completed >= total and self._termination_reason is None
        method = self.method or "unbound"
        metrics = self.metrics.summarize(
            completed_steps=completed,
            total_steps=total,
            success=success,
        )
        return TrackingRolloutResult(
            method=method,
            backend=self.backend,
            protocol_id=self.protocol_id,
            reference_name=self.reference.name,
            fps=self.reference.fps,
            qpos=np.asarray(self.qpos_history, dtype=np.float32),
            reference_qpos=np.asarray(self.reference_history, dtype=np.float32),
            actions=np.asarray(self.action_history, dtype=np.float32),
            metrics=metrics,
            termination_reason=self._termination_reason,
        )

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
