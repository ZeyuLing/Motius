#!/usr/bin/env python3
"""Evaluate OpenTrack dagger ONNX checkpoints without importing JAX OpenTrack code."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.evaluation.gentrack.canonical_rollouts import (
    qpos_to_body_arrays,
    save_body,
    save_qpos,
    write_reference_from_qpos,
    write_run_config,
)


ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
OBS_JOINT_NAMES = ACTION_JOINT_NAMES
FEET_ALL_SITES = ["left_foot", "right_foot", "left_foot_top", "right_foot_top"]
DEFAULT_QPOS = np.float32(
    [
        0,
        0,
        0.79,
        1,
        0,
        0,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        -0.1,
        0,
        0,
        0.3,
        -0.2,
        0,
        0,
        0,
        0,
        0.2,
        0.3,
        0,
        1.28,
        0,
        0,
        0,
        0.2,
        -0.3,
        0,
        1.28,
        0,
        0,
        0,
    ]
)
KPS = np.float32(
    [
        100,
        100,
        100,
        200,
        80,
        20,
        100,
        100,
        100,
        200,
        80,
        20,
        300,
        300,
        300,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
        90,
        60,
        20,
        60,
        20,
        20,
        20,
    ]
)
KDS = np.float32(
    [
        2,
        2,
        2,
        4,
        2,
        1,
        2,
        2,
        2,
        4,
        2,
        1,
        10,
        10,
        10,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    ]
)
TORQUE_LIMIT = np.float32(
    [
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        139,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ]
)
MESHES_BY_BODY = {
    "pelvis": ["pelvis.stl", "pelvis_contour_link.stl"],
    "head": [],
    "left_hip_pitch_link": ["left_hip_pitch_link.stl"],
    "left_hip_roll_link": ["left_hip_roll_link.stl"],
    "left_hip_yaw_link": ["left_hip_yaw_link.stl"],
    "left_knee_link": ["left_knee_link.stl"],
    "left_ankle_pitch_link": ["left_ankle_pitch_link.stl"],
    "left_ankle_roll_link": ["left_ankle_roll_link.stl"],
    "right_hip_pitch_link": ["right_hip_pitch_link.stl"],
    "right_hip_roll_link": ["right_hip_roll_link.stl"],
    "right_hip_yaw_link": ["right_hip_yaw_link.stl"],
    "right_knee_link": ["right_knee_link.stl"],
    "right_ankle_pitch_link": ["right_ankle_pitch_link.stl"],
    "right_ankle_roll_link": ["right_ankle_roll_link.stl"],
    "waist_yaw_link": ["waist_yaw_link_rev_1_0.stl"],
    "waist_roll_link": ["waist_roll_link_rev_1_0.stl"],
    "torso_link": ["torso_link_rev_1_0.stl", "logo_link.stl", "head_link.stl"],
    "left_shoulder_pitch_link": ["left_shoulder_pitch_link.stl"],
    "left_shoulder_roll_link": ["left_shoulder_roll_link.stl"],
    "left_shoulder_yaw_link": ["left_shoulder_yaw_link.stl"],
    "left_elbow_link": ["left_elbow_link.stl"],
    "left_wrist_roll_link": ["left_wrist_roll_link.stl"],
    "left_wrist_pitch_link": ["left_wrist_pitch_link.stl"],
    "left_wrist_yaw_link": ["left_wrist_yaw_link.stl", "left_rubber_hand.stl"],
    "left_rubber_hand": [],
    "right_shoulder_pitch_link": ["right_shoulder_pitch_link.stl"],
    "right_shoulder_roll_link": ["right_shoulder_roll_link.stl"],
    "right_shoulder_yaw_link": ["right_shoulder_yaw_link.stl"],
    "right_elbow_link": ["right_elbow_link.stl"],
    "right_wrist_roll_link": ["right_wrist_roll_link.stl"],
    "right_wrist_pitch_link": ["right_wrist_pitch_link.stl"],
    "right_wrist_yaw_link": ["right_wrist_yaw_link.stl", "right_rubber_hand.stl"],
    "right_rubber_hand": [],
}


def _qpos_qvel_slices(joint_names: list[str], jnt_type: np.ndarray) -> tuple[dict, dict]:
    qpos_slices = {}
    qvel_slices = {}
    qpos_i = 0
    qvel_i = 0
    for name, typ in zip(joint_names, jnt_type):
        typ = int(typ)
        if typ == mujoco.mjtJoint.mjJNT_FREE:
            qpos_slices[name] = slice(qpos_i, qpos_i + 7)
            qvel_slices[name] = slice(qvel_i, qvel_i + 6)
            qpos_i += 7
            qvel_i += 6
        elif typ in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qpos_slices[name] = slice(qpos_i, qpos_i + 1)
            qvel_slices[name] = slice(qvel_i, qvel_i + 1)
            qpos_i += 1
            qvel_i += 1
        else:
            raise ValueError(f"Unsupported joint type {typ} for {name}")
    return qpos_slices, qvel_slices


def _quat_step_angvel_wxyz(qpos: np.ndarray, fps: float) -> np.ndarray:
    quat = qpos[:, 3:7]
    inv = np.concatenate([quat[:, :1], -quat[:, 1:]], axis=1)
    q1 = inv[:-1]
    q2 = quat[1:]
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    s = 2 * (w**2) - 1
    angle = np.arccos(np.clip(s, -1, 1))
    axis = np.stack([x, y, z], axis=1)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-9)
    return axis * angle[:, None] * fps


def _build_qvel(qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) * fps
    qvel[1:, 3:6] = _quat_step_angvel_wxyz(qpos, fps)
    qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) * fps
    return qvel


def _resample_qpos(qpos: np.ndarray, source_fps: float, output_fps: float) -> np.ndarray:
    if abs(float(source_fps) - float(output_fps)) < 1e-6:
        return qpos.astype(np.float32, copy=False)

    n = qpos.shape[0]
    if n < 2:
        return qpos.astype(np.float32, copy=False)
    duration = (n - 1) / float(source_fps)
    src_t = np.arange(n, dtype=np.float64) / float(source_fps)
    out_n = int(round(duration * float(output_fps))) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / float(output_fps)
    dst_t[-1] = min(dst_t[-1], src_t[-1])

    out = np.empty((out_n, qpos.shape[1]), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    src_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out[:, 3:7] = out_xyzw[:, [3, 0, 1, 2]]
    for i in range(7, qpos.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    return out


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.outer(q, q)
    return np.array(
        [
            [
                q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                2 * (q[1, 2] - q[0, 3]),
                2 * (q[1, 3] + q[0, 2]),
            ],
            [
                2 * (q[1, 2] + q[0, 3]),
                q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                2 * (q[2, 3] - q[0, 1]),
            ],
            [
                2 * (q[1, 3] - q[0, 2]),
                2 * (q[2, 3] + q[0, 1]),
                q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
            ],
        ],
        dtype=np.float32,
    )


def _fps_after_stride(fps: float, stride: int):
    out = fps / max(int(stride), 1)
    rounded = round(out)
    return int(rounded) if abs(out - rounded) < 1e-6 else float(out)


def _body_descriptors(body_names: list[str]) -> list[dict]:
    return [
        {
            "name": name,
            "meshes": [
                {"file": mesh, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
                for mesh in MESHES_BY_BODY.get(name, [])
            ],
        }
        for name in body_names
    ]


class OpenTrackRollout:
    def __init__(self, xml: Path, config: dict, onnx: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.ref_data = mujoco.MjData(self.model)
        self.config = config
        self.dt = float(config["env_config"]["ctrl_dt"])
        self.sim_dt = float(config["env_config"]["sim_dt"])
        self.action_scale = float(config["env_config"]["action_scale"])
        self.obs_scale_joint_vel = float(config["env_config"]["obs_scales_config"]["joint_vel"])
        self.obs_scale_dif_joint_vel = float(
            config["env_config"]["obs_scales_config"]["dif_joint_vel"]
        )
        self.obs_keys = list(config["env_config"]["obs_keys"])
        self.action_ids = np.array([self.model.actuator(n).id for n in ACTION_JOINT_NAMES])
        self.obs_ids = np.array([self.model.actuator(n).id for n in OBS_JOINT_NAMES])
        self.default_dof = DEFAULT_QPOS[7:].copy()
        self.model_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.model_qpos_slices, self.model_qvel_slices = _qpos_qvel_slices(
            self.model_joint_names, self.model.jnt_type
        )
        self.pelvis_imu_site_id = self.model.site("imu_in_pelvis").id
        self.feet_all_site_id = np.array([self.model.site(n).id for n in FEET_ALL_SITES])
        self.valid_body_ids = np.array([i for i in range(1, self.model.nbody)])
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in self.valid_body_ids
        ]
        self.body_descriptors = _body_descriptors(self.body_names)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(onnx),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.output_name = self.sess.get_outputs()[0].name

    def _frame_from_data(self, data: mujoco.MjData) -> dict:
        return {
            "body_pos": data.xpos[self.valid_body_ids].astype(float).tolist(),
            "body_quat": data.xquat[self.valid_body_ids].astype(float).tolist(),
        }

    def _body_arrays_from_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return qpos_to_body_arrays(self.model, qpos, self.valid_body_ids)

    def _robot_frames_payload(self, frames: list[dict], frame_stride: int) -> dict:
        fps = _fps_after_stride(1.0 / self.dt, frame_stride)
        return {
            "type": "robot_frames",
            "robot": "g1",
            "fps": fps,
            "num_frames": len(frames),
            "num_bodies": len(self.body_descriptors),
            "bodies": self.body_descriptors,
            "frames": frames,
        }

    def _sensor(self, name: str) -> np.ndarray:
        sid = self.model.sensor(name).id
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim]

    def _obs(self, ref_qpos: np.ndarray, ref_qvel: np.ndarray, last_motor_targets: np.ndarray) -> np.ndarray:
        self.ref_data.qpos[:] = ref_qpos
        self.ref_data.qvel[:] = ref_qvel
        mujoco.mj_forward(self.model, self.ref_data)

        gyro_pelvis = self._sensor("gyro_pelvis")
        gvec_pelvis = self.data.site_xmat[self.pelvis_imu_site_id].reshape(3, 3).T @ np.array(
            [0, 0, -1], dtype=np.float32
        )
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        dif_joint_pos = ref_qpos[7:] - joint_pos
        dif_joint_vel = ref_qvel[6:] - joint_vel
        ref_feet_height = self.ref_data.site_xpos[self.feet_all_site_id, 2]
        root_rot = quat_to_mat(ref_qpos[3:7])
        state_dict = {
            "gyro_pelvis": gyro_pelvis * self.obs_scale_joint_vel,
            "gvec_pelvis": gvec_pelvis,
            "joint_pos": (joint_pos - self.default_dof)[self.obs_ids],
            "joint_vel": joint_vel[self.obs_ids] * self.obs_scale_joint_vel,
            "last_motor_targets": last_motor_targets,
            "dif_joint_pos": dif_joint_pos,
            "dif_joint_vel": dif_joint_vel * self.obs_scale_dif_joint_vel,
            "ref_feet_height": ref_feet_height,
            "ref_root_height": np.array([ref_qpos[2]], dtype=np.float32),
            "ref_root_linvel": root_rot.T @ ref_qvel[:3] * self.obs_scale_joint_vel,
            "ref_root_angvel": ref_qvel[3:6] * self.obs_scale_joint_vel,
        }
        return np.hstack([state_dict[k] for k in self.obs_keys]).astype(np.float32)

    def _load_motion(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        pack = np.load(path, allow_pickle=True)
        qpos_src = pack["qpos"].astype(np.float32)
        qvel_src = pack["qvel"].astype(np.float32) if "qvel" in pack.files else None
        source_fps = float(np.asarray(pack["frequency"]).reshape(-1)[0]) if "frequency" in pack.files else 1.0 / self.dt

        if qpos_src.shape[1] == self.model.nq:
            qpos = qpos_src
            qvel = qvel_src if qvel_src is not None and qvel_src.shape[1] == self.model.nv else _build_qvel(qpos, source_fps)
        else:
            if "joint_names" not in pack.files:
                raise ValueError(
                    f"{path}: qpos dim {qpos_src.shape[1]} != model.nq {self.model.nq} and no joint_names"
                )
            source_joint_names = [str(x) for x in pack["joint_names"].tolist()]
            if "jnt_type" in pack.files:
                source_jnt_type = np.asarray(pack["jnt_type"])
            else:
                source_jnt_type = np.array([mujoco.mjtJoint.mjJNT_FREE] + [mujoco.mjtJoint.mjJNT_HINGE] * (len(source_joint_names) - 1))
            src_qpos_slices, src_qvel_slices = _qpos_qvel_slices(source_joint_names, source_jnt_type)

            # Match OpenTrack's TrajectoryHandler.filter_and_extend behavior:
            # joints missing from a source trajectory are appended with qpos=0/qvel=0,
            # not with the policy's standing DEFAULT_QPOS.
            qpos = np.zeros((qpos_src.shape[0], self.model.nq), dtype=np.float32)
            qvel = np.zeros((qpos_src.shape[0], self.model.nv), dtype=np.float32)
            for name in self.model_joint_names:
                if name not in src_qpos_slices:
                    continue
                qpos[:, self.model_qpos_slices[name]] = qpos_src[:, src_qpos_slices[name]]
                if qvel_src is not None and name in src_qvel_slices:
                    qvel[:, self.model_qvel_slices[name]] = qvel_src[:, src_qvel_slices[name]]

        qpos = _resample_qpos(qpos, source_fps, 1.0 / self.dt)
        qvel = _build_qvel(qpos, 1.0 / self.dt)
        return qpos.astype(np.float32), qvel.astype(np.float32)

    def reference_motion_frames(
        self,
        path: Path,
        max_steps: Optional[int] = None,
        frame_stride: int = 1,
    ) -> dict:
        qpos, qvel = self._load_motion(path)
        n_steps = qpos.shape[0] - 1
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        frames = []
        for i in range(0, n_steps + 1, max(int(frame_stride), 1)):
            self.ref_data.qpos[:] = qpos[i]
            self.ref_data.qvel[:] = qvel[i]
            mujoco.mj_forward(self.model, self.ref_data)
            frames.append(self._frame_from_data(self.ref_data))
        if n_steps % max(int(frame_stride), 1) != 0:
            self.ref_data.qpos[:] = qpos[n_steps]
            self.ref_data.qvel[:] = qvel[n_steps]
            mujoco.mj_forward(self.model, self.ref_data)
            frames.append(self._frame_from_data(self.ref_data))
        return self._robot_frames_payload(frames, frame_stride)

    def evaluate_motion(
        self,
        path: Path,
        max_steps: Optional[int] = None,
        capture_frames: bool = False,
        capture_qpos: bool = False,
        frame_stride: int = 1,
    ) -> dict:
        qpos, qvel = self._load_motion(path)
        if qpos.shape[0] < 3:
            raise ValueError(f"Motion too short: {path}")

        self.data.qpos[:] = qpos[0]
        self.data.qvel[:] = qvel[0]
        self.data.ctrl[:] = qpos[0, 7:]
        mujoco.mj_forward(self.model, self.data)
        self.ref_data.qpos[:] = qpos[0]
        self.ref_data.qvel[:] = qvel[0]
        mujoco.mj_forward(self.model, self.ref_data)
        last_motor_targets = self.data.qpos[7:].copy()
        prev_body_pos = self.data.xpos[self.valid_body_ids].copy()
        prev_ref_body_pos = self.ref_data.xpos[self.valid_body_ids].copy()
        rollout_frames = [self._frame_from_data(self.data)] if capture_frames else []
        rollout_qpos = [self.data.qpos.copy()] if capture_qpos else []

        n_steps = qpos.shape[0] - 1
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)

        root_err = []
        body_err = []
        xy_aligned_body_err = []
        local_body_err = []
        body_vel_err = []
        local_body_vel_err = []
        body_acc_err = []
        local_body_acc_err = []
        joint_err = []
        max_joint_err = []
        root_height_err = []
        height = []
        finite = True
        prev_body_vel = None
        prev_ref_body_vel = None
        prev_local_body_vel = None
        prev_ref_local_body_vel = None
        for i in range(1, n_steps + 1):
            obs = self._obs(qpos[i], qvel[i], last_motor_targets)
            action = self.sess.run([self.output_name], {"obs": obs.reshape(1, -1)})[0][0]
            lower_targets = qpos[i, 7:][self.action_ids] + action * self.action_scale
            motor_targets = self.default_dof.copy()
            motor_targets[self.action_ids] = lower_targets
            last_motor_targets = motor_targets.copy()

            for _ in range(int(round(self.dt / self.sim_dt))):
                torques = KPS * (motor_targets - self.data.qpos[7:]) + KDS * (-self.data.qvel[6:])
                self.data.ctrl[:] = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
                mujoco.mj_step(self.model, self.data)
                if not np.all(np.isfinite(self.data.qpos)):
                    finite = False
                    break
            if not finite:
                break

            root_err.append(float(np.linalg.norm(self.data.qpos[:3] - qpos[i, :3])))
            root_height_err.append(float(abs(self.data.qpos[2] - qpos[i, 2])))
            joint_abs = np.abs(self.data.qpos[7:] - qpos[i, 7:])
            joint_err.append(float(np.mean(joint_abs)))
            max_joint_err.append(float(np.max(joint_abs)))
            current_body_pos = self.data.xpos[self.valid_body_ids].copy()
            self.ref_data.qpos[:] = qpos[i]
            self.ref_data.qvel[:] = qvel[i]
            mujoco.mj_forward(self.model, self.ref_data)
            ref_body_pos = self.ref_data.xpos[self.valid_body_ids].copy()
            current_root_mat = quat_to_mat(self.data.qpos[3:7])
            ref_root_mat = quat_to_mat(qpos[i, 3:7])
            current_body_local = (
                current_body_pos - self.data.qpos[:3][None]
            ) @ current_root_mat
            ref_body_local = (ref_body_pos - qpos[i, :3][None]) @ ref_root_mat
            body_err.append(
                float(
                    np.mean(
                        np.linalg.norm(
                            current_body_pos - ref_body_pos,
                            axis=-1,
                        )
                    )
                )
            )
            ref_body_xy_aligned = ref_body_pos.copy()
            ref_body_xy_aligned[:, :2] += self.data.qpos[:2][None] - qpos[i, :2][None]
            xy_aligned_body_err.append(
                float(np.mean(np.linalg.norm(current_body_pos - ref_body_xy_aligned, axis=-1)))
            )
            local_body_err.append(
                float(np.mean(np.linalg.norm(current_body_local - ref_body_local, axis=-1)))
            )
            body_vel = (current_body_pos - prev_body_pos) / self.dt
            ref_body_vel = (ref_body_pos - prev_ref_body_pos) / self.dt
            body_vel_err.append(float(np.mean(np.linalg.norm(body_vel - ref_body_vel, axis=-1))))
            prev_current_root_mat = quat_to_mat(self.data.qpos[3:7])
            prev_ref_root_mat = quat_to_mat(qpos[i, 3:7])
            local_body_vel = (
                (current_body_pos - prev_body_pos) / self.dt
            ) @ prev_current_root_mat
            ref_local_body_vel = (
                (ref_body_pos - prev_ref_body_pos) / self.dt
            ) @ prev_ref_root_mat
            local_body_vel_err.append(
                float(np.mean(np.linalg.norm(local_body_vel - ref_local_body_vel, axis=-1)))
            )
            if prev_body_vel is not None:
                body_acc = (body_vel - prev_body_vel) / self.dt
                ref_body_acc = (ref_body_vel - prev_ref_body_vel) / self.dt
                local_body_acc = (local_body_vel - prev_local_body_vel) / self.dt
                ref_local_body_acc = (ref_local_body_vel - prev_ref_local_body_vel) / self.dt
                body_acc_err.append(float(np.mean(np.linalg.norm(body_acc - ref_body_acc, axis=-1))))
                local_body_acc_err.append(
                    float(np.mean(np.linalg.norm(local_body_acc - ref_local_body_acc, axis=-1)))
                )
            prev_body_vel = body_vel
            prev_ref_body_vel = ref_body_vel
            prev_local_body_vel = local_body_vel
            prev_ref_local_body_vel = ref_local_body_vel
            prev_body_pos = current_body_pos
            prev_ref_body_pos = ref_body_pos
            height.append(float(self.data.qpos[2]))
            if capture_qpos:
                rollout_qpos.append(self.data.qpos.copy())
            if capture_frames and (i % max(int(frame_stride), 1) == 0 or i == n_steps):
                rollout_frames.append(self._frame_from_data(self.data))

        if not root_err:
            root_err = [float("inf")]
            body_err = [float("inf")]
            xy_aligned_body_err = [float("inf")]
            local_body_err = [float("inf")]
            body_vel_err = [float("inf")]
            local_body_vel_err = [float("inf")]
            body_acc_err = [float("inf")]
            local_body_acc_err = [float("inf")]
            joint_err = [float("inf")]
            max_joint_err = [float("inf")]
            root_height_err = [float("inf")]
            height = [float("-inf")]

        raw_global_mpjpe_m = float(np.mean(body_err))
        xy_aligned_mpjpe_m = float(np.mean(xy_aligned_body_err))
        local_mpjpe_m = float(np.mean(local_body_err))
        root_err_m = float(np.mean(root_err))
        root_height_err_m = float(np.mean(root_height_err))
        paper_failed = (not finite) or local_mpjpe_m > 0.2 or root_height_err_m > 0.2
        strict_failed = (
            paper_failed
            or root_err_m > 1.0
            or float(np.max(max_joint_err)) > 0.7
        )
        failed = (
            (not finite)
            or max(local_body_err) > 0.75
            or min(height) < 0.25
            or max(max_joint_err) > 2.5
        )
        result = {
            "motion": path.stem,
            "steps": int(len(root_err)),
            "success": bool(not failed),
            "root_err_mean": root_err_m,
            "root_err_max": float(np.max(root_err)),
            "root_height_err_mean": root_height_err_m,
            "root_height_err_max": float(np.max(root_height_err)),
            "raw_body_err_mean": raw_global_mpjpe_m,
            "raw_body_err_max": float(np.max(body_err)),
            "body_err_mean": xy_aligned_mpjpe_m,
            "body_err_max": float(np.max(xy_aligned_body_err)),
            "xy_aligned_body_err_mean": xy_aligned_mpjpe_m,
            "xy_aligned_body_err_max": float(np.max(xy_aligned_body_err)),
            "local_body_err_mean": local_mpjpe_m,
            "local_body_err_max": float(np.max(local_body_err)),
            "body_vel_err_mean": float(np.mean(body_vel_err)),
            "local_body_vel_err_mean": float(np.mean(local_body_vel_err)),
            "body_acc_err_mean": float(np.mean(body_acc_err)) if body_acc_err else float("nan"),
            "local_body_acc_err_mean": float(np.mean(local_body_acc_err)) if local_body_acc_err else float("nan"),
            "raw_global_mpjpe_m": raw_global_mpjpe_m,
            "raw_global_mpjpe_mm": raw_global_mpjpe_m * 1000.0,
            "xy_aligned_mpjpe_m": xy_aligned_mpjpe_m,
            "xy_aligned_mpjpe_mm": xy_aligned_mpjpe_m * 1000.0,
            "mpjpe_m": xy_aligned_mpjpe_m,
            "mpjpe_mm": xy_aligned_mpjpe_m * 1000.0,
            "local_mpjpe_m": local_mpjpe_m,
            "local_mpjpe_mm": local_mpjpe_m * 1000.0,
            "mpjve_mps": float(np.mean(body_vel_err)),
            "local_mpjve_mps": float(np.mean(local_body_vel_err)),
            "mpjae_mps2": float(np.mean(body_acc_err)) if body_acc_err else float("nan"),
            "local_mpjae_mps2": float(np.mean(local_body_acc_err)) if local_body_acc_err else float("nan"),
            "joint_err_mean": float(np.mean(joint_err)),
            "max_joint_err_mean": float(np.mean(max_joint_err)),
            "max_joint_err_max": float(np.max(max_joint_err)),
            "min_height": float(np.min(height)),
            "paper_success": bool(not paper_failed),
            "strict_success": bool(not strict_failed),
        }
        if capture_frames:
            result["_robot_frames"] = self._robot_frames_payload(rollout_frames, frame_stride)
        if capture_qpos:
            result["_rollout_qpos"] = np.stack(rollout_qpos, axis=0).astype(np.float32)
            result["_rollout_fps"] = float(1.0 / self.dt)
        return result


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-motions", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--frames-manifest", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--canonical-root", type=Path, default=None)
    parser.add_argument("--canonical-split", default=None)
    parser.add_argument("--canonical-method", default="any2track")
    parser.add_argument("--canonical-output-fps", type=float, default=30.0)
    args = parser.parse_args()

    names = None
    if args.manifest is not None:
        names = json.loads(args.manifest.read_text())
    paths = [args.motion_dir / f"{name}.npz" for name in names] if names else sorted(args.motion_dir.glob("*.npz"))
    if args.max_motions is not None:
        paths = paths[: args.max_motions]

    runner = OpenTrackRollout(args.xml, json.loads(args.config.read_text()), args.onnx)
    rows = []
    viz_rows = []
    if args.frames_dir is not None:
        (args.frames_dir / "reference").mkdir(parents=True, exist_ok=True)
        (args.frames_dir / "opentrack").mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc="Any2Track eval"):
        row = runner.evaluate_motion(
            path,
            max_steps=args.max_steps,
            capture_frames=args.frames_dir is not None,
            capture_qpos=args.canonical_root is not None,
            frame_stride=args.frame_stride,
        )
        rollout_payload = row.pop("_robot_frames", None)
        rollout_qpos = row.pop("_rollout_qpos", None)
        rollout_fps = float(row.pop("_rollout_fps", 1.0 / runner.dt))
        rows.append(row)
        if args.canonical_root is not None:
            if not args.canonical_split:
                raise ValueError("--canonical-split is required with --canonical-root")
            ref_qpos, _ = runner._load_motion(path)
            n_steps = ref_qpos.shape[0] - 1
            if args.max_steps is not None:
                n_steps = min(n_steps, args.max_steps)
            ref_qpos = ref_qpos[: n_steps + 1]
            meta = {
                "source": str(path),
                "runner": "eval_opentrack_onnx_mujoco.py",
                "xml": str(args.xml),
                "config": str(args.config),
                "onnx": str(args.onnx),
                "max_steps": args.max_steps,
                "output_fps": args.canonical_output_fps,
            }
            write_reference_from_qpos(
                args.canonical_root,
                args.canonical_split,
                path.stem,
                ref_qpos,
                source_fps=1.0 / runner.dt,
                model=runner.model,
                target_fps=args.canonical_output_fps,
                metadata=meta,
            )
            if rollout_qpos is not None and rollout_qpos.size:
                save_qpos(
                    args.canonical_root,
                    args.canonical_split,
                    args.canonical_method,
                    path.stem,
                    rollout_qpos,
                    source_fps=rollout_fps,
                    target_fps=args.canonical_output_fps,
                    metadata={**meta, "execution_frames_source": int(rollout_qpos.shape[0])},
                )
                exec_qpos30 = _resample_qpos(rollout_qpos, rollout_fps, args.canonical_output_fps)
                body_pos, body_quat, body_names = runner._body_arrays_from_qpos(exec_qpos30)
                save_body(
                    args.canonical_root,
                    args.canonical_split,
                    args.canonical_method,
                    path.stem,
                    body_pos,
                    body_quat,
                    body_names,
                    source_fps=args.canonical_output_fps,
                    target_fps=args.canonical_output_fps,
                    metadata={**meta, "execution_frames_source": int(rollout_qpos.shape[0])},
                )
        if args.frames_dir is not None and rollout_payload is not None:
            ref_payload = runner.reference_motion_frames(
                path,
                max_steps=args.max_steps,
                frame_stride=args.frame_stride,
            )
            ref_path = (args.frames_dir / "reference" / f"{path.stem}.json").resolve()
            rollout_path = (args.frames_dir / "opentrack" / f"{path.stem}.json").resolve()
            ref_path.write_text(json.dumps(ref_payload))
            rollout_path.write_text(json.dumps(rollout_payload))
            metrics = {
                "success": row["success"],
                "paper_success": row["paper_success"],
                "g_mpjpe_mm": row["mpjpe_mm"],
                "root_frame_mpjpe_mm": row["local_mpjpe_mm"],
                "root_err_mean_m": row["root_err_mean"],
                "min_height_m": row["min_height"],
            }
            viz_rows.append(
                {
                    "iteration": len(viz_rows),
                    "iteration_label": f"Case {len(viz_rows):02d}  ·  {path.stem}",
                    "prompt_id": path.stem,
                    "prompt": f"LAFAN1-G1 motion {path.stem}",
                    "columns": {
                        "reference": {
                            "status": "ready",
                            "title": "LAFAN1-G1 reference",
                            "path": str(ref_path),
                            "metrics": {},
                        },
                        "opentrack": {
                            "status": "ready",
                            "title": "Any2Track",
                            "path": str(rollout_path),
                            "metrics": metrics,
                        },
                    },
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_motions": len(rows),
        "success_rate": _mean(rows, "success"),
        "root_err_mean": _mean(rows, "root_err_mean"),
        "root_err_max_mean": _mean(rows, "root_err_max"),
        "root_height_err_mean": _mean(rows, "root_height_err_mean"),
        "root_height_err_max_mean": _mean(rows, "root_height_err_max"),
        "body_err_mean": _mean(rows, "body_err_mean"),
        "body_err_max_mean": _mean(rows, "body_err_max"),
        "local_body_err_mean": _mean(rows, "local_body_err_mean"),
        "local_body_err_max_mean": _mean(rows, "local_body_err_max"),
        "body_vel_err_mean": _mean(rows, "body_vel_err_mean"),
        "local_body_vel_err_mean": _mean(rows, "local_body_vel_err_mean"),
        "body_acc_err_mean": _mean(rows, "body_acc_err_mean"),
        "local_body_acc_err_mean": _mean(rows, "local_body_acc_err_mean"),
        "mpjpe_m": _mean(rows, "mpjpe_m"),
        "mpjpe_mm": _mean(rows, "mpjpe_mm"),
        "local_mpjpe_m": _mean(rows, "local_mpjpe_m"),
        "local_mpjpe_mm": _mean(rows, "local_mpjpe_mm"),
        "mpjve_mps": _mean(rows, "mpjve_mps"),
        "local_mpjve_mps": _mean(rows, "local_mpjve_mps"),
        "mpjae_mps2": _mean(rows, "mpjae_mps2"),
        "local_mpjae_mps2": _mean(rows, "local_mpjae_mps2"),
        "paper_success_rate": _mean(rows, "paper_success"),
        "strict_success_rate": _mean(rows, "strict_success"),
        "joint_err_mean": _mean(rows, "joint_err_mean"),
        "max_joint_err_mean": _mean(rows, "max_joint_err_mean"),
        "max_joint_err_max_mean": _mean(rows, "max_joint_err_max"),
        "min_height_mean": _mean(rows, "min_height"),
    }
    args.output_json.write_text(json.dumps({"summary": summary, "motions": rows}, indent=2) + "\n")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["motion"])
        writer.writeheader()
        writer.writerows(rows)
    if args.frames_manifest is not None:
        args.frames_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "project": "Any2Track LAFAN1-G1 Visualization",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_from": {
                "motion_dir": str(args.motion_dir),
                "xml": str(args.xml),
                "config": str(args.config),
                "onnx": str(args.onnx),
                "max_steps": args.max_steps,
                "frame_stride": args.frame_stride,
            },
            "group_label": "case",
            "column_order": [
                {"key": "reference", "title": "LAFAN1-G1 reference", "color": "raw"},
                {"key": "opentrack", "title": "Any2Track", "color": "track"},
            ],
            "rows": viz_rows,
        }
        args.frames_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.canonical_root is not None:
        for rep in ("g1_qpos30", "g1_body30"):
            write_run_config(
                args.canonical_root,
                args.canonical_split or "unknown",
                rep,
                args.canonical_method,
                {
                    "method": args.canonical_method,
                    "runner": "motius.models.gentrack.any2track_runtime",
                    "motion_dir": str(args.motion_dir),
                    "manifest": str(args.manifest) if args.manifest else None,
                    "xml": str(args.xml),
                    "config": str(args.config),
                    "onnx": str(args.onnx),
                    "max_steps": args.max_steps,
                    "output_fps": args.canonical_output_fps,
                    "num_rows": len(rows),
                },
            )


if __name__ == "__main__":
    main()
