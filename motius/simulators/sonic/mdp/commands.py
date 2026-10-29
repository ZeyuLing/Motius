# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 commands components; see licenses/sonic."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import copy
import dataclasses
import glob
import os
from typing import TYPE_CHECKING
import easydict
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_apply_yaw
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from isaaclab.utils.math import sample_uniform
import numpy as np
import torch
import motius.motion.representation.sonic.order as joint_utils
import motius.motion.geometry.sonic.rotations as rotations
import motius.configuration.sonic.components as common
import motius.motion.representation.sonic.order as order_converter
import motius.motion.geometry.sonic.transform as torch_transform
import motius.datasets.sonic_motion_robot as motion_lib_robot

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
INACTIVE_OBJECT_BASE_OFFSET = torch.tensor([1000.0, 0.0, -50.0])
INACTIVE_OBJECT_Z_SPACING = 10.0


def _init_variable_frames(
    enabled: bool,
    min_frames: int,
    num_future_frames: int,
    step: int,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Initialize variable frame support tensors.

    Returns (per_env_num_frames, frame_choices) or (None, None) if disabled.
    """
    if not enabled:
        return (None, None)
    assert step > 0, f"variable_frames_step must be positive, got {step}"
    per_env_num_frames = torch.full(
        (num_envs,), num_future_frames, device=device, dtype=torch.long
    )
    frame_choices = torch.arange(min_frames, num_future_frames + 1, step, device=device)
    assert (
        len(frame_choices) > 0
    ), f"No valid frame choices: variable_frames_min={min_frames} > num_future_frames={num_future_frames}"
    return (per_env_num_frames, frame_choices)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = None
    force = None


class TrackingCommand(CommandTerm):
    """Provide reference motion trajectories for motion-tracking RL.

    This is the primary command term for SONIC-style humanoid control. It manages
    a motion library of pre-recorded motion clips (robot joint trajectories, SMPL
    poses, and optionally object trajectories) and serves reference frames to the
    policy and reward system at each simulation step.

    Key responsibilities:
        - Load and index motion clips from the motion library (MotionLibRobot).
        - Sample motion IDs and start times for each environment at episode reset.
        - Advance the time cursor each step and provide current + multi-future
          reference frames (positions, orientations, velocities, joint states).
        - Transform reference quantities into robot-local, egocentric, or
          heading-canonicalized coordinate frames for observation terms.
        - Handle DOF mismatch between motion data and robot (e.g., extra finger
          joints not present in the motion library).
        - Support adaptive sampling, contact-based initialization, variable future
          frame counts, encoder mode sampling (G1/SMPL/teleop), and multi-object
          scene management.

    The command exposes ~60 property-based accessors that observation and reward
    terms query each step. Properties are named by the pattern::

        {quantity}[_multi_future][_dif][_{frame}]

    where ``quantity`` is body_pos, joint_pos, smpl_pose, etc.; ``multi_future``
    means values for all ``num_future_frames`` reference frames stacked;
    ``dif`` is the difference relative to the robot; and ``frame`` is ``w``
    (world), ``l`` (robot-local / de-headed), or ``b`` (body).
    """

    cfg: TrackingCommandCfg

    def __init__(
        self,
        cfg: TrackingCommandCfg,
        env: ManagerBasedRLEnv,
        max_num_load_motions: int = None,
    ):
        """Initialize the tracking command with motion library and body mappings.

        Loads motion data, builds DOF/body index mappings between the motion
        library (MuJoCo ordering) and the IsaacLab robot, pre-allocates per-env
        tensors for motion IDs, time steps, and future frame offsets, and
        optionally sets up height-map raycasting.

        Args:
            cfg: Configuration dataclass specifying motion files, body names,
                future frame settings, encoder sampling probabilities, etc.
            env: The manager-based RL environment that owns this command.
            max_num_load_motions: Cap on how many unique motion clips to load
                into GPU memory. Defaults to ``min(num_envs, 1024)`` or the
                full library when ``use_paired_motions`` is enabled.
        """
        super().__init__(cfg, env)
        self.is_evaluating = False
        self.cmd_body_names = self.cfg.body_names
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = (
            torch.tensor(
                self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device
            )
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.reward_point_body_indices = [
            self.robot.body_names.index(name) for name in self.cfg.reward_point_body
        ]
        self.reward_point_body_offsets = (
            torch.tensor(
                self.cfg.reward_point_body_offset,
                dtype=torch.float32,
                device=self.device,
            )
            .view(1, -1, 3)
            .repeat(self.num_envs, 1, 1)
        )
        self.reward_point_body_indices_motion = [
            self.cfg.body_names.index(name) for name in self.cfg.reward_point_body
        ]
        self.down_dir = (
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        isaac_lab_joints = env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"]
        self.isaaclab_to_mujoco_dof = env.cfg.isaaclab_to_mujoco_mapping[
            "isaaclab_to_mujoco_dof"
        ]
        self.mujoco_to_isaaclab_dof = env.cfg.isaaclab_to_mujoco_mapping[
            "mujoco_to_isaaclab_dof"
        ]
        self.lower_joint_indices_mujoco = list(range(12))
        self.lower_joint_isaaclab_indices = [
            self.isaaclab_to_mujoco_dof[i] for i in self.lower_joint_indices_mujoco
        ]
        self.isaaclab_to_mujoco_body = env.cfg.isaaclab_to_mujoco_mapping[
            "isaaclab_to_mujoco_body"
        ]
        self.mujoco_to_isaaclab_body = env.cfg.isaaclab_to_mujoco_mapping[
            "mujoco_to_isaaclab_body"
        ]
        self.running_ref_root_height = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.body_indexes_data = [
            isaac_lab_joints.index(name) for name in self.cfg.body_names
        ]
        if self.cfg.motion_lib_cfg is not None:
            motion_lib_cfg = easydict.EasyDict(self.cfg.motion_lib_cfg)
        else:
            motion_lib_cfg = easydict.EasyDict(
                {
                    "motion_file": self.cfg.motion_file,
                    "smpl_motion_file": getattr(self.cfg, "smpl_motion_file", None),
                    "asset": {
                        "assetRoot": str(Path(__file__).resolve().parents[4] / "checkpoints/robots/sonic/robot_description/mjcf"),
                        "assetFileName": "g1_29dof_rev_1_0.xml",
                        "urdfFileName": "",
                    },
                    "extend_config": [],
                    "target_fps": 50,
                    "multi_thread": True,
                    "filter_motion_keys": self.cfg.filter_motion_keys,
                }
            )
        filter_keys = self.cfg.filter_motion_keys
        if filter_keys is None:
            filter_keys = motion_lib_cfg.get("filter_motion_keys", None)
        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
                "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
                "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
                "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
                "body_indexes": self.body_indexes,
                "body_indexes_data": self.body_indexes_data,
                "filter_motion_keys": filter_keys,
                "lower_joint_indices_mujoco": self.lower_joint_indices_mujoco,
                "cat_upper_body_poses": self.cfg.cat_upper_body_poses,
                "cat_upper_body_poses_prob": self.cfg.cat_upper_body_poses_prob,
                "randomize_heading": self.cfg.randomize_heading,
                "freeze_frame_aug": self.cfg.freeze_frame_aug,
                "freeze_frame_aug_prob": self.cfg.freeze_frame_aug_prob,
                "randomize_wrist_poses": self.cfg.randomize_wrist_poses,
                "randomize_wrist_prob": self.cfg.randomize_wrist_prob,
                "randomize_wrist_std": self.cfg.randomize_wrist_std,
            }
        )
        self.motion_lib = motion_lib_robot.MotionLibRobot(
            motion_lib_cfg, self.num_envs, self.device
        )
        if max_num_load_motions is None:
            if self.cfg.use_paired_motions:
                self.max_num_load_motions = self.motion_lib._num_unique_motions
            else:
                self.max_num_load_motions = min(self.num_envs, 1024)
        else:
            self.max_num_load_motions = max_num_load_motions
        self.motion_lib.load_motions_for_training(
            max_num_seqs=self.max_num_load_motions
        )
        self.use_adaptive_sampling = self.motion_lib.use_adaptive_sampling
        self._load_contact_data()
        self.robot_num_dof = self.robot.num_joints
        self.motion_lib_num_dof = self.cfg.motion_lib_num_dof
        if self.motion_lib_num_dof is None:
            self.motion_lib_num_dof = self.robot_num_dof
        self.extra_num_dof = self.robot_num_dof - self.motion_lib_num_dof
        self.has_dof_mismatch = self.extra_num_dof > 0
        if self.has_dof_mismatch:
            self.body_joint_indices = joint_utils.get_body_joint_indices(self.robot)
            self.extra_joint_indices = joint_utils.get_hand_joint_indices(self.robot)
            self.extra_default_positions = torch.tensor(
                self.cfg.hand_default_positions or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
            self.extra_default_velocities = torch.tensor(
                self.cfg.hand_default_velocities or [0.0] * self.extra_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
        if self.cfg.use_paired_motions:
            self.motion_ids = (
                torch.arange(self.num_envs, device=self.device)
                % self.motion_lib._num_motions
            )
        elif getattr(self.cfg, "sample_unique_motions", False):
            num_available = len(self.motion_lib._curr_motion_ids)
            if self.num_envs > num_available:
                raise ValueError(
                    f"sample_unique_motions=True requires num_envs ({self.num_envs}) <= num_available_motions ({num_available})"
                )
            perm = torch.randperm(num_available, device=self.device)[: self.num_envs]
            self.motion_ids = perm
            print(
                f"[TrackingCommand] Sampled {self.num_envs} unique motions (no duplicates)"
            )
        else:
            self.motion_ids = self.motion_lib.sample_motions(self.num_envs)
        self.motion_start_time_steps = self.motion_lib.sample_time_steps(
            self.motion_ids, truncate_time=None
        )
        if self.cfg.sample_from_n_initial_frames is not None:
            n_frames = self.cfg.sample_from_n_initial_frames
            self.motion_start_time_steps = torch.randint(
                0,
                n_frames,
                (self.num_envs,),
                dtype=self.motion_start_time_steps.dtype,
                device=self.device,
            )
        elif self.cfg.start_from_first_frame:
            self.motion_start_time_steps.zero_()
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._object_position_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.num_future_frames = self.cfg.num_future_frames
        self.frame_skips = self.cfg.dt_future_ref_frames // (
            1.0 / motion_lib_cfg.target_fps
        )
        steps_exact = self.cfg.dt_future_ref_frames * motion_lib_cfg.target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-09:
            import warnings

            warnings.warn(
                f"dt_future_ref_frames={self.cfg.dt_future_ref_frames} * target_fps={motion_lib_cfg.target_fps} = {steps_exact} is not an integer; using frame_skips={self.frame_skips} so effective ref-frame spacing is {self.frame_skips / motion_lib_cfg.target_fps:.4f}s (not {self.cfg.dt_future_ref_frames}s). Velocity-based losses may use incorrect dt.",
                stacklevel=2,
            )
        self.future_time_steps_init = (
            (
                torch.arange(
                    self.num_future_frames, device=self.device, dtype=torch.long
                )
                * self.frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )
        if self.cfg.smpl_num_future_frames is None:
            self.smpl_num_future_frames = self.num_future_frames
        else:
            self.smpl_num_future_frames = self.cfg.smpl_num_future_frames
        if self.cfg.smpl_dt_future_ref_frames is None:
            self.smpl_dt_future_ref_frames = self.cfg.dt_future_ref_frames
        else:
            self.smpl_dt_future_ref_frames = self.cfg.smpl_dt_future_ref_frames
        self.smpl_frame_skips = self.smpl_dt_future_ref_frames // (
            1.0 / motion_lib_cfg.target_fps
        )
        self.smpl_future_time_steps_init = (
            (
                torch.arange(
                    self.smpl_num_future_frames, device=self.device, dtype=torch.long
                )
                * self.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(self.num_envs, 1)
        )
        self.future_motion_ids = self.motion_ids.repeat_interleave(
            self.num_future_frames
        )
        self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(
            self.smpl_num_future_frames
        )
        self.variable_frames_enabled = getattr(
            self.cfg, "variable_frames_enabled", False
        )
        self.per_env_num_frames, self._frame_choices = _init_variable_frames(
            self.variable_frames_enabled,
            getattr(self.cfg, "variable_frames_min", 16),
            self.num_future_frames,
            getattr(self.cfg, "variable_frames_step", 4),
            self.num_envs,
            self.device,
        )
        self.encoder_sample_probs_dict = self.cfg.encoder_sample_probs
        self.optimize_encoders_ratio_for_CHIP = getattr(
            self.cfg, "optimize_encoders_ratio_for_CHIP", False
        )
        self.encoder_sample_probs = None
        if self.encoder_sample_probs_dict is not None:
            self.encoder_sample_probs = torch.tensor(
                list(self.encoder_sample_probs_dict.values()), device=self.device
            )
            self.encoder_sample_probs = (
                self.encoder_sample_probs / self.encoder_sample_probs.sum()
            )
            self.encoder_sample_probs_no_smpl_dict = copy.deepcopy(
                self.encoder_sample_probs_dict
            )
            if "smpl" in self.encoder_sample_probs_no_smpl_dict:
                self.encoder_sample_probs_no_smpl_dict["smpl"] = 0.0
            self.encoder_sample_probs_no_smpl = torch.tensor(
                list(self.encoder_sample_probs_no_smpl_dict.values()),
                device=self.device,
            )
            self.encoder_sample_probs_no_smpl = (
                self.encoder_sample_probs_no_smpl
                / self.encoder_sample_probs_no_smpl.sum()
            )
            self.g1_encoder_index = list(self.encoder_sample_probs_dict.keys()).index(
                "g1"
            )
            if "smpl" in self.encoder_sample_probs_dict:
                self.smpl_encoder_index = list(
                    self.encoder_sample_probs_dict.keys()
                ).index("smpl")
            else:
                self.smpl_encoder_index = None
            if "teleop" in self.encoder_sample_probs_dict:
                self.teleop_encoder_index = list(
                    self.encoder_sample_probs_dict.keys()
                ).index("teleop")
            else:
                self.teleop_encoder_index = None
            if "soma" in self.encoder_sample_probs_dict:
                self.soma_encoder_index = list(
                    self.encoder_sample_probs_dict.keys()
                ).index("soma")
                encoder_sample_probs_no_soma_dict = copy.deepcopy(
                    self.encoder_sample_probs_dict
                )
                encoder_sample_probs_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_soma_dict.values()), device=self.device
                )
                no_soma_sum = self.encoder_sample_probs_no_soma.sum()
                if no_soma_sum > 0:
                    self.encoder_sample_probs_no_soma = (
                        self.encoder_sample_probs_no_soma / no_soma_sum
                    )
                else:
                    self.encoder_sample_probs_no_soma[self.g1_encoder_index] = 1.0
                encoder_sample_probs_no_smpl_no_soma_dict = copy.deepcopy(
                    self.encoder_sample_probs_no_smpl_dict
                )
                if "soma" in encoder_sample_probs_no_smpl_no_soma_dict:
                    encoder_sample_probs_no_smpl_no_soma_dict["soma"] = 0.0
                self.encoder_sample_probs_no_smpl_no_soma = torch.tensor(
                    list(encoder_sample_probs_no_smpl_no_soma_dict.values()),
                    device=self.device,
                )
                no_smpl_no_soma_sum = self.encoder_sample_probs_no_smpl_no_soma.sum()
                if no_smpl_no_soma_sum > 0:
                    self.encoder_sample_probs_no_smpl_no_soma = (
                        self.encoder_sample_probs_no_smpl_no_soma / no_smpl_no_soma_sum
                    )
                else:
                    self.encoder_sample_probs_no_smpl_no_soma[self.g1_encoder_index] = (
                        1.0
                    )
            else:
                self.soma_encoder_index = None
                self.encoder_sample_probs_no_soma = None
                self.encoder_sample_probs_no_smpl_no_soma = None
            self.teleop_sample_prob_when_smpl = self.cfg.teleop_sample_prob_when_smpl
            self.encoder_index = torch.zeros(
                (self.num_envs, self.encoder_sample_probs.shape[0]),
                dtype=torch.long,
                device=self.device,
            )
            if "smpl" in self.encoder_sample_probs_dict:
                assert (
                    self.smpl_encoder_index > self.g1_encoder_index
                ), f"SMPL encoder index {self.smpl_encoder_index} must be greater than G1 encoder index {self.g1_encoder_index} to when both exist!"
        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.use_ref_motion_root_quat_w_as_anchor = False
        self.ref_motion_root_rotation_noise = None
        self.num_bodies = len(self.cfg.body_names)
        self._multi_object_mode = False
        self._object_names = []
        self._active_object_name = None
        if hasattr(self._env, "scene") and hasattr(self._env.scene, "rigid_objects"):
            rigid_obj_keys = list(self._env.scene.rigid_objects.keys())
            multi_obj_keys = [k for k in rigid_obj_keys if k.startswith("object_")]
            if len(multi_obj_keys) > 0:
                self._multi_object_mode = True
                for key in multi_obj_keys:
                    safe_name = key[7:]
                    original_name = safe_name.replace("_", "-")
                    self._object_names.append(safe_name)
                print(
                    f"[Multi-Object Mode] Detected {len(self._object_names)} objects in scene"
                )
        self._table_meta_cache = {}
        if self.cfg.use_height_map:
            try:
                from simple_raycaster.raycaster import MultiMeshRaycaster
            except ImportError:
                command_install = "pip install -e git+https://github.com/Agent-3154/simple-raycaster.git@197daa6dcb146c5ce3e675a173328e17df6b9777#egg=simple-raycaster"
                raise ImportError(
                    f"simple-raycaster is required for height map observation. Install with: {command_install}"
                )
            import omni

            prim_path_patterns = ["/World/envs/env_\\d+/Object"]
            stage = omni.usd.get_context().get_stage()
            self.height_map = MultiMeshRaycaster.from_prim_paths(
                paths=prim_path_patterns, stage=stage, device=str(self.device)
            )
            mesh_filters = [
                [f"/World/envs/env_{i}/Object"] for i in range(self.num_envs)
            ]
            self.num_mesh_per_cam, self.mesh_ids, self.cam_ids = (
                self.height_map.get_mesh_ids(mesh_filters, device=self.device)
            )
            lin = torch.linspace(
                -0.5 * self.cfg.height_map_size,
                0.5 * self.cfg.height_map_size,
                int(self.cfg.height_map_size / self.cfg.height_map_resolution) + 1,
                device=self.device,
            )
            grid_x, grid_y = torch.meshgrid(lin, lin)
            zeros = torch.zeros_like(grid_x)
            scan_offsets = torch.stack([grid_x, grid_y, zeros], dim=-1).view(-1, 3)
            scan_offsets[:, 2] = -1.0
            ray_dirs_local = torch.nn.functional.normalize(scan_offsets, dim=-1)
            self.ray_dirs_local = ray_dirs_local.expand(self.num_envs, -1, -1)
            self.num_rays = self.ray_dirs_local.shape[1]
            self.num_rays_x, self.num_rays_y = grid_x.shape
            self.scan_dot_pos_w = torch.zeros(
                self.num_envs, self.num_rays_x, self.num_rays_y, 3, device=self.device
            )

    @classmethod
    def create_offline(cls, motion_lib_cfg, device):
        """Create a MotionCommand without env/sim for offline kinematic use.

        All @property methods that only use motion_lib data work unchanged.
        Properties that require robot state (self.robot) will use the reference
        motion root as the anchor (equivalent to use_ref_motion_root_quat_w_as_anchor=True).

        Expected keys in motion_lib_cfg:
            num_future_frames: Number of future reference frames.
            dt_future_ref_frames: Time delta between future reference frames (seconds).
        """
        inst = object.__new__(cls)
        inst._offline = True
        inst._debug_vis_handle = None
        num_envs = 1

        class _MinimalEnv:
            pass

        _env = _MinimalEnv()
        _env.device = device
        _env.num_envs = num_envs
        inst._env = _env
        motion_lib_cfg = (
            easydict.EasyDict(motion_lib_cfg)
            if not isinstance(motion_lib_cfg, easydict.EasyDict)
            else motion_lib_cfg
        )
        num_future_frames = motion_lib_cfg.get("num_future_frames", 8)
        dt_future_ref_frames = motion_lib_cfg.get("dt_future_ref_frames", 0.1)
        isaaclab_to_mujoco_mapping = (
            order_converter.G1Converter().get_isaaclab_to_mujoco_mapping()
        )
        motion_lib_cfg.update(
            {
                "mujoco_to_isaaclab_body": isaaclab_to_mujoco_mapping[
                    "mujoco_to_isaaclab_body"
                ],
                "mujoco_to_isaaclab_dof": isaaclab_to_mujoco_mapping[
                    "mujoco_to_isaaclab_dof"
                ],
                "isaaclab_to_mujoco_body": isaaclab_to_mujoco_mapping[
                    "isaaclab_to_mujoco_body"
                ],
                "isaaclab_to_mujoco_dof": isaaclab_to_mujoco_mapping[
                    "isaaclab_to_mujoco_dof"
                ],
            }
        )
        isaaclab_joints = isaaclab_to_mujoco_mapping["isaaclab_joints"]
        body_names = motion_lib_cfg.get("body_names", None)
        if "body_indexes_data" not in motion_lib_cfg:
            if body_names is not None:
                motion_lib_cfg.body_indexes_data = [
                    isaaclab_joints.index(name) for name in body_names
                ]
            else:
                num_bodies_full = len(
                    isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_body"]
                )
                motion_lib_cfg.body_indexes_data = list(range(num_bodies_full))
        inst.motion_lib = motion_lib_robot.MotionLibRobot(
            motion_lib_cfg, num_envs, device
        )
        max_num_motions = motion_lib_cfg.get("max_num_motions", None)
        inst.motion_lib.load_motions_for_training(max_num_seqs=max_num_motions)
        target_fps = motion_lib_cfg.get("target_fps", 50)
        inst.num_future_frames = num_future_frames
        inst.frame_skips = dt_future_ref_frames // (1.0 / target_fps)
        steps_exact = dt_future_ref_frames * target_fps
        if abs(steps_exact - round(steps_exact)) > 1e-09:
            import warnings

            warnings.warn(
                f"dt_future_ref_frames={dt_future_ref_frames} * target_fps={target_fps} = {steps_exact} is not an integer; using frame_skips={inst.frame_skips}",
                stacklevel=2,
            )
        inst.future_time_steps_init = (
            (
                torch.arange(num_future_frames, device=device, dtype=torch.long)
                * inst.frame_skips
            )
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        anchor_body = motion_lib_cfg.get("anchor_body", None)
        if body_names is not None and anchor_body is not None:
            inst.motion_anchor_body_index = body_names.index(anchor_body)
        else:
            inst.motion_anchor_body_index = 0
        inst.num_bodies = len(motion_lib_cfg.body_indexes_data)
        if isaaclab_to_mujoco_mapping is not None:
            inst.isaaclab_to_mujoco_dof = isaaclab_to_mujoco_mapping[
                "isaaclab_to_mujoco_dof"
            ]
            inst.mujoco_to_isaaclab_dof = isaaclab_to_mujoco_mapping[
                "mujoco_to_isaaclab_dof"
            ]
        inst.lower_joint_indices_mujoco = list(range(12))
        if hasattr(inst, "isaaclab_to_mujoco_dof"):
            inst.lower_joint_isaaclab_indices = [
                inst.isaaclab_to_mujoco_dof[i] for i in inst.lower_joint_indices_mujoco
            ]
        inst.variable_frames_enabled = motion_lib_cfg.get(
            "variable_frames_enabled", False
        )
        inst.per_env_num_frames, inst._frame_choices = _init_variable_frames(
            inst.variable_frames_enabled,
            motion_lib_cfg.get("variable_frames_min", 16),
            num_future_frames,
            motion_lib_cfg.get("variable_frames_step", 4),
            num_envs,
            device,
        )
        inst.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.motion_start_time_steps = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        inst.future_motion_ids = inst.motion_ids.repeat_interleave(num_future_frames)
        inst.motion_num_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        inst.use_ref_motion_root_quat_w_as_anchor = True
        inst.ref_motion_root_rotation_noise = None
        smpl_num_future_frames = motion_lib_cfg.get("smpl_num_future_frames", None)
        smpl_dt_future_ref_frames = motion_lib_cfg.get(
            "smpl_dt_future_ref_frames", None
        )
        inst.smpl_num_future_frames = (
            smpl_num_future_frames
            if smpl_num_future_frames is not None
            else num_future_frames
        )
        inst.smpl_dt_future_ref_frames = (
            smpl_dt_future_ref_frames
            if smpl_dt_future_ref_frames is not None
            else dt_future_ref_frames
        )
        inst.smpl_frame_skips = inst.smpl_dt_future_ref_frames // (1.0 / target_fps)
        smpl_steps_exact = inst.smpl_dt_future_ref_frames * target_fps
        if abs(smpl_steps_exact - round(smpl_steps_exact)) > 1e-09:
            import warnings

            warnings.warn(
                f"smpl_dt_future_ref_frames={inst.smpl_dt_future_ref_frames} * target_fps={target_fps} = {smpl_steps_exact} is not an integer; using smpl_frame_skips={inst.smpl_frame_skips}",
                stacklevel=2,
            )
        inst.smpl_future_time_steps_init = (
            (
                torch.arange(
                    inst.smpl_num_future_frames, device=device, dtype=torch.long
                )
                * inst.smpl_frame_skips
            )
            .view(1, -1)
            .repeat(num_envs, 1)
        )
        inst.smpl_future_motion_ids = inst.motion_ids.repeat_interleave(
            inst.smpl_num_future_frames
        )
        inst.encoder_sample_probs_dict = None
        inst.encoder_sample_probs = None
        inst.is_evaluating = False
        inst.metrics = {}
        return inst

    def set_is_evaluating(self, is_evaluating: bool = True):
        """Toggle evaluation mode, which disables reset randomizations."""
        self.is_evaluating = is_evaluating

    def forward_motion_samples(self, env_ids: Sequence[int]):
        """Assign sequential motion IDs and reset time steps for given envs.

        Used during paired/evaluation mode to deterministically cycle through
        motions. Updates motion IDs, start times, motion lengths, and caches
        the initial body pose for relative-frame computation.

        Args:
            env_ids: Environment indices to reassign motions for.
        """
        self.motion_ids[env_ids] = (
            torch.arange(self.num_envs).to(self.device) % self.motion_lib._num_motions
        )[env_ids]
        sampled_times = self.motion_lib.sample_time_steps(
            self.motion_ids[env_ids], truncate_time=None
        )
        if self.cfg.sample_from_n_initial_frames is not None:
            n_frames = self.cfg.sample_from_n_initial_frames
            sampled_times = torch.randint(
                0,
                n_frames,
                (len(env_ids),),
                dtype=sampled_times.dtype,
                device=self.device,
            )
        elif self.cfg.start_from_first_frame:
            sampled_times.zero_()
        self.motion_start_time_steps[env_ids] = sampled_times
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
            self.motion_ids[env_ids]
        )
        self.time_steps[env_ids] = 0
        self.body_pos_relative_w[env_ids] = self.motion_lib.get_body_pos_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )
        self.body_quat_relative_w[env_ids] = self.motion_lib.get_body_quat_w(
            self.motion_ids[env_ids], self.motion_start_time_steps[env_ids]
        )

    @property
    def command(self) -> torch.Tensor:
        """Return current-frame joint positions and velocities concatenated.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_dof)``.
        """
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def command_z(self) -> torch.Tensor:
        """Return reference root height (z) for the current frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z

    @property
    def command_z_multi_future(self) -> torch.Tensor:
        """Return reference root height (z) for the first future frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        return self.root_z_multi_future

    @property
    def command_vel(self) -> torch.Tensor:
        """Return reference root velocity (2D linear + 1D angular) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return torch.cat([self.root_lin_vel_b_2d, self.root_ang_vel_b_1d], dim=1)

    @property
    def command_max(self) -> torch.Tensor:
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_w,
                self.body_quat_w,
                self.body_lin_vel_w,
                self.body_ang_vel_w,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max(self) -> torch.Tensor:
        """Return all body state (pos, quat, lin_vel, ang_vel) flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_w,
                self.body_quat_w,
                self.body_lin_vel_w,
                self.body_ang_vel_w,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l(self) -> torch.Tensor:
        """Return body state differences in robot-local frame, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_dif_l.view(self.num_envs, -1),
                self.body_quat_dif_l.view(self.num_envs, -1),
                self.body_lin_vel_l,
                self.body_ang_vel_l,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_diff_l_multi_future(self) -> torch.Tensor:
        """Return multi-future body state diffs and local poses concatenated.

        Returns:
            Tensor of shape ``(num_envs, <variable>)`` depending on body count and future frames.
        """
        return torch.cat(
            [
                self.body_pos_dif_l_multi_future.view(self.num_envs, -1),
                self.body_quat_dif_l_multi_future.view(self.num_envs, -1),
                self.body_lin_vel_dif_l_multi_future,
                self.body_ang_vel_dif_l_multi_future,
                self.bod_pos_local_multi_future,
                self.body_quat_local_multi_future,
                self.body_lin_vel_l_multi_future,
                self.body_ang_vel_l_multi_future,
            ],
            dim=-1,
        ).view(self.num_envs, -1)

    @property
    def command_max_multi_future(self) -> torch.Tensor:
        """Return all body state for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 13)``.
        """
        return torch.cat(
            [
                self.body_pos_w_multi_future,
                self.body_quat_w_multi_future,
                self.body_lin_vel_w_multi_future,
                self.body_ang_vel_w_multi_future,
            ],
            dim=1,
        )

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Return joint positions and velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, 2 * num_future_frames * num_dof)``.
        """
        return torch.cat(
            [self.joint_pos_multi_future, self.joint_vel_multi_future], dim=1
        )

    @property
    def command_multi_future_joint_pos(self) -> torch.Tensor:
        """Return joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.joint_pos_multi_future

    @property
    def command_multi_future_joint_body_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions for all future frames.

        Body positions are expressed relative to each frame's anchor position,
        rotated into the anchor's yaw-inverse frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(
                self.future_motion_ids, self.future_time_steps
            )[:, self.motion_anchor_body_index]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
        )
        body_pos_relative_to_anchor_multi_frame = (
            body_pos_multi_frame - anchor_pos_multi_frame
        )
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(
            self.num_envs, self.num_future_frames, 1, 4
        ).expand(self.num_envs, self.num_future_frames, self.num_bodies, 4)
        body_pos_multi_frame = quat_apply_yaw(
            quat_inv(anchor_quat_w_repeat), body_pos_relative_to_anchor_multi_frame
        ).reshape(self.num_envs, -1)
        return torch.cat([self.joint_pos_multi_future, body_pos_multi_frame], dim=1)

    @property
    def command_multi_future_joint_body_abs_pos(self) -> torch.Tensor:
        """Return joint positions and anchor-relative body positions (without rotation).

        Unlike ``command_multi_future_joint_body_pos``, the body position deltas
        are not rotated into the anchor's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        body_pos_multi_frame = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).reshape(self.num_envs, -1)
        anchor_pos_multi_frame = (
            self.motion_lib.get_body_pos_w(
                self.future_motion_ids, self.future_time_steps
            )[:, self.motion_anchor_body_index]
            .view(self.num_envs, self.num_future_frames, 1, 3)
            .expand(self.num_envs, self.num_future_frames, self.num_bodies, 3)
            .reshape(self.num_envs, -1)
        )
        body_pos_relative_to_anchor_multi_frame = (
            body_pos_multi_frame - anchor_pos_multi_frame
        )
        return torch.cat(
            [self.joint_pos_multi_future, body_pos_relative_to_anchor_multi_frame],
            dim=1,
        )

    @property
    def command_multi_future_joint_body_diff_pos(self) -> torch.Tensor:
        """Return joint positions and robot-relative body position differences.

        Computes body positions in robot-anchor-relative frame: positions are
        translated so the XY comes from the robot root and Z from the reference
        anchor, then rotated by the heading difference. The result is the
        difference from the current robot body positions, expressed in the
        robot's local frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof + num_future_frames * num_bodies * 3)``.
        """
        N, F, B = (self.num_envs, self.num_future_frames, self.num_bodies)
        anchor_pos_w_repeat = self.anchor_pos_w_multi_future.view(N, F, 1, 3).expand(
            N, F, B, 3
        )
        anchor_quat_w_repeat = self.anchor_quat_w_multi_future.view(N, F, 1, 4).expand(
            N, F, B, 4
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, None, :].expand(
            N, F, B, 3
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, None, :].expand(
            N, F, B, 4
        )
        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
        body_pos_relative_w_multi_frame = delta_pos_w + quat_apply(
            delta_ori_w,
            self.body_pos_w_multi_future.view(N, F, B, 3) - anchor_pos_w_repeat,
        )
        body_pos_dif = body_pos_relative_w_multi_frame - self.robot_body_pos_w.view(
            N, 1, B, 3
        ).expand(N, F, B, 3)
        body_pos_relative_to_robot_anchor_multi_frame = quat_apply_yaw(
            quat_inv(robot_anchor_quat_w_repeat), body_pos_dif
        ).reshape(N, -1)
        return torch.cat(
            [
                self.joint_pos_multi_future,
                body_pos_relative_to_robot_anchor_multi_frame,
            ],
            dim=1,
        )

    @property
    def command_multi_future_lower_body(self) -> torch.Tensor:
        return torch.cat(
            [
                self.joint_pos_lower_body_multi_future,
                self.joint_vel_lower_body_multi_future,
            ],
            dim=1,
        )

    @property
    def command_multi_future_lower_body_joint_pos(self) -> torch.Tensor:
        return self.joint_pos_lower_body_multi_future

    @property
    def num_bodies_full(self) -> int:
        """Get the full number of bodies (all bodies, not just selected body_indexes)."""
        return self.motion_lib.num_bodies_full

    @property
    def egocentric_joint_positions_multi_future(self) -> torch.Tensor:
        """Get body positions (including pelvis) in egocentric frame for all future frames.

        Egocentric frame = projected root frame (heading/yaw only rotation, z=0 projected).
        For each future frame, positions are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 3] joint positions
                          in egocentric (projected root) frame
        """
        N, F, B = (self.num_envs, self.num_future_frames, self.num_bodies_full)
        body_pos_w = self.motion_lib.get_body_pos_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 3)
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)
        anchor_pos_w = body_pos_w[:, :, anchor_body_idx_full, :]
        anchor_pos_projected = anchor_pos_w.clone()
        anchor_pos_projected[..., 2] = 0
        anchor_pos_projected_expanded = anchor_pos_projected.view(N, F, 1, 3).expand(
            N, F, B, 3
        )
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]
        anchor_heading_quat = torch_transform.get_heading_q(
            anchor_quat_w.reshape(-1, 4)
        ).reshape(N, F, 4)
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(
            N, F, B, 4
        )
        body_pos_relative = body_pos_w - anchor_pos_projected_expanded
        body_pos_egocentric = quat_apply(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)),
            body_pos_relative.reshape(-1, 3),
        ).reshape(N, F, B, 3)
        return body_pos_egocentric

    @property
    def egocentric_joint_rotations_multi_future(self) -> torch.Tensor:
        """Get body rotations (including pelvis) in egocentric frame for all future frames.

        Returns 6D rotation representation (first two columns of rotation matrix).
        For each future frame, rotations are relative to that frame's projected root.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 6] joint rotations
                          in 6D representation (first two columns of rotation matrix)
        """
        N, F, B = (self.num_envs, self.num_future_frames, self.num_bodies_full)
        body_quat_w = self.motion_lib.get_body_quat_w_full(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, B, 4)
        anchor_body_idx_full = self.motion_lib.m_cfg.get("anchor_body_idx_full", 0)
        anchor_quat_w = body_quat_w[:, :, anchor_body_idx_full, :]
        anchor_heading_quat = torch_transform.get_heading_q(
            anchor_quat_w.reshape(-1, 4)
        ).reshape(N, F, 4)
        anchor_heading_quat_expanded = anchor_heading_quat.view(N, F, 1, 4).expand(
            N, F, B, 4
        )
        body_quat_egocentric = quat_mul(
            quat_inv(anchor_heading_quat_expanded.reshape(-1, 4)),
            body_quat_w.reshape(-1, 4),
        ).reshape(N, F, B, 4)
        mat = matrix_from_quat(body_quat_egocentric)
        body_rot_6d = rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, B, 6)
        return body_rot_6d

    @property
    def egocentric_joint_transforms_multi_future(self) -> torch.Tensor:
        """Combined egocentric joint transforms (positions + rotations) for all future frames.

        Returns:
            torch.Tensor: [num_envs, num_future_frames, num_bodies_full, 9]
                          (3 for position + 6 for 6D rotation)
        """
        positions = self.egocentric_joint_positions_multi_future
        rotations_ = self.egocentric_joint_rotations_multi_future
        return torch.cat([positions, rotations_], dim=-1)

    @property
    def root_transforms_relative_to_first_frame(self) -> torch.Tensor:
        """Get root transforms (position + rotation) relative to the first reference frame's
        projected root (z=0, heading only).

        Position: Delta from first frame's projected root position, rotated to first frame's heading frame.
        Rotation: Relative rotation from first frame's heading quaternion (6D representation).

        Returns:
            torch.Tensor: [num_envs, num_future_frames, 9]
                          (3 for relative position + 6 for 6D relative rotation)
        """
        N, F = (self.num_envs, self.num_future_frames)
        root_pos_w = self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 3)
        root_quat_w = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(N, F, 4)
        first_frame_pos = root_pos_w[:, 0:1, :]
        first_frame_pos_projected = first_frame_pos.clone()
        first_frame_pos_projected[..., 2] = 0
        first_frame_quat = root_quat_w[:, 0:1, :]
        first_frame_heading = torch_transform.get_heading_q(
            first_frame_quat.reshape(-1, 4)
        ).reshape(N, 1, 4)
        delta_pos_w = root_pos_w - first_frame_pos_projected.expand(N, F, 3)
        delta_pos_local = quat_apply(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)),
            delta_pos_w.reshape(-1, 3),
        ).reshape(N, F, 3)
        relative_quat = quat_mul(
            quat_inv(first_frame_heading.expand(N, F, 4).reshape(-1, 4)),
            root_quat_w.reshape(-1, 4),
        ).reshape(N, F, 4)
        mat = matrix_from_quat(relative_quat)
        relative_rot_6d = rotations.mat_to_rot6d_first_two_cols(mat).reshape(N, F, 6)
        return torch.cat([delta_pos_local, relative_rot_6d], dim=-1)

    @property
    def smpl_joints(self) -> torch.Tensor:
        """Return SMPL joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_smpl_joints, 3)``.
        """
        return self.motion_lib.get_smpl_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_joints_multi_future(self) -> torch.Tensor:
        """Return SMPL joint positions for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, num_smpl_joints, 3)``.
        """
        smpl_joints_mf = self.motion_lib.get_smpl_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_joints_mf.shape[1:]
        )

    @property
    def smpl_transl_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *smpl_transl_mf.shape[1:]
        )

    @property
    def smpl_transl_z_multi_future(self) -> torch.Tensor:
        smpl_transl_mf = self.motion_lib.get_smpl_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return smpl_transl_mf[..., 1:2].view(
            self.num_envs, self.smpl_num_future_frames, 1
        )

    @property
    def smpl_pose(self) -> torch.Tensor:
        """Return full SMPL pose (root + body joints) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 72)`` (24 joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def smpl_body_pose(self) -> torch.Tensor:
        """Return SMPL body pose (excluding root) in axis-angle for current frame.

        Returns:
            Tensor of shape ``(num_envs, 69)`` (23 body joints * 3 axis-angle).
        """
        return self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[..., 3:]

    @property
    def smpl_body_pose_6d(self) -> torch.Tensor:
        """Return SMPL body pose in 6D rotation representation for current frame.

        Returns:
            Tensor of shape ``(num_envs, 23 * 6)``.
        """
        smpl_body_pose = self.smpl_body_pose.reshape(-1, 23, 3)
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    def smpl_root_ytoz_up(self, root_quat_y_up: torch.Tensor) -> torch.Tensor:
        """Convert SMPL root quaternion from Y-up to Z-up convention.

        Args:
            root_quat_y_up: Root quaternions in Y-up frame, ``(N, 4)``.

        Returns:
            Root quaternions rotated to Z-up frame, ``(N, 4)``.
        """
        base_rot = torch_transform.angle_axis_to_quaternion(
            torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
        )
        root_quat_z_up = rotations.quat_mul(
            base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up, w_last=False
        )
        return root_quat_z_up

    @property
    def smpl_pose_noheading(self) -> torch.Tensor:
        """Return SMPL pose with root heading removed (preserving pitch/roll).

        Converts SMPL root from Y-up if needed, removes the base rotation offset,
        extracts and removes heading, then re-encodes as axis-angle.

        Returns:
            Tensor of shape ``(num_envs, 72)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(
            -1, 4
        )
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_heading_inv = rotations.calc_heading_quat_inv(root_quat, w_last=False)
        root_quat_noheading = rotations.quat_mul(
            root_heading_inv, root_quat, w_last=False
        )
        smpl_pose[..., :3] = torch_transform.quaternion_to_angle_axis(
            root_quat_noheading
        ).view(smpl_pose.shape[:-1] + (3,))
        return smpl_pose

    @property
    def smpl_root_quat_w(self) -> torch.Tensor:
        """Return SMPL root quaternion in Z-up world frame for current frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(
            -1, 4
        )
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def smpl_root_quat_w_multi_future(self) -> torch.Tensor:
        """Return SMPL root quaternions in Z-up world frame for all SMPL future frames.

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames, 4)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(
            -1, 4
        )
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False).view(
            self.num_envs, self.smpl_num_future_frames, 4
        )
        return root_quat

    @property
    def smpl_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """Return SMPL root orientation relative to robot orientation for all SMPL future frames.

        Computes ``quat_inv(robot_anchor) * smpl_root`` and returns 6D rotation
        matrix representation (first 2 columns).

        Returns:
            Tensor of shape ``(num_envs, smpl_num_future_frames * 6)``.
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(
            -1, 4
        )
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def smpl_root_quat_w_dif_refheading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by the first SMPL future frame's heading.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the heading
        of the first SMPL future frame instead of the robot's orientation.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        smpl_root_quat = torch_transform.angle_axis_to_quaternion(
            smpl_pose[..., :3]
        ).view(-1, 4)
        if self.motion_lib.smpl_y_up:
            smpl_root_quat = self.smpl_root_ytoz_up(smpl_root_quat)
        smpl_root_quat = rotations.remove_smpl_base_rot(smpl_root_quat, w_last=False)
        smpl_root_quat = smpl_root_quat.view(
            self.num_envs, self.smpl_num_future_frames, 4
        )
        ref_first_heading = torch_transform.get_heading_q(smpl_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            smpl_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_root_quat_w_dif_heading_multi_future(self) -> torch.Tensor:
        """SMPL root orientation canonicalized by robot heading (yaw) only.

        Same extraction as smpl_root_quat_w_dif_l_multi_future but uses the
        robot's heading (yaw) instead of full orientation for canonicalization.

        Returns:
            torch.Tensor: 6D rotation matrix representation,
                shape (num_envs, smpl_num_future_frames * 6)
        """
        smpl_pose = self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        root_quat = torch_transform.angle_axis_to_quaternion(smpl_pose[..., :3]).view(
            -1, 4
        )
        if self.motion_lib.smpl_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_smpl_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.smpl_num_future_frames, -1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def smpl_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).reshape(self.num_envs, self.smpl_num_future_frames, -1)

    @property
    def smpl_body_pose_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_smpl_pose(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )[..., 3:].reshape(self.num_envs, -1)

    @property
    def smpl_body_pose_multi_future_6d(self) -> torch.Tensor:
        smpl_body_pose = self.smpl_body_pose_multi_future.reshape(
            -1, self.smpl_num_future_frames, 23, 3
        )
        quat = torch_transform.angle_axis_to_quaternion(smpl_body_pose)
        mat = matrix_from_quat(quat)
        smpl_body_pose_6d = mat[..., :2].reshape(smpl_body_pose.shape[0], -1)
        return smpl_body_pose_6d

    @property
    def soma_joints(self) -> torch.Tensor:
        """SOMA joints in Z-up body-local frame. Stored Z-up in PKL (same as SMPL)."""
        return self.motion_lib.get_soma_joints(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def soma_joints_multi_future(self) -> torch.Tensor:
        """SOMA joints multi-future in Z-up body-local frame."""
        soma_joints_mf = self.motion_lib.get_soma_joints(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_joints_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_joints_mf.shape[1:]
        )

    @property
    def soma_root_quat_w(self) -> torch.Tensor:
        """SOMA root quaternion in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat

    @property
    def soma_root_quat_w_multi_future(self) -> torch.Tensor:
        """SOMA root quaternion multi-future in Z-up world frame (wxyz)."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        return root_quat.view(self.num_envs, self.smpl_num_future_frames, 4)

    @property
    def soma_root_quat_w_dif_l_multi_future(self) -> torch.Tensor:
        """SOMA root orientation relative to robot anchor, as 6D rotation matrix."""
        root_quat = self.motion_lib.get_soma_root_quat(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        if self.motion_lib.soma_y_up:
            root_quat = self.smpl_root_ytoz_up(root_quat)
        root_quat = rotations.remove_bvh_base_rot(root_quat, w_last=False)
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.smpl_num_future_frames, 1
                )
            ),
            root_quat.view(self.num_envs, self.smpl_num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def soma_transl_multi_future(self) -> torch.Tensor:
        """SOMA hips translation multi-future. Stored Y-up (same as SMPL transl)."""
        soma_transl_mf = self.motion_lib.get_soma_transl(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        )
        return soma_transl_mf.view(
            self.num_envs, self.smpl_num_future_frames, *soma_transl_mf.shape[1:]
        )

    @property
    def object_root_pos(self) -> torch.Tensor:
        """Return object root position in world frame with env origin and z-offset.

        Returns:
            Tensor of shape ``(num_envs, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos = object_root_pos.clone()
            object_root_pos[..., 2] += z_offset
        return object_root_pos + self._env.scene.env_origins[:, None, :]

    @property
    def object_root_quat(self) -> torch.Tensor:
        return self.motion_lib.get_object_root_quat(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    def _get_object_pos_with_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object position with randomization offset applied.

        This method also resamples the offset for the given env_ids.
        Called at environment reset to place object with random offset.

        Args:
            env_ids: Environment indices being reset

        Returns:
            Object positions with offset applied, shape [len(env_ids), 3]
        """
        obj_pos = self.object_root_pos[env_ids, 0].clone()
        if self.cfg.object_position_randomize:
            rand_cfg = self.cfg.object_position_randomization or {}
            x_range = rand_cfg.get("x", 0.0)
            y_range = rand_cfg.get("y", 0.0)
            z_range = rand_cfg.get("z", 0.0)
            self._object_position_offset[env_ids, 0] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * x_range
            self._object_position_offset[env_ids, 1] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * y_range
            self._object_position_offset[env_ids, 2] = (
                torch.rand(len(env_ids), device=self.device) * 2 - 1
            ) * z_range
            obj_pos = obj_pos + self._object_position_offset[env_ids]
        return obj_pos

    @property
    def object_root_pos_multi_future(self) -> torch.Tensor:
        """Return object root positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, num_objects, 3)``.
        """
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.future_motion_ids, self.future_time_steps
        )
        object_root_pos_view = object_root_pos.view(
            self.num_envs, self.num_future_frames, -1, 3
        )
        z_offset = getattr(self.cfg, "object_z_offset", 0.0)
        if z_offset != 0.0:
            object_root_pos_view = object_root_pos_view.clone()
            object_root_pos_view[..., 2] += z_offset
        return object_root_pos_view + self._env.scene.env_origins[:, None, None, :]

    @property
    def object_root_quat_multi_future(self) -> torch.Tensor:
        object_root_quat = self.motion_lib.get_object_root_quat(
            self.future_motion_ids, self.future_time_steps
        )
        return object_root_quat.view(self.num_envs, self.num_future_frames, -1, 4)

    def _get_contact_center_world(self, hand: str) -> torch.Tensor | None:
        """Get object contact center in world frame for the given hand.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs, 3) in world frame, or None if not available.
        """
        contact_center = self.motion_lib.get_object_contact_center(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )
        if contact_center is None:
            return None
        obj_pos = self.object_root_pos[:, 0, :]
        obj_quat = self.object_root_quat[:, 0, :]
        import motius.motion.geometry.sonic.rotations as rotations

        rotated_center = rotations.quat_rotate(obj_quat, contact_center, w_last=False)
        world_center = rotated_center + obj_pos
        return world_center

    @property
    def object_contact_center_left(self) -> torch.Tensor | None:
        """Get left hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("left_hand")

    @property
    def object_contact_center_right(self) -> torch.Tensor | None:
        """Get right hand object contact center in world frame. Shape: (num_envs, 3)."""
        return self._get_contact_center_world("right_hand")

    def get_in_contact(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get binary in_contact label for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with 1.0 if in contact, 0.0 otherwise,
            or None if not available.
        """
        return self.motion_lib.get_object_in_contact(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    def get_hand_action(self, hand: str = "right_hand") -> torch.Tensor | None:
        """Get discrete hand action (open/closed) for the given hand at current timestep.

        Args:
            hand: "left_hand" or "right_hand"

        Returns:
            Tensor of shape (num_envs,) with -1.0 = open, +1.0 = closed,
            or None if not available.
        """
        return self.motion_lib.get_hand_action(
            self.motion_ids, self.motion_start_time_steps + self.time_steps, hand=hand
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return reference joint positions for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_pos(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_pos_multi_future(self) -> torch.Tensor:
        """Return reference joint positions for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_pos(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def joint_pos_multi_future_for_smpl(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(
            self.smpl_future_motion_ids, self.smpl_future_time_steps
        ).view(self.num_envs, -1)

    @property
    def joint_pos_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_pos(
            self.future_motion_ids, self.future_time_steps
        )[..., self.lower_joint_isaaclab_indices].view(self.num_envs, -1)

    @property
    def joint_vel(self) -> torch.Tensor:
        """Return reference joint velocities for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_dof)``.
        """
        return self.motion_lib.get_dof_vel(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def joint_vel_multi_future(self) -> torch.Tensor:
        """Return reference joint velocities for all future frames, flattened.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_dof)``.
        """
        return self.motion_lib.get_dof_vel(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def root_pos_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def root_quat_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def root_z(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, 2:3].view(self.num_envs, -1)

    @property
    def root_z_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_root_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, 2:3].view(self.num_envs, -1)

    @property
    def joint_vel_lower_body_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_dof_vel(
            self.future_motion_ids, self.future_time_steps
        )[..., self.lower_joint_isaaclab_indices].view(self.num_envs, -1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return reference body positions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference body positions in world frame for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * num_bodies * 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        return body_pos_w_env.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_w(self) -> torch.Tensor:
        """Return position difference (reference - robot) in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w
        return body_pos_dif

    @property
    def body_pos_dif_w_multi_future(self) -> torch.Tensor:
        body_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 3)
        body_pos_w_env = body_pos_w + self._env.scene.env_origins[:, None, None, :]
        body_pos_dif = body_pos_w_env - self.robot_body_pos_w[:, None, :, :]
        return body_pos_dif.reshape(self.num_envs, -1)

    @property
    def body_pos_dif_l(self) -> torch.Tensor:
        """Return body position difference de-headed into robot-local frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 3)``.
        """
        body_pos_dif_w = self.body_pos_dif_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
            1, self.num_bodies, 1
        )
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def body_pos_dif_l_multi_future(self) -> torch.Tensor:
        body_pos_dif_w = self.body_pos_dif_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_dif_l = quat_apply_yaw(quat_inv(root_quat), body_pos_dif_w)
        return deheaded_dif_l

    @property
    def root_lin_vel_b_2d(self) -> torch.Tensor:
        """Return reference root linear velocity (XY only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 2)``.
        """
        root_lin_vel_w = self.motion_lib.get_root_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_lin_vel_l = quat_apply_yaw(quat_inv(root_quat), root_lin_vel_w)[:, :2]
        return root_lin_vel_l

    @property
    def root_ang_vel_b_1d(self) -> torch.Tensor:
        """Return reference root angular velocity (yaw only) in body frame.

        Returns:
            Tensor of shape ``(num_envs, 1)``.
        """
        root_ang_vel_w = self.motion_lib.get_root_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_quat = self.anchor_quat_w.view(self.num_envs, 1, 4)
        root_ang_vel_l = quat_apply_yaw(quat_inv(root_quat), root_ang_vel_w)[:, 2:3]
        return root_ang_vel_l

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return reference body quaternions in world frame for the current frame.

        Returns:
            Tensor of shape ``(num_envs, num_bodies, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_quat_dif_w(self) -> torch.Tensor:
        body_quat_w = self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        body_quat_dif = quat_mul(quat_inv(body_quat_w), self.robot_body_quat_w)
        return body_quat_dif

    @property
    def body_quat_dif_w_multi_future(self) -> torch.Tensor:
        ref_body_quat_w = self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, -1, 4)
        robot_body_quat_w = self.robot_body_quat_w.view(
            self.num_envs, 1, self.num_bodies, 4
        ).repeat(1, self.num_future_frames, 1, 1)
        body_quat_dif = quat_mul(quat_inv(ref_body_quat_w), robot_body_quat_w)
        return body_quat_dif

    @property
    def anchor_heading_quat(self) -> torch.Tensor:
        """Return robot anchor heading quaternion (yaw-only, pitch/roll removed).

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return torch_transform.get_heading_q(self.robot_anchor_quat_w)

    @property
    def root_rot_dif_l(self) -> torch.Tensor:
        """Return reference root orientation relative to robot orientation in 6D repr.

        Returns:
            Tensor of shape ``(num_envs, 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )
        root_rot_dif_w = quat_mul(quat_inv(self.robot_anchor_quat_w), ref_root_quat)
        mat = matrix_from_quat(root_rot_dif_w)
        deheaded_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return deheaded_rot_dif_l_mat

    @property
    def root_rot_dif_l_multi_future(self) -> torch.Tensor:
        """Return reference root orientation relative to robot for all future frames.

        Uses the full robot orientation for canonicalization (preserves heading diff).

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 6)``.
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        root_rot_dif = quat_mul(
            quat_inv(
                self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
                    1, self.num_future_frames, 1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_l_mat

    @property
    def root_rot_dif_heading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by robot heading (yaw) only.

        Unlike root_rot_dif_l_multi_future which uses the full robot orientation,
        this version only uses the heading (yaw) for canonicalization. This preserves
        the reference motion's pitch/roll relative to gravity while removing the
        heading difference.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        root_rot_dif = quat_mul(
            quat_inv(
                self.anchor_heading_quat.view(self.num_envs, 1, 4).expand(
                    -1, self.num_future_frames, -1
                )
            ),
            ref_root_quat.view(self.num_envs, self.num_future_frames, 4),
        )
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_heading_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_dif_heading_mat

    @property
    def root_rot_dif_refheading_multi_future(self) -> torch.Tensor:
        """Reference root orientation canonicalized by the first future frame's heading.

        Instead of using the robot's current heading for canonicalization, this uses
        the heading of the first (immediate) target frame from the reference motion.
        This makes the trajectory representation independent of the robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation (first 2 columns),
                shape (num_envs, num_future_frames * 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        root_rot_dif = quat_mul(
            quat_inv(
                ref_first_heading.view(self.num_envs, 1, 4).expand(
                    -1, self.num_future_frames, -1
                )
            ),
            ref_root_quat,
        )
        mat = matrix_from_quat(root_rot_dif)
        return mat[..., :2].reshape(mat.shape[0], -1)

    @property
    def heading_diff_robot_ref(self) -> torch.Tensor:
        """Relative heading rotation from robot heading to reference first frame heading.

        Computes quat_mul(quat_inv(robot_heading), ref_first_heading), expressing
        the reference motion's heading direction in the robot's heading frame.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, self.num_future_frames, 4)
        ref_first_heading = torch_transform.get_heading_q(ref_root_quat[:, 0, :])
        heading_diff = quat_mul(quat_inv(self.anchor_heading_quat), ref_first_heading)
        mat = matrix_from_quat(heading_diff)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def raw_root_quat_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        return ref_root_quat.reshape(self.num_envs, self.num_future_frames, 4)

    @property
    def root_rot_w_multi_future(self) -> torch.Tensor:
        ref_root_quat = self.motion_lib.get_root_quat_w(
            self.future_motion_ids, self.future_time_steps
        )
        mat = matrix_from_quat(ref_root_quat)
        root_rot_w_mat = mat[..., :2].reshape(mat.shape[0], -1)
        return root_rot_w_mat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_lin_vel_l(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
            1, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_lin_vel_l_multi_future(self) -> torch.Tensor:
        body_lin_vel_w = self.body_lin_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_lin_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def body_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.future_motion_ids, self.future_time_steps
        ).view(self.num_envs, -1)

    @property
    def body_ang_vel_l(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 4).repeat(
            1, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def body_ang_vel_l_multi_future(self) -> torch.Tensor:
        body_ang_vel_w = self.body_ang_vel_w_multi_future
        root_quat = self.robot_anchor_quat_w.view(self.num_envs, 1, 1, 4).repeat(
            1, self.num_future_frames, self.num_bodies, 1
        )
        deheaded_vel_l = quat_apply_yaw(quat_inv(root_quat), body_ang_vel_w)
        return deheaded_vel_l

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Return reference anchor body position in world frame.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        return (
            self.motion_lib.get_body_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )[:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 3)``.
        """
        anchor_pos_w = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].view(
            self.num_envs, self.num_future_frames, -1
        )
        anchor_pos_w_env = anchor_pos_w + self._env.scene.env_origins[:, None, :]
        return anchor_pos_w_env.reshape(self.num_envs, -1)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Return reference anchor body quaternion in world frame.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_multi_future(self) -> torch.Tensor:
        """Return reference anchor quaternions for all future frames.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames * 4)``.
        """
        return self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].reshape(self.num_envs, -1)

    @property
    def anchor_ori_refheading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by its own heading.

        Uses get_heading_q(anchor_quat_w) as the canonical frame, removing the
        heading component and preserving pitch/roll relative to gravity.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        ref_heading = torch_transform.get_heading_q(self.anchor_quat_w)
        ori = quat_mul(quat_inv(ref_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_ori_heading(self) -> torch.Tensor:
        """Current anchor orientation canonicalized by robot heading (yaw).

        Uses get_heading_q(robot_anchor_quat_w) as the canonical frame, preserving
        the reference motion's pitch/roll relative to gravity while removing the
        robot's heading.

        Returns:
            torch.Tensor: 6D rotation matrix representation, shape (num_envs, 6)
        """
        robot_heading = self.anchor_heading_quat
        ori = quat_mul(quat_inv(robot_heading), self.anchor_quat_w)
        mat = matrix_from_quat(ori)
        return mat[..., :2].reshape(self.num_envs, -1)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_lin_vel_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].view(self.num_envs, -1)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_ang_vel_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.motion_anchor_body_index].view(self.num_envs, -1)

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]

    @property
    def reward_point_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]

    @property
    def vr_3point_body_quat_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.vr_3point_body_indices_motion].view(
            self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1
        )

    @property
    def head_orn_w_multi_future(self) -> torch.Tensor:
        return self.motion_lib.get_body_quat_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.vr_3point_body_indices_motion[2]].view(
            self.num_envs, self.num_future_frames, -1
        )

    @property
    def reward_point_body_pos_w(self) -> torch.Tensor:
        reward_point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.reward_point_body_indices_motion]
        return (
            reward_point_original
            + quat_apply(self.reward_point_body_quat_w, self.reward_point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        """Return reference VR 3-point body positions (with offsets) in world frame.

        The 3 points are typically left wrist, right wrist, and head. Offsets
        allow tracking a point displaced from the body origin (e.g., palm center).

        Returns:
            Tensor of shape ``(num_envs, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )[:, self.vr_3point_body_indices_motion]
        return (
            vr_3point_original
            + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets)
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def vr_3point_body_pos_w_multi_future(self) -> torch.Tensor:
        """Return reference VR 3-point positions for all future frames in world frame.

        Returns:
            Tensor of shape ``(num_envs, num_future_frames, 3, 3)``.
        """
        vr_3point_original = self.motion_lib.get_body_pos_w(
            self.future_motion_ids, self.future_time_steps
        )[:, self.vr_3point_body_indices_motion].view(
            self.num_envs, self.num_future_frames, len(self.cfg.vr_3point_body), -1
        )
        vr_3point_offset_extend = self.vr_3point_body_offsets.unsqueeze(1).repeat(
            1, self.num_future_frames, 1, 1
        )
        vr_3point_pos_w = (
            vr_3point_original
            + quat_apply(
                self.vr_3point_body_quat_w_multi_future, vr_3point_offset_extend
            )
            + self._env.scene.env_origins[:, None, None, :]
        )
        return vr_3point_pos_w

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Return the robot's current anchor body position in world frame.

        In offline mode, falls back to the reference motion root position.

        Returns:
            Tensor of shape ``(num_envs, 3)``.
        """
        if getattr(self, "_offline", False):
            return self.motion_lib.get_root_pos_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Return the robot's current anchor body quaternion in world frame.

        When ``use_ref_motion_root_quat_w_as_anchor`` is True or in offline mode,
        returns the reference motion root orientation (optionally with added noise)
        instead of the simulated robot state.

        Returns:
            Tensor of shape ``(num_envs, 4)``.
        """
        if self.use_ref_motion_root_quat_w_as_anchor or getattr(
            self, "_offline", False
        ):
            ref_root_quat = self.motion_lib.get_root_quat_w(
                self.motion_ids, self.motion_start_time_steps + self.time_steps
            )
            if self.ref_motion_root_rotation_noise is not None:
                ref_root_quat = quat_mul(
                    ref_root_quat, self.ref_motion_root_rotation_noise
                )
            return ref_root_quat
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_lin_vel_w is not available in offline mode"
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        assert not getattr(
            self, "_offline", False
        ), "robot_anchor_ang_vel_w is not available in offline mode"
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_vr_3point_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.vr_3point_body_indices]

    @property
    def robot_reward_point_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[
            :, self.reward_point_body_indices
        ] + quat_apply(
            self.robot.data.body_quat_w[:, self.reward_point_body_indices],
            self.reward_point_body_offsets,
        )

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.vr_3point_body_indices] + quat_apply(
            self.robot.data.body_quat_w[:, self.vr_3point_body_indices],
            self.vr_3point_body_offsets,
        )

    @property
    def feet_l(self) -> torch.Tensor:
        return self.motion_lib.get_feet_l(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def feet_r(self) -> torch.Tensor:
        return self.motion_lib.get_feet_r(
            self.motion_ids, self.motion_start_time_steps + self.time_steps
        )

    @property
    def episode_encoder_index(self) -> torch.Tensor:
        return self.encoder_index

    def _update_metrics(self):
        """Compute tracking error metrics between reference motion and robot state.

        Populates ``self.metrics`` with per-env errors for anchor position/rotation,
        body position/rotation, and joint position/velocity. When there is a DOF
        mismatch (extra finger joints), only body joints are compared.
        """
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        if self.has_dof_mismatch:
            robot_body_joint_pos = self.robot_joint_pos[:, self.body_joint_indices]
            robot_body_joint_vel = self.robot_joint_vel[:, self.body_joint_indices]
            self.metrics["error_joint_pos"] = torch.abs(
                self.joint_pos - robot_body_joint_pos
            ).mean(dim=-1)
            self.metrics["error_joint_vel"] = torch.abs(
                self.joint_vel - robot_body_joint_vel
            ).mean(dim=-1)
        else:
            self.metrics["error_joint_pos"] = torch.abs(
                self.joint_pos - self.robot_joint_pos
            ).mean(dim=-1)
            self.metrics["error_joint_vel"] = torch.abs(
                self.joint_vel - self.robot_joint_vel
            ).mean(dim=-1)

    def resample_all_commands(self):
        """Resample motion clips and reset state for all environments at once."""
        self._resample_command(torch.arange(self.num_envs))

    def _load_contact_data(self):
        """Load contact data from file or directory and validate frame counts against motion library.

        Supports both:
        - Single file: loads one pkl file
        - Directory: loads all pkl files and merges them

        Sets:
            self._contact_data: Raw contact data dict
            self._first_contact_frame: Dict mapping motion_name -> first contact frame index
        """
        import joblib

        self._first_contact_frame = None
        self._contact_data = None
        self._first_contact_lookup = None
        self._per_env_first_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._motion_contact_flags = None
        if self.cfg.contact_file is None or not os.path.exists(self.cfg.contact_file):
            self._derive_first_contact_from_in_contact_labels()
            return
        if os.path.isfile(self.cfg.contact_file):
            contact_data = joblib.load(self.cfg.contact_file)
        elif os.path.isdir(self.cfg.contact_file):
            contact_data = {}
            pkl_files = glob.glob(os.path.join(self.cfg.contact_file, "*.pkl"))
            for pkl_file in sorted(pkl_files):
                expected_key = os.path.splitext(os.path.basename(pkl_file))[0]
                try:
                    data = joblib.load(pkl_file)
                    if expected_key in data:
                        contact_data[expected_key] = data[expected_key]
                except Exception as e:
                    print(f"  Warning: Failed to load {pkl_file}: {e}")
            print(
                f"[TrackingCommand] Loaded {len(contact_data)} contact sequences from {self.cfg.contact_file}"
            )
        else:
            print(
                f"[TrackingCommand] Warning: contact_file path invalid: {self.cfg.contact_file}"
            )
            return
        motion_keys_set = set(self.motion_lib.curr_motion_keys)
        filtered_contact_data = {
            k: v for k, v in contact_data.items() if k in motion_keys_set
        }
        if len(filtered_contact_data) < len(contact_data):
            print(
                f"[TrackingCommand] Filtered contact data: {len(filtered_contact_data)}/{len(contact_data)} motions match loaded motion keys"
            )
        self._contact_data = filtered_contact_data
        first_contact_frames = {}
        for motion_name, motion_data in filtered_contact_data.items():
            object_contact = motion_data.get("object", None)
            if object_contact is not None:
                contact_per_frame = (object_contact != 0).sum(axis=1)
                contact_frames = np.where(contact_per_frame > 0)[0]
                if len(contact_frames) > 0:
                    first_contact_frames[motion_name] = int(contact_frames[0])
                else:
                    first_contact_frames[motion_name] = object_contact.shape[0]
            else:
                first_contact_frames[motion_name] = 0
        self._first_contact_frame = first_contact_frames
        print(f"[TrackingCommand] Loaded contact data from {self.cfg.contact_file}")
        for motion_name, first_frame in first_contact_frames.items():
            print(f"  {motion_name}: first contact at frame {first_frame}")
        self._validate_and_align_contact_frame_counts(filtered_contact_data)
        self._build_first_contact_lookup()
        self._build_motion_contact_flags()

    def _derive_first_contact_from_in_contact_labels(self):
        """Derive first contact frames from motion lib's in_contact labels.

        When no contact_file is provided, but the motion lib has per-frame
        in_contact labels (from contact_points_left_hand/right_hand in the
        object motion data), derive the first contact frame for each motion.
        """
        hand = getattr(self.cfg, "sample_before_contact_hand", "right_hand")
        side = "left" if hand == "left_hand" else "right"
        attr = f"_motion_object_in_contact_{side}"
        if not hasattr(self.motion_lib, attr):
            return
        in_contact_tensor = getattr(self.motion_lib, attr)
        length_starts = self.motion_lib.length_starts
        num_motions = len(self.motion_lib.curr_motion_keys)
        first_contact_frames = {}
        for motion_idx in range(num_motions):
            motion_key = self.motion_lib.curr_motion_keys[motion_idx]
            start = length_starts[motion_idx].item()
            num_frames = self.motion_lib._motion_num_frames[motion_idx].item()
            end = start + num_frames
            motion_in_contact = in_contact_tensor[start:end]
            contact_indices = torch.nonzero(motion_in_contact > 0.5, as_tuple=False)
            if len(contact_indices) > 0:
                first_contact_frames[motion_key] = int(contact_indices[0].item())
            else:
                first_contact_frames[motion_key] = num_frames
        self._first_contact_frame = first_contact_frames
        print(
            f"[TrackingCommand] Derived first contact from in_contact labels ({hand}):"
        )
        for motion_name, first_frame in first_contact_frames.items():
            print(f"  {motion_name}: first contact at frame {first_frame}")
        self._build_first_contact_lookup()

    def _build_first_contact_lookup(self):
        """Build a tensor for efficient per-env first contact frame lookup."""
        if self._first_contact_frame is None:
            self._first_contact_lookup = None
            return
        num_motions = len(self.motion_lib.curr_motion_keys)
        self._first_contact_lookup = torch.zeros(
            num_motions, device=self.device, dtype=torch.long
        )
        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key in self._first_contact_frame:
                self._first_contact_lookup[motion_idx] = self._first_contact_frame[
                    motion_key
                ]
            else:
                self._first_contact_lookup[motion_idx] = 0

    def _build_motion_contact_flags(self):
        """Preprocess contact flags for each motion: motion_id -> (num_frames,) bool tensor."""
        if self._contact_data is None or len(self._contact_data) == 0:
            self._motion_contact_flags = None
            return
        self._motion_contact_flags = {}
        for motion_idx, motion_key in enumerate(self.motion_lib.curr_motion_keys):
            if motion_key not in self._contact_data:
                continue
            motion_contact_data = self._contact_data[motion_key]
            object_contact = motion_contact_data.get("object", None)
            if object_contact is not None:
                if not isinstance(object_contact, torch.Tensor):
                    object_contact_tensor = torch.from_numpy(object_contact).to(
                        self.device
                    )
                else:
                    object_contact_tensor = object_contact.to(self.device)
                contact_per_frame = (object_contact_tensor != 0).sum(dim=1)
                contact_flags = contact_per_frame > 0
                self._motion_contact_flags[motion_idx] = contact_flags

    def _update_per_env_first_contact(self, env_ids):
        """Update _per_env_first_contact for given env_ids based on their assigned motion."""
        if self._first_contact_lookup is None:
            return
        motion_ids = self.motion_ids[env_ids]
        self._per_env_first_contact[env_ids] = self._first_contact_lookup[motion_ids]

    def _validate_and_align_contact_frame_counts(self, contact_data: dict):
        """Validate contact data frame counts match the loaded motion frame counts.

        Allows a tolerance of ±3 frames due to slight duration differences between
        GRAB (120Hz) and robot motion (30Hz) sources.

        Args:
            contact_data: Dict mapping motion_name -> contact arrays

        Raises:
            AssertionError: If frame count difference exceeds 3 frames
        """
        FRAME_TOLERANCE = 3
        for motion_name, motion_contact in contact_data.items():
            contact_frames = None
            if "object" in motion_contact and motion_contact["object"] is not None:
                contact_frames = motion_contact["object"].shape[0]
            elif "body" in motion_contact and motion_contact["body"] is not None:
                contact_frames = motion_contact["body"].shape[0]
            if contact_frames is None:
                continue
            motion_idx = None
            for idx, key in enumerate(self.motion_lib.curr_motion_keys):
                if key == motion_name:
                    motion_idx = idx
                    break
            if motion_idx is None:
                print(
                    f"[TrackingCommand] Warning: Contact motion '{motion_name}' not found in loaded motions"
                )
                continue
            motion_frames = int(self.motion_lib._motion_num_frames[motion_idx].item())
            frame_diff = abs(contact_frames - motion_frames)
            if frame_diff == 0:
                print(f"  {motion_name}: {contact_frames} frames (exact match)")
            elif frame_diff <= FRAME_TOLERANCE:
                print(
                    f"  {motion_name}: {contact_frames} frames (motion lib: {motion_frames}, diff: {contact_frames - motion_frames})"
                )
            else:
                raise AssertionError(
                    f"[TrackingCommand] Frame count mismatch too large for motion '{motion_name}':\n  Contact data: {contact_frames} frames\n  Motion lib: {motion_frames} frames\n  Difference: {frame_diff} frames (max allowed: ±{FRAME_TOLERANCE})"
                )

    def _load_table_meta(self, motion_key: str):
        """Load and cache table meta info (table_pos, table_quat, table_size) for a motion.
        Derives meta path from motion_file path. Falls back to motion file if meta not found.

        For GeniHOI data:
            - table_pos: [x, y, z] center position of the table
            - table_quat: [w, x, y, z] quaternion (identity for cuboid tables)
            - table_size: [width, depth, thickness] dimensions of the table

        For GRAB data:
            - table_pos, table_quat: from meta file
            - table_size: not provided (uses USD with scene_scale)
        """
        import joblib

        try:
            motion_file = (
                self.cfg.motion_lib_cfg.get("motion_file", "")
                if self.cfg.motion_lib_cfg
                else ""
            )
            if motion_file:
                if os.path.isdir(motion_file):
                    meta_dir = motion_file.replace("/robot", "/meta")
                elif "/robot" in motion_file:
                    meta_dir = os.path.dirname(motion_file).replace("/robot", "/meta")
                else:
                    meta_dir = "data/motion_lib_grab/meta"
                meta_file = os.path.join(meta_dir, f"{motion_key}.pkl")
            else:
                meta_file = f"data/motion_lib_grab/meta/{motion_key}.pkl"
            if os.path.exists(meta_file):
                meta = joblib.load(meta_file)
                self._table_meta_cache[motion_key] = {
                    "table_pos": torch.tensor(
                        meta.get("table_pos", [0.0, 0.0, 0.8]), device=self.device
                    ).float(),
                    "table_quat": torch.tensor(
                        meta.get("table_quat", [1.0, 0.0, 0.0, 0.0]), device=self.device
                    ).float(),
                    "table_size": torch.tensor(
                        meta.get("table_size", [1.0, 0.6, 0.04]), device=self.device
                    ).float(),
                }
                return
            if motion_file and os.path.isfile(motion_file):
                motion_file_data = joblib.load(motion_file)
                for mk, motion_data in motion_file_data.items():
                    if mk in self._table_meta_cache:
                        continue
                    if (
                        isinstance(motion_data, dict)
                        and "table_pos" in motion_data
                        and ("table_quat" in motion_data)
                    ):
                        self._table_meta_cache[mk] = {
                            "table_pos": torch.tensor(
                                motion_data["table_pos"], device=self.device
                            ).float(),
                            "table_quat": torch.tensor(
                                motion_data["table_quat"], device=self.device
                            ).float(),
                        }
                    else:
                        self._table_meta_cache[mk] = None
                if motion_key in self._table_meta_cache:
                    return
        except Exception:
            pass
        self._table_meta_cache[motion_key] = None

    def _sample_before_contact(
        self, env_ids: Sequence[int], sampled_times: torch.Tensor
    ) -> torch.Tensor:
        """Sample timestamps before the first contact frame for contact-based initialization.

        Args:
            env_ids: Environment indices to resample
            sampled_times: Originally sampled time steps

        Returns:
            Modified sampled_times with timestamps clamped to before first contact
        """
        if self._first_contact_frame is None or len(self._first_contact_frame) == 0:
            return sampled_times
        curr_motion_keys = getattr(self.motion_lib, "curr_motion_keys", None)
        for i, env_idx in enumerate(env_ids):
            motion_id = self.motion_ids[env_idx].item()
            if curr_motion_keys is not None and motion_id < len(curr_motion_keys):
                motion_key = curr_motion_keys[motion_id]
            else:
                motion_key = list(self._first_contact_frame.keys())[0]
            first_contact = None
            if motion_key in self._first_contact_frame:
                first_contact = self._first_contact_frame[motion_key]
            else:
                for contact_key in self._first_contact_frame.keys():
                    if contact_key in motion_key or motion_key in contact_key:
                        first_contact = self._first_contact_frame[contact_key]
                        break
                if first_contact is None:
                    first_contact = list(self._first_contact_frame.values())[0]
            margin = getattr(self.cfg, "sample_before_contact_margin", 10)
            if first_contact > margin:
                sampled_times[i] = torch.randint(
                    0,
                    first_contact - margin,
                    (1,),
                    device=self.device,
                    dtype=sampled_times.dtype,
                )
            else:
                sampled_times[i] = 0
        return sampled_times

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion clips, reset robot state, and position objects for given envs.

        This is the main episode-reset handler. It performs the following in order:

        1. Sample new motion IDs and start times (respecting evaluation mode,
           paired motions, multi-object mode, and adaptive sampling).
        2. Resample encoder mode (G1/SMPL/teleop) per env.
        3. Apply pose and velocity randomization (skipped during evaluation).
        4. Handle DOF mismatch by mapping motion lib joints to robot joints.
        5. Write joint state and root state to the simulator.
        6. Position scene objects (single or multi-object mode) and tables.
        7. Cache body-relative poses for reward computation.

        Args:
            env_ids: Environment indices being reset.
        """
        self.time_steps[env_ids] = 0
        if self.variable_frames_enabled and len(env_ids) > 0:
            idx = torch.randint(
                0, len(self._frame_choices), (len(env_ids),), device=self.device
            )
            self.per_env_num_frames[env_ids] = self._frame_choices[idx]
        if len(env_ids) > 0:
            if self.is_evaluating:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions
                )[env_ids]
                self.motion_start_time_steps[env_ids] = 0
            elif self.cfg.use_paired_motions:
                self.motion_ids[env_ids] = (
                    torch.arange(self.num_envs).to(self.device)
                    % self.motion_lib._num_motions
                )[env_ids]
            elif self._multi_object_mode:
                new_motion_id = self.motion_lib.sample_motions(1)[0]
                self.motion_ids[env_ids] = new_motion_id
                self.motion_start_time_steps[env_ids] = (
                    self.motion_lib.sample_time_steps(
                        self.motion_ids[env_ids], truncate_time=None
                    )
                )
            else:
                if self.use_adaptive_sampling:
                    sampled_ids, sampled_times = (
                        self.motion_lib.sample_motion_ids_and_time_steps(len(env_ids))
                    )
                    self.motion_ids[env_ids] = sampled_ids.to(self.motion_ids.dtype)
                    sampled_times = sampled_times.to(self.motion_start_time_steps.dtype)
                else:
                    self.motion_ids[env_ids] = self.motion_lib.sample_motions(
                        len(env_ids)
                    )
                    sampled_times = self.motion_lib.sample_time_steps(
                        self.motion_ids[env_ids], truncate_time=None
                    )
                if self.cfg.sample_from_n_initial_frames is not None:
                    n_frames = self.cfg.sample_from_n_initial_frames
                    sampled_times = torch.randint(
                        0,
                        n_frames,
                        (len(env_ids),),
                        dtype=sampled_times.dtype,
                        device=self.device,
                    )
                elif self.cfg.start_from_first_frame:
                    sampled_times.zero_()
                if (
                    self.cfg.sample_before_contact
                    and self._first_contact_frame is not None
                ):
                    sampled_times = self._sample_before_contact(env_ids, sampled_times)
                self.motion_start_time_steps[env_ids] = sampled_times
            if self.encoder_sample_probs is not None:
                has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]]
                if self.soma_encoder_index is not None and hasattr(
                    self.motion_lib, "motion_has_soma"
                ):
                    has_soma = self.motion_lib.motion_has_soma[self.motion_ids[env_ids]]
                    sampling_cases = [
                        (env_ids[has_smpl & has_soma], self.encoder_sample_probs),
                        (
                            env_ids[has_smpl & ~has_soma],
                            self.encoder_sample_probs_no_soma,
                        ),
                        (
                            env_ids[~has_smpl & has_soma],
                            self.encoder_sample_probs_no_smpl,
                        ),
                        (
                            env_ids[~has_smpl & ~has_soma],
                            self.encoder_sample_probs_no_smpl_no_soma,
                        ),
                    ]
                else:
                    sampling_cases = [
                        (env_ids[has_smpl], self.encoder_sample_probs),
                        (env_ids[~has_smpl], self.encoder_sample_probs_no_smpl),
                    ]
                for subset_ids, probs in sampling_cases:
                    if len(subset_ids) > 0:
                        encoder_index = torch.multinomial(
                            probs, len(subset_ids), replacement=True
                        ).to(self.device)
                        self.encoder_index[subset_ids] = 0
                        self.encoder_index[subset_ids, encoder_index] = 1
                if self.smpl_encoder_index is not None and (
                    not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_smpl = self.encoder_index[env_ids, self.smpl_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_smpl
                    )
                    if (
                        self.teleop_encoder_index is not None
                        and self.teleop_sample_prob_when_smpl > 0.0
                    ):
                        smpl_env_ids = env_ids[use_smpl.bool()]
                        if len(smpl_env_ids) > 0:
                            sample_teleop = (
                                torch.rand(len(smpl_env_ids), device=self.device)
                                < self.teleop_sample_prob_when_smpl
                            )
                            self.encoder_index[
                                smpl_env_ids, self.teleop_encoder_index
                            ] = (
                                self.encoder_index[
                                    smpl_env_ids, self.teleop_encoder_index
                                ]
                                | sample_teleop.long()
                            )
                if self.soma_encoder_index is not None and (
                    not self.optimize_encoders_ratio_for_CHIP
                ):
                    use_soma = self.encoder_index[env_ids, self.soma_encoder_index]
                    self.encoder_index[env_ids, self.g1_encoder_index] = (
                        self.encoder_index[env_ids, self.g1_encoder_index] | use_soma
                    )
            self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                self.motion_ids[env_ids]
            )
            if self.num_future_frames > 1:
                self.future_motion_ids = self.motion_ids.repeat_interleave(
                    self.num_future_frames
                )
                self.smpl_future_motion_ids = self.motion_ids.repeat_interleave(
                    self.smpl_num_future_frames
                )
                self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(
                    self.motion_ids[env_ids]
                )
            if self._first_contact_frame is not None:
                self._update_per_env_first_contact(env_ids)
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        self.running_ref_root_height[env_ids] = self.anchor_pos_w[env_ids, 2]
        if not self.is_evaluating:
            range_list = [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_pos[env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(
                rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
            )
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]
        motion_lib_joint_pos = self.joint_pos.clone()
        motion_lib_joint_vel = self.joint_vel.clone()
        if self.has_dof_mismatch:
            joint_pos = torch.zeros(
                self.num_envs,
                self.robot_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
            joint_vel = torch.zeros(
                self.num_envs,
                self.robot_num_dof,
                dtype=torch.float32,
                device=self.device,
            )
            joint_pos[:, self.body_joint_indices] = motion_lib_joint_pos
            joint_vel[:, self.body_joint_indices] = motion_lib_joint_vel
            joint_pos[:, self.extra_joint_indices] = self.extra_default_positions
            joint_vel[:, self.extra_joint_indices] = self.extra_default_velocities
        else:
            joint_pos = motion_lib_joint_pos
            joint_vel = motion_lib_joint_vel
        if not self.is_evaluating:
            joint_pos += sample_uniform(
                *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
            )
            joint_vel += sample_uniform(
                *self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device
            )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )
        if self._multi_object_mode and len(self._object_names) > 0:
            active_motion_id = (
                self.motion_ids[env_ids[0]].item() if len(env_ids) > 0 else 0
            )
            active_motion_key = self.motion_lib.curr_motion_keys[active_motion_id]
            active_obj_safe_name = active_motion_key.replace("-", "_")
            self._active_object_name = active_motion_key
            active_key = f"object_{active_obj_safe_name}"
            if active_key in self._env.scene.rigid_objects:
                obj = self._env.scene[active_key]
                obj_pos = self._get_object_pos_with_offset(env_ids)
                zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                obj.write_root_state_to_sim(
                    torch.cat(
                        [obj_pos, self.object_root_quat[env_ids, 0], zero_vel], dim=-1
                    ),
                    env_ids=env_ids,
                )
            else:
                print(
                    f"[Warning] Active object '{active_key}' not found in scene. Available: {list(self._env.scene.rigid_objects.keys())[:5]}..."
                )
            inactive_base = INACTIVE_OBJECT_BASE_OFFSET.to(self.device)
            inactive_quat = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self.device
            ).expand(len(env_ids), -1)
            inactive_idx = 0
            for safe_name in self._object_names:
                obj_key = f"object_{safe_name}"
                if obj_key != active_key and obj_key in self._env.scene.rigid_objects:
                    z_offset = torch.tensor(
                        [0.0, 0.0, -inactive_idx * INACTIVE_OBJECT_Z_SPACING],
                        device=self.device,
                    )
                    inactive_pos = (
                        self._env.scene.env_origins[env_ids] + inactive_base + z_offset
                    )
                    inactive_zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
                    inactive_state = torch.cat(
                        [inactive_pos, inactive_quat, inactive_zero_vel], dim=-1
                    )
                    self._env.scene[obj_key].write_root_state_to_sim(
                        inactive_state, env_ids=env_ids
                    )
                    inactive_idx += 1
        elif "object" in self._env.scene.rigid_objects:
            obj = self._env.scene["object"]
            obj_pos = self._get_object_pos_with_offset(env_ids)
            zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
            obj.write_root_state_to_sim(
                torch.cat(
                    [obj_pos, self.object_root_quat[env_ids, 0], zero_vel], dim=-1
                ),
                env_ids=env_ids,
            )
        if "table" in self._env.scene.rigid_objects:
            table = self._env.scene["table"]
            table_pos_list = []
            table_quat_list = []
            motion_ids_cpu = self.motion_ids.cpu()
            for i, env_idx in enumerate(env_ids):
                motion_idx = motion_ids_cpu[env_idx].item()
                motion_key = self.motion_lib.curr_motion_keys[motion_idx]
                if motion_key not in self._table_meta_cache:
                    self._load_table_meta(motion_key)
                cached = self._table_meta_cache.get(motion_key)
                if cached is not None:
                    pos = cached["table_pos"].clone()
                    quat = cached["table_quat"].clone()
                    pos = pos + self._env.scene.env_origins[env_idx]
                    table_pos_list.append(pos)
                    table_quat_list.append(quat)
                else:
                    pos = self.object_root_pos[env_idx, 0].clone()
                    pos[2] = 0.76
                    pos[1] -= 0.15
                    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._env.device)
                    table_pos_list.append(pos)
                    table_quat_list.append(quat)
            if len(table_pos_list) > 0:
                table_pos = torch.stack(table_pos_list, dim=0)
                table_quat = torch.stack(table_quat_list, dim=0)
                if self.cfg.table_offset is not None:
                    table_offset = torch.tensor(
                        self.cfg.table_offset,
                        device=self._env.device,
                        dtype=table_pos.dtype,
                    )
                    table_pos = table_pos + table_offset
                table_root_pose = torch.cat([table_pos, table_quat], dim=-1)
                table.write_root_pose_to_sim(table_root_pose, env_ids=env_ids)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = torch_transform.get_heading_q(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

    def _update_command(self):
        """Advance the motion time cursor by one step and handle episode wrap-around.

        Called once per simulation step. Updates adaptive sampling statistics,
        increments the time cursor, resamples any environments that have reached
        the end of their motion clip, updates the running reference root height
        EMA, and optionally performs height-map raycasting for object-aware
        observations.
        """
        if self.use_adaptive_sampling:
            with common.Timer("update_adaptive_sampling"):
                cur_time_steps = self.motion_start_time_steps + self.time_steps
                self.motion_lib.update_adaptive_sampling(
                    self._env.reset_terminated, self.motion_ids, cur_time_steps
                )
        self.time_steps += 1
        env_ids = torch.where(
            self.time_steps + self.motion_start_time_steps
            >= self.motion_lib.get_time_step_total(self.motion_ids)
        )[0]
        self._resample_command(env_ids)
        ema_alpha = 0.1
        self.running_ref_root_height = (
            ema_alpha * self.anchor_pos_w[:, 2]
            + (1 - ema_alpha) * self.running_ref_root_height
        )
        if self.cfg.use_height_map:
            root_pos_w = self.robot.data.root_pos_w
            root_quat_w = self.robot.data.root_quat_w
            ray_starts_w = root_pos_w.unsqueeze(1).expand(-1, self.num_rays, -1)
            root_quat_expanded = (
                torch_transform.get_heading_q(root_quat_w)
                .unsqueeze(1)
                .expand(-1, self.num_rays, -1)
            )
            ray_dirs_w = quat_apply(root_quat_expanded, self.ray_dirs_local)
            scan_dot_pos_w, _ = self.height_map.raycast_fused(
                self.object_root_pos,
                self.object_root_quat,
                ray_starts_w,
                ray_dirs_w,
                n_mesh_per_cam=self.num_mesh_per_cam,
                mesh_ids_flattened=self.mesh_ids,
                cam_ids_flattened=self.cam_ids,
                min_dist=0.0,
                max_dist=self.cfg.height_map_max_dist,
            )
            denom = ray_starts_w[:, :, 2] - scan_dot_pos_w[:, :, 2]
            denom = torch.clamp_min(denom, 1e-08)
            scale = torch.clamp(ray_starts_w[:, :, 2] / denom, max=1.0)
            self.scan_dot_pos_w[:] = (
                ray_starts_w + (scan_dot_pos_w - ray_starts_w) * scale.unsqueeze(-1)
            ).view(self.num_envs, self.num_rays_x, self.num_rays_y, 3)

    @property
    def future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for all future reference frames.

        Clamps to the last valid frame of each motion to avoid out-of-bounds access.

        Returns:
            Flattened tensor of shape ``(num_envs * num_future_frames,)``.
        """
        return (
            torch.clip(
                self.future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    @property
    def smpl_future_time_steps(self) -> torch.Tensor:
        """Compute absolute time-step indices for SMPL future reference frames.

        SMPL future frames may use different count and spacing than robot frames.

        Returns:
            Flattened tensor of shape ``(num_envs * smpl_num_future_frames,)``.
        """
        return (
            torch.clip(
                self.smpl_future_time_steps_init
                + self.time_steps[:, None]
                + self.motion_start_time_steps[:, None],
                max=self.motion_num_steps[:, None] - 1,
            )
            .flatten()
            .long()
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of debug visualization markers.

        Lazily initializes feet contact markers, height-map dot markers, and
        contact center sphere markers on first enable. Subsequent calls toggle
        visibility without re-creating prims.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(
                    self.cfg.body_pos_visualizer_cfg.replace(
                        prim_path="/Visuals/goal_marker_sphere"
                    )
                )
                self.feet_contact_goal_visualizers = []
                for name in self.cfg.feet_body_names:
                    self.feet_contact_goal_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.feet_contact_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )
            self.goal_pos_visualizer.set_visibility(True)
            for i in range(len(self.cfg.feet_body_names)):
                self.feet_contact_goal_visualizers[i].set_visibility(True)
            if self.cfg.use_height_map:
                if not hasattr(self, "height_map_visualizer"):
                    height_map_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/height_map",
                        markers={
                            "scan_dots": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 1.0, 0.0)
                                ),
                            )
                        },
                    )
                    self.height_map_visualizer = VisualizationMarkers(height_map_cfg)
                self.height_map_visualizer.set_visibility(True)
            if not hasattr(self, "contact_center_visualizers"):
                self.contact_center_visualizers = None
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "feet_contact_goal_visualizers"):
                for vis in self.feet_contact_goal_visualizers:
                    vis.set_visibility(False)
            if hasattr(self, "height_map_visualizer"):
                self.height_map_visualizer.set_visibility(False)
            if (
                hasattr(self, "contact_center_visualizers")
                and self.contact_center_visualizers
            ):
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization marker positions each render frame.

        Draws current robot body frames, reference (goal) body frames, feet
        contact indicators, height-map hit points, and object contact centers.

        Args:
            event: Render event from the simulation (unused).
        """
        if not self.robot.is_initialized:
            return
        if not hasattr(self, "goal_pos_visualizer"):
            return
        self.goal_pos_visualizer.visualize(self.body_pos_w.view(-1, 3))
        if hasattr(self, "feet_contact_goal_visualizers"):
            for i in range(len(self.cfg.body_names)):
                if self.cfg.body_names[i] == "left_ankle_roll_link":
                    self.feet_contact_goal_visualizers[0].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_l.int().reshape(-1),
                    )
                if self.cfg.body_names[i] == "right_ankle_roll_link":
                    self.feet_contact_goal_visualizers[1].visualize(
                        translations=self.body_pos_relative_w[:, i],
                        marker_indices=self.feet_r.int().reshape(-1),
                    )
        if self.cfg.use_height_map:
            self.height_map_visualizer.visualize(
                translations=self.scan_dot_pos_w.view(-1, 3)
            )
        if (
            hasattr(self, "contact_center_visualizers")
            and self.contact_center_visualizers is None
        ):
            if hasattr(self, "motion_lib"):
                self.contact_center_visualizers = {}
                if hasattr(self.motion_lib, "_motion_object_contact_center_left"):
                    left_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_left",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 0.0, 1.0)
                                ),
                            )
                        },
                    )
                    self.contact_center_visualizers["left_hand"] = VisualizationMarkers(
                        left_cfg
                    )
                if hasattr(self.motion_lib, "_motion_object_contact_center_right"):
                    right_cfg = VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/contact_center_right",
                        markers={
                            "contact": sim_utils.SphereCfg(
                                radius=0.05,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(0.0, 1.0, 1.0)
                                ),
                            )
                        },
                    )
                    self.contact_center_visualizers["right_hand"] = (
                        VisualizationMarkers(right_cfg)
                    )
                for vis in self.contact_center_visualizers.values():
                    vis.set_visibility(True)
        if self.contact_center_visualizers:
            hidden_pos = torch.tensor([[0.0, 0.0, -1000.0]], device=self._env.device)
            for hand, visualizer in self.contact_center_visualizers.items():
                contact_center = self.motion_lib.get_object_contact_center(
                    self.motion_ids,
                    self.motion_start_time_steps + self.time_steps,
                    hand=hand,
                )
                if contact_center is None:
                    visualizer.visualize(translations=hidden_pos)
                    continue
                valid_mask = torch.norm(contact_center, dim=-1) > 1e-06
                world_center = self._get_contact_center_world(hand)
                world_center[~valid_mask] = hidden_pos
                visualizer.visualize(translations=world_center)

    def set_motion_state(self, motion_ids, time_steps, motion_start_time_steps=None):
        """Update which motion clip/frame this command serves (for offline use)."""
        self.motion_ids = motion_ids
        self.time_steps = time_steps
        self.motion_start_time_steps = (
            motion_start_time_steps
            if motion_start_time_steps is not None
            else torch.zeros_like(motion_ids)
        )
        self.future_motion_ids = motion_ids.repeat_interleave(self.num_future_frames)
        self.smpl_future_motion_ids = motion_ids.repeat_interleave(
            self.smpl_num_future_frames
        )
        self.motion_num_steps = self.motion_lib.get_time_step_total(motion_ids)


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for TrackingCommand motion tracking.

    Controls motion library loading, future reference frame layout, encoder
    sampling probabilities (G1/SMPL/teleop), episode initialization strategy,
    object handling, and debug visualization marker styles.
    """

    class_type: type = TrackingCommand
    asset_name: str = dataclasses.MISSING
    motion_lib_cfg: dict = None
    motion_file: str = None
    smpl_motion_file: str = None
    filter_motion_keys: list[str] = None
    use_paired_motions: bool = False
    contact_file: str = None
    sample_before_contact: bool = False
    sample_before_contact_margin: int = 10
    sample_before_contact_hand: str = "right_hand"
    anchor_body: str = dataclasses.MISSING
    body_names: list[str] = dataclasses.MISSING
    vr_3point_body: list[str] = []
    vr_3point_body_offset: list[list[float]] = []
    reward_point_body: list[str] = []
    reward_point_body_offset: list[list[float]] = []
    force_push_body: list[str] = []
    force_push_body_offset: list[list[float]] = []
    num_future_frames: int = 1
    dt_future_ref_frames: float = 0.1
    randomize_heading: bool = False
    variable_frames_enabled: bool = False
    variable_frames_min: int = 16
    variable_frames_step: int = 4
    freeze_frame_aug: bool = False
    freeze_frame_aug_prob: float = 0.1
    smpl_num_future_frames: int = None
    smpl_dt_future_ref_frames: float = None
    smpl_num_future_frames: int = None
    smpl_dt_future_ref_frames: float = None
    encoder_sample_probs: dict[str, float] = None
    optimize_encoders_ratio_for_CHIP: bool = False
    teleop_sample_prob_when_smpl: float = 0.0
    start_from_first_frame: bool = False
    sample_unique_motions: bool = False
    sample_from_n_initial_frames: int = None
    object_position_randomize: bool = False
    object_position_randomization: dict[str, float] = None
    table_offset: list[float] = None
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    joint_velocity_range: tuple[float, float] = (-0, 0)
    body_pos_visualizer_cfg: VisualizationMarkersCfg = (
        DEFORMABLE_TARGET_MARKER_CFG.replace(prim_path="/Visuals/goal_marker_sphere")
    )
    body_pos_visualizer_cfg.markers["target"].radius = 0.05
    body_pos_visualizer_cfg.markers["target"].visual_material = (
        sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
    )
    feet_contact_visualizer_cfg: VisualizationMarkersCfg = (
        DEFORMABLE_TARGET_MARKER_CFG.replace(prim_path="/Visuals/goal_marker_sphere")
    )
    feet_contact_visualizer_cfg.markers = {
        "target_small": sim_utils.SphereCfg(
            radius=0.0001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_big": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
    waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    left_eef_deps = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ]
    right_eef_deps = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ]
    force_push_body: list[str] = []
    force_push_body_offset: list[list[float]] = []
    feet_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    cat_upper_body_poses: bool = False
    cat_upper_body_poses_prob: float = 0.5
    randomize_wrist_poses: bool = False
    randomize_wrist_prob: float = 0.3
    randomize_wrist_std: float = 0.1
    use_height_map: bool = False
    height_map_resolution: float = 0.15
    height_map_size: float = 1.5
    height_map_max_dist: float = 5.0
    motion_lib_num_dof: int | None = None
    hand_default_positions: list[float] | None = None
    hand_default_velocities: list[float] | None = None
    object_z_offset: float = 0.0


def _get_body_indexes(
    command: TrackingCommand, body_names: list[str] | None
) -> list[int]:
    """Return indices into command.cfg.body_names for the requested subset.

    Args:
        command: TrackingCommand instance whose cfg.body_names defines the full body list.
        body_names: Subset of body names to select. If None, return all indices.

    Returns:
        List of integer indices into ``command.cfg.body_names``.
    """
    return [
        idx
        for idx, body_name in enumerate(command.cfg.body_names)
        if body_names is None or body_name in body_names
    ]
