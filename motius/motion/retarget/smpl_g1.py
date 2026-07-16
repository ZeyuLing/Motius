"""SMPL / SMPL-H / SMPL-X 22-joint humanoid → Unitree G1 29-DOF retargeting.

The single retargeting backend is ``GMRSMPLToG1Retargeter`` (high quality): it
uses an in-tree vendored copy of General Motion Retargeting (GMR, mink-based
inverse kinematics) at ``motius/motion/retarget/_gmr``. It solves a per-frame IK problem that matches
G1 link poses to SMPL-X body targets and post-processes the result into a
ready-to-use, ground-aligned, Z-up G1 motion (``dof_pos`` + floating-base root).
This is what you want for visualization, deployment, or physics tracking. GMR's
extra deps (``mink``, ``daqp``, ``smplx``, ``mujoco``) are imported lazily, so
importing this module never requires them — they are only needed when you
actually call the retargeter.

Overview
--------
HyMotion T2M generates 201-dim (or 135-dim) motion in SMPL format:
  - 22 joints × 6 (rotation_6d row-major) + 3 (translation)

Unitree G1 has 29 DOF:
  - Legs: 2 × 6 DOF (hip_pitch/roll/yaw, knee, ankle_pitch/roll)
  - Waist: 3 DOF (yaw, roll, pitch)
  - Arms: 2 × 7 DOF (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)

A previous fast analytic (per-frame Euler decomposition) backend was removed: it
produced low-quality, visibly broken poses. Use GMR for everything.
"""

from __future__ import annotations

import os
import pathlib
import pickle
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from motius.motion.representation.rotation import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

__all__ = [
    'SMPL_JOINT_NAMES',
    'G1_JOINT_NAMES',
    'G1_JOINT_LIMITS',
    'GMR_Y_UP_FROM_Z_UP',
    'GMR_Z_UP_FROM_Y_UP',
    'GMRSMPLToG1Retargeter',
]


# ============================================================================
# Constants
# ============================================================================

SMPL_JOINT_NAMES = [
    'Pelvis',      # 0
    'L_Hip',       # 1
    'R_Hip',       # 2
    'Spine1',      # 3
    'L_Knee',      # 4
    'R_Knee',      # 5
    'Spine2',      # 6
    'L_Ankle',     # 7
    'R_Ankle',     # 8
    'Spine3',      # 9
    'L_Foot',      # 10
    'R_Foot',      # 11
    'Neck',        # 12
    'L_Collar',    # 13
    'R_Collar',    # 14
    'Head',        # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',     # 18
    'R_Elbow',     # 19
    'L_Wrist',     # 20
    'R_Wrist',     # 21
]

# G1 29-DOF joint names in order
G1_JOINT_NAMES = [
    # Left leg (6 DOF)
    'left_hip_pitch_joint',     # 0
    'left_hip_roll_joint',      # 1
    'left_hip_yaw_joint',       # 2
    'left_knee_joint',          # 3
    'left_ankle_pitch_joint',   # 4
    'left_ankle_roll_joint',    # 5
    # Right leg (6 DOF)
    'right_hip_pitch_joint',    # 6
    'right_hip_roll_joint',     # 7
    'right_hip_yaw_joint',      # 8
    'right_knee_joint',         # 9
    'right_ankle_pitch_joint',  # 10
    'right_ankle_roll_joint',   # 11
    # Waist (3 DOF)
    'waist_yaw_joint',          # 12
    'waist_roll_joint',         # 13
    'waist_pitch_joint',        # 14
    # Left arm (7 DOF)
    'left_shoulder_pitch_joint',  # 15
    'left_shoulder_roll_joint',   # 16
    'left_shoulder_yaw_joint',    # 17
    'left_elbow_joint',           # 18
    'left_wrist_roll_joint',      # 19
    'left_wrist_pitch_joint',     # 20
    'left_wrist_yaw_joint',       # 21
    # Right arm (7 DOF)
    'right_shoulder_pitch_joint',  # 22
    'right_shoulder_roll_joint',   # 23
    'right_shoulder_yaw_joint',    # 24
    'right_elbow_joint',           # 25
    'right_wrist_roll_joint',      # 26
    'right_wrist_pitch_joint',     # 27
    'right_wrist_yaw_joint',       # 28
]

# G1 joint limits (radians) from URDF.
# Format: (lower, upper) for each of the 29 DOF.
# Source: unitree g1_29dof.urdf typical values.
G1_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    # Left leg
    'left_hip_pitch_joint':     (-2.5307, 2.8798),
    'left_hip_roll_joint':      (-0.5236, 2.9671),
    'left_hip_yaw_joint':       (-2.7576, 2.7576),
    'left_knee_joint':          (-0.2618, 2.0944),
    'left_ankle_pitch_joint':   (-0.8727, 0.5236),
    'left_ankle_roll_joint':    (-0.2618, 0.2618),
    # Right leg
    'right_hip_pitch_joint':    (-2.5307, 2.8798),
    'right_hip_roll_joint':     (-2.9671, 0.5236),
    'right_hip_yaw_joint':      (-2.7576, 2.7576),
    'right_knee_joint':         (-0.2618, 2.0944),
    'right_ankle_pitch_joint':  (-0.8727, 0.5236),
    'right_ankle_roll_joint':   (-0.2618, 0.2618),
    # Waist
    'waist_yaw_joint':          (-2.6180, 2.6180),
    'waist_roll_joint':         (-0.5236, 0.5236),
    'waist_pitch_joint':        (-0.5236, 0.5236),
    # Left arm
    'left_shoulder_pitch_joint':  (-3.0892, 2.6927),
    'left_shoulder_roll_joint':   (-1.5882, 2.2515),
    'left_shoulder_yaw_joint':    (-2.6180, 2.6180),
    'left_elbow_joint':           (-1.0472, 2.0944),
    'left_wrist_roll_joint':      (-1.9722, 1.9722),
    'left_wrist_pitch_joint':     (-0.3491, 0.3491),
    'left_wrist_yaw_joint':       (-0.5236, 0.5236),
    # Right arm
    'right_shoulder_pitch_joint': (-3.0892, 2.6927),
    'right_shoulder_roll_joint':  (-2.2515, 1.5882),
    'right_shoulder_yaw_joint':   (-2.6180, 2.6180),
    'right_elbow_joint':          (-1.0472, 2.0944),
    'right_wrist_roll_joint':     (-1.9722, 1.9722),
    'right_wrist_pitch_joint':    (-0.3491, 0.3491),
    'right_wrist_yaw_joint':      (-0.5236, 0.5236),
}


# ============================================================================
# GMR retargeter (high quality, mink inverse-kinematics)
# ============================================================================

# Repository root: .../motius/motion/retarget/smpl_g1.py -> parents[3]
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Vendored GMR lives in-tree (motius/motion/retarget/_gmr) so the library has
# Robot MJCF + IK configs ship inside the package;
# only the SMPL-X body models are user data, resolved from the repo checkpoints.
_VENDOR_DIR = pathlib.Path(__file__).resolve().parent / '_gmr'
_DEFAULT_SMPLX_MODEL_DIR = pathlib.Path(
    os.environ.get('MOTIUS_SMPL_MODEL_DIR', _REPO_ROOT / 'checkpoints' / 'body_models')
)

# Rotation (xyzw) that maps GMR's SMPL-X (Y-up, with the pelvis facing offset
# baked into the IK config) world frame onto the standard MuJoCo / robot Z-up
# frame. The retarget root pose is left-multiplied by its inverse. This exact
# offset is what produces an upright, correctly-facing G1 in MuJoCo and was
# validated against the rendered jog reference.
_GMR_ZUP_ROT_OFFSET_XYZW = (-0.5, -0.5, -0.5, 0.5)

# GMR maps SMPL Y-up coordinates to MuJoCo Z-up as [x, y, z] -> [z, x, y].
# Keep the inverse public so downstream renderers do not guess a different
# horizontal-axis convention and accidentally rotate or flip the robot.
GMR_Z_UP_FROM_Y_UP = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
)
GMR_Y_UP_FROM_Z_UP = GMR_Z_UP_FROM_Y_UP.T.copy()

# GMR robot key -> vendored MuJoCo XML (used both as GMR's retarget target model
# and for the headless ground-alignment FK pass).
_GMR_ROBOT_XML = {
    'unitree_g1': _VENDOR_DIR / 'assets' / 'unitree_g1' / 'g1_mocap_29dof.xml',
}

# G1 mechanical joint limits matching GMR's g1 model (g1_bm.xml). These differ
# slightly from the reference ``G1_JOINT_LIMITS`` above (e.g. knee/wrist ranges)
# and are the correct bounds to clamp GMR IK output against.
_GMR_G1_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    'left_hip_pitch_joint': (-2.5307, 2.8798),
    'left_hip_roll_joint': (-0.5236, 2.9671),
    'left_hip_yaw_joint': (-2.7576, 2.7576),
    'left_knee_joint': (-0.087267, 2.8798),
    'left_ankle_pitch_joint': (-0.87267, 0.5236),
    'left_ankle_roll_joint': (-0.2618, 0.2618),
    'right_hip_pitch_joint': (-2.5307, 2.8798),
    'right_hip_roll_joint': (-2.9671, 0.5236),
    'right_hip_yaw_joint': (-2.7576, 2.7576),
    'right_knee_joint': (-0.087267, 2.8798),
    'right_ankle_pitch_joint': (-0.87267, 0.5236),
    'right_ankle_roll_joint': (-0.2618, 0.2618),
    'waist_yaw_joint': (-2.618, 2.618),
    'waist_roll_joint': (-0.52, 0.52),
    'waist_pitch_joint': (-0.52, 0.52),
    'left_shoulder_pitch_joint': (-3.0892, 2.6704),
    'left_shoulder_roll_joint': (-1.5882, 2.2515),
    'left_shoulder_yaw_joint': (-2.618, 2.618),
    'left_elbow_joint': (-1.0472, 2.0944),
    'left_wrist_roll_joint': (-1.97222, 1.97222),
    'left_wrist_pitch_joint': (-1.61443, 1.61443),
    'left_wrist_yaw_joint': (-1.61443, 1.61443),
    'right_shoulder_pitch_joint': (-3.0892, 2.6704),
    'right_shoulder_roll_joint': (-2.2515, 1.5882),
    'right_shoulder_yaw_joint': (-2.618, 2.618),
    'right_elbow_joint': (-1.0472, 2.0944),
    'right_wrist_roll_joint': (-1.97222, 1.97222),
    'right_wrist_pitch_joint': (-1.61443, 1.61443),
    'right_wrist_yaw_joint': (-1.61443, 1.61443),
}


class GMRSMPLToG1Retargeter:
    """High-quality SMPL/SMPL-H/SMPL-X → Unitree G1 retargeting via GMR (mink IK).

    This wraps the in-tree vendored GMR (``motius.motion.retarget._gmr``,
    ``GeneralMotionRetargeting``) and turns it into
    a clean, in-repo library API. It solves a per-frame inverse-kinematics problem
    that matches G1 link poses to SMPL-X body targets, then post-processes the
    output into a deployment-ready G1 motion.

    Pipeline (per call)::

        SMPL-X (root_orient/pose_body/trans/betas)
          -> GMR.load_smplx_file (FK + auto human-height)
          -> GMR.get_smplx_data_offline_fast (fps alignment)
          -> GeneralMotionRetargeting (mink IK, posture-cost temporal reg.)
          -> per-frame ground offset (set_ground_offset)
          -> qpos = [root_pos(3), root_quat_wxyz(4), dof(29)]
          -> joint-limit clamp (soft) + Savitzky-Golay temporal smoothing
          -> (optional) Y-up -> Z-up root transform
          -> (optional) global ground alignment (lowest geom -> z=0)

    Output dict (frame = Z-up MuJoCo/robot frame by default)::

        {
          'dof_pos':           (T, 29) float32,   # G1 joint positions (rad)
          'root_pos':          (T, 3)  float32,   # floating-base translation (m)
          'root_orient_quat':  (T, 4)  float32,   # wxyz (MuJoCo convention)
          'root_rot':          (T, 4)  float32,   # xyzw (scipy/ProtoMotions)
          'fps':               float,
          'joint_names':       list[str],         # 29 G1 joint names
          'dof':               29,
        }

    The result is directly consumable by :meth:`to_mujoco_qpos` (-> (T, 36)) and
    :meth:`save_pkl` (GMR/ProtoMotions-style pkl with xyzw root).

    Notes:
        - GMR deps (``mink``, ``daqp``, ``smplx``, ``mujoco``) are imported lazily.
          A clear ``ImportError`` is raised if they're missing.
        - SMPL-X body models are user data, resolved from ``smplx_model_dir``
          (default ``<repo>/checkpoints/body_models``; smplx looks under ``smplx/``).
        - ``mujoco_zup=True`` + ``ground_align=True`` (defaults) make the output a
          drop-in qpos for a standard Z-up G1 MuJoCo model. Set both False to get
          GMR's raw solver frame.
    """

    def __init__(
        self,
        robot: str = 'unitree_g1',
        tgt_fps: int = 30,
        posture_cost: float = 20.0,
        actual_human_height: Optional[float] = None,
        offset_to_ground: bool = False,
        clamp_limits: bool = True,
        soft_clamp: bool = True,
        smooth: bool = True,
        mujoco_zup: bool = True,
        ground_align: bool = True,
        ground_clearance: float = 0.0,
        smplx_model_dir: Optional[Union[str, pathlib.Path]] = None,
        robot_xml: Optional[Union[str, pathlib.Path]] = None,
    ):
        self.robot = robot
        self.tgt_fps = int(tgt_fps)
        self.posture_cost = float(posture_cost)
        self.actual_human_height = actual_human_height
        self.offset_to_ground = bool(offset_to_ground)
        self.clamp_limits = bool(clamp_limits)
        self.soft_clamp = bool(soft_clamp)
        self.smooth = bool(smooth)
        self.mujoco_zup = bool(mujoco_zup)
        self.ground_align = bool(ground_align)
        self.ground_clearance = float(ground_clearance)

        # SMPL-X body models are user data (smplx.create looks under <dir>/smplx).
        self.smplx_model_dir = (
            pathlib.Path(smplx_model_dir) if smplx_model_dir else _DEFAULT_SMPLX_MODEL_DIR
        )
        if robot_xml is not None:
            self.robot_xml = pathlib.Path(robot_xml)
        else:
            self.robot_xml = _GMR_ROBOT_XML.get(robot)

        self._gmr_loaded = False

    # ------------------------------------------------------------------ deps
    def _ensure_gmr(self):
        """Lazily import the in-tree vendored GMR (and its runtime deps).

        Imports from ``motius.motion.retarget._gmr``. Only the heavy runtime
        deps (mink/daqp/mujoco/smplx)
        are optional and produce a clear error if missing.
        """
        if self._gmr_loaded:
            return
        try:
            from motius.motion.retarget._gmr import (  # noqa
                GeneralMotionRetargeting,
                load_smplx_file,
                get_smplx_data_offline_fast,
            )
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                'Failed to import the vendored GMR backend. It needs extra deps '
                'not in pyproject.toml: install them with\n'
                '  python3 -m pip install mink daqp smplx mujoco scipy\n'
                f'(underlying error: {e})'
            ) from e
        self._GMR = GeneralMotionRetargeting
        self._load_smplx_file = load_smplx_file
        self._get_smplx_data_offline_fast = get_smplx_data_offline_fast
        self._gmr_loaded = True

    # ------------------------------------------------------------- entrypoints
    def retarget_smplx_file(self, smplx_file: Union[str, pathlib.Path]) -> Dict[str, np.ndarray]:
        """Retarget from an SMPL-X NPZ file.

        The NPZ must contain ``root_orient`` (T,3), ``pose_body`` (T,63),
        ``trans`` (T,3), ``betas`` (>=10,), and ``gender``.
        """
        return self._run(str(smplx_file))

    def retarget_smplx(
        self,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        trans: np.ndarray,
        betas: Optional[np.ndarray] = None,
        gender: str = 'neutral',
        fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Retarget from in-memory SMPL-X arrays.

        Args:
            root_orient: (T, 3) global orientation axis-angle.
            pose_body:   (T, 63) body pose axis-angle (21 joints).
            trans:       (T, 3) root translation (m).
            betas:       (>=10,) shape params (default zeros(16)).
            gender:      'neutral' | 'male' | 'female'.
            fps:         source frame rate.
        """
        root_orient = np.asarray(root_orient, np.float32).reshape(-1, 3)
        pose_body = np.asarray(pose_body, np.float32).reshape(root_orient.shape[0], -1)
        assert pose_body.shape[1] == 63, f'pose_body must be (T,63), got {pose_body.shape}'
        trans = np.asarray(trans, np.float32).reshape(-1, 3)
        if betas is None:
            betas = np.zeros(16, np.float32)
        betas = np.asarray(betas, np.float32).reshape(-1)
        npz = {
            'root_orient': root_orient,
            'pose_body': pose_body,
            'trans': trans,
            'betas': betas,
            'gender': str(gender),
            'mocap_frame_rate': np.int64(round(fps)),
        }
        return self._run_from_arrays(npz)

    def retarget_smplh(
        self,
        poses: np.ndarray,
        trans: np.ndarray,
        betas: Optional[np.ndarray] = None,
        gender: str = 'neutral',
        fps: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Retarget from SMPL-H / SMPL pose parameters.

        Args:
            poses: (T, >=66) axis-angle pose. The first 3 dims are the global
                   orientation and the next 63 are the 21-joint body pose; any
                   trailing hand/face dims (e.g. SMPL-H 156, SMPL 72) are ignored.
            trans: (T, 3) root translation (m).
            betas: shape params (default zeros(16)). Accepts (16,) or (1,16).
            gender: 'neutral' | 'male' | 'female'.
            fps: source frame rate.
        """
        poses = np.asarray(poses, np.float32)
        assert poses.ndim == 2 and poses.shape[1] >= 66, (
            f'poses must be (T,>=66) axis-angle, got {poses.shape}'
        )
        root_orient = poses[:, 0:3]
        pose_body = poses[:, 3:66]
        if betas is not None:
            betas = np.asarray(betas, np.float32).reshape(-1)
        return self.retarget_smplx(
            root_orient, pose_body, trans, betas=betas, gender=gender, fps=fps
        )

    def retarget_from_motion135(
        self,
        motion_135: np.ndarray,
        fps: float = 30.0,
        betas: Optional[np.ndarray] = None,
        gender: str = 'neutral',
    ) -> Dict[str, np.ndarray]:
        """Retarget from HyMotion 135-dim SMPL motion ([transl(3), rot6d(132)]).

        rot6d is decoded explicitly as row-major, then converted to axis-angle
        into SMPL-X global_orient/body_pose before running GMR.

        Note: this assumes the motion is already in SMPL canonical (Y-up) frame,
        matching what GMR's SMPL-X loader expects.
        """
        motion_135 = np.asarray(motion_135, np.float32)
        assert motion_135.ndim == 2 and motion_135.shape[1] == 135, (
            f'motion_135 must be (T,135), got {motion_135.shape}'
        )
        T = motion_135.shape[0]
        transl = motion_135[:, 0:3]
        rot6d = motion_135[:, 3:135].reshape(T, 22, 6)
        mats = rotation_6d_to_matrix(
            torch.from_numpy(rot6d.reshape(-1, 6).astype(np.float32)),
            convention='row',
        ).reshape(T, 22, 3, 3)
        aa = matrix_to_axis_angle(mats).reshape(T, 22, 3).cpu().numpy()
        root_orient = aa[:, 0]
        pose_body = aa[:, 1:22].reshape(T, 63)
        return self.retarget_smplx(
            root_orient, pose_body, transl, betas=betas, gender=gender, fps=fps
        )

    # --------------------------------------------------------------- internals
    def _run_from_arrays(self, npz: Dict) -> Dict[str, np.ndarray]:
        """Write a temp SMPL-X NPZ then delegate to GMR's file loader.

        Using GMR's own ``load_smplx_file`` keeps the FK-based human-height
        estimation identical to the reference pipeline.
        """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        try:
            np.savez(path, **npz)
            return self._run(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _run(self, smplx_file: str) -> Dict[str, np.ndarray]:
        self._ensure_gmr()

        smplx_data, body_model, smplx_output, auto_h = self._load_smplx_file(
            smplx_file, str(self.smplx_model_dir)
        )
        height = self.actual_human_height if self.actual_human_height is not None else auto_h

        frames, aligned_fps = self._get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=self.tgt_fps
        )

        retarget = self._GMR(
            actual_human_height=height,
            src_human='smplx',
            tgt_robot=self.robot,
            posture_cost=self.posture_cost,
        )
        ground_offset = self._compute_ground_offset(retarget, frames)
        retarget.set_ground_offset(ground_offset)

        qpos_list = [
            retarget.retarget(fd, offset_to_ground=self.offset_to_ground)
            for fd in frames
        ]
        qpos = np.asarray(qpos_list, dtype=np.float64)
        root_pos = qpos[:, 0:3]
        root_wxyz = qpos[:, 3:7]
        dof = qpos[:, 7:]

        if self.clamp_limits:
            dof, _ = self._clamp_joint_limits(dof, soft=self.soft_clamp)
        if self.smooth:
            dof = self._smooth(dof)

        if self.mujoco_zup:
            root_pos, root_wxyz = self._yup_to_zup(root_pos, root_wxyz)
        if self.ground_align:
            root_pos = self._ground_align(root_pos, root_wxyz, dof)

        root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
        return {
            'dof_pos': dof.astype(np.float32),
            'root_pos': root_pos.astype(np.float32),
            'root_orient_quat': root_wxyz.astype(np.float32),  # wxyz (MuJoCo)
            'root_rot': root_xyzw.astype(np.float32),          # xyzw (scipy)
            'fps': float(aligned_fps),
            'joint_names': list(G1_JOINT_NAMES),
            'dof': dof.shape[1],
        }

    @staticmethod
    def _compute_ground_offset(retarget, frames) -> float:
        """Lowest body Z across all frames (mirrors GMR fbx_offline grounding)."""
        offset = np.inf
        for frame_data in frames:
            human_data = retarget.to_numpy(frame_data)
            human_data = retarget.scale_human_data(
                human_data, retarget.human_root_name, retarget.human_scale_table
            )
            human_data = retarget.offset_human_data(
                human_data, retarget.pos_offsets1, retarget.rot_offsets1
            )
            for _, (pos, _quat) in human_data.items():
                if pos[2] < offset:
                    offset = float(pos[2])
        return offset

    @staticmethod
    def _clamp_joint_limits(dof_pos, soft=True):
        clamped = dof_pos.copy()
        num_clamped = 0
        for i, name in enumerate(G1_JOINT_NAMES):
            if name not in _GMR_G1_JOINT_LIMITS:
                continue
            lo, hi = _GMR_G1_JOINT_LIMITS[name]
            below = clamped[:, i] < lo
            above = clamped[:, i] > hi
            if soft:
                mid = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                scale = 0.9
                clamped[:, i] = mid + half * np.tanh((clamped[:, i] - mid) / (half * scale))
            else:
                clamped[:, i] = np.clip(clamped[:, i], lo, hi)
            num_clamped += int(np.sum(below) + np.sum(above))
        return clamped, num_clamped

    @staticmethod
    def _smooth(dof_pos):
        try:
            from scipy.signal import savgol_filter
        except Exception:
            return dof_pos
        T = dof_pos.shape[0]
        win = min(7, T if T % 2 == 1 else T - 1)
        if win >= 5:
            return savgol_filter(dof_pos, window_length=win, polyorder=3, axis=0)
        return dof_pos

    @staticmethod
    def _yup_to_zup(root_pos, root_wxyz):
        from scipy.spatial.transform import Rotation as R

        rot_offset = R.from_quat(list(_GMR_ZUP_ROT_OFFSET_XYZW))  # xyzw
        root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
        pos = np.asarray(root_pos) @ GMR_Z_UP_FROM_Y_UP.T
        xyzw = (rot_offset.inv() * R.from_quat(root_xyzw)).as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        return pos, wxyz

    def _ground_align(self, root_pos, root_wxyz, dof):
        """Shift the whole clip vertically so the lowest *robot* geom rests on z=0.

        The mjcf ships a world-attached ground ``plane`` geom (always at z=0); it
        must be excluded or it pins the per-frame minimum to 0 and the robot is
        never pulled down to its real feet (i.e. it floats). We only consider
        geoms owned by an actual robot body (``geom_bodyid != 0``).
        """
        if self.robot_xml is None or not os.path.isfile(str(self.robot_xml)):
            return root_pos
        try:
            import mujoco
        except Exception:
            return root_pos
        m = mujoco.MjModel.from_xml_path(str(self.robot_xml))
        d = mujoco.MjData(m)
        robot_geoms = np.where(np.asarray(m.geom_bodyid) != 0)[0]
        if robot_geoms.size == 0:
            robot_geoms = np.arange(m.ngeom)
        T = root_pos.shape[0]
        gmin = np.inf
        for t in range(T):
            d.qpos[:3] = root_pos[t]
            d.qpos[3:7] = root_wxyz[t]
            d.qpos[7:] = dof[t]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            gmin = min(gmin, float(d.geom_xpos[robot_geoms, 2].min()))
        root_pos = root_pos.copy()
        root_pos[:, 2] -= (gmin - self.ground_clearance)
        return root_pos

    # ----------------------------------------------------------------- outputs
    def to_mujoco_qpos(self, result: Dict[str, np.ndarray]) -> np.ndarray:
        """[root_pos(3), root_quat_wxyz(4), dof(29)] -> (T, 36) MuJoCo qpos."""
        dof = result['dof_pos']
        T, n = dof.shape
        qpos = np.zeros((T, 7 + n), dtype=np.float32)
        qpos[:, 0:3] = result['root_pos']
        qpos[:, 3:7] = result['root_orient_quat']
        qpos[:, 7:] = dof
        return qpos

    def save_pkl(self, result: Dict[str, np.ndarray], output_path: str) -> str:
        """Save GMR/ProtoMotions-style pkl (root_rot is xyzw)."""
        motion_data = {
            'fps': result['fps'],
            'root_pos': np.asarray(result['root_pos']),
            'root_rot': np.asarray(result['root_rot']),  # xyzw
            'dof_pos': np.asarray(result['dof_pos']),
            'local_body_pos': None,
            'link_body_list': None,
        }
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)
        return output_path

    @staticmethod
    def to_asap_pkl(result: Dict[str, np.ndarray], output_path: str) -> str:
        """Save an ASAP/HumanoidVerse-compatible pkl from a GMR result.

        Includes finite-difference ``dof_vel`` / ``root_vel`` expected by ASAP.
        """
        fps = float(result['fps'])
        dof = np.asarray(result['dof_pos'])
        root_pos = np.asarray(result['root_pos'])
        motion_data = {
            'fps': fps,
            'joint_names': list(result.get('joint_names') or G1_JOINT_NAMES),
            'dof': int(result.get('dof') or dof.shape[1]),
            'dof_pos': dof,                                       # (T, 29)
            'root_pos': root_pos,                                 # (T, 3)
            'root_orient_quat': np.asarray(result['root_orient_quat']),  # wxyz
            'dof_vel': np.gradient(dof, 1.0 / fps, axis=0).astype(np.float32),
            'root_vel': np.gradient(root_pos, 1.0 / fps, axis=0).astype(np.float32),
        }
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)
        return output_path
