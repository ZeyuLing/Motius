"""AIST++ SMPL-24 joint representation used by music-to-dance methods."""

from __future__ import annotations

from pathlib import Path

import numpy as np


AISTPP_SMPL24_JOINT_DIM = 72
AISTPP_MOTION_FPS = 60.0
AISTPP_SMPL24_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def as_aistpp_smpl24_joints(motion) -> np.ndarray:
    """Validate and reshape AIST++ global SMPL-24 joint positions."""

    joints = np.asarray(motion, dtype=np.float32)
    if joints.ndim == 2 and joints.shape[-1] == AISTPP_SMPL24_JOINT_DIM:
        joints = joints.reshape(-1, 24, 3)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(
            "AIST++ SMPL-24 joints must have shape (T,24,3) or (T,72), "
            f"got {joints.shape}"
        )
    if not np.isfinite(joints).all():
        raise ValueError("AIST++ SMPL-24 joints contain NaN or infinite values")
    return joints


def aistpp_smpl24_to_smpl22_joints(motion) -> np.ndarray:
    """Select the exact 22-joint SMPL body subset shared with Motius."""

    return as_aistpp_smpl24_joints(motion)[:, :22].copy()


def aistpp_smpl24_fk(
    smpl_poses,
    smpl_trans,
    smpl_scaling,
    rest_offsets,
) -> np.ndarray:
    """Materialize AIST++ SMPL-24 joints from pose and calibrated offsets.

    This is joint-level SMPL forward kinematics. It intentionally does not
    synthesize vertices or shape-dependent skinning.
    """

    from scipy.spatial.transform import Rotation

    poses = np.asarray(smpl_poses, dtype=np.float64)
    if poses.ndim == 2 and poses.shape[1] == 72:
        poses = poses.reshape(-1, 24, 3)
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"smpl_poses must have shape (T,24,3), got {poses.shape}")
    translation = np.asarray(smpl_trans, dtype=np.float64).reshape(-1, 3)
    if len(translation) != len(poses):
        raise ValueError("smpl_trans and smpl_poses have different frame counts")
    scale = float(np.asarray(smpl_scaling).reshape(-1)[0])
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"smpl_scaling must be finite and non-zero, got {scale}")
    offsets = np.asarray(rest_offsets, dtype=np.float64)
    if offsets.shape != (24, 3):
        raise ValueError(f"rest_offsets must have shape (24,3), got {offsets.shape}")

    local_rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    local_rotations = local_rotations.reshape(len(poses), 24, 3, 3)
    global_rotations = np.empty_like(local_rotations)
    joints = np.empty((len(poses), 24, 3), dtype=np.float64)
    global_rotations[:, 0] = local_rotations[:, 0]
    joints[:, 0] = translation / scale
    for joint in range(1, 24):
        parent = int(AISTPP_SMPL24_PARENTS[joint])
        global_rotations[:, joint] = np.einsum(
            "tij,tjk->tik",
            global_rotations[:, parent],
            local_rotations[:, joint],
        )
        joints[:, joint] = joints[:, parent] + np.einsum(
            "tij,j->ti", global_rotations[:, parent], offsets[joint]
        )
    return joints.astype(np.float32)


def aistpp_smpl24_to_motion135(
    motion,
    *,
    model_dir: str | Path | None = None,
    source_fps: float = AISTPP_MOTION_FPS,
    target_fps: float = 30.0,
    gender: str = "male",
    device=None,
    **ik_kwargs,
) -> np.ndarray:
    """Fit AIST++ positions to Motius SMPL ``motion135`` using position IK.

    AIST++ stores positions rather than joint rotations, so this route is
    intentionally lossy. The first 22 joints are the standard SMPL body chain;
    the final two hand joints are outside the Motius SMPL-22 bridge.
    """

    from motius.motion.retarget.hml263_smpl import retarget_hml263_clip

    result = retarget_hml263_clip(
        aistpp_smpl24_to_smpl22_joints(motion),
        model_dir=model_dir,
        source_fps=source_fps,
        target_fps=target_fps,
        gender=gender,
        device=device,
        rotation_init="position_ik",
        **ik_kwargs,
    )
    return result["motion_135"]


__all__ = [
    "AISTPP_MOTION_FPS",
    "AISTPP_SMPL24_JOINT_DIM",
    "AISTPP_SMPL24_PARENTS",
    "aistpp_smpl24_fk",
    "aistpp_smpl24_to_motion135",
    "aistpp_smpl24_to_smpl22_joints",
    "as_aistpp_smpl24_joints",
]
