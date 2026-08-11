"""G1-native motion representation for text-to-motion fine-tuning.

Retargeted G1 AMP NPZ files store the full robot
state (30 body links + 29 DOFs).  For a *generative* T2M model we only need a
compact, FK-invertible target: the pelvis (root) pose plus the 29 joint angles.
Everything else (body link positions / rotations / velocities) is recoverable
via MuJoCo forward kinematics, so we drop it from the learning target.

Representation (``G1_MOTION_DIM = 38`` per frame)::

    [0:3]   pelvis translation (x, y, z)   -- MuJoCo world frame, Z-up, metres
    [3:9]   pelvis rotation 6D             -- HyMotion row-major 6D (first two
                                              columns of the rotation matrix)
    [9:38]  29 DOF joint angles            -- radians, AMP ``dof_positions`` order

Decoding to MuJoCo ``qpos`` (root xyz + quat_wxyz + 29 DOF = 36) is exact, so a
generated 38-d motion can be passed directly to MuJoCo or a robot-motion export
path without another SMPL-to-G1 retargeting step.

Conventions:
  * AMP ``body_rotations`` are stored as quaternion (x, y, z, w) -- xyzw.
  * ``quaternion_to_matrix`` expects (w, x, y, z) -- wxyz.
  * 6D is row-major ``[R00, R01, R10, R11, R20, R21]`` (HyMotion convention).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from motius.motion.representation.rotation import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

# ---- layout ----
G1_NUM_DOF = 29
G1_TRANSL_DIM = 3
G1_ROT6D_DIM = 6
G1_MOTION_DIM = G1_TRANSL_DIM + G1_ROT6D_DIM + G1_NUM_DOF  # 38
G1_QPOS_DIM = 3 + 4 + G1_NUM_DOF  # 36 (transl + quat_wxyz + dof)

# pelvis is the first body link in the AMP export (``body_names[0]``).
PELVIS_BODY_IDX = 0


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion xyzw -> wxyz."""
    return quat_xyzw[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion wxyz -> xyzw."""
    return quat_wxyz[..., [1, 2, 3, 0]]


def _canonicalize_root(transl: torch.Tensor, rmat: torch.Tensor):
    """Express the pelvis trajectory in a per-clip canonical frame.

    The retargeted AMP clips keep their raw MuJoCo *world* pelvis pose: the
    ground-plane (X, Y) start position and the global heading (yaw about the
    Z-up axis) are arbitrary per clip and **not** a function of the caption.
    A T2M model therefore cannot predict them and regresses translation toward
    the dataset mean (verified: even a static clip far from the mean shows ~1 m
    translation error).  We remove these two nuisance d.o.f. so that every clip
    starts at ground origin facing the canonical +X direction, making the
    target a deterministic function of the text.

    Concretely (Z is the up/height axis -- confirmed from the data: pelvis Z
    ~0.73 m standing, smallest per-dim std):

      * subtract the frame-0 ground position (X, Y); **keep Z** (real height,
        which is physically meaningful: crawl ~0.26 m vs stand ~0.76 m);
      * rotate the whole trajectory and the root orientation about +Z by
        ``-yaw0`` so the frame-0 heading is zero.

    DOF joint angles are parent-relative and unaffected.  This is lossy only in
    the absolute world placement, which is irrelevant for the PhysFlow loop
    (the robot is spawned at the canonical origin).
    """
    transl = transl.clone().float()
    rmat = rmat.clone().float()

    x0 = float(transl[0, 0]); y0 = float(transl[0, 1])
    transl[:, 0] -= x0
    transl[:, 1] -= y0

    R0 = rmat[0]
    yaw0 = torch.atan2(R0[1, 0], R0[0, 0])
    c = torch.cos(-yaw0); s = torch.sin(-yaw0)
    z = torch.zeros((), dtype=c.dtype)
    o = torch.ones((), dtype=c.dtype)
    Rz = torch.stack([
        torch.stack([c, -s, z]),
        torch.stack([s, c, z]),
        torch.stack([z, z, o]),
    ])  # (3, 3); rotation about +Z preserves the height channel.

    transl = transl @ Rz.T                       # rotate ground trajectory
    rmat = torch.einsum('ij,tjk->tik', Rz, rmat)  # rotate root orientation
    return transl, rmat


def encode_g1_motion(
    npz: Dict[str, np.ndarray],
    canonicalize: bool = True,
    root_velocity: bool = True,
) -> np.ndarray:
    """Encode a loaded G1 AMP npz dict into the (T, 38) generative target.

    Args:
        npz: mapping with at least ``body_positions`` (T, 30, 3),
            ``body_rotations`` (T, 30, 4) xyzw, ``dof_positions`` (T, 29).
        canonicalize: if True (default) express the root in a per-clip
            canonical frame (frame-0 ground origin + zero heading), so the
            target is a deterministic function of the caption.  See
            :func:`_canonicalize_root`.
        root_velocity: if True (default) store the ground-plane (X, Y)
            translation as a **per-frame velocity** (delta) instead of the
            absolute position; the height channel (Z) stays absolute.  This is
            the standard HumanML3D-style root representation: absolute position
            spans a large, sparse range (a run reaches several metres) that a
            flow-matching model under-shoots, whereas the per-frame velocity is
            small, bounded and easy to learn -- the absolute trajectory is
            recovered by ``cumsum`` at decode (see :func:`decode_g1_to_qpos`).

    Layout (root_velocity=True): ``[vx, vy, z_height]`` for channels [0:3].

    Returns:
        ``np.float32`` array of shape (T, 38).
    """
    body_pos = np.asarray(npz['body_positions'], dtype=np.float32)
    body_rot = np.asarray(npz['body_rotations'], dtype=np.float32)
    dof = np.asarray(npz['dof_positions'], dtype=np.float32)

    transl = torch.from_numpy(body_pos[:, PELVIS_BODY_IDX, :])  # (T, 3)
    quat_xyzw = torch.from_numpy(body_rot[:, PELVIS_BODY_IDX, :])  # (T, 4)
    quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
    rmat = quaternion_to_matrix(quat_wxyz)  # (T, 3, 3)
    if canonicalize:
        transl, rmat = _canonicalize_root(transl, rmat)
    rot6d = matrix_to_rotation_6d(rmat, convention="row")  # (T, 6)

    transl_feat = transl
    if root_velocity:
        # ground-plane (X, Y) -> per-frame delta (frame 0 has zero velocity);
        # keep Z (height) absolute -- it is bounded & physically meaningful.
        vel_xy = torch.zeros_like(transl[:, :2])
        vel_xy[1:] = transl[1:, :2] - transl[:-1, :2]
        transl_feat = torch.cat([vel_xy, transl[:, 2:3]], dim=-1)  # (T, 3)

    motion = torch.cat([transl_feat, rot6d, torch.from_numpy(dof)], dim=-1)  # (T, 38)
    return motion.numpy().astype(np.float32)


def encode_g1_qpos(
    qpos: np.ndarray,
    canonicalize: bool = True,
    root_velocity: bool = True,
) -> np.ndarray:
    """Encode MuJoCo G1 qpos (root xyz + quat_wxyz + 29 DOF) into (T, 38).

    This is the inverse-side adapter for generated artifacts exported as qpos
    CSVs. It uses the same canonical root and velocity convention as
    :func:`encode_g1_motion`, so qpos rollouts can be scored by TMR-G1.
    """
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != G1_QPOS_DIM:
        raise ValueError(f"Expected qpos shape (T, {G1_QPOS_DIM}), got {arr.shape}.")

    transl = torch.from_numpy(arr[:, :3])
    quat_wxyz = torch.from_numpy(arr[:, 3:7])
    dof = torch.from_numpy(arr[:, 7:])
    rmat = quaternion_to_matrix(quat_wxyz)
    if canonicalize:
        transl, rmat = _canonicalize_root(transl, rmat)
    rot6d = matrix_to_rotation_6d(rmat, convention="row")

    transl_feat = transl
    if root_velocity:
        vel_xy = torch.zeros_like(transl[:, :2])
        vel_xy[1:] = transl[1:, :2] - transl[:-1, :2]
        transl_feat = torch.cat([vel_xy, transl[:, 2:3]], dim=-1)

    motion = torch.cat([transl_feat, rot6d, dof], dim=-1)
    return motion.numpy().astype(np.float32)


def decode_g1_to_qpos(motion: torch.Tensor, root_velocity: bool = True) -> torch.Tensor:
    """Decode (T, 38) [or (B, T, 38)] generative motion into MuJoCo qpos.

    Args:
        motion: (..., T, 38) generative representation.
        root_velocity: must match the flag used in :func:`encode_g1_motion`.
            When True, channels [0:2] are per-frame ground velocity and are
            integrated back to absolute (X, Y) via cumulative sum over the
            frame axis; channel [2] is absolute height.

    Returns qpos with layout ``[transl(3), quat_wxyz(4), dof(29)]`` -> last dim 36.
    """
    transl_feat = motion[..., 0:G1_TRANSL_DIM]
    rot6d = motion[..., G1_TRANSL_DIM:G1_TRANSL_DIM + G1_ROT6D_DIM]
    dof = motion[..., G1_TRANSL_DIM + G1_ROT6D_DIM:]
    if root_velocity:
        xy = torch.cumsum(transl_feat[..., 0:2], dim=-2)  # integrate ground velocity
        transl = torch.cat([xy, transl_feat[..., 2:3]], dim=-1)
    else:
        transl = transl_feat
    rmat = rotation_6d_to_matrix(rot6d, convention="row")  # (..., 3, 3)
    quat_wxyz = matrix_to_quaternion(rmat)  # (..., 4) wxyz (pytorch3d convention)
    return torch.cat([transl, quat_wxyz, dof], dim=-1)
