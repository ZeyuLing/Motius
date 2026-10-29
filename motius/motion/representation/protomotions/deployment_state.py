# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected G1 state and heading geometry for native MuJoCo deployment."""
import numpy as np


def mujoco_wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo quaternion (wxyz) to ProtoMotions xyzw convention.

    Works on any array whose last dimension is 4.

    Args:
        wxyz: Quaternion(s) in wxyz order, shape ``(..., 4)``.

    Returns:
        Quaternion(s) in xyzw order, same shape.
    """
    return wxyz[..., [1, 2, 3, 0]]


def compute_anchor_rot_np(
    rigid_body_rot: np.ndarray,
    anchor_body_index: int,
) -> np.ndarray:
    """Extract the anchor body's orientation from the full body-rotation array.

    During deployment this corresponds to the **torso IMU** quaternion.

    Args:
        rigid_body_rot: Body orientations, shape ``[num_bodies, 4]`` (xyzw).
        anchor_body_index: 0-based index of the anchor body (e.g. 16 for
            ``torso_link`` on the G1).

    Returns:
        Anchor body orientation, shape ``[4,]`` (xyzw).
    """
    return rigid_body_rot[anchor_body_index]


def _extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract the yaw-only quaternion from a full orientation (xyzw).

    Decomposes the orientation into yaw (rotation about world Z) and
    returns a quaternion that represents only that yaw.

    Args:
        q_xyzw: Unit quaternion ``[x, y, z, w]``, shape ``[4,]``.

    Returns:
        Yaw-only quaternion ``[x, y, z, w]``, shape ``[4,]``.
    """
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Yaw-only quaternion: rotation about Z by `yaw`
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _quat_mul_np(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (pure NumPy).

    Works on single quaternions ``[4,]`` or batched ``[..., 4]``.
    """
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    ).astype(np.float32)


def _quat_conjugate_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats) of an xyzw quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def compute_yaw_offset_np(
    robot_quat_xyzw: np.ndarray,
    motion_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a yaw-only heading offset between robot and motion frames.

    Returns a quaternion ``R_offset`` such that::

        R_offset * motion_body_rot ≈ aligned_body_rot

    where ``aligned_body_rot`` is in the robot's heading frame.

    At ``t=0``, this makes ``quat_inv(robot_anchor) * R_offset * motion_anchor``
    near-identity, which is the correct initial condition for the obs function.

    In simulation (robot starts at motion frame 0), the offset is near-identity
    and has no effect.  On the real robot, it compensates for arbitrary startup
    heading.

    Args:
        robot_quat_xyzw:  Robot anchor body orientation at t=0, ``[4,]`` (xyzw).
        motion_quat_xyzw: Motion anchor body orientation at frame 0, ``[4,]`` (xyzw).

    Returns:
        Yaw-only offset quaternion ``[4,]`` (xyzw).
    """
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)
    # offset = yaw(robot) * yaw(motion)^-1
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(
    offset_quat_xyzw: np.ndarray,
    body_rots_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a heading offset to an array of body rotations.

    Computes ``offset * body_rot`` for every quaternion in the array.

    Args:
        offset_quat_xyzw: Yaw-only offset ``[4,]`` (xyzw), from
            :func:`compute_yaw_offset_np`.
        body_rots_xyzw: Body rotations, shape ``[..., 4]`` (xyzw).
            Typically ``[num_future_steps, num_bodies, 4]``.

    Returns:
        Aligned body rotations, same shape as input.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    # Broadcast offset [4,] against flat [N, 4]
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = _quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)
