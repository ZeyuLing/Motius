# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion representation helpers: velocity, heading, masks, and rotation of features."""

import einops
import torch

from motius.motion.representation.kimodo.geometry import cont6d_to_matrix, matrix_to_cont6d
from motius.motion.body_models.kimodo import SkeletonBase
from motius.utils.kimodo import ensure_batched


def diff_angles(angles: torch.Tensor, fps: float) -> torch.Tensor:
    """Compute frame-to-frame angular differences in radians, scaled by fps.

    Args:
        angles: [..., T] batched sequences of rotation angles in radians.
        fps: Sampling rate used to convert frame differences to per-second rate.

    Returns:
        [..., T-1] difference between consecutive angles (rad/s).
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos_diff = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
    sin_diff = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]

    # should be close to angles.diff() but more robust
    # multiply by fps = 1 / dt
    angles_diff = fps * torch.arctan2(sin_diff, cos_diff)
    return angles_diff


@ensure_batched(positions=4, lengths=1)
def compute_vel_xyz(
    positions: torch.Tensor,
    fps: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the velocities from positions: dx/dt.

    Works with batches. The last velocity is duplicated to keep the same size.
        Args:
            positions (torch.Tensor): [..., T, J, 3] xyz positions of a human skeleton
            fps (float): frame per seconds
            lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, positions should not be batched

        Returns:
            velocity (torch.Tensor): [..., T, J, 3] velocities computed from the positions
    """
    device = positions.device

    if lengths is None:
        assert positions.shape[0] == 1, "If lengths is not provided, the input should not be batched."
        lengths = torch.tensor([len(positions)], device=device)

    # compute velocities with fps
    velocity = fps * (positions[:, 1:] - positions[:, :-1])
    # pading the velocity vector
    vel_pad = torch.zeros_like(velocity[:, 0])
    velocity, _ = einops.pack([velocity, vel_pad], "batch * nbjoints dim")

    # repeat the previous velocity at the last valid frame of each sequence. Static-shape
    # formulation: an index_put driven by `lengths` is data-dependent and hostile to
    # torch.compile (this runs inside the compiled denoiser via global_root_to_local_root).
    time_idx = torch.arange(velocity.shape[1], device=device)
    is_last_valid = time_idx[None, :] == (lengths - 1)[:, None]  # [B, T]
    velocity_prev = torch.cat([torch.zeros_like(velocity[:, :1]), velocity[:, :-1]], dim=1)
    velocity = torch.where(is_last_valid[:, :, None, None], velocity_prev, velocity)
    return velocity


@ensure_batched(root_rot_angles=2, lengths=1)
def compute_vel_angle(
    root_rot_angles: torch.Tensor,
    fps: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the local root rotation velocity: dtheta/dt.

    Args:
        root_rot_angles (torch.Tensor): [..., T] rotation angle (in radian)
        fps (float): frame per seconds
        lengths (Optional[torch.Tensor]): [...] size of each input batched. If not provided, root_rot_angles should not be batched

    Returns:
        local_root_rot_vel (torch.Tensor): [..., T] local root rotation velocity (in radian/s)
    """
    device = root_rot_angles.device
    if lengths is None:
        assert root_rot_angles.shape[0] == 1, "If lengths is not provided, the input should not be batched."
        lengths = torch.tensor([len(root_rot_angles)], device=device)

    local_root_rot_vel = diff_angles(root_rot_angles, fps)
    pad_rot_vel_angles = torch.zeros_like(root_rot_angles[:, 0])
    local_root_rot_vel, _ = einops.pack(
        [local_root_rot_vel, pad_rot_vel_angles],
        "batch *",
    )
    # repeat the previous value at the last valid frame (static-shape formulation, see
    # compute_vel_xyz)
    time_idx = torch.arange(local_root_rot_vel.shape[1], device=device)
    is_last_valid = time_idx[None, :] == (lengths - 1)[:, None]  # [B, T]
    vel_prev = torch.cat([torch.zeros_like(local_root_rot_vel[:, :1]), local_root_rot_vel[:, :-1]], dim=1)
    local_root_rot_vel = torch.where(is_last_valid, vel_prev, local_root_rot_vel)
    return local_root_rot_vel


@ensure_batched(posed_joints=4)
def compute_heading_angle(posed_joints: torch.Tensor, skeleton: SkeletonBase) -> torch.Tensor:
    """Compute the heading direction from joint positions using the hip vector.

    Args:
        posed_joints: [B, T, J, 3] global joint positions.
        skeleton: Skeleton instance used to get hip joint indices.

    Returns:
        [B] heading angle in radians.
    """
    # compute root heading for the sequence from hip positions
    r_hip, l_hip = skeleton.hip_joint_idx
    diff = posed_joints[:, :, r_hip] - posed_joints[:, :, l_hip]
    heading_angle = torch.atan2(diff[..., 2], -diff[..., 0])
    return heading_angle


def length_to_mask(
    length: torch.Tensor | list,
    max_len: int | None = None,
    device=None,
) -> torch.Tensor:
    """Convert sequence lengths to a boolean validity mask.

    Args:
        length: Sequence lengths, either a tensor ``[B]`` or a Python list.
        max_len: Optional mask width. If omitted, uses ``max(length)``.
        device: Optional device. When ``length`` is a list, this controls where
            the new tensor is created.

    Returns:
        A boolean tensor of shape ``[B, max_len]`` where ``True`` marks valid
        timesteps.
    """
    if isinstance(length, list):
        if device is None:
            device = "cpu"
        length = torch.tensor(length, device=device)

    # Use requested device for output; move length if needed so mask and length match
    if device is not None:
        target = torch.device(device)
        if length.device != target:
            length = length.to(target)
    device = length.device

    if max_len is None:
        max_len = max(length)

    mask = torch.arange(max_len, device=device).expand(len(length), max_len) < length.unsqueeze(1)
    return mask


class RotateFeatures:
    """Helper that applies a global heading rotation to motion features."""

    def __init__(self, angle: torch.Tensor):
        """Precompute 2D and 3D rotation matrices for a batch of angles.

        Args:
            angle: Rotation angle(s) in radians, shaped ``[B]``.
        """
        self.angle = angle

        ## Create the necessary rotations matrices
        cos, sin = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones_like(angle), torch.zeros_like(angle)

        # 2D rotation transposed (sin are -sin)
        self.corrective_mat_2d_T = torch.stack((cos, sin, -sin, cos), -1).reshape(angle.shape + (2, 2))
        # 3D rotation on Y axis
        self.corrective_mat_Y = torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), -1).reshape(
            angle.shape + (3, 3)
        )
        self.corrective_mat_Y_T = self.corrective_mat_Y.transpose(-2, -1).contiguous()

    @staticmethod
    def _align_mat(mat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Insert singleton dims so the per-angle matrices broadcast against ``tensor``.

        ``mat`` has the angle's batch shape plus the trailing matrix dims; ``tensor`` is
        ``[*angle_shape, ..., C]``. A per-sample angle ``[B]`` applied to ``[B, T, J, 3]``
        needs ``[B, 1, 3, 3]`` (plain ``@`` would misalign ``T`` with ``B``); a per-frame
        angle ``[B, T]`` on the same tensor already aligns and is left unchanged.
        """
        pad = tensor.dim() - mat.dim()
        assert pad >= 0, f"angle batch shape {mat.shape[:-2]} has more dims than positions {tensor.shape}"
        return mat.reshape(mat.shape[:-2] + (1,) * pad + mat.shape[-2:])

    def rotate_positions(self, positions: torch.Tensor):
        """Rotate 3D positions around the Y axis."""
        return positions @ self._align_mat(self.corrective_mat_Y_T, positions)

    def rotate_2d_positions(self, positions_2d: torch.Tensor):
        """Rotate 2D ``(x, z)`` vectors in the ground plane."""
        return positions_2d @ self._align_mat(self.corrective_mat_2d_T, positions_2d)

    def rotate_rotations(self, rotations: torch.Tensor):
        """Left-multiply global rotation matrices by the heading correction."""
        # "Rotate" the global rotations
        # which means add an extra Y rotation after the transform
        # so at the left R' = R_y R
        # (since we use the convention x' = R x)
        # "bik,btdkj->btdij"

        B, T, J = rotations.shape[:3]
        BTJ = B * T * J
        return (
            self.corrective_mat_Y[:, None, None].expand(B, T, J, 3, 3).reshape(BTJ, 3, 3) @ rotations.reshape(BTJ, 3, 3)
        ).reshape(B, T, J, 3, 3)

    def rotate_6d_rotations(self, rotations_6d: torch.Tensor):
        """Rotate 6D rotation features via matrix conversion."""
        return matrix_to_cont6d(self.rotate_rotations(cont6d_to_matrix(rotations_6d)))
