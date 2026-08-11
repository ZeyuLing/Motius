"""Load pre-processed o6dp_1103 motion representation.

The o6dp_1103 format stores absolute-translation + rot6d + root-invariant
joint coordinates (RIC) as a flat NumPy array.

For 52 joints (``joints_num=52``), the o6dp_1103 layout is 471 dims:
  - ``[0:3]``    abs translation (3)
  - ``[3:9]``    root global rot6d (6)
  - ``[9:315]``  body local rot6d ((52-1)*6=306)
  - ``[315:471]`` RIC joints 3D (52*3=156)

For 22 joints (``joints_num=22``), the o6dp_1103 layout is 201 dims:
  - ``[0:3]``    abs translation (3)
  - ``[3:9]``    root global rot6d (6)
  - ``[9:135]``  body local rot6d ((22-1)*6=126)
  - ``[135:201]`` RIC joints 3D (22*3=66)

This transform loads preprocessed npy files and extracts the requested joint
subset.  It also supports the official HY-Motion T2M training path where
``motions_o6dp_v0922`` files are 272/632-dim and are reformulated to the
201-dim o6dp_1103 training target inside the dataset.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


def _rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert row-major 6D rotations into orthonormal matrices."""

    columns = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    first = F.normalize(columns[..., 0], dim=-1)
    second = columns[..., 1]
    second = F.normalize(
        second - (first * second).sum(dim=-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _recover_root_translation(
    yaw_velocity: torch.Tensor,
    xz_velocity_body: torch.Tensor,
    y_translation: torch.Tensor,
    root_rotation_init: torch.Tensor,
) -> torch.Tensor:
    """Integrate the public HY-Motion T2M root-velocity representation."""

    local_x = root_rotation_init[..., 0]
    local_z = root_rotation_init[..., 2]
    forward = F.normalize(local_z[[0, 2]], dim=-1, eps=1e-8)
    fallback = F.normalize(
        torch.stack((-local_x[2], local_x[0])), dim=-1, eps=1e-8
    )
    yaw = torch.where(
        local_z[1].abs() > 0.50,
        torch.atan2(fallback[0], fallback[1]),
        torch.atan2(forward[0], forward[1]),
    )

    yaw_delta = yaw_velocity.squeeze(-1)
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    if yaw_delta.shape[0] >= 3:
        yaw_delta = torch.clamp(yaw_delta, -math.radians(60.0), math.radians(60.0))
        yaw_delta = F.avg_pool1d(
            F.pad(yaw_delta[None, None], (1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )[0, 0]

    yaw_sequence = yaw + torch.cumsum(yaw_delta, dim=0)
    yaw_step = yaw_sequence[1:] - yaw_sequence[:-1]
    yaw_step = (yaw_step + torch.pi) % (2 * torch.pi) - torch.pi
    yaw_sequence = torch.cat(
        [yaw_sequence[:1], yaw_sequence[:1] + torch.cumsum(yaw_step, dim=0)]
    )

    velocity = xz_velocity_body
    if velocity.shape[0] >= 3:
        velocity = F.avg_pool1d(
            F.pad(velocity.T[None], (1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )[0].T

    cos_yaw = torch.cos(yaw_sequence[:-1])
    sin_yaw = torch.sin(yaw_sequence[:-1])
    x_world = cos_yaw * velocity[:-1, 0] + sin_yaw * velocity[:-1, 1]
    z_world = -sin_yaw * velocity[:-1, 0] + cos_yaw * velocity[:-1, 1]
    xz_world = torch.stack((x_world, z_world), dim=-1)
    xz_translation = torch.zeros(
        yaw_sequence.shape[0], 2, dtype=velocity.dtype, device=velocity.device
    )
    xz_translation[1:] = torch.cumsum(xz_world, dim=0)
    return torch.cat(
        (xz_translation[:, :1], y_translation, xz_translation[:, 1:]), dim=-1
    )


def _extract_22j_from_52j(motion_52j: np.ndarray) -> np.ndarray:
    """Extract 22-joint 201-dim representation from 52-joint 471-dim.

    Args:
        motion_52j: (T, 471) array in o6dp_1103 52-joint format.

    Returns:
        (T, 201) array in o6dp_1103 22-joint format.
    """
    T = motion_52j.shape[0]

    # Parse 52-joint layout
    transl = motion_52j[:, 0:3]            # (T, 3)
    root_rot6d = motion_52j[:, 3:9]        # (T, 6)
    body_rot6d_52 = motion_52j[:, 9:315]   # (T, 306) = 51 joints * 6
    ric_52 = motion_52j[:, 315:471]        # (T, 156) = 52 joints * 3

    # Extract first 21 body joints (skip hand joints after index 21)
    body_rot6d_22 = body_rot6d_52[:, :21 * 6]  # (T, 126)
    # Extract first 22 RIC joints
    ric_22 = ric_52[:, :22 * 3]                 # (T, 66)

    # Concatenate: [transl(3), root_rot6d(6), body_rot6d(126), ric(66)] = 201
    return np.concatenate([transl, root_rot6d, body_rot6d_22, ric_22], axis=-1)


def _reformulate_0922_to_1103_22j(
    motion_0922: np.ndarray,
    motion_fps: int = 30,
) -> np.ndarray:
    """Convert official 272/632-dim o6dp_0922 files to 201-dim o6dp_1103."""
    motion = torch.from_numpy(motion_0922).float()
    D = motion.shape[1]
    if D not in (272, 632):
        raise ValueError(f'Expected 272/632-dim o6dp_0922 motion, got {D}')

    translation_vel = motion[:, :4].clone()
    non_translation_part = motion[:, 4:].clone()
    root_rotmat_init = _rot6d_to_rotation_matrix(
        non_translation_part[:, 0:6].clone()
    )
    translation_reconstructed_xyz = _recover_root_translation(
        translation_vel[:, 0:1].clone(),
        translation_vel[:, 1:3].clone(),
        translation_vel[:, 3:4].clone(),
        root_rotmat_init[0, ...].clone(),
    )

    num_joints = 52 if D == 632 else 22
    o6d_start = 10
    rifke_start = o6d_start + (num_joints - 1) * 6

    motion_1103 = torch.cat(
        [
            translation_reconstructed_xyz,
            motion[:, 4:10],
            motion[:, o6d_start:o6d_start + 21 * 6],
            motion[:, rifke_start:rifke_start + 22 * 3],
        ],
        dim=1,
    )
    # Keep the signature close to the official reformulation.  The fps is only
    # needed by o6dp_1103_rel, which this training transform intentionally
    # does not expose.
    _ = motion_fps
    return motion_1103.numpy().astype(np.float32)


@TRANSFORMS.register_module(force=True)
class LoadO6dp(BaseTransform):
    """Load pre-processed o6dp_1103 motion npy files.

    Supports loading both 22-joint (201-dim) and 52-joint (471-dim) npy files.
    When ``joints_num=22`` and the file is 471-dim, automatically extracts
    the 22-joint subset.

    Parameters
    ----------
    key : str
        Key to read the motion path from ``results[f'{key}_path']``
        and write the result to ``results[key]``.
    joints_num : int
        Target number of joints: 22 (body-only, 201-dim) or 52 (with hands, 471-dim).
    transl_aug_prob : float
        Probability of applying Y-axis rotation augmentation.
    transl_aug_yaw_deg : float
        Max yaw rotation range in degrees for augmentation.
    transl_aug_offset_std : tuple
        Std of XZ-plane offset augmentation (Y forced to 0).
    motion_fps : int
        FPS used by the legacy o6dp_0922 -> o6dp_1103 reformulation.
    """

    def __init__(
        self,
        key: str = 'motion',
        joints_num: int = 22,
        transl_aug_prob: float = 0.75,
        transl_aug_yaw_deg: float = 180.0,
        transl_aug_offset_std: Tuple[float, float, float] = (1.0, 0.0, 1.0),
        motion_fps: int = 30,
    ):
        super().__init__()
        assert joints_num in (22, 52), f"joints_num must be 22 or 52, got {joints_num}"
        self.key = key
        self.joints_num = joints_num
        self.expected_dim = 3 + 6 + (joints_num - 1) * 6 + joints_num * 3
        # 22 joints -> 201, 52 joints -> 471
        self.motion_fps = int(motion_fps)

        self.transl_aug_prob = float(transl_aug_prob)
        self.transl_aug_yaw_deg = float(transl_aug_yaw_deg)
        self.transl_aug_offset_std = np.asarray(transl_aug_offset_std, dtype=np.float32)

    def _sample_augmentation(self):
        """Sample Y-axis rotation and XZ offset augmentation."""
        do_aug = (self.transl_aug_prob > 0.0) and (
            np.random.rand() < self.transl_aug_prob
        )
        if not do_aug:
            return False, 0.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        yaw_deg = float(np.random.uniform(-self.transl_aug_yaw_deg, self.transl_aug_yaw_deg))
        yaw = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

        sx, _, sz = self.transl_aug_offset_std
        offset = np.array(
            [np.random.normal(0, float(sx)), 0.0, np.random.normal(0, float(sz))],
            dtype=np.float32,
        )
        return True, yaw_deg, R_y, offset

    def _apply_augmentation(
        self,
        motion: np.ndarray,
        R_y: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        """Apply Y-axis rotation + XZ offset to o6dp_1103 motion.

        Rotates translation, root rotation, and RIC joint positions.
        Body-relative rotations are unchanged (they are parent-relative).
        """
        T = motion.shape[0]
        J = self.joints_num
        D = self.expected_dim

        motion = motion.copy()

        # 1. Rotate and offset translation
        transl = motion[:, 0:3]  # (T, 3)
        transl = transl @ R_y.T + offset[None, :]
        motion[:, 0:3] = transl

        # 2. Rotate root orientation (rot6d)
        # root_rot6d (6) represents first two columns of rotation matrix
        # For row-major: [R00,R01, R10,R11, R20,R21]
        # R_new = R_y @ R_old
        root6d = motion[:, 3:9].reshape(T, 3, 2)  # row-major: each row is [col0, col1]
        # Reconstruct full rotation from first two columns
        col0 = root6d[:, :, 0]  # (T, 3)
        col1 = root6d[:, :, 1]  # (T, 3)
        col0_new = (R_y[None, :, :] @ col0[:, :, None]).squeeze(-1)
        col1_new = (R_y[None, :, :] @ col1[:, :, None]).squeeze(-1)
        root6d_new = np.stack([col0_new, col1_new], axis=-1).reshape(T, 6)
        motion[:, 3:9] = root6d_new

        # 3. Rotate RIC joint positions
        ric_start = 3 + 6 + (J - 1) * 6
        ric = motion[:, ric_start:ric_start + J * 3].reshape(T, J, 3)
        ric = (R_y[None, None, :, :] @ ric[:, :, :, None]).squeeze(-1)
        motion[:, ric_start:ric_start + J * 3] = ric.reshape(T, J * 3)

        return motion

    def transform(self, results: Dict) -> Dict:
        path = results[f'{self.key}_path']
        if isinstance(path, (list, tuple)):
            raise NotImplementedError("LoadO6dp does not support multi-person loading yet.")

        path = str(path)
        motion = np.load(path).astype(np.float32)  # (T, D)

        # Handle dimension mismatch: extract 22-joint from 52-joint
        if motion.shape[1] in (272, 632) and self.joints_num == 22:
            motion = _reformulate_0922_to_1103_22j(motion, motion_fps=self.motion_fps)
        elif motion.shape[1] == 471 and self.joints_num == 22:
            motion = _extract_22j_from_52j(motion)
        elif motion.shape[1] != self.expected_dim:
            raise ValueError(
                f"LoadO6dp: expected {self.expected_dim}-dim motion, got {motion.shape[1]}-dim "
                f"from {path}. Set joints_num appropriately."
            )

        # Augmentation
        do_aug, yaw_deg, R_y, offset = self._sample_augmentation()
        if do_aug:
            motion = self._apply_augmentation(motion, R_y, offset)

        out = torch.from_numpy(motion)
        if torch.any(torch.isnan(out)):
            raise ValueError(f"NaN in {path} after loading/augmentation.")

        results[self.key] = out  # (T, D)
        results['num_person'] = 1
        results['num_frames'] = int(motion.shape[0])
        results['num_joints'] = self.joints_num
        results['aug_yaw_deg'] = yaw_deg if do_aug else 0.0
        results['aug_offset'] = offset.tolist() if do_aug else [0.0, 0.0, 0.0]
        return results
