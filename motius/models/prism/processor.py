"""PRISM's native 138D SMPL-H body representation processor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


def _stats_block(stats: dict, name: str, subkey: Optional[str] = None):
    block = stats[name]
    if subkey is not None:
        block = block.get(subkey, block.get("rot6d"))
    return block["mean"], block["std"]


class PRISMMotionProcessor(nn.Module):
    """Normalize and decode PRISM motion138 tensors.

    The native tensor is ``[absolute translation (3), translation velocity
    (3), 22 local body rotations in column-major rot6d (132)]``.
    """

    def __init__(self, stats_file: str, eps: float = 1e-6):
        super().__init__()
        self.stats_file = str(stats_file)
        stats = json.loads(Path(stats_file).read_text())
        means, stds = [], []
        for name, subkey in (
            ("transl", None),
            ("transl_vel", None),
            ("global_orient", "rotation_6d"),
            ("body_pose", "rotation_6d"),
        ):
            mean, std = _stats_block(stats, name, subkey)
            means.extend(mean)
            stds.extend(std)
        mean = torch.tensor(means, dtype=torch.float32)
        std = torch.tensor(stds, dtype=torch.float32).clamp_min(float(eps))
        if mean.shape != (138,) or std.shape != (138,):
            raise ValueError(
                f"PRISM motion stats must have 138 values, got {mean.shape}/{std.shape}"
            )
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean.to(motion)) / self.std.to(motion)

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std.to(motion) + self.mean.to(motion)

    @staticmethod
    def inv_convert_transl(
        transl: Union[np.ndarray, torch.Tensor],
        use_rollout: Union[bool, str] = "xz_rollout_y_absolute",
    ):
        mode = use_rollout.lower().replace("-", "_") if isinstance(use_rollout, str) else use_rollout
        if mode is False or mode == "absolute":
            return transl[..., :3]
        pos0 = transl[..., :1, :3]
        velocity = transl[..., 1:, 3:]
        if torch.is_tensor(transl):
            rollout = torch.cumsum(torch.cat([pos0, velocity], dim=-2), dim=-2)
        else:
            rollout = np.cumsum(np.concatenate([pos0, velocity], axis=-2), axis=-2)
        if mode is True or mode == "rollout":
            return rollout
        if mode in {"xz_rollout_y_absolute", "xz_rollout_y_abs", "hybrid"}:
            output = rollout.clone() if torch.is_tensor(rollout) else rollout.copy()
            output[..., 1] = transl[..., :3][..., 1]
            return output
        raise ValueError(f"Unsupported PRISM translation decode mode: {use_rollout}")

    def smplx_dict_to_motion_vector(self, smplx_dict: dict) -> torch.Tensor:
        transl = np.asarray(smplx_dict["transl"], dtype=np.float32)
        global_orient = np.asarray(smplx_dict["global_orient"], dtype=np.float32)
        body_pose = np.asarray(smplx_dict["body_pose"], dtype=np.float32)
        body_pose = body_pose.reshape(len(transl), 21, 3)
        axis_angle = np.concatenate([global_orient[:, None], body_pose], axis=1)
        rot6d = matrix_to_rotation_6d(
            axis_angle_to_matrix(axis_angle), convention="column"
        ).reshape(len(transl), 132)
        velocity = np.zeros_like(transl)
        velocity[1:] = transl[1:] - transl[:-1]
        motion = np.concatenate([transl, velocity, rot6d], axis=-1)
        return torch.from_numpy(motion.astype(np.float32, copy=False))

    @staticmethod
    def transl_pose_to_smplx_dict(
        transl,
        poses,
        mocap_framerate: float = 30.0,
        betas=None,
        expression=None,
        gender: str = "neutral",
        rot_type: str = "axis_angle",
        to_numpy: bool = True,
    ) -> dict:
        if rot_type != "axis_angle":
            raise ValueError("PRISM pipeline passes axis-angle poses to this method")
        if torch.is_tensor(transl):
            transl = transl.detach().cpu().numpy()
        if torch.is_tensor(poses):
            poses = poses.detach().cpu().numpy()
        transl = np.asarray(transl, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        if poses.shape[-1] != 66:
            raise ValueError(f"Expected 22 body joints (66D axis-angle), got {poses.shape}")
        frames = len(poses)
        full_pose = np.concatenate(
            [poses, np.zeros((frames, 99), dtype=np.float32)], axis=-1
        )
        return {
            "trans": transl,
            "transl": transl,
            "poses": full_pose,
            "global_orient": full_pose[:, :3],
            "body_pose": full_pose[:, 3:66],
            "jaw_pose": full_pose[:, 66:69],
            "leye_pose": full_pose[:, 69:72],
            "reye_pose": full_pose[:, 72:75],
            "left_hand_pose": full_pose[:, 75:120],
            "right_hand_pose": full_pose[:, 120:165],
            "gender": gender,
            "betas": np.zeros((10,), dtype=np.float32) if betas is None else betas,
            "expression": (
                np.zeros((frames, 10), dtype=np.float32)
                if expression is None
                else expression
            ),
            "mocap_framerate": float(mocap_framerate),
        }

    @staticmethod
    def normalize_smplx_dict(smplx_dict: dict, smplx_model=None) -> dict:
        """Canonicalize frame-0 yaw/XZ and put the lowest root height at zero."""
        global_orient = np.asarray(smplx_dict["global_orient"], dtype=np.float64)
        transl = np.asarray(smplx_dict["transl"], dtype=np.float64).copy()
        root_rot = axis_angle_to_matrix(global_orient)
        forward = root_rot[0] @ np.array([0.0, 0.0, 1.0])
        yaw = np.arctan2(forward[0], forward[2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        yaw_rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        from scipy.spatial.transform import Rotation

        corrected = Rotation.from_matrix(yaw_rot[None] @ root_rot).as_rotvec()
        transl = (yaw_rot @ transl.T).T
        transl[:, 0] -= transl[0, 0]
        transl[:, 2] -= transl[0, 2]
        transl[:, 1] -= transl[:, 1].min()
        poses = np.asarray(smplx_dict["poses"], dtype=np.float32).copy()
        poses[:, :3] = corrected.astype(np.float32)
        smplx_dict.update(
            global_orient=corrected.astype(np.float32),
            transl=transl.astype(np.float32),
            trans=transl.astype(np.float32),
            poses=poses,
        )
        return smplx_dict

    def post_hoc_static_refine(self, *args, **kwargs):
        raise RuntimeError("PRISM static refinement is not part of the public body-22 release")

    def smooth_smplx_dict(self, *args, **kwargs):
        raise RuntimeError("PRISM SmoothNet weights are not part of the public release")
