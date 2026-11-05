"""HumanML3D feature masks for CondMDI motion control."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch


JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
LOWER_BODY = (0, 1, 2, 4, 5, 7, 8, 10, 11)
PELVIS_FEET = (0, 10, 11)
PELVIS_VR = (0, 15, 20, 21)


def _feature_correspondence():
    pos = np.zeros((22, 263), dtype=bool)
    pos[0, 1:4] = True
    for joint in range(1, 22):
        pos[joint, 4 + 3 * (joint - 1): 4 + 3 * joint] = True

    rot = np.zeros((22, 263), dtype=bool)
    rot[0, 0] = True
    rot_start = 4 + 21 * 3
    for joint in range(1, 22):
        rot[joint, rot_start + 6 * (joint - 1): rot_start + 6 * joint] = True

    vel = np.zeros((22, 263), dtype=bool)
    vel_start = rot_start + 21 * 6
    for joint in range(22):
        vel[joint, vel_start + 3 * joint: vel_start + 3 * (joint + 1)] = True

    contact = np.zeros((22, 263), dtype=bool)
    contact[7, -4] = contact[10, -3] = contact[8, -2] = contact[11, -1] = True
    return pos, rot, vel, contact


MAT_POS, MAT_ROT, MAT_VEL, MAT_CONTACT = _feature_correspondence()


def joint_mask_to_feature_mask(joint_mask: torch.Tensor, feature_mode: str = "pos_rot_vel"):
    if feature_mode not in {"pos", "pos_rot", "pos_rot_vel"}:
        raise ValueError("feature_mode must be pos, pos_rot, or pos_rot_vel")
    matrices = [MAT_POS, MAT_CONTACT]
    if feature_mode in {"pos_rot", "pos_rot_vel"}:
        matrices.append(MAT_ROT)
    if feature_mode == "pos_rot_vel":
        matrices.append(MAT_VEL)
    mapping = torch.from_numpy(np.logical_or.reduce(matrices)).to(joint_mask.device)
    return torch.einsum("bjt,jf->bft", joint_mask.float(), mapping.float()).bool().unsqueeze(2)


def build_observation_mask(
    lengths: Sequence[int],
    n_frames: int,
    mode: str = "none",
    transition_length: int = 10,
    feature_mode: str = "pos_rot_vel",
    joint_indices: Optional[Iterable[int]] = None,
    keyframe_indices: Optional[Sequence[Sequence[int]]] = None,
):
    """Build a ``(B,263,1,T)`` feature mask for common CondMDI controls."""
    mask = torch.zeros((len(lengths), 22, n_frames), dtype=torch.bool)
    for batch_index, length_value in enumerate(lengths):
        length = max(1, min(int(length_value), n_frames))
        if keyframe_indices is not None:
            frames = [int(x) for x in keyframe_indices[batch_index] if 0 <= int(x) < length]
            joints = range(22) if joint_indices is None else joint_indices
            for joint in joints:
                mask[batch_index, int(joint), frames] = True
            continue
        if mode in {"none", "uncond"}:
            continue
        if mode == "start":
            frames = [0]
        elif mode == "first_last":
            frames = [0, length - 1]
        elif mode == "sparse":
            frames = list(range(0, length, max(1, int(transition_length))))
        elif mode == "prefix":
            frames = list(range(min(length, int(transition_length))))
        elif mode == "suffix":
            frames = list(range(max(0, length - int(transition_length)), length))
        elif mode == "middle":
            width = min(length, int(transition_length))
            start = (length - width) // 2
            frames = list(range(0, start)) + list(range(start + width, length))
        elif mode in {"trajectory", "pelvis"}:
            mask[batch_index, 0, :length] = True
            continue
        elif mode == "lower_body":
            mask[batch_index, list(LOWER_BODY), :length] = True
            continue
        elif mode == "pelvis_feet":
            mask[batch_index, list(PELVIS_FEET), :length] = True
            continue
        elif mode == "pelvis_vr":
            mask[batch_index, list(PELVIS_VR), :length] = True
            continue
        elif mode == "joints":
            if joint_indices is None:
                raise ValueError("mode='joints' requires joint_indices")
            mask[batch_index, list(joint_indices), :length] = True
            continue
        else:
            raise ValueError(f"unsupported CondMDI control mode: {mode}")
        joints = list(range(22) if joint_indices is None else joint_indices)
        if frames:
            for joint in joints:
                mask[batch_index, joint, frames] = True
    return joint_mask_to_feature_mask(mask, feature_mode=feature_mode)


__all__ = [
    "JOINT_NAMES",
    "build_observation_mask",
    "joint_mask_to_feature_mask",
]
