"""Conversions between standard HML263 and CondMDI's absolute-root variant."""

from __future__ import annotations

import torch

def _qrot(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quaternion[..., 1:]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2 * (quaternion[..., :1] * uv + uuv)


def relative_to_absolute(motion: torch.Tensor) -> torch.Tensor:
    """Convert physical-scale standard HML263 to CondMDI absolute-root HML263."""
    if motion.shape[-1] != 263:
        raise ValueError(f"expected (..., T, 263), got {tuple(motion.shape)}")
    root_angle = torch.zeros_like(motion[..., 0])
    root_angle[..., 1:] = motion[..., :-1, 0]
    root_angle = torch.cumsum(root_angle, dim=-1)
    root_quat = torch.zeros(root_angle.shape + (4,), dtype=motion.dtype, device=motion.device)
    root_quat[..., 0] = torch.cos(root_angle)
    root_quat[..., 2] = torch.sin(root_angle)
    root_pos = torch.zeros(motion.shape[:-1] + (3,), dtype=motion.dtype, device=motion.device)
    root_pos[..., 1:, [0, 2]] = motion[..., :-1, 1:3]
    inverse_quat = root_quat.clone()
    inverse_quat[..., 1:] *= -1
    root_pos = _qrot(inverse_quat, root_pos)
    root_pos = torch.cumsum(root_pos, dim=-2)
    root_pos[..., 1] = motion[..., 3]
    output = motion.clone()
    output[..., 0] = root_angle
    output[..., 1:3] = root_pos[..., [0, 2]]
    return output


def absolute_to_relative(motion: torch.Tensor) -> torch.Tensor:
    """Convert physical-scale CondMDI absolute-root HML263 to standard HML263."""
    if motion.shape[-1] != 263:
        raise ValueError(f"expected (..., T, 263), got {tuple(motion.shape)}")
    output = motion.clone()
    root_angle = motion[..., 0]
    root_pos = motion[..., 1:4][..., [0, 2, 1]]

    relative_pos = torch.zeros_like(root_pos)
    relative_pos[..., 1:, [0, 2]] = root_pos[..., 1:, [0, 2]] - root_pos[..., :-1, [0, 2]]
    root_quat = torch.zeros(root_angle.shape + (4,), dtype=motion.dtype, device=motion.device)
    root_quat[..., 0] = torch.cos(root_angle)
    root_quat[..., 2] = torch.sin(root_angle)
    relative_pos = _qrot(root_quat, relative_pos)
    relative_pos[..., :-1, :] = relative_pos[..., 1:, :].clone()
    relative_pos[..., 1] = motion[..., 3]

    relative_angle = torch.zeros_like(root_angle)
    relative_angle[..., :-1] = root_angle[..., 1:] - root_angle[..., :-1]
    output[..., 0] = relative_angle
    output[..., 1:4] = relative_pos[..., [0, 2, 1]]
    return output


__all__ = ["absolute_to_relative", "relative_to_absolute"]
