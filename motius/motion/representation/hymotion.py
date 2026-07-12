"""HY-Motion-201 O6DP conversion helpers.

The official 201-dimensional layout is ``motion135`` followed by all 22
pelvis-relative SMPL body-joint positions. The pelvis triplet is retained and
is therefore zero by construction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .specs import HYMOTION201, MOTION135


def _as_tensor(value) -> tuple[Tensor, bool]:
    is_numpy = isinstance(value, np.ndarray)
    tensor = torch.as_tensor(value, dtype=torch.float32) if is_numpy else value
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected numpy.ndarray or torch.Tensor, got {type(value)!r}")
    return tensor, is_numpy


def _same_type(tensor: Tensor, as_numpy: bool):
    return tensor.detach().cpu().numpy() if as_numpy else tensor


def validate_hymotion201(motion, *, pelvis_eps: float | None = None):
    """Validate shape and, optionally, the redundant pelvis RIC channels."""

    tensor, is_numpy = _as_tensor(motion)
    if tensor.shape[-1] != HYMOTION201.dim:
        raise ValueError(f"expected last dim {HYMOTION201.dim}, got {tensor.shape[-1]}")
    if pelvis_eps is not None and tensor[..., 135:138].abs().max().item() > pelvis_eps:
        raise ValueError("HY-Motion-201 pelvis RIC channels [135:138] must be zero")
    return _same_type(tensor, is_numpy)


def hymotion201_to_motion135(motion):
    """Drop the redundant joint-position block from HY-Motion-201."""

    tensor, is_numpy = _as_tensor(motion)
    if tensor.shape[-1] != HYMOTION201.dim:
        raise ValueError(f"expected last dim {HYMOTION201.dim}, got {tensor.shape[-1]}")
    return _same_type(tensor[..., :MOTION135.dim], is_numpy)


def motion135_to_hymotion201(motion, bone_offsets):
    """Append 22 pelvis-relative FK joints to row-major ``motion135``."""

    tensor, is_numpy = _as_tensor(motion)
    if tensor.shape[-1] < MOTION135.dim:
        raise ValueError(f"expected at least {MOTION135.dim} channels, got {tensor.shape[-1]}")
    offsets = torch.as_tensor(bone_offsets, dtype=tensor.dtype, device=tensor.device)
    if offsets.shape != (22, 3):
        raise ValueError(f"bone_offsets must have shape (22,3), got {tuple(offsets.shape)}")

    from motius.motion.skeleton.fk import motion135_to_fk

    joints, _, _, _ = motion135_to_fk(tensor[..., :135], offsets)
    relative = joints - joints[..., :1, :]
    result = torch.cat([tensor[..., :135], relative.reshape(*tensor.shape[:-1], 66)], dim=-1)
    return _same_type(result, is_numpy)


def hymotion201_to_joints(motion, *, source: str = "stored", bone_offsets=None):
    """Decode HY-Motion-201 joints from stored RIC or from rotation-channel FK.

    ``source='stored'`` preserves the model's explicit position prediction.
    ``source='fk'`` checks rotation/position consistency and requires offsets.
    """

    tensor, is_numpy = _as_tensor(motion)
    if tensor.shape[-1] != HYMOTION201.dim:
        raise ValueError(f"expected last dim {HYMOTION201.dim}, got {tensor.shape[-1]}")
    if source == "stored":
        relative = tensor[..., 135:201].reshape(*tensor.shape[:-1], 22, 3)
        joints = relative + tensor[..., None, 0:3]
    elif source == "fk":
        if bone_offsets is None:
            raise ValueError("bone_offsets is required for source='fk'")
        offsets = torch.as_tensor(bone_offsets, dtype=tensor.dtype, device=tensor.device)
        from motius.motion.skeleton.fk import motion135_to_fk

        joints, _, _, _ = motion135_to_fk(tensor[..., :135], offsets)
    else:
        raise ValueError("source must be 'stored' or 'fk'")
    return _same_type(joints, is_numpy)


__all__ = [
    "validate_hymotion201",
    "hymotion201_to_motion135",
    "motion135_to_hymotion201",
    "hymotion201_to_joints",
]
