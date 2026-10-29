"""Dataset-statistic transforms used by motion Pipelines and Trainers."""
from __future__ import annotations
import numpy as np
import torch


def affine_normalize(motion, mean, std):
    return (motion - mean) / std


def affine_denormalize(motion, mean, std):
    return motion * std + mean


def matched_denormalize(motion, mean, std):
    """Match the input dtype/device before applying FlowMDM statistics."""
    return motion * std.to(motion) + mean.to(motion)


def range_denormalize(motion, offset, scale):
    """Invert EDGE's clamped range transform in its original operation order."""
    motion = motion.clamp(-1, 1)
    return (motion - offset.to(motion)) / scale.to(motion)


def array_normalize(motion, mean, std):
    if isinstance(motion, np.ndarray):
        return (motion - mean.detach().cpu().numpy()) / std.detach().cpu().numpy()
    motion = torch.as_tensor(motion, device=mean.device)
    return (motion - mean) / std


def array_denormalize(motion, mean, std):
    if isinstance(motion, np.ndarray):
        return motion * std.detach().cpu().numpy() + mean.detach().cpu().numpy()
    motion = torch.as_tensor(motion, device=mean.device)
    return motion * std + mean
