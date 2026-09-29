from __future__ import annotations
import torch


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (B, N, J, 3).
        fps (float).
    Returns:
        jitter (B, N-3).
    """
    pred_jitter = torch.norm(
        (joints[..., 3:, :, :] - 3 * joints[..., 2:-1, :, :] + 3 * joints[..., 1:-2, :, :] - joints[..., :-3, :, :])
        * (fps**3),
        dim=-1,
    ).mean(dim=-1)

    return pred_jitter.mean()
