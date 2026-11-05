"""Shared EDGE noise schedule outside neural modules and inference orchestration."""

import math
import torch


def cosine_alphas_cumprod(steps: int = 1_000, cosine_s: float = 8e-3) -> torch.Tensor:
    timeline = torch.arange(steps + 1, dtype=torch.float64) / steps + cosine_s
    alphas = torch.cos(timeline / (1.0 + cosine_s) * math.pi / 2.0).square()
    alphas = alphas / alphas[0]
    betas = (1.0 - alphas[1:] / alphas[:-1]).clamp(0.0, 0.999)
    # Source converts betas to float32 before cumulative multiplication.
    return torch.cumprod(1.0 - betas.float(), dim=0)
