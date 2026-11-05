from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def flow_grpo_transition(model_output: Tensor, latents: Tensor, sigma: Tensor, sigma_next: Tensor, sigma_first_next: Tensor, eta: float, next_latents: Tensor | None=None, frame_mask: Tensor | None=None, transition_noise: Tensor | None=None) -> tuple[Tensor, Tensor]:
    """Sample/replay one stochastic Flow-GRPO transition and its log-prob.

    ``model_output`` is in decreasing-sigma coordinates. For this repository's
    rectified-flow model, callers pass ``-predict_flow(...)``.
    """
    if eta <= 0:
        raise ValueError('eta must be positive for a stochastic policy update')
    dtype = model_output.dtype
    device = model_output.device
    batch = model_output.shape[0]

    def expand(value: Tensor) -> Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch)
        return value.reshape(batch, *[1] * (model_output.ndim - 1))
    sigma_e = expand(sigma)
    sigma_next_e = expand(sigma_next)
    sigma_first_next_e = expand(sigma_first_next)
    dt = sigma_next_e - sigma_e
    if torch.any(dt >= 0):
        raise ValueError('sigma schedule must be strictly decreasing')
    safe_denom_sigma = torch.where(sigma_e == 1, sigma_first_next_e, sigma_e)
    std_dev = torch.sqrt(sigma_e / (1 - safe_denom_sigma).clamp_min(1e-06)) * float(eta)
    transition_std = std_dev * torch.sqrt((-dt).clamp_min(1e-12))
    mean = latents * (1 + std_dev.square() / (2 * sigma_e.clamp_min(1e-06)) * dt) + model_output * (1 + std_dev.square() * (1 - sigma_e) / (2 * sigma_e.clamp_min(1e-06))) * dt
    if next_latents is not None and transition_noise is not None:
        raise ValueError('next_latents and transition_noise are mutually exclusive')
    if next_latents is None:
        if transition_noise is None:
            transition_noise = torch.randn_like(mean)
        else:
            transition_noise = transition_noise.to(device=device, dtype=dtype)
            if transition_noise.shape != mean.shape:
                raise ValueError(f'transition_noise shape {tuple(transition_noise.shape)} does not match transition mean {tuple(mean.shape)}')
        next_latents = mean + transition_std * transition_noise
    per_element = -0.5 * ((next_latents.detach() - mean) / transition_std.clamp_min(1e-08)).square() - torch.log(transition_std.clamp_min(1e-08)) - 0.5 * math.log(2.0 * math.pi)
    if frame_mask is None:
        log_prob = per_element.flatten(1).mean(dim=1)
    else:
        mask = frame_mask.to(device=device, dtype=per_element.dtype)
        if mask.ndim == per_element.ndim - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(per_element)
        log_prob = (per_element * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1)
    return (next_latents, log_prob)
