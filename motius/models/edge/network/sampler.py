"""Official EDGE DDIM and overlapping-window stitching protocol."""

from __future__ import annotations

import math

import torch

from motius.motion.representation.rotation import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from .motion import matrix_to_rotation_6d, rotation_6d_to_matrix


def cosine_alphas_cumprod(
    steps: int = 1_000, cosine_s: float = 8e-3
) -> torch.Tensor:
    timeline = torch.arange(steps + 1, dtype=torch.float64) / steps + cosine_s
    alphas = torch.cos(timeline / (1.0 + cosine_s) * math.pi / 2.0).square()
    alphas = alphas / alphas[0]
    betas = (1.0 - alphas[1:] / alphas[:-1]).clamp(0.0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


def _randn(shape, *, device, dtype, generator):
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


@torch.inference_mode()
def edge_ddim_sample(
    model,
    condition: torch.Tensor,
    *,
    representation_dim: int = 151,
    guidance_weight: float = 2.0,
    sampling_steps: int = 50,
    eta: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample normalized 5-second EDGE windows with the released sampler."""

    if condition.ndim != 3 or condition.shape[1:] != (150, 4_800):
        raise ValueError(f"EDGE condition must have shape (B,150,4800), got {condition.shape}")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    batch = len(condition)
    device = condition.device
    dtype = next(model.parameters()).dtype
    condition = condition.to(dtype=dtype)
    alphas = cosine_alphas_cumprod().to(device=device, dtype=dtype)
    times = torch.linspace(-1, len(alphas) - 1, sampling_steps + 1).int().tolist()[::-1]
    pairs = list(zip(times[:-1], times[1:]))
    weights = torch.linspace(0.0, guidance_weight * 2.0, sampling_steps).clamp_max(
        guidance_weight
    )
    sample = _randn(
        (batch, 150, representation_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    long_mode = batch > 1
    for index, (time, next_time) in enumerate(pairs):
        timestep = torch.full((batch,), time, device=device, dtype=torch.long)
        weight = float(weights[index]) if long_mode else float(guidance_weight)
        predicted = model.guided_forward(sample, condition, timestep, weight).clamp(-1, 1)
        if next_time < 0:
            sample = predicted
            continue
        alpha = alphas[time]
        alpha_next = alphas[next_time]
        sigma = eta * torch.sqrt(
            (1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)
        )
        coefficient = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0))
        predicted_noise = (
            torch.sqrt(1.0 / alpha) * sample - predicted
        ) / torch.sqrt(1.0 / alpha - 1.0)
        noise = _randn(sample.shape, device=device, dtype=dtype, generator=generator)
        sample = predicted * torch.sqrt(alpha_next) + coefficient * predicted_noise + sigma * noise
        if long_mode and time > 0:
            sample[1:, :75] = sample[:-1, 75:]
    return sample


def _quaternion_slerp(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor):
    dot = (left * right).sum(dim=-1, keepdim=True)
    right = torch.where(dot < 0, -right, right)
    dot = dot.abs().clamp(max=1.0)
    close = (1.0 - dot) < 0.01
    omega = torch.acos(dot)
    denominator = torch.sin(omega).clamp_min(1e-8)
    amount_left = torch.sin((1.0 - weight) * omega) / denominator
    amount_right = torch.sin(weight * omega) / denominator
    spherical = amount_left * left + amount_right * right
    linear = (1.0 - weight) * left + weight * right
    return torch.nn.functional.normalize(torch.where(close, linear, spherical), dim=-1)


def stitch_edge_windows(windows: torch.Tensor) -> torch.Tensor:
    """Stitch unnormalized EDGE windows with the official 50% overlap."""

    if windows.ndim != 3 or windows.shape[1:] != (150, 151):
        raise ValueError(f"Expected EDGE windows (N,150,151), got {windows.shape}")
    if len(windows) == 1:
        return windows[0]
    half = 75
    count = len(windows)
    output_frames = 150 + half * (count - 1)
    fade_out = torch.ones((150, 1), device=windows.device, dtype=windows.dtype)
    fade_in = torch.ones_like(fade_out)
    fade_out[half:] = torch.linspace(1, 0, half, device=windows.device, dtype=windows.dtype)[:, None]
    fade_in[:half] = torch.linspace(0, 1, half, device=windows.device, dtype=windows.dtype)[:, None]

    contacts = windows[..., :4]
    root = windows[..., 4:7].clone()
    rotations = rotation_6d_to_matrix(windows[..., 7:].reshape(count, 150, 24, 6))
    root[:-1] *= fade_out
    root[1:] *= fade_in
    full_root = torch.zeros((output_frames, 3), device=windows.device, dtype=windows.dtype)
    full_contacts = torch.zeros((output_frames, 4), device=windows.device, dtype=windows.dtype)
    contact_weight = torch.zeros((output_frames, 1), device=windows.device, dtype=windows.dtype)
    for index in range(count):
        start = index * half
        full_root[start : start + 150] += root[index]
        full_contacts[start : start + 150] += contacts[index]
        contact_weight[start : start + 150] += 1
    full_contacts /= contact_weight

    left = matrix_to_quaternion(rotations[:-1, half:])
    right = matrix_to_quaternion(rotations[1:, :half])
    weight = torch.linspace(0, 1, half, device=windows.device, dtype=windows.dtype)
    merged = quaternion_to_matrix(
        _quaternion_slerp(left, right, weight[None, :, None, None])
    )
    full_rotations = torch.zeros(
        (output_frames, 24, 3, 3), device=windows.device, dtype=windows.dtype
    )
    full_rotations[:half] = rotations[0, :half]
    cursor = half
    for overlap in merged:
        full_rotations[cursor : cursor + half] = overlap
        cursor += half
    full_rotations[cursor : cursor + half] = rotations[-1, half:]
    rotation6d = matrix_to_rotation_6d(full_rotations).reshape(output_frames, 144)
    return torch.cat((full_contacts, full_root, rotation6d), dim=-1)


__all__ = ["cosine_alphas_cumprod", "edge_ddim_sample", "stitch_edge_windows"]
