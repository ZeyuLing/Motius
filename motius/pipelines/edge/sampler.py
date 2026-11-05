"""Official EDGE DDIM and overlapping-window stitching protocol."""

from __future__ import annotations

import numpy as np

import torch

from motius.motion.representation.rotation import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from motius.motion.representation.edge import matrix_to_rotation_6d
from motius.motion.representation.edge import rotation_6d_to_matrix


from motius.diffusion.edge import cosine_alphas_cumprod


def _randn(shape, *, device, dtype, generator):
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def guided_prediction(model, sample, condition, times, guidance_weight):
    unconditional = model(sample, condition, times, cond_drop_prob=1)
    conditional = model(sample, condition, times, cond_drop_prob=0)
    return unconditional + (conditional - unconditional) * guidance_weight


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
        raise ValueError(
            f"EDGE condition must have shape (B,150,4800), got {condition.shape}"
        )
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    batch = len(condition)
    device = condition.device
    dtype = next(model.parameters()).dtype
    condition = condition.to(dtype=dtype)
    alphas = cosine_alphas_cumprod()
    sqrt_recip = torch.sqrt(1.0 / alphas).to(device=device, dtype=dtype)
    sqrt_recipm1 = torch.sqrt(1.0 / alphas - 1.0).to(device=device, dtype=dtype)
    alphas = alphas.to(device=device, dtype=dtype)
    times = torch.linspace(-1, len(alphas) - 1, sampling_steps + 1).int().tolist()[::-1]
    pairs = list(zip(times[:-1], times[1:]))
    weights = np.clip(
        np.linspace(0, guidance_weight * 2.0, sampling_steps), None, guidance_weight
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
        predicted = guided_prediction(model, sample, condition, timestep, weight).clamp(
            -1, 1
        )
        if next_time < 0:
            sample = predicted
            continue
        alpha = alphas[time]
        alpha_next = alphas[next_time]
        sigma = eta * torch.sqrt(
            (1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)
        )
        coefficient = torch.sqrt(1.0 - alpha_next - sigma**2)
        predicted_noise = (sqrt_recip[time] * sample - predicted) / sqrt_recipm1[time]
        noise = _randn(sample.shape, device=device, dtype=dtype, generator=generator)
        sample = (
            predicted * torch.sqrt(alpha_next)
            + coefficient * predicted_noise
            + sigma * noise
        )
        if long_mode and time > 0:
            sample[1:, :75] = sample[:-1, 75:]
    return sample


def _quaternion_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res


def stitch_edge_windows(windows: torch.Tensor, *, return_pose=False):
    """Original axis-angle/quaternion stitching; optional exact FK inputs."""
    from motius.motion.representation.edge_rotation import (
        matrix_to_axis_angle,
        axis_angle_to_quaternion,
        quaternion_to_axis_angle,
        axis_angle_to_matrix,
    )

    if windows.ndim != 3 or windows.shape[1:] != (150, 151):
        raise ValueError(f"Expected EDGE windows (N,150,151), got {windows.shape}")
    count, frames, _ = windows.shape
    root = windows[..., 4:7].clone()
    rotations = rotation_6d_to_matrix(windows[..., 7:].reshape(count, frames, 24, 6))
    angles = matrix_to_axis_angle(rotations)
    if count == 1:
        return (windows[0], angles[0], root[0]) if return_pose else windows[0]
    half = frames // 2
    output_frames = frames + half * (count - 1)
    fade_out = torch.ones((1, frames, 1)).to(root.device)
    fade_in = torch.ones((1, frames, 1)).to(root.device)
    fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(root.device)
    fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(root.device)
    root[:-1] *= fade_out
    root[1:] *= fade_in
    full_root = torch.zeros((output_frames, 3)).to(root.device)
    full_contacts = torch.zeros((output_frames, 4)).to(root.device)
    contact_weight = torch.zeros((output_frames, 1)).to(root.device)
    for index, part in enumerate(root):
        start = index * half
        full_root[start : start + frames] += part
        full_contacts[start : start + frames] += windows[index, :, :4]
        contact_weight[start : start + frames] += 1
    full_contacts /= contact_weight
    weight = torch.linspace(0, 1, half)[None, :, None].to(root.device)
    left = axis_angle_to_quaternion(angles[:-1, half:])
    right = axis_angle_to_quaternion(angles[1:, :half])
    merged = quaternion_to_axis_angle(_quaternion_slerp(left, right, weight))
    full_angles = torch.zeros((output_frames, 24, 3)).to(root.device)
    full_angles[:half] += angles[0, :half]
    cursor = half
    for overlap in merged:
        full_angles[cursor : cursor + half] += overlap
        cursor += half
    full_angles[cursor : cursor + half] += angles[-1, half:]
    rotation6d = matrix_to_rotation_6d(axis_angle_to_matrix(full_angles)).reshape(
        output_frames, 144
    )
    motion = torch.cat((full_contacts, full_root, rotation6d), dim=-1)
    return (motion, full_angles, full_root) if return_pose else motion


__all__ = ["cosine_alphas_cumprod", "edge_ddim_sample", "stitch_edge_windows"]
