"""Inference-time conditioning helpers from the official CondMDI runtime."""

from __future__ import annotations

import numpy as np


def get_gradient_schedule(schedule_name=None, num_diffusion_steps=1000, scale=0.05):
    if schedule_name is None:
        return np.ones(num_diffusion_steps)
    if schedule_name == "first-half":
        return np.concatenate(
            (np.ones(num_diffusion_steps // 2), np.zeros(num_diffusion_steps - num_diffusion_steps // 2))
        )
    if schedule_name == "last-half":
        return np.concatenate(
            (np.zeros(num_diffusion_steps // 2), np.ones(num_diffusion_steps - num_diffusion_steps // 2))
        )
    timesteps = np.arange(num_diffusion_steps)
    if schedule_name == "exponential":
        return np.exp(-scale * timesteps[::-1])
    scale /= 5
    if schedule_name == "sigmoid":
        return 1 / (1 + np.exp(scale * (-timesteps + num_diffusion_steps / 2)))
    if schedule_name == "half-sigmoid":
        return 1 / (1 + np.exp(scale * -timesteps))
    raise ValueError(f"unknown reconstruction-guidance schedule: {schedule_name}")


def requires_reconstruction_guidance(model_kwargs, denoising_step):
    options = model_kwargs.get("y", {})
    if not options.get("reconstruction_guidance", False):
        return False
    return (denoising_step >= options["stop_recguidance_at"]).all()


def requires_imputation(model_kwargs, denoising_step):
    options = model_kwargs.get("y", {})
    if not options.get("imputate", False):
        return False
    return (denoising_step >= options["stop_imputation_at"]).all()
