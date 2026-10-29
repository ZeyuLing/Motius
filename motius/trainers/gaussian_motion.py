"""Prepared-motion training for the released MDM-family diffusion objectives."""

import numpy as np
import torch
from motius.motion.conditioning.clip_text import tokenize_clip
from motius.trainers.base_trainer import BaseTrainer


def unwrap_network(network):
    while hasattr(network, "module") or hasattr(network, "_orig_mod"):
        network = network.module if hasattr(network, "module") else network._orig_mod
    return network


class _NetworkForward:
    """Keep DDP forward ownership while exposing original architecture metadata."""

    def __init__(self, network):
        self.network = network

    def __call__(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(unwrap_network(self.network), name)


def uniform_timesteps(num_steps, batch_size, device):
    """Original OpenAI/MDM UniformSampler draws, including NumPy RNG ownership."""
    weights = np.ones([num_steps])
    probabilities = weights / np.sum(weights)
    indices = np.random.choice(len(probabilities), size=(batch_size,), p=probabilities)
    importance = 1 / (len(probabilities) * probabilities[indices])
    return (
        torch.from_numpy(indices).long().to(device),
        torch.from_numpy(importance).float().to(device),
    )


class GaussianMotionTrainer(BaseTrainer):
    """The native diffusion runtime owns the exact method-specific loss math.

    Inputs are normalized motion tensors and prepared observations/hints. Corpus
    normalization and random observation selection belong to dataset transforms.
    """

    def __init__(self, bundle, *, method, schedule_sampler=None):
        super().__init__(bundle)
        self.method = method
        self.schedule_sampler = schedule_sampler
        options = vars(bundle._args) if method == "mdm" else bundle.config
        if any(
            options.get(name, 0.0)
            for name in ["lambda_rcxyz", "lambda_fc", "lambda_vel_rcxyz"]
        ):
            raise ValueError(
                "Nonzero body-geometry losses require an explicit body codec"
            )
        if method == "mdm":
            from motius.diffusion.mdm.factory import create_gaussian_diffusion

            self.diffusion = create_gaussian_diffusion(bundle._args)
        elif method == "condmdi":
            from motius.diffusion.condmdi.factory import build_diffusion

            self.diffusion = build_diffusion(bundle.config)
        elif method == "omnicontrol":
            from motius.diffusion.omnicontrol.factory import build_diffusion

            self.diffusion = build_diffusion(
                bundle.config,
                mean=bundle.mean,
                std=bundle.std,
                raw_mean=bundle.raw_mean,
                raw_std=bundle.raw_std,
            )
        else:
            raise ValueError("Unknown Gaussian motion objective: " + method)
        bundle._trainable_modules = ["net"]
        bundle._save_ckpt_modules = ["net"]
        clip_model = getattr(unwrap_network(bundle.net), "clip_model", None)
        if clip_model is not None:
            clip_model.requires_grad_(False).eval()

    def train_step(self, batch):
        motion = batch["motion"].to(self.bundle.device)
        if motion.ndim == 3:
            motion = motion.permute(0, 2, 1).unsqueeze(2)
        if motion.ndim != 4:
            raise ValueError("Expected normalized motion [B,T,D] or [B,D,1,T]")
        lengths = torch.as_tensor(
            batch.get(
                "motion_lens", batch.get("lengths", [motion.shape[-1]] * len(motion))
            ),
            device=motion.device,
        )
        conditions = dict(batch.get("conditions", {}))
        conditions.update(
            lengths=lengths,
            mask=(
                torch.arange(motion.shape[-1], device=motion.device)[None]
                < lengths[:, None]
            )[:, None, None],
        )
        captions = batch.get("text")
        if captions is not None:
            conditions["text_tokens"] = tokenize_clip(
                captions, context_length=22, device=motion.device
            )
        elif "text_tokens" in conditions:
            conditions["text_tokens"] = conditions["text_tokens"].to(motion.device)
        else:
            raise ValueError("Provide captions or prepared CLIP token tensors")
        timesteps = batch.get("timesteps")
        if timesteps is None:
            if self.schedule_sampler is None:
                timesteps, weights = uniform_timesteps(
                    self.diffusion.num_timesteps, len(motion), motion.device
                )
            else:
                timesteps, weights = self.schedule_sampler.sample(
                    len(motion), motion.device
                )
        else:
            timesteps = torch.as_tensor(timesteps, device=motion.device).long()
            weights = torch.ones(len(motion), device=motion.device)
        weights = torch.as_tensor(
            batch.get("sample_weights", weights), device=motion.device
        )
        noise = batch.get("noise")
        noise = torch.randn_like(motion) if noise is None else noise.to(motion)
        kwargs = {"y": conditions}
        if self.method == "condmdi":
            if "obs_mask" not in batch:
                raise ValueError(
                    "CondMDI training requires a prepared observation mask"
                )
            kwargs["obs_mask"] = batch["obs_mask"].to(
                device=motion.device, dtype=torch.bool
            )
            kwargs["obs_x0"] = batch.get("obs_x0", motion).to(motion)
        if self.method == "omnicontrol":
            if "hint" in batch:
                conditions["hint"] = batch["hint"].to(motion)
            if "hint" not in conditions:
                raise ValueError("OmniControl training requires prepared joint hints")
            conditions["hint"] = conditions["hint"].to(motion)
            for name in ["hint_axis_mask", "hint_frame_mask"]:
                if name in batch:
                    conditions[name] = batch[name].to(motion.device)
        clip_model = getattr(unwrap_network(self.bundle.net), "clip_model", None)
        if clip_model is not None:
            clip_model.eval()
        terms = self.diffusion.training_losses(
            _NetworkForward(self.bundle.net),
            motion,
            timesteps,
            model_kwargs=kwargs,
            noise=noise,
        )
        if self.training and hasattr(self.schedule_sampler, "update_with_local_losses"):
            self.schedule_sampler.update_with_local_losses(
                timesteps, terms["loss"].detach()
            )
        return {
            "loss": (terms["loss"] * weights).mean(),
            **{
                self.method + "_" + name: value.detach().mean()
                for name, value in terms.items()
            },
        }

    def val_step(self, batch):
        # OmniControl's original spatial guide differentiates the input motion.
        with torch.enable_grad():
            return {
                "val_" + name: value.detach()
                for name, value in self.train_step(batch).items()
            }
