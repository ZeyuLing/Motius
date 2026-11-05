"""Native online/target latent-consistency distillation."""

import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.gaussian_motion import unwrap_network
from motius.motion.conditioning.motionlcm_text import encode_text
from .schedule import (
    DDIMSolver,
    append_dims,
    guidance_scale_embedding,
    predicted_origin,
    scalings_for_boundary_conditions,
)


@TRAINERS.register_module()
class MotionLCMTrainer(BaseTrainer):
    """Original text-only distillation with teacher CFG and a target EMA.

    The trainer owns optimization so the target changes after a successful
    student update and before the shared checkpoint hook saves either root.
    A bare objective call without injected optimizers remains differentiable.
    """

    trainer_controls_optimization = True

    def __init__(
        self,
        bundle,
        *,
        teacher_weights_path=None,
        num_ddim_timesteps=50,
        w_min=5.0,
        w_max=15.0,
        uncond_batch_size=256,
        loss_type="huber",
        huber_c=0.001,
        ema_decay=0.95,
        max_grad_norm=1.0,
    ):
        super().__init__(bundle)
        if not hasattr(bundle, "teacher_denoiser"):
            if teacher_weights_path is None:
                raise ValueError(
                    "Distillation requires explicit pretrained teacher weights"
                )
            bundle.initialize_distillation(teacher_weights_path)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.uncond_batch_size = int(uncond_batch_size)
        self.loss_type, self.huber_c = loss_type, float(huber_c)
        self.ema_decay, self.max_grad_norm = float(ema_decay), float(max_grad_norm)
        if loss_type not in {"huber", "l2"} or self.uncond_batch_size <= 0:
            raise ValueError("Use huber/l2 and a positive unconditional batch size")
        if not 0 <= self.ema_decay <= 1 or self.huber_c <= 0:
            raise ValueError("Invalid consistency EMA or Huber scale")
        from diffusers import DDPMScheduler

        config = bundle._sched_cfg
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_train_timesteps"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            beta_schedule=config["beta_schedule"],
            prediction_type="epsilon",
        )
        self.alpha_schedule = self.noise_scheduler.alphas_cumprod.sqrt()
        self.sigma_schedule = (1 - self.noise_scheduler.alphas_cumprod).sqrt()
        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            timesteps=config["num_train_timesteps"],
            ddim_timesteps=num_ddim_timesteps,
        )
        self.topk = config["num_train_timesteps"] // num_ddim_timesteps
        self.uncond_prompt_embeds = None
        bundle._trainable_modules = ["denoiser"]
        bundle._frozen_modules = [
            "vae",
            "text_encoder",
            "teacher_denoiser",
            "target_denoiser",
        ]
        bundle._save_ckpt_modules = ["denoiser", *bundle._frozen_modules]
        for name in bundle._frozen_modules:
            getattr(bundle, name).requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        online = unwrap_network(self.bundle.denoiser)
        for target, source in zip(
            self.bundle.target_denoiser.parameters(), online.parameters()
        ):
            target.detach().mul_(self.ema_decay).add_(source, alpha=1 - self.ema_decay)

    def _objective(self, batch):
        bundle = self.bundle
        device = bundle.device
        motion = batch["motion"].to(device).float()
        lengths = (
            torch.as_tensor(batch.get("motion_lens", batch.get("length")))
            .long()
            .tolist()
        )
        if (
            len(lengths) != len(motion)
            or not lengths
            or min(lengths) <= 0
            or max(lengths) > motion.shape[1]
        ):
            raise ValueError("Invalid MotionLCM valid-frame lengths")
        motion = motion[:, : max(lengths)]
        if len(motion) > self.uncond_batch_size:
            raise ValueError(
                "Minibatch exceeds the source unconditional embedding batch"
            )
        # Original frozen VAE, teacher and target retain training-time dropout.
        for name in ["vae", "teacher_denoiser", "target_denoiser"]:
            getattr(bundle, name).train(self.training)
        bundle.text_encoder.eval()
        if self.uncond_prompt_embeds is None:
            self.uncond_prompt_embeds = encode_text(
                bundle.text_encoder, [""] * self.uncond_batch_size
            )
        self.alpha_schedule = self.alpha_schedule.to(device)
        self.sigma_schedule = self.sigma_schedule.to(device)
        self.solver.to(device)
        with torch.no_grad():
            latents, _ = bundle.vae.encode(motion, lengths)
            latents = latents.permute(1, 0, 2)
        prompt_embeds = encode_text(bundle.text_encoder, batch["text"])
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        index = torch.randint(
            0, len(self.solver.ddim_timesteps), (bsz,), device=device
        ).long()
        start = self.solver.ddim_timesteps[index]
        end = start - self.topk
        end = torch.where(end < 0, torch.zeros_like(end), end)
        skip_start, out_start = [
            append_dims(x, latents.ndim)
            for x in scalings_for_boundary_conditions(start)
        ]
        skip, out = [
            append_dims(x, latents.ndim) for x in scalings_for_boundary_conditions(end)
        ]
        noisy = self.noise_scheduler.add_noise(latents, noise, start)
        w = (self.w_max - self.w_min) * torch.rand((bsz,)) + self.w_min
        online = unwrap_network(bundle.denoiser)
        embedding = guidance_scale_embedding(w, embedding_dim=online.time_cond_proj_dim)
        w = append_dims(w, latents.ndim).to(device=device, dtype=latents.dtype)
        embedding = embedding.to(device=device, dtype=latents.dtype)

        def origin(prediction, timestep, sample):
            return predicted_origin(
                prediction,
                timestep,
                sample,
                "epsilon",
                self.alpha_schedule,
                self.sigma_schedule,
            )

        predicted = bundle.denoiser(
            noisy, start, timestep_cond=embedding, encoder_hidden_states=prompt_embeds
        )
        model_pred = skip_start * noisy + out_start * origin(predicted, start, noisy)
        with torch.no_grad():
            conditional = bundle.teacher_denoiser(
                noisy, start, encoder_hidden_states=prompt_embeds
            )
            cond_x0 = origin(conditional, start, noisy)
            unconditional = bundle.teacher_denoiser(
                noisy, start, encoder_hidden_states=self.uncond_prompt_embeds[:bsz]
            )
            uncond_x0 = origin(unconditional, start, noisy)
            pred_x0 = cond_x0 + w * (cond_x0 - uncond_x0)
            pred_noise = conditional + w * (conditional - unconditional)
            previous = self.solver.ddim_step(pred_x0, pred_noise, index)
            target_noise = bundle.target_denoiser(
                previous.float(),
                end,
                timestep_cond=embedding,
                encoder_hidden_states=prompt_embeds,
            )
            target = skip * previous + out * origin(target_noise, end, previous)
        if self.loss_type == "l2":
            return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return torch.mean(
            torch.sqrt((model_pred.float() - target.float()) ** 2 + self.huber_c**2)
            - self.huber_c
        )

    def train_step(self, batch):
        loss = self._objective(batch)
        if not self.optimizers:
            return dict(loss=loss)
        if len(self.optimizers) != 1:
            raise ValueError("MotionLCM uses one online-student optimizer")
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError(
                "This source distillation recipe requires accumulation_steps=1"
            )
        name, optimizer = next(iter(self.optimizers.items()))
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(
            self.bundle.denoiser.parameters(), self.max_grad_norm
        )
        optimizer.step()
        scheduler = self.get_lr_scheduler(name)
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if not self.accelerator.optimizer_step_was_skipped:
            self.update_target()
        return dict(loss=None, consistency_loss=loss.detach())

    def val_step(self, batch):
        with torch.no_grad():
            return dict(val_loss=self._objective(batch).detach())
