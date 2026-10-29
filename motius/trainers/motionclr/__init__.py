"""Native MotionCLR noise construction and masked reconstruction objective."""

import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.gaussian_motion import unwrap_network
from motius.motion.conditioning.clip_text import tokenize_clip


@TRAINERS.register_module()
class MotionCLRTrainer(BaseTrainer):
    def __init__(
        self,
        bundle,
        *,
        diffusion_steps=1000,
        beta_schedule="linear",
        prediction_type="sample",
    ):
        super().__init__(bundle)
        from diffusers import DDPMScheduler

        self.diffusion_steps = int(diffusion_steps)
        self.prediction_type = prediction_type
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_schedule=beta_schedule,
            variance_type="fixed_small",
            prediction_type=prediction_type,
            clip_sample=False,
        )
        bundle._trainable_modules = ["network"]
        bundle._save_ckpt_modules = ["network"]
        unwrap_network(bundle.network).clip_model.requires_grad_(False).eval()

    def train_step(self, batch):
        motion = batch["motion"].detach().float().to(self.bundle.device)
        if motion.ndim != 3 or motion.shape[-1] != 263:
            raise ValueError("MotionCLR expects normalized HumanML263 [B,T,263]")
        lengths = (
            torch.as_tensor(batch["motion_lens"], device=motion.device)
            .long()
            .clamp_max(motion.shape[1])
        )
        mask = (
            torch.arange(motion.shape[1], device=motion.device)[None] < lengths[:, None]
        ).float()
        noise = torch.randn_like(motion)
        timesteps = torch.randint(
            0, self.diffusion_steps, (len(motion),), device=motion.device
        )
        noisy = self.noise_scheduler.add_noise(motion, noise, timesteps)
        net = unwrap_network(self.bundle.network)
        net.clip_model.eval()
        predicted = self.bundle.network(
            noisy, timesteps, text=tokenize_clip(batch["text"])
        )
        if self.prediction_type == "sample":
            target = motion
        elif self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(motion, noise, timesteps)
        else:
            raise ValueError(
                "Unsupported MotionCLR prediction type: " + self.prediction_type
            )
        per_frame = torch.nn.functional.mse_loss(
            predicted, target, reduction="none"
        ).mean(dim=-1)
        per_sample = (per_frame * mask).sum(-1) / mask.sum(-1)
        loss = (per_sample * torch.ones_like(timesteps)).mean()
        return dict(loss=loss, loss_mot_rec=loss.detach())

    def val_step(self, batch):
        with torch.no_grad():
            return {
                "val_" + name: value.detach()
                for name, value in self.train_step(batch).items()
            }
