"""Native DART primitive-denoiser and VAE training graphs."""

from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.diffusion.dart import create_gaussian_diffusion
from .objective import denoiser_loss, vae_loss


@TRAINERS.register_module()
class DARTTrainer(BaseTrainer):
    trainer_controls_optimization = True

    def __init__(
        self,
        bundle,
        motion_processor=None,
        stage="denoiser",
        loss_weights=None,
        ema_decay=None,
        grad_clip=None,
        learning_rate=None,
        anneal_lr=None,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if stage not in {"denoiser", "vae"}:
            raise ValueError(stage)
        self.stage = stage
        if motion_processor is None:
            from motius.motion.conditioning.dart_training import DARTTrainingProcessor

            motion_processor = DARTTrainingProcessor(bundle)
        self.train_dataset = motion_processor
        defaults = dict(
            weight_latent_rec=1.0,
            weight_feature_rec=1.0,
            weight_smpl_joints_rec=0.0,
            weight_joints_consistency=0.0,
            weight_joints_delta=10000.0,
            weight_transl_delta=10000.0,
            weight_orient_delta=10000.0,
            weight_kl=1e-6,
            weight_rec=1.0,
        )
        if stage == "vae":
            defaults.update(
                weight_smpl_joints_rec=10.0,
                weight_joints_consistency=10.0,
                weight_joints_delta=100.0,
                weight_transl_delta=100.0,
                weight_orient_delta=100.0,
            )
        source_args = None
        if hasattr(bundle, "artifact_dir"):
            from motius.configuration.dart import read_arguments

            checkpoint = (
                bundle.denoiser_checkpoint
                if stage == "denoiser"
                else bundle.denoiser_args.mvae_path
            )
            source_args = read_arguments(
                Path(bundle.artifact_dir) / Path(checkpoint).parent / "args.yaml"
            ).train_args
            defaults.update(
                {
                    key: value
                    for key, value in vars(source_args).items()
                    if key.startswith("weight_")
                }
            )
        defaults.update(loss_weights or {})
        self.ema_decay = float(
            ema_decay
            if ema_decay is not None
            else getattr(source_args, "ema_decay", 0.999)
        )
        self.grad_clip = float(
            grad_clip
            if grad_clip is not None
            else getattr(source_args, "grad_clip", 1.0)
        )
        self.learning_rate = float(
            learning_rate
            if learning_rate is not None
            else getattr(source_args, "learning_rate", 1e-4)
        )
        self.anneal_lr = bool(
            anneal_lr
            if anneal_lr is not None
            else getattr(source_args, "anneal_lr", True)
        )
        self.total_steps = sum(
            getattr(source_args, f"stage{stage_index}_steps", 100000)
            for stage_index in [1, 2, 3]
        )
        self.args = SimpleNamespace(train_args=SimpleNamespace(**defaults))
        self.rec_criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)
        self.diffusion = create_gaussian_diffusion(
            bundle.denoiser_args.diffusion_args, enable_ddim=False
        )
        bundle.requires_grad_(False)
        name = "denoiser_model" if stage == "denoiser" else "vae_model"
        self.active_name = name
        getattr(bundle, name).requires_grad_(True).train()
        bundle._trainable_modules = [name]
        # Keep the decoder/denoiser paired with the trained primitive network.
        # The frozen counterpart is also needed for a self-contained resume.
        bundle._save_ckpt_modules = ["denoiser_model", "vae_model"]
        bundle._frozen_modules = [
            "vae_model" if stage == "denoiser" else "denoiser_model",
            "clip_model",
        ]
        if self.ema_decay > 0:
            self.ema_name = name + "_ema"
            if not hasattr(bundle, self.ema_name):
                setattr(bundle, self.ema_name, deepcopy(getattr(bundle, name)))
            getattr(bundle, self.ema_name).requires_grad_(False).eval()
            bundle._save_ckpt_modules.append(self.ema_name)
            bundle._frozen_modules.append(self.ema_name)
        else:
            self.ema_name = None

    @property
    def device(self):
        return self.bundle.device

    def _objective(self, batch):
        if hasattr(self.train_dataset, "to"):
            self.train_dataset.to(self.device)
        motion = batch["motion"].to(self.device)
        h = self.train_dataset.history_length
        f = self.train_dataset.future_length
        ground_history = motion[:, :h]
        future = motion[:, -f:]
        history = batch.get("history_motion", ground_history).to(self.device)
        condition = {"y": dict(batch.get("condition", {}))}
        if self.stage == "vae":
            prediction, latent, distribution = self.bundle.reconstruct_motion(
                future, history
            )
            return vae_loss(
                self,
                motion,
                condition,
                history,
                future,
                prediction,
                latent,
                distribution,
            )
        with torch.no_grad():
            encoding_history = (
                ground_history
                if self.bundle.denoiser_args.train_rollout_history == "gt"
                else history
            )
            latent, _ = self.bundle.encode_motion(
                future, encoding_history, self.bundle.denoiser_args.rescale_latent
            )
        start = latent.permute(1, 0, 2)
        timestep = batch.get("timesteps")
        if timestep is None:
            # Original UniformSampler uses NumPy's categorical draw.
            import numpy as np

            timestep = torch.from_numpy(
                np.random.choice(
                    self.diffusion.num_timesteps,
                    size=len(motion),
                    p=np.ones(self.diffusion.num_timesteps)
                    / self.diffusion.num_timesteps,
                )
            ).long()
        timestep = timestep.to(self.device)
        noise = batch.get("noise")
        noise = torch.randn_like(start) if noise is None else noise.to(start)
        noisy = self.diffusion.q_sample(start, timestep, noise)
        y = {
            "text_embedding": batch["text_embedding"].to(self.device),
            "history_motion_normalized": history,
        }
        prediction = self.bundle.predict_latent(
            noisy, self.diffusion._scale_timesteps(timestep), y
        ).permute(1, 0, 2)
        decoded = self.bundle.decode_motion(
            prediction,
            history,
            nfuture=f,
            scale_latent=self.bundle.denoiser_args.rescale_latent,
        )
        return denoiser_loss(
            self, motion, condition, history, future, decoded, latent, prediction, None
        )

    @torch.no_grad()
    def update_ema(self):
        if self.ema_name is None:
            return
        model = getattr(self.bundle, self.active_name)
        while hasattr(model, "module"):
            model = model.module
        for online, average in zip(
            model.parameters(), getattr(self.bundle, self.ema_name).parameters()
        ):
            average.mul_(self.ema_decay).add_(online, alpha=1 - self.ema_decay)

    def train_step(self, batch):
        terms = self._objective(batch)
        if not self.optimizers:
            return terms
        if (
            len(self.optimizers) != 1
            or self.accelerator.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "The original DART primitive recipe uses one optimizer and accumulation one"
            )
        name, optimizer = next(iter(self.optimizers.items()))
        if self.anneal_lr:
            rate = (
                1.0 - self.get_global_step() / self.total_steps
            ) * self.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = rate
        optimizer.zero_grad()
        self.accelerator.backward(terms["loss"])
        if self.grad_clip > 0:
            self.accelerator.clip_grad_norm_(
                getattr(self.bundle, self.active_name).parameters(), self.grad_clip
            )
        optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.update_ema()
        scheduler = self.get_lr_scheduler(name)
        if scheduler is not None:
            scheduler.step()
        return {
            key: None if key == "loss" else value.detach()
            for key, value in terms.items()
        }

    @torch.no_grad()
    def calibrate_latent(self, batch, *, use_ema=True):
        """Original VAE latent calibration on an explicitly supplied training batch."""
        if self.stage != "vae":
            raise ValueError("Latent calibration belongs to the VAE stage")
        model = getattr(
            self.bundle,
            self.ema_name if use_ema and self.ema_name else self.active_name,
        )
        while hasattr(model, "module"):
            model = model.module
        mode = model.training
        model.eval()
        motion = batch["motion"].to(self.device)
        latent, _ = model.encode(
            motion[:, -self.train_dataset.future_length :],
            motion[:, : self.train_dataset.history_length],
        )
        mean = latent.mean()
        std = (latent - mean).pow(2).mean().sqrt()
        model.register_buffer("latent_mean", mean)
        model.register_buffer("latent_std", std)
        model.train(mode)
        return mean, std

    @torch.no_grad()
    def val_step(self, batch):
        return {
            "val_" + key: value.detach()
            for key, value in self._objective(batch).items()
        }
