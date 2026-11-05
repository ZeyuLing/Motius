"""Native MLD VAE and latent-diffusion objectives."""

import numpy as np
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.gaussian_motion import unwrap_network
from motius.motion.conditioning.motionlcm_text import encode_text
from motius.motion.representation.humanml import recover_from_ric


@TRAINERS.register_module()
class MLDTrainer(BaseTrainer):
    """Original two-stage objectives; corpus transforms remain outside the Bundle.

    Frozen VAE parameters retain the source training-time dropout behavior during
    diffusion training. Text conditioning is prepared by the external processor.
    """

    def __init__(
        self,
        bundle,
        *,
        stage="diffusion",
        guidance_uncondp=0.1,
        lambda_rec=1.0,
        lambda_joint=1.0,
        lambda_kl=1e-4,
        source_vae_training=True,
    ):
        super().__init__(bundle)
        if stage not in {"vae", "diffusion"}:
            raise ValueError("Use a separate VAE or diffusion stage")
        self.stage = stage
        self.guidance_uncondp = float(guidance_uncondp)
        self.lambda_rec = float(lambda_rec)
        self.lambda_joint = float(lambda_joint)
        self.lambda_kl = float(lambda_kl)
        self.source_vae_training = bool(source_vae_training)
        self.predict_epsilon = (
            bundle._sched_cfg.get("prediction_type", "epsilon") == "epsilon"
        )
        from diffusers import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=bundle._sched_cfg["num_train_timesteps"],
            beta_start=bundle._sched_cfg["beta_start"],
            beta_end=bundle._sched_cfg["beta_end"],
            beta_schedule=bundle._sched_cfg["beta_schedule"],
            prediction_type="epsilon" if self.predict_epsilon else "sample",
        )
        bundle._trainable_modules = ["vae"] if stage == "vae" else ["denoiser"]
        bundle._save_ckpt_modules = ["vae", "denoiser"] + (
            ["text_encoder"] if bundle.text_encoder is not None else []
        )
        bundle._frozen_modules = ["denoiser"] if stage == "vae" else ["vae"]
        if bundle.text_encoder is not None:
            bundle._frozen_modules.append("text_encoder")
        if not hasattr(bundle, "_mld_intrinsic_trainability"):
            bundle._mld_intrinsic_trainability = {
                root: {
                    name: parameter.requires_grad
                    for name, parameter in getattr(bundle, root).named_parameters()
                }
                for root in ["vae", "denoiser"]
            }
        for root in bundle._trainable_modules:
            for name, parameter in getattr(bundle, root).named_parameters():
                parameter.requires_grad_(bundle._mld_intrinsic_trainability[root][name])
        for name in bundle._frozen_modules:
            getattr(bundle, name).requires_grad_(False)

    def train_step(self, batch):
        features = batch["motion"].to(self.bundle.device).float()
        lengths = (
            torch.as_tensor(
                batch.get("motion_lens", batch.get("length")), device=features.device
            )
            .long()
            .tolist()
        )
        if (
            len(lengths) != len(features)
            or not lengths
            or min(lengths) <= 0
            or max(lengths) > features.shape[1]
        ):
            raise ValueError(
                "MLD requires positive valid lengths within the supplied motion"
            )
        # The original corpus collator pads to the local minibatch maximum.
        # A prepared file may have a larger common storage length.
        features = features[:, : max(lengths)]
        vae = unwrap_network(self.bundle.vae)
        if self.stage == "vae":
            reconstructed, latent, distribution = self.bundle.vae(features, lengths)
            # Original VAE stage also evaluates a second stochastic encoding.
            # It advances dropout/posterior RNG even though no cycle loss uses it.
            vae.encode(reconstructed, lengths)
            joints_rst = recover_from_ric(
                reconstructed * self.bundle.std + self.bundle.mean, 22
            )
            joints_ref = recover_from_ric(
                features * self.bundle.std + self.bundle.mean, 22
            )
            reference = torch.distributions.Normal(
                torch.zeros_like(distribution.loc), torch.ones_like(distribution.scale)
            )
            frames = min(features.shape[1], reconstructed.shape[1])
            feature_loss = F.smooth_l1_loss(
                reconstructed[:, :frames], features[:, :frames], reduction="mean"
            )
            joint_loss = F.smooth_l1_loss(joints_rst, joints_ref, reduction="mean")
            kl_loss = torch.distributions.kl_divergence(distribution, reference).mean()
            loss = 0.0 + self.lambda_rec * feature_loss
            loss = loss + self.lambda_joint * joint_loss
            loss = loss + self.lambda_kl * kl_loss
            return dict(
                loss=loss,
                recons_feature=feature_loss.detach(),
                recons_joints=joint_loss.detach(),
                kl_motion=kl_loss.detach(),
            )
        vae.train(self.training and self.source_vae_training)
        with torch.no_grad():
            latent, _ = vae.encode(features, lengths)
        captions = [
            "" if np.random.rand(1) < self.guidance_uncondp else text
            for text in batch["text"]
        ]
        self.bundle.text_encoder.eval()
        conditions = encode_text(self.bundle.text_encoder, captions)
        latent = latent.permute(1, 0, 2)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (len(latent),),
            device=latent.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(latent.clone(), noise, timesteps)
        prediction = self.bundle.denoiser(
            sample=noisy, timestep=timesteps, encoder_hidden_states=conditions
        )
        target = noise if self.predict_epsilon else latent
        loss = F.mse_loss(prediction, target, reduction="mean")
        return dict(loss=loss, inst_loss=loss.detach())

    def val_step(self, batch):
        with torch.no_grad():
            return {
                "val_" + name: value.detach()
                for name, value in self.train_step(batch).items()
            }
