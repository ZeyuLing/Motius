"""Native EDGE reconstruction, velocity, FK and foot-skate objectives."""

import copy
import torch
import torch.nn.functional as F
from einops import reduce
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.representation.edge import edge_axis_angle_forward_kinematics
from motius.motion.representation.edge_rotation import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)
from motius.diffusion.edge import cosine_alphas_cumprod


def extract(values, timesteps, shape):
    return values.gather(-1, timesteps).reshape(
        len(timesteps), *((1,) * (len(shape) - 1))
    )


@TRAINERS.register_module()
class EDGETrainer(BaseTrainer):
    """Original fixed-window, x0-prediction objective on prepared EDGE tensors.

    FK intentionally uses the normalized predictions: this is the source
    objective, whose unnormalization lines are commented out. Iterative
    inference, audio extraction and corpus processing remain outside the Bundle.
    """

    trainer_controls_optimization = True

    def __init__(self, bundle, cond_drop_prob=0.25, ema_decay=0.9999):
        super().__init__(bundle)
        self.cond_drop_prob = float(cond_drop_prob)
        self.ema_decay = float(ema_decay)
        if not hasattr(bundle, "ema_network"):
            bundle.ema_network = copy.deepcopy(bundle.network)
        bundle.ema_network.requires_grad_(False)
        alphas = cosine_alphas_cumprod(bundle.config["sampling"]["train_timesteps"])
        self.sqrt_alphas_cumprod = alphas.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas).sqrt()
        self.p2_loss_weight = (1 + alphas / (1 - alphas)) ** 0
        bundle._trainable_modules = ["network"]
        bundle._save_ckpt_modules = ["network", "ema_network"]
        bundle._frozen_modules = ["ema_network"]

    def _objective(self, batch):
        motion = batch["motion"].to(self.bundle.device).float()
        condition = batch["music_features"].to(self.bundle.device).float()
        if motion.shape[1:] != (150, 151) or condition.shape != (
            len(motion),
            150,
            4800,
        ):
            raise ValueError(
                "EDGE requires prepared motion [B,150,151] and music [B,150,4800]"
            )
        device = motion.device
        timesteps = torch.randint(
            0, len(self.sqrt_alphas_cumprod), (len(motion),), device=device
        ).long()
        noise = torch.randn_like(motion)
        noisy = (
            extract(self.sqrt_alphas_cumprod.to(device), timesteps, motion.shape)
            * motion
            + extract(
                self.sqrt_one_minus_alphas_cumprod.to(device), timesteps, motion.shape
            )
            * noise
        )
        prediction = self.bundle.network(
            noisy, condition, timesteps, cond_drop_prob=self.cond_drop_prob
        )
        weights = extract(self.p2_loss_weight.to(device), timesteps, (len(motion), 1))
        reconstruction = (
            reduce(
                F.mse_loss(prediction, motion, reduction="none"),
                "b ... -> b (...)",
                "mean",
            )
            * weights
        )
        contact, prediction = torch.split(
            prediction, (4, prediction.shape[2] - 4), dim=2
        )
        _, target = torch.split(motion, (4, motion.shape[2] - 4), dim=2)
        target_velocity = target[:, 1:] - target[:, :-1]
        prediction_velocity = prediction[:, 1:] - prediction[:, :-1]
        velocity = (
            reduce(
                F.mse_loss(prediction_velocity, target_velocity, reduction="none"),
                "b ... -> b (...)",
                "mean",
            )
            * weights
        )
        b, frames, _ = prediction.shape
        pred_angles = matrix_to_axis_angle(
            rotation_6d_to_matrix(prediction[:, :, 3:].reshape(b, frames, -1, 6))
        )
        target_angles = matrix_to_axis_angle(
            rotation_6d_to_matrix(target[:, :, 3:].reshape(b, frames, -1, 6))
        )
        pred_joints = edge_axis_angle_forward_kinematics(
            pred_angles, prediction[:, :, :3]
        )
        target_joints = edge_axis_angle_forward_kinematics(
            target_angles, target[:, :, :3]
        )
        fk = (
            reduce(
                F.mse_loss(pred_joints, target_joints, reduction="none"),
                "b ... -> b (...)",
                "mean",
            )
            * weights
        )
        feet = pred_joints[:, :, [7, 8, 10, 11]]
        foot_velocity = torch.zeros_like(feet)
        foot_velocity[:, :-1] = feet[:, 1:] - feet[:, :-1]
        foot_velocity[~(contact > 0.95)] = 0
        foot = reduce(
            F.mse_loss(
                foot_velocity, torch.zeros_like(foot_velocity), reduction="none"
            ),
            "b ... -> b (...)",
            "mean",
        )
        losses = (
            0.636 * reconstruction.mean(),
            2.964 * velocity.mean(),
            0.646 * fk.mean(),
            10.942 * foot.mean(),
        )
        return dict(
            loss=sum(losses),
            reconstruction_loss=losses[0].detach(),
            velocity_loss=losses[1].detach(),
            fk_loss=losses[2].detach(),
            foot_loss=losses[3].detach(),
        )

    @torch.no_grad()
    def update_ema(self):
        network = self.bundle.network
        while hasattr(network, "module"):
            network = network.module
        for target, source in zip(
            self.bundle.ema_network.parameters(), network.parameters()
        ):
            target.copy_(target * self.ema_decay + (1 - self.ema_decay) * source)

    def train_step(self, batch):
        values = self._objective(batch)
        if not self.optimizers:
            return values
        if (
            len(self.optimizers) != 1
            or self.accelerator.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "Original EDGE recipe uses one optimizer and accumulation one"
            )
        name, optimizer = next(iter(self.optimizers.items()))
        optimizer.zero_grad()
        self.accelerator.backward(values["loss"])
        optimizer.step()
        scheduler = self.get_lr_scheduler(name)
        if scheduler is not None:
            scheduler.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.update_ema()
        return {
            key: None if key == "loss" else value.detach()
            for key, value in values.items()
        }

    def val_step(self, batch):
        with torch.no_grad():
            return {
                "val_" + key: value.detach()
                for key, value in self._objective(batch).items()
            }
