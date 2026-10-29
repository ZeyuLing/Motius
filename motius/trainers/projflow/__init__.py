"""Native prepared-joint flow-matching objective for the released ACMDM prior."""

import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.trainers.gaussian_motion import unwrap_network
from motius.motion.conditioning.clip_text import encode_clip_text


@TRAINERS.register_module()
class ProjFlowTrainer(BaseTrainer):
    """Original linear-path velocity loss, with data/noise outside the network."""

    def __init__(self, bundle):
        super().__init__(bundle)
        bundle._trainable_modules = ["net"]
        bundle._save_ckpt_modules = ["net"]
        unwrap_network(bundle.net).clip_model.requires_grad_(False).eval()

    def train_step(self, batch):
        motion = batch["motion"].detach().float().to(self.bundle.device)
        if motion.ndim != 4 or motion.shape[-2:] != (22, 3):
            raise ValueError("Provide normalized SMPL-22 joint positions [B,T,22,3]")
        lengths = torch.as_tensor(batch["motion_lens"], device=motion.device).long()
        mask = (
            torch.arange(motion.shape[1], device=motion.device)[None] < lengths[:, None]
        )
        motion = torch.where(mask[:, :, None, None], motion, torch.zeros_like(motion))
        target = motion.clone().permute(0, 3, 1, 2).detach()
        net = unwrap_network(self.bundle.net)
        net.clip_model.eval()
        with torch.no_grad():
            conditions = encode_clip_text(net, batch["text"])
        # Preserve the source randn_like memory format and CPU uniform-time RNG.
        noise = torch.randn_like(target)
        times = torch.rand((len(target),)) * (1 - 0) + 0
        times = times.to(target)
        expanded = times.view(len(target), 1, 1, 1)
        noisy = expanded * target + (1 - expanded) * noise
        velocity = 1 * target + (-1) * noise
        attention = (
            mask.unsqueeze(-1)
            .repeat(1, 1, net.patches_per_frame)
            .flatten(1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        predicted = self.bundle.net(
            noisy, times, conds=conditions, force_mask=False, attention_mask=attention
        )
        per_frame = torch.mean((predicted - velocity) ** 2, dim=(1, 3))
        loss = (per_frame * mask).sum() / mask.sum()
        return {"loss": loss, "projflow_velocity_mse": loss.detach()}

    def val_step(self, batch):
        with torch.no_grad():
            return {
                "val_" + name: value.detach()
                for name, value in self.train_step(batch).items()
            }
