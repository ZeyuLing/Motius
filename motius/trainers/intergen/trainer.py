"""InterGen training graph and interaction geometry objective."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.diffusion.intergen.factory import create_diffusion
from motius.diffusion.intergen.gaussian_diffusion import create_named_schedule_sampler
from motius.motion.representation.intergen import MotionNormalizerTorch
from .losses import InterLoss, GeometricLoss


@TRAINERS.register_module()
class InterGenTrainer(BaseTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        cfg = bundle.network.cfg
        self.diffusion = create_diffusion(cfg, [cfg.DIFFUSION_STEPS])
        self.sampler = create_named_schedule_sampler(cfg.SAMPLER, self.diffusion)
        self.normalizer = MotionNormalizerTorch(bundle.mean_path, bundle.std_path)
        self.t_bar = cfg.T_BAR

    def compute_loss(self, batch):
        device = self.bundle.device
        cond = encode_clip_text(self.bundle, batch["text"])
        x_start = batch["motions"].to(device)
        B, T = x_start.shape[:2]
        # Retain the source RNG call order (text dropout, condition mask, t, noise).
        dropped = torch.bernoulli(torch.ones(B, device=device) * 0.1).view(B, 1)
        cond = cond * (1. - dropped)
        lengths = batch["motion_lens"].to(device)
        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])
        mask = mask.float()[..., None].repeat(1, 1, 2)
        t, _ = self.sampler.sample(B, device)
        target = self.normalizer.forward(x_start.reshape(B, T, 2, -1))
        normalized = target.reshape(B, T, -1)
        noisy = self.diffusion.q_sample(normalized, t, noise=torch.randn_like(normalized))
        prediction = self.diffusion._wrap_model(self.bundle.predict_motion)(
            noisy, t, mask=mask, cond=cond).reshape(B, T, 2, -1)
        mask = mask.reshape(B, T, 2, 1)
        timestep_mask = (t <= self.t_bar).float()
        inter = InterLoss("l2", 22, self.normalizer)
        inter(prediction, target, mask, timestep_mask)
        loss_a = GeometricLoss("l2", 22, "A", self.normalizer)
        loss_a(prediction[..., 0, :], target[..., 0, :], mask[..., 0, :], timestep_mask)
        loss_b = GeometricLoss("l2", 22, "B", self.normalizer)
        loss_b(prediction[..., 1, :], target[..., 1, :], mask[..., 0, :], timestep_mask)
        losses = {**loss_a.losses, **loss_b.losses, **inter.losses}
        losses["total"] = loss_a.losses["A"] + loss_b.losses["B"] + inter.losses["total"]
        return losses

    def train_step(self, batch):
        losses = self.compute_loss(batch)
        return {"loss": losses["total"], **{f"intergen_{k}": v.detach() for k, v in losses.items()}}

    def val_step(self, batch):
        return {f"val_intergen_{k}": v.detach() for k, v in self.compute_loss(batch).items()}
