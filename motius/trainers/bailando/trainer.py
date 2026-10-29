"""Native Bailando supervised GPT and root-velocity decoder objectives."""
import torch
from torch.nn import functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class BailandoTrainer(BaseTrainer):
    def __init__(self, bundle, stage="gpt", **kwargs):
        super().__init__(bundle, **kwargs)
        if stage not in {"gpt", "root"}:
            raise ValueError("Bailando stage must be 'gpt' or 'root'")
        self.stage = stage
        bundle.gpt.requires_grad_(stage == "gpt")
        bundle.gpt.critic_net.requires_grad_(False)
        bundle.vqvae.requires_grad_(False)
        if stage == "root":
            bundle.vqvae.vqvae_down.decoders_root.requires_grad_(True)
        bundle._trainable_modules = ["gpt" if stage == "gpt" else "vqvae"]
        bundle._save_ckpt_modules = ["vqvae", "gpt"]
        bundle._frozen_modules = ["vqvae" if stage == "gpt" else "gpt"]

    def train_step(self, batch):
        if self.stage == "gpt":
            up, down = self.bundle.predict_codes(
                tuple(value.to(self.bundle.device) for value in batch["tokens"]),
                batch["music_features"].to(self.bundle.device))
            target_up, target_down = (value.to(self.bundle.device) for value in batch["targets"])
            loss_up = F.cross_entropy(up.reshape(-1, up.shape[-1]), target_up.reshape(-1), reduction="none")
            loss_down = F.cross_entropy(down.reshape(-1, down.shape[-1]), target_down.reshape(-1), reduction="none")
            return {"loss": (loss_up + loss_down).mean(), "bailando_upper_ce": loss_up.mean().detach(),
                    "bailando_lower_ce": loss_down.mean().detach()}
        result = self.bundle.reconstruct(batch["motion"].to(self.bundle.device))
        output = result["lower"]
        channels = self.bundle.config["vqvae"]["joint_channel"]
        target = result["lower_target"].float()[..., :channels]
        reconstruction = target.new_zeros(())
        acceleration = target.new_zeros(())
        metrics = {}
        for level in reversed(range(len(output["velocities"]))):
            value = output["velocities"][level]
            term = (value - target).abs().mean()
            reconstruction += term
            acceleration += ((value[:, 1:] - value[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
            metrics[f"bailando_recons_l{level+1}"] = term.detach()
        weight = self.bundle.config["vqvae"]["down_half"].get("acc", 0.)
        loss = reconstruction + weight * acceleration
        return {"loss": loss, "bailando_velocity": reconstruction.detach(),
                "bailando_acceleration": acceleration.detach(), **metrics}

    @torch.no_grad()
    def val_step(self, batch):
        return {f"val_{key}": value.detach() for key, value in self.train_step(batch).items()}
