"""Native MotionMillion corrupted-token CE and released FSQ reconstruction."""

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .reconstruction import ReConsLoss
from motius.motion.conditioning.motionmillion_text import prepare_motionmillion_text


@TRAINERS.register_module()
class MotionMillionTrainer(BaseTrainer):
    def __init__(
        self,
        bundle,
        *,
        stage="ar",
        pkeep=1.0,
        if_maxtest=False,
        recons_loss="l1_smooth",
        commit=0.02,
        loss_vel=0.1,
        acc_loss=0.0,
        acc_vel_loss=0.0,
        warmup_root_loss=0.0,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if stage not in {"ar", "vqvae"}:
            raise ValueError("MotionMillion stage must be ar or vqvae")
        if recons_loss not in {"l1", "l2", "l1_smooth"}:
            raise ValueError("Unsupported original reconstruction loss")
        self.stage, self.pkeep, self.if_maxtest = stage, float(pkeep), bool(if_maxtest)
        self.recons_loss = recons_loss
        self.reconstruction = ReConsLoss(recons_loss, 22)
        self.commit, self.loss_vel = float(commit), float(loss_vel)
        self.acc_loss, self.acc_vel_loss, self.warmup_root_loss = (
            float(acc_loss),
            float(acc_vel_loss),
            float(warmup_root_loss),
        )
        roots = [
            name
            for name in ["ar", "vqvae", "text_model"]
            if getattr(bundle, name, None) is not None
        ]
        if stage not in roots:
            raise ValueError("The selected neural stage was not loaded")
        if not hasattr(bundle, "_motionmillion_trainable_parameters"):
            bundle._motionmillion_trainable_parameters = {
                root: frozenset(
                    name
                    for name, p in getattr(bundle, root).named_parameters()
                    if p.requires_grad
                )
                for root in roots
            }
        for root in roots:
            for name, p in getattr(bundle, root).named_parameters():
                p.requires_grad_(
                    root == stage
                    and name in bundle._motionmillion_trainable_parameters[root]
                )
        bundle._trainable_modules = [stage]
        bundle._save_ckpt_modules = roots
        bundle._frozen_modules = [root for root in roots if root != stage]

    def train_step(self, batch):
        if self.stage == "vqvae":
            target = batch["motion"].to(self.bundle.device)
            prediction, commitment, perplexity, *_ = self.bundle.vqvae(target)
            reconstruction = self.reconstruction(prediction, target)
            velocity = self.reconstruction.forward_vel(prediction, target)
            loss = reconstruction + self.commit * commitment + self.loss_vel * velocity
            if self.acc_loss:
                loss = loss + self.acc_loss * self.reconstruction.forward_acc(
                    prediction, target
                )
            if self.acc_vel_loss:
                loss = loss + self.acc_vel_loss * self.reconstruction.forward_acc_vel(
                    prediction, target
                )
            if self.warmup_root_loss and batch.get("warmup", False):
                loss = loss + self.warmup_root_loss * self.reconstruction.forward_root(
                    prediction, target
                )
            return dict(
                loss=loss,
                motionmillion_reconstruction=reconstruction.detach(),
                motionmillion_velocity=velocity.detach(),
                motionmillion_perplexity=torch.as_tensor(
                    perplexity, device=loss.device
                ).detach(),
            )
        target = batch["tokens"].to(self.bundle.device)
        if "text_features" in batch:
            features = batch["text_features"].to(self.bundle.device)
            mask = batch["text_mask"].to(self.bundle.device)
        else:
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.bundle.device.type, enabled=False),
            ):
                features, mask = self.bundle.encode_text(
                    prepare_motionmillion_text(
                        self.bundle.text_model_name, batch["text"]
                    )
                )
        probability = np.random.rand(1)[0] if self.pkeep == -1 else self.pkeep
        keep = (
            torch.bernoulli(
                probability * torch.ones(target.shape, device=target.device)
            )
            .round()
            .to(torch.int64)
        )
        size = self.bundle.vqvae.vqvae.quantizer.codebook_size
        random = torch.randint_like(target, size)
        corrupted = keep * target + (1 - keep) * random
        logits = (
            self.bundle.ar(corrupted, features, mask)
            .contiguous()[..., :-1, :]
            .contiguous()
        )
        labels = target[..., 1:].contiguous().to(torch.int64)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=size + 1
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = (
            probabilities.argmax(-1)
            if self.if_maxtest
            else Categorical(probabilities).sample()
        )
        valid = labels != size + 1
        accuracy = ((selected == labels) & valid).sum() / valid.sum()
        return dict(loss=loss, motionmillion_accuracy=accuracy.detach())

    @torch.no_grad()
    def val_step(self, batch):
        return {
            "val_" + key: value.detach()
            for key, value in self.train_step(batch).items()
        }
