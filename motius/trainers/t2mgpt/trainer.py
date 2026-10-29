"""Original T2M-GPT token corruption and VQ reconstruction stages."""

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.conditioning.clip_text import encode_clip_text


@TRAINERS.register_module()
class T2MGPTTrainer(BaseTrainer):
    def __init__(
        self,
        bundle,
        *,
        stage="gpt",
        pkeep=0.5,
        if_maxtest=False,
        recons_loss="l1_smooth",
        commit=0.02,
        loss_vel=0.1,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if stage not in {"gpt", "vqvae"}:
            raise ValueError("T2M-GPT stage must be gpt or vqvae")
        if recons_loss not in {"l1", "l2", "l1_smooth"}:
            raise ValueError("Unsupported original reconstruction loss")
        self.stage, self.pkeep, self.if_maxtest = stage, float(pkeep), bool(if_maxtest)
        self.recons_loss, self.commit, self.loss_vel = (
            recons_loss,
            float(commit),
            float(loss_vel),
        )
        roots = [
            name
            for name in ("gpt", "vqvae", "clip_model")
            if getattr(bundle, name, None) is not None
        ]
        if not hasattr(bundle, "_t2mgpt_trainable_parameters"):
            bundle._t2mgpt_trainable_parameters = {
                root: frozenset(
                    name
                    for name, parameter in getattr(bundle, root).named_parameters()
                    if parameter.requires_grad
                )
                for root in roots
            }
        for root in roots:
            for name, parameter in getattr(bundle, root).named_parameters():
                parameter.requires_grad_(
                    root == stage and name in bundle._t2mgpt_trainable_parameters[root]
                )
        bundle._trainable_modules = [stage]
        bundle._save_ckpt_modules = roots
        bundle._frozen_modules = [name for name in roots if name != stage]

    def train_step(self, batch):
        if self.stage == "vqvae":
            motion = batch["motion"].to(self.bundle.device)
            prediction, commitment, perplexity = self.bundle.vqvae(motion)
            function = {
                "l1": F.l1_loss,
                "l2": F.mse_loss,
                "l1_smooth": F.smooth_l1_loss,
            }[self.recons_loss]
            reconstruction = function(prediction[..., :263], motion[..., :263])
            velocity = function(prediction[..., 4:67], motion[..., 4:67])
            loss = reconstruction + self.commit * commitment + self.loss_vel * velocity
            return {
                "loss": loss,
                "t2mgpt_reconstruction": reconstruction.detach(),
                "t2mgpt_commitment": commitment.detach(),
                "t2mgpt_velocity": velocity.detach(),
                "t2mgpt_perplexity": perplexity.detach(),
            }
        target = batch["tokens"].to(self.bundle.device)
        lengths = batch["token_lens"].to(self.bundle.device)
        with torch.no_grad():
            condition = encode_clip_text(self.bundle, batch["text"])
        inputs = target[:, :-1]
        probability = np.random.rand(1)[0] if self.pkeep == -1 else self.pkeep
        mask = torch.bernoulli(
            probability * torch.ones(inputs.shape, device=inputs.device)
        )
        mask = mask.round().to(dtype=torch.int64)
        network = self.bundle.gpt
        while hasattr(network, "module"):
            network = network.module
        random = torch.randint_like(inputs, network.num_vq)
        corrupted = mask * inputs + (1 - mask) * random
        logits = self.bundle.gpt(corrupted, condition).contiguous()
        loss = 0.0
        correct = 0
        for index in range(len(target)):
            prediction = logits[index][: lengths[index] + 1]
            labels = target[index][: lengths[index] + 1]
            loss += F.cross_entropy(prediction, labels) / len(target)
            probabilities = torch.softmax(prediction, dim=-1)
            if self.if_maxtest:
                _, selected = torch.max(probabilities, dim=-1)
            else:
                selected = Categorical(probabilities).sample()
            correct += (selected.flatten(0) == labels.flatten(0)).sum().item()
        accuracy = loss.new_tensor(correct) / (lengths + 1).sum()
        return {"loss": loss, "t2mgpt_accuracy": accuracy.detach()}

    @torch.no_grad()
    def val_step(self, batch):
        return {
            "val_" + key: value.detach()
            for key, value in self.train_step(batch).items()
        }
