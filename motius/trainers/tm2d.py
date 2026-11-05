"""Original TM2D joint text/music token objectives at the prepared-data boundary."""

import torch
from torch.nn import functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class TM2DTrainer(BaseTrainer):
    def __init__(
        self, bundle, label_smoothing=False, lambda_a2d=1.0, lambda_t2m=1.0, **kwargs
    ):
        super().__init__(bundle, **kwargs)
        from motius.models.tm2d.network import JointTokenNetwork

        self.label_smoothing = bool(label_smoothing)
        self.lambda_a2d, self.lambda_t2m = float(lambda_a2d), float(lambda_t2m)
        bundle.requires_grad_(False)
        bundle.audio_transformer.requires_grad_(True)
        bundle.text_transformer.requires_grad_(True)
        if not hasattr(bundle, "joint_network"):
            bundle.joint_network = JointTokenNetwork(
                bundle.audio_transformer, bundle.text_transformer
            )
        bundle._trainable_modules = ["joint_network"]
        bundle._save_ckpt_modules = [
            "joint_network",
            "vq_encoder",
            "quantizer",
            "vq_decoder",
        ]
        bundle._frozen_modules = ["vq_encoder", "quantizer", "vq_decoder"]

    def _token_loss(self, logits, target):
        prediction = logits.view(-1, logits.shape[-1]).clone()
        gold = target.contiguous().view(-1).clone()
        mask = gold.ne(self.bundle.config["motion_pad_id"])
        if self.label_smoothing:
            epsilon, classes = 0.1, prediction.shape[-1]
            one_hot = torch.zeros_like(prediction).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (classes - 1)
            value = -(one_hot * F.log_softmax(prediction, dim=1)).sum(dim=1)
            loss = value.masked_select(mask).sum()
        else:
            loss = F.cross_entropy(
                prediction,
                gold,
                ignore_index=self.bundle.config["motion_pad_id"],
                reduction="sum",
            )
        return loss / mask.sum().item()

    def train_step(self, batch):
        values = {name: value.to(self.bundle.device) for name, value in batch.items()}
        dance, motion = values["dance_tokens"].long(), values["motion_tokens"].long()
        audio_logits, text_logits = self.bundle.joint_network(
            values["audio_features"],
            values["audio_lengths"],
            dance[:, :-1],
            values["text_tokens"].long(),
            motion[:, :-1],
        )
        audio_loss = self._token_loss(audio_logits, dance[:, 1:])
        text_loss = self._token_loss(text_logits, motion[:, 1:])
        return dict(
            loss=audio_loss * self.lambda_a2d + text_loss * self.lambda_t2m,
            tm2d_audio=audio_loss.detach(),
            tm2d_text=text_loss.detach(),
        )

    @torch.no_grad()
    def val_step(self, batch):
        return {
            "val_" + name: value.detach()
            for name, value in self.train_step(batch).items()
        }
