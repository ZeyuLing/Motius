"""Native TM2T motion-token to text Transformer training."""

import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import cal_performance


@TRAINERS.register_module()
class TM2TTrainer(BaseTrainer):
    def __init__(self, bundle, *, label_smoothing=False, **kwargs):
        super().__init__(bundle, **kwargs)
        self.label_smoothing = bool(label_smoothing)
        bundle.vq_encoder.requires_grad_(False)
        bundle.quantizer.requires_grad_(False)
        bundle._trainable_modules = ["transformer"]
        bundle._save_ckpt_modules = ["transformer", "vq_encoder", "quantizer"]
        bundle._frozen_modules = ["vq_encoder", "quantizer"]

    def train_step(self, batch):
        motion = (
            batch["motion_tokens"]
            .detach()
            .to(device=self.bundle.device, dtype=torch.long)
        )
        words = (
            batch["text_tokens"]
            .detach()
            .to(device=self.bundle.device, dtype=torch.long)
        )
        prediction = self.bundle.transformer(motion, words[:, :-1])
        prediction = prediction.view(-1, prediction.shape[-1]).clone()
        gold = words[:, 1:].contiguous().view(-1).clone()
        loss, _, correct, count = cal_performance(
            prediction, gold, self.bundle.text_pad_index, smoothing=self.label_smoothing
        )
        # Original gradients use the summed CE; normalized values are for logging.
        return {
            "loss": loss,
            "tm2t_mean_ce": loss.detach() / count,
            "tm2t_accuracy": loss.new_tensor(correct / count),
        }

    @torch.no_grad()
    def val_step(self, batch):
        return {
            "val_" + key: value.detach()
            for key, value in self.train_step(batch).items()
        }
