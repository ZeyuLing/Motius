"""Native masked-token stage trainer for InterMask."""
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import compute_objective


@TRAINERS.register_module()
class InterMaskTrainer(BaseTrainer):
    def train_step(self, batch):
        loss, accuracy, *_ = compute_objective(
            self.bundle.token_network, self.bundle.predict_logits,
            batch["tokens"].to(self.bundle.device), batch["text"],
            batch["token_lens"].to(self.bundle.device))
        return {"loss": loss, "intermask_accuracy": torch.as_tensor(accuracy, device=loss.device).detach()}

    def val_step(self, batch):
        result = self.train_step(batch)
        return {"val_intermask_loss": result["loss"].detach(),
                "val_intermask_accuracy": result["intermask_accuracy"]}
