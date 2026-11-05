"""YOLOX training through the shared Motius optimization protocol."""
import torch

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .yolox_objective import YOLOXObjective


@TRAINERS.register_module()
class YOLOXTrainer(BaseTrainer):
    def __init__(self, bundle, use_l1=False, **kwargs):
        super().__init__(bundle, **kwargs)
        bundle.model.head.use_l1 = use_l1
        self.objective = YOLOXObjective(bundle.model.head.num_classes, use_l1=use_l1)

    def train_step(self, batch):
        images, labels = batch["images"], batch["labels"]
        prediction = self.bundle.predict(images)
        loss, iou, confidence, classification, l1, foreground = self.objective(
            imgs=images, labels=labels, **prediction)
        return {"loss": loss, "loss_iou": iou.detach(),
                "loss_confidence": confidence.detach(),
                "loss_classification": classification.detach(),
                "loss_l1": l1.detach() if torch.is_tensor(l1) else l1,
                "foreground": foreground}

    @torch.no_grad()
    def val_step(self, batch):
        return {"predictions": self.bundle.predict(batch["images"])}
