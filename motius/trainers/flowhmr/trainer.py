"""Motius optimization of the unmodified official FlowHMR objective."""

import torch

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


@TRAINERS.register_module()
class FlowHMRTrainer(BaseTrainer):
    def train_step(self, batch):
        batch = to_device(batch, next(self.bundle.parameters()).device)
        lengths = batch["length"]
        frames = batch["inputs"]["feature"]["feature"].shape[1]
        if lengths.ndim != 1 or (lengths <= 0).any() or (lengths > frames).any():
            raise ValueError("length must contain positive frame counts within the padded batch")
        result = self.bundle.training_forward(batch, self.get_global_step())
        return {"loss": result["loss"],
                **{f"loss_{k}": v.detach() for k, v in result["loss_dict"].items()}}

    @torch.no_grad()
    def val_step(self, batch):
        # Loss validation handles arbitrary batch sizes and masks; upstream
        # multi-seed geometry evaluation assumes batch size one.
        return self.train_step(batch)
