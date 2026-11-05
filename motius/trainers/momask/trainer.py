"""MoMask supervised base-token and residual-token training."""
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import compute_base_objective, compute_residual_objective


@TRAINERS.register_module()
class MoMaskTrainer(BaseTrainer):
    def __init__(self, bundle, stage="base", **kwargs):
        super().__init__(bundle, **kwargs)
        if stage not in {"base", "residual"}:
            raise ValueError("MoMask stage must be 'base' or 'residual'")
        self.stage = stage
        self.module_name = "t2m_transformer" if stage == "base" else "res_transformer"
        for name in ("t2m_transformer", "res_transformer", "vq_model", "length_estimator"):
            module = getattr(bundle, name, None)
            if module is not None:
                module.requires_grad_(name == self.module_name)
                if hasattr(module, "clip_model"):
                    module.clip_model.requires_grad_(False).eval()
        bundle._trainable_modules = [self.module_name]
        bundle._save_ckpt_modules = [self.module_name]
        bundle._frozen_modules = [name for name in ("t2m_transformer", "res_transformer", "vq_model", "length_estimator")
                                  if name != self.module_name and getattr(bundle, name, None) is not None]

    def train_step(self, batch):
        network = self.bundle.unwrap_network(getattr(self.bundle, self.module_name))
        predict = self.bundle.predict_base if self.stage == "base" else self.bundle.predict_residual
        objective = compute_base_objective if self.stage == "base" else compute_residual_objective
        loss, _, accuracy = objective(network, predict,
            batch["tokens"].to(self.bundle.device), batch["text"], batch["token_lens"].to(self.bundle.device))
        return {"loss": loss, "momask_accuracy": torch.as_tensor(accuracy, device=loss.device).detach()}

    @torch.no_grad()
    def val_step(self, batch):
        return {f"val_{key}": value.detach() for key, value in self.train_step(batch).items()}
