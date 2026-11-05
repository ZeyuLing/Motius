"""Native training of the released six-anchor MaskControl logits regularizer."""
import torch
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import compute_control_objective


@TRAINERS.register_module()
class MaskControlTrainer(BaseTrainer):
    CONTROL_MODULES = ("seqTransEncoder_control", "encoder_control", "first_zero_linear", "mid_zero_linear")

    def __init__(self, bundle, *, control="random", xent_weight=1., control_weight=1., **kwargs):
        super().__init__(bundle, **kwargs)
        if control not in {"random", "cross"}:
            raise ValueError("The released six-anchor architecture supports random and cross training masks")
        self.control, self.xent_weight, self.control_weight = control, float(xent_weight), float(control_weight)
        bundle.requires_grad_(False)
        network = bundle.unwrap_network(bundle.control_model)
        for name in self.CONTROL_MODULES:
            getattr(network, name).requires_grad_(True)
        bundle._trainable_modules = ["control_model"]
        bundle._save_ckpt_modules = ["control_model"]
        bundle._frozen_modules = [name for name in ("vq_model", "residual_model", "length_estimator")
                                  if getattr(bundle, name, None) is not None]

    def train_step(self, batch):
        network = self.bundle.unwrap_network(self.bundle.control_model)
        # Upstream ctrl_train freezes/evaluates the base network and trains only these blocks.
        network.eval()
        for name in self.CONTROL_MODULES:
            getattr(network, name).train()
        _, ce, _, accuracy, control = compute_control_objective(
            network, self.bundle.predict_logits, self.bundle.vq_model,
            self.bundle.control_mean, self.bundle.control_std, self.bundle.mask_emb_vq,
            self.control, batch["text"], batch["motion_lens"].to(self.bundle.device),
            batch["motion"].to(self.bundle.device))
        return {"loss": self.xent_weight * ce + self.control_weight * control,
                "maskcontrol_ce": ce.detach(), "maskcontrol_control": control.detach(),
                "maskcontrol_accuracy": torch.as_tensor(accuracy, device=ce.device)}

    @torch.no_grad()
    def val_step(self, batch):
        result=self.train_step(batch)
        return {"val_"+key:value.detach() for key,value in result.items()}
