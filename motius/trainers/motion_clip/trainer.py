"""Native MotionCLIP contrastive training graph."""

import torch
import torch.distributed as dist
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.representation.motion_clip.processor import MotionCLIPProcessor
from .loss import clip_loss, gather_with_grad


@TRAINERS.register_module()
class MotionCLIPTrainer(BaseTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        self.processor = MotionCLIPProcessor(bundle)

    def objective(self, output):
        model = getattr(self.bundle.motionclip_model, "module", self.bundle.motionclip_model)
        if model.training:
            motion = gather_with_grad(output.motion_embeds)
            text = gather_with_grad(output.text_embeds)
            scale = torch.clamp(
                model.logit_scale.exp(), max=100.0
            )
            loss = clip_loss(torch.matmul(text, motion.t()) * scale)
            if dist.is_initialized() and dist.get_world_size() > 1:
                loss = loss * dist.get_world_size()
            return loss
        return clip_loss(output.logits_per_text)

    def train_step(self, batch):
        prepared = self.processor.prepare_batch(
            batch["motion"], batch["caption"], batch["num_frames"]
        )
        output = self.bundle(**prepared)
        return dict(loss=self.objective(output))

    @torch.no_grad()
    def val_step(self, batch):
        prepared = self.processor.prepare_batch(
            batch["motion"], batch["caption"], batch["num_frames"]
        )
        output = self.bundle(**prepared)
        return dict(
            loss=self.objective(output),
            text_embeds=output.text_embeds,
            motion_embeds=output.motion_embeds,
        )
