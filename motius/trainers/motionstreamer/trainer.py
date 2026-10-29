"""Native MotionStreamer reconstruction and two-forward autoregressive training."""

import torch
import random
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from .objective import (
    forward_loss_withmask_2_forward,
    forward_loss_withmask_2_forward_streaming,
)
from .reconstruction import ReConsLoss


@TRAINERS.register_module()
class MotionStreamerTrainer(BaseTrainer):
    @classmethod
    def export_policy(
        cls, checkpoint, artifact, output, *, include_text_encoder=True, device="cpu"
    ):
        from pathlib import Path
        from motius.models.motionstreamer import MotionStreamerBundle

        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if not {"tae", "ar"} <= state.keys():
            raise ValueError("Expected a native MotionStreamer Runner checkpoint")
        bundle = MotionStreamerBundle.from_pretrained(
            artifact, device=device, load_text_model=include_text_encoder
        )
        bundle.load_state_dict_selective(
            state,
            strict=True,
            exclude_module_keys=None if include_text_encoder else ["text_model"],
        )
        return bundle.save_pretrained(output, include_text_encoder=include_text_encoder)

    def __init__(
        self,
        bundle,
        *,
        stage="ar",
        total_steps=300000,
        batch_multiplier=4,
        root_weight=7.0,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if stage not in {"tae", "ar", "streaming"}:
            raise ValueError("MotionStreamer stage must be tae, ar or streaming")
        self.stage = stage
        self.total_steps = int(total_steps)
        self.batch_multiplier = int(batch_multiplier)
        self.root_weight = float(root_weight)
        self.text_frontend = None
        self.reconstruction = ReConsLoss(motion_dim=272)
        trainable = "tae" if stage == "tae" else "ar"
        frozen = "ar" if stage == "tae" else "tae"
        bundle._trainable_modules = [trainable]
        bundle._frozen_modules = [frozen, "text_model"]
        getattr(bundle, trainable).requires_grad_(True)
        getattr(bundle, frozen).requires_grad_(False).eval()

    def train_step(self, batch):
        if self.stage == "tae":
            target = batch["motion"].to(self.bundle.device).float()
            prediction, mean, log_variance = self.bundle.tae(target)
            reconstruction = self.reconstruction(prediction, target)
            divergence = self.reconstruction.forward_KL(mean, log_variance)
            root = self.reconstruction.forward_root(prediction, target)
            return dict(
                loss=reconstruction + divergence + self.root_weight * root,
                reconstruction=reconstruction.detach(),
                divergence=divergence.detach(),
                root=root.detach(),
            )
        latents = batch["latents"] if "latents" in batch else batch["tokens"][:, :-1, :]
        latents = latents.to(self.bundle.device)
        if "text_features" in batch:
            features = batch["text_features"].to(self.bundle.device)
        else:
            if self.text_frontend is None:
                from motius.motion.representation.motionstreamer_text import (
                    SentenceT5Frontend,
                )

                self.text_frontend = SentenceT5Frontend(
                    self.bundle.text_model, self.bundle.text_model_name
                )
            captions = list(batch["caption"])
            for index in random.sample(range(len(captions)), int(len(captions) * 0.1)):
                captions[index] = ""
            features = (
                torch.from_numpy(self.text_frontend.encode(captions))
                .float()
                .to(self.bundle.device)
            )
        lengths = batch["lengths"].to(self.bundle.device)
        step = int(batch.get("step", getattr(self.runner, "global_step", 0)))
        arguments = dict(
            latents=latents,
            trans=self.bundle.ar,
            m_lens=lengths,
            feat_text=features,
            step=step,
            total_steps=self.total_steps,
            diffmlps_batch_mul=self.batch_multiplier,
        )
        if self.stage == "streaming":
            arguments["A_token_length"] = batch["prefix_lengths"].to(self.bundle.device)
            loss = forward_loss_withmask_2_forward_streaming(**arguments)
        else:
            loss = forward_loss_withmask_2_forward(**arguments)
        return dict(loss=loss)

    @torch.no_grad()
    def val_step(self, batch):
        return self.train_step(batch)
