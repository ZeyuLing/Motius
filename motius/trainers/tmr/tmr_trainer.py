"""Motius-native trainer for TMR evaluators."""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Optional
from copy import deepcopy

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


@TRAINERS.register_module()
class TMRTrainer(BaseTrainer):
    """Train a :class:`TMRBundle` through ``AccelerateRunner``."""

    def __init__(self, bundle, lmd=None, temperature=None, threshold_selfsim=None,
                 log_prefix="tmr", **kwargs):
        super().__init__(bundle, **kwargs)
        self.log_prefix = str(log_prefix)
        self.lmd = deepcopy(dict(recons=1.0,latent=1.0e-5,kl=1.0e-5,contrastive=.1) if lmd is None else lmd)
        self.temperature = .1 if temperature is None else float(temperature)
        self.threshold_selfsim = .8 if threshold_selfsim is None else float(threshold_selfsim)


    def _device(self) -> torch.device:
        return next(self.bundle.tmr.parameters()).device

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = _move_to_device(batch, self._device())
        losses = self.compute_loss(batch)
        out: Dict[str, Any] = {"loss": losses["loss"]}
        for name, value in losses.items():
            out[f"{self.log_prefix}_{name}"] = value.detach()
        return out

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = _move_to_device(batch, self._device())
        losses = self.compute_loss(batch)
        return {
            f"val_{self.log_prefix}_{name}": value.detach()
            for name, value in losses.items()
        }

    @staticmethod
    def _kl_loss(q, p) -> Tensor:
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        log_var_ratio = logvar_q - logvar_p
        mean_term = (mu_p - mu_q).pow(2) / logvar_p.exp()
        divergence = 0.5 * (log_var_ratio.exp() + mean_term - 1 - log_var_ratio)
        return divergence.mean()

    def _contrastive_loss(
        self,
        text_latents: Tensor,
        motion_latents: Tensor,
        sentence_embeddings: Optional[Tensor],
    ) -> Tensor:
        text_latents = F.normalize(text_latents, dim=-1)
        motion_latents = F.normalize(motion_latents, dim=-1)
        logits = text_latents @ motion_latents.T / self.temperature
        if sentence_embeddings is not None and self.threshold_selfsim:
            similarities = sentence_embeddings @ sentence_embeddings.T
            similarities = similarities - similarities.diag().diag()
            indices = torch.where(similarities > (2 * self.threshold_selfsim - 1))
            logits[indices] = -torch.inf
        labels = torch.arange(len(logits), device=logits.device)
        return (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute the released TMR reconstruction and retrieval objective."""
        core = self.bundle._unwrap_tmr(self.bundle.tmr)
        reference = batch["motion_x_dict"]["x"]
        output = self.bundle.reconstruct_pair(batch["text_x_dict"], batch["motion_x_dict"])
        text_latents, text_dist = output["text_latents"], output["text_dist"]
        motion_latents, motion_dist = output["motion_latents"], output["motion_dist"]
        text_reconstruction = output["text_reconstruction"]
        motion_reconstruction = output["motion_reconstruction"]

        losses: Dict[str, Tensor] = {}
        losses["recons_text"] = F.smooth_l1_loss(
            text_reconstruction, reference
        )
        losses["recons_motion"] = F.smooth_l1_loss(
            motion_reconstruction, reference
        )
        losses["recons"] = losses["recons_text"] + losses["recons_motion"]
        if core.vae:
            zero_dist = (
                torch.zeros_like(motion_dist[0]),
                torch.zeros_like(motion_dist[1]),
            )
            losses["kl"] = (
                self._kl_loss(text_dist, motion_dist)
                + self._kl_loss(motion_dist, text_dist)
                + self._kl_loss(motion_dist, zero_dist)
                + self._kl_loss(text_dist, zero_dist)
            )
        losses["latent"] = F.smooth_l1_loss(text_latents, motion_latents)
        losses["contrastive"] = self._contrastive_loss(
            text_latents,
            motion_latents,
            batch.get("sent_emb"),
        )
        losses["loss"] = sum(
            self.lmd[name] * value
            for name, value in losses.items()
            if name in self.lmd
        )
        return losses
