"""Motius-native trainer for TMR evaluators."""

from __future__ import annotations

from typing import Any, Dict

import torch

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

    def __init__(self, bundle, log_prefix: str = "tmr", **kwargs):
        super().__init__(bundle)
        self.log_prefix = str(log_prefix)

    def _device(self) -> torch.device:
        return next(self.bundle.tmr.parameters()).device

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = _move_to_device(batch, self._device())
        losses = self.bundle.compute_loss(batch)
        out: Dict[str, Any] = {"loss": losses["loss"]}
        for name, value in losses.items():
            out[f"{self.log_prefix}_{name}"] = value.detach()
        return out

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = _move_to_device(batch, self._device())
        losses = self.bundle.compute_loss(batch)
        return {
            f"val_{self.log_prefix}_{name}": value.detach()
            for name, value in losses.items()
        }
