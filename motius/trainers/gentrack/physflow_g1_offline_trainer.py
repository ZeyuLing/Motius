"""Offline reward-filtered SFT control for the G1-native generator."""

from __future__ import annotations

from typing import Any, Dict

import torch

from motius.registry import TRAINERS
from motius.trainers.gentrack.physflow_g1_trainer import PhysFlowG1Trainer


@TRAINERS.register_module()
class PhysFlowG1OfflineTrainer(PhysFlowG1Trainer):
    """Train only on a frozen, pre-scored candidate pool.

    The dataset supplies ``motion`` as the selected G0 candidate and
    ``gt_motion`` as the matched public-data anchor.  This trainer never calls
    a tracker or samples the live generator, so it is a strict offline control.
    """

    def __init__(
        self,
        bundle,
        offline_anchor_weight: float = 0.01,
        gt_anchor_interval: int = 4,
        **kwargs,
    ) -> None:
        kwargs.setdefault("enable_reward", False)
        kwargs.setdefault("num_samples", 1)
        super().__init__(bundle=bundle, **kwargs)
        self.offline_anchor_weight = float(offline_anchor_weight)
        self.gt_anchor_interval = max(0, int(gt_anchor_interval))

    def _target_loss(
        self,
        batch: Dict[str, Any],
        target_key: str,
        length_key: str,
        *,
        anchor_weight: float,
    ) -> Dict[str, Any]:
        target = self.bundle.normalize_motion(
            batch[target_key].to(self.bundle._device()).float()
        ).detach()
        lengths = batch[length_key]
        batch_size = int(target.shape[0])
        out = self.bundle.sft_loss_g1(
            batch["text_vec_raw"],
            list(batch["text_ctxt_raw"]),
            batch["text_ctxt_raw_length"],
            target,
            lengths,
            good_mask=torch.ones(batch_size),
            anchor_weight=anchor_weight,
        )
        return out

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        step = self._global_step()
        use_gt = (
            self.gt_anchor_interval > 0
            and step % self.gt_anchor_interval == 0
            and "gt_motion" in batch
        )
        if use_gt:
            out = self._target_loss(
                batch,
                "gt_motion",
                "gt_length",
                anchor_weight=0.0,
            )
        else:
            out = self._target_loss(
                batch,
                "motion",
                "tgt_length",
                anchor_weight=self.offline_anchor_weight,
            )
        result: Dict[str, Any] = {
            "loss": out["loss"],
            "loss_offline_sft": out["sft_mse"],
            "update_is_gt_anchor": torch.tensor(float(use_gt)),
        }
        if "anchor_mse" in out:
            result["loss_reference_anchor"] = out["anchor_mse"]
        return result
