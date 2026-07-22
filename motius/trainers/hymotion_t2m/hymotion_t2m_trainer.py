"""HyMotion-T2M Trainer: flow-matching training for text-to-motion generation.

Unlike HyMotion-M2M, this trainer does NOT use VACE conditioning.
The input to the transformer is just x_t (noisy motion), purely text-conditioned.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths


@TRAINERS.register_module(force=True)
class HyMotionT2MTrainer(BaseTrainer):
    """Trainer for HyMotion-T2M flow-matching text-to-motion generation.

    Training forward:
      1. Get motion from batch, normalize
      2. Prepare text embeddings (pre-encoded, online, or null)
      3. Sample timesteps, create x_t via flow matching interpolation
      4. Forward through bundle.predict_flow() (NO VACE context)
      5. Compute loss via bundle.m2m_loss
    """

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
        **kwargs,
    ):
        super().__init__(bundle)
        self.val_num_steps = val_num_steps
        self.max_text_len = max_text_len

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.bundle.motion_transformer.parameters()).device

        # Get motion (this is both source and target for pure generation).
        motion = batch['motion'].to(device)
        B, L, D = motion.shape

        # Build padding mask from tgt_length or num_frames
        tgt_length = batch.get('tgt_length')
        if tgt_length is None:
            tgt_length = batch.get('num_frames')
        if tgt_length is None:
            tgt_length = [L] * B
        elif isinstance(tgt_length, (list, tuple)) and all(x is None for x in tgt_length):
            tgt_length = [L] * B
        if isinstance(tgt_length, Tensor):
            tgt_length = tgt_length.tolist()
        tgt_padding_mask = _length_to_mask(
            torch.tensor(tgt_length, dtype=torch.long, device=device), L
        )

        # Normalize motion
        x1 = self.bundle.normalize_motion(motion)

        # Prepare text: 3 paths (pre-encoded, online, null)
        if batch.get('text_vec_raw') is not None:
            vtxt_input = batch['text_vec_raw'].to(device)
            ctxt_raw = batch['text_ctxt_raw']
            pad_len = self.max_text_len
            if isinstance(ctxt_raw, (list, tuple)):
                feat_dim = ctxt_raw[0].shape[-1]
                ctxt_padded = ctxt_raw[0].new_zeros(len(ctxt_raw), pad_len, feat_dim)
                for i, t in enumerate(ctxt_raw):
                    seq = min(t.shape[0], pad_len)
                    ctxt_padded[i, :seq] = t[:seq]
                ctxt_input = ctxt_padded.to(device)
            else:
                cur_len = ctxt_raw.shape[1]
                if cur_len < pad_len:
                    ctxt_input = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len)).to(device)
                else:
                    ctxt_input = ctxt_raw[:, :pad_len].to(device)
            ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)

            # For null-embedding samples, force-replace with learned null embeddings
            null_mask = (ctxt_length == 0)
            if null_mask.any():
                null_v = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
                null_c = self.bundle.null_ctxt_input.expand_as(ctxt_input)
                vtxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(vtxt_input), null_v, vtxt_input
                )
                ctxt_input = torch.where(
                    null_mask.view(B, 1, 1).expand_as(ctxt_input), null_c, ctxt_input
                )

            vtxt_input, ctxt_input = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
            )
        elif 'caption' in batch and batch['caption'] is not None:
            captions = batch['caption']
            if isinstance(captions, torch.Tensor):
                captions = captions.tolist()
            captions = [c if c is not None else '' for c in captions]
            with torch.no_grad():
                text_feats = self.bundle.encode_text(captions)
            vtxt_input = text_feats['text_vec_raw'].to(device)
            ctxt_input = text_feats['text_ctxt_raw'].to(device)
            ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input = self.bundle.mask_text_cond(
                vtxt_input, ctxt_input,
                force_mask=False,
                cond_mask_prob=self.bundle.cond_mask_prob,
            )
        else:
            # Null text (unconditional)
            vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
            ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
            ctxt_length = torch.tensor([1], device=device).expand(B)
            ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

        # Flow matching: sample t, build x_t
        x0 = torch.randn_like(x1)

        if self.bundle.pred_type == 'x1':
            z = torch.randn(B, dtype=x1.dtype, device=device) * 0.8 + (-0.8)
            timesteps = torch.sigmoid(z)
        else:
            timesteps = torch.rand(B, dtype=x1.dtype, device=device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t) * x0 + t * x1

        # Forward: NO VACE context, just x_t
        pred = self.bundle.predict_flow(
            x_input=x_t,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        # Compute loss
        if self.bundle.pred_type == 'velocity':
            gt_velocity = x1 - x0
            pred_velocity = pred
            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
            )
        elif self.bundle.pred_type == 'x1':
            t_eps = 0.05
            gt_velocity = (x1 - x_t) / (1 - t).clamp_min(t_eps)
            pred_velocity = (pred - x_t) / (1 - t).clamp_min(t_eps)
            losses = self.bundle.m2m_loss(
                pred_vel=pred_velocity,
                gt_vel=gt_velocity,
                pred_x1=pred,
                gt_x1=x1,
                data_mask_temporal=tgt_padding_mask,
                global_step=self.get_global_step(),
            )
        else:
            raise ValueError(f'Unsupported pred_type: {self.bundle.pred_type}')

        loss = self.sum_train_losses(losses)
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()
        return result

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from motius.pipelines.hymotion_t2m.hymotion_t2m_pipeline import (
            HyMotionT2MPipeline,
        )

        pipeline = HyMotionT2MPipeline(
            bundle=self.bundle,
            num_steps=self.val_num_steps,
        )
        with torch.no_grad():
            output = pipeline(batch)
        return {'preds': output}
