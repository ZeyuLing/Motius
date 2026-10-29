"""MoGenDiT-specific denoising training objective from the recovered snapshot."""
from __future__ import annotations

import math
import random
import torch
from torch.nn import functional as F

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.diffusion.mogendit.resample import UniformSampler
from .mask_scheduler import MotionMaskScheduler
from .utils import random_index

DEFAULT_MASK_MODES = dict(random_frame=0.2, random_phrase=0.2, random_start_end=0.2,
                          block_trans=0.1, joint_only=0.1, uncond=0.2)


def weighted_masked_loss(pred, target, weight, mask, l1_weight, l2_weight):
    """Original weighted valid-element mean; empty velocity masks contribute zero."""
    if not mask.any():
        return pred.sum() * 0
    weights = weight.reshape(-1, 1, 1)
    loss = pred.new_zeros(())
    if l1_weight:
        loss = loss + l1_weight * (weights * F.l1_loss(pred, target, reduction='none'))[mask].mean()
    if l2_weight:
        loss = loss + l2_weight * (weights * F.mse_loss(pred, target, reduction='none'))[mask].mean()
    return loss


@TRAINERS.register_module()
class MoGenDiTTrainer(BaseTrainer):
    """Clean x0 supervision with native masks, corruption and rigidity consistency.

    Optimizer/EMA/checkpoint orchestration lives in the training entrypoint or
    Motius runner. ``train_step`` returns a differentiable loss, without stepping.
    Only the original effective uniform timestep sampling is supported.
    """
    def __init__(self, bundle, *, motion_degradation=True, degrade_rate=0.5,
                 keyframe_modes=None, l1_weight_x0=1.0, l2_weight_x0=0.0,
                 consis_loss=True, l1_weight_consis=1.0, l2_weight_consis=0.0,
                 consis_start_step=10000, schedule_sampler_type='uniform'):
        super().__init__(bundle)
        if schedule_sampler_type != 'uniform':
            raise ValueError('Only uniform sampling is supported; the upstream loss-aware sampler was never updated')
        self.keyframe_modes = dict(DEFAULT_MASK_MODES if keyframe_modes is None else keyframe_modes)
        from motius.motion.representation.mogendit.motion_representation import OccamMotionRep
        from motius.diffusion.mogendit import create_diffusion
        from .corruption import MoGenDiTCorruption
        from .kinematic_loss import MoGenDiTKinematicLoss
        self.motion_rep = OccamMotionRep(global_pose=bundle.config["global_pose"], fps=bundle.config["fps"])
        self.diffusion = create_diffusion()
        self.corruption = MoGenDiTCorruption(self.motion_rep)
        self.kinematic_loss = MoGenDiTKinematicLoss(self.motion_rep)
        self.mask_scheduler = MotionMaskScheduler(self.motion_rep)
        valid_modes = set(self.mask_scheduler.temporal_mask_modes + self.mask_scheduler.spatial_mask_modes) - {'fix_start_end'}
        if (not self.keyframe_modes or set(self.keyframe_modes) - valid_modes
                or any(not math.isfinite(v) or v < 0 for v in self.keyframe_modes.values())
                or not math.isclose(sum(self.keyframe_modes.values()), 1.0, abs_tol=1e-6)):
            raise ValueError('keyframe_modes must have supported modes and nonnegative weights summing to 1')
        if not 0 <= degrade_rate <= 1:
            raise ValueError('degrade_rate must be in [0,1]')
        weights = [l1_weight_x0, l2_weight_x0, l1_weight_consis, l2_weight_consis]
        if any(not math.isfinite(v) or v < 0 for v in weights) or l1_weight_x0 + l2_weight_x0 <= 0:
            raise ValueError('Loss weights must be finite, nonnegative; x0 loss must be positive')
        if consis_loss and l1_weight_consis + l2_weight_consis <= 0:
            raise ValueError('Enabled consistency loss must have a positive weight')
        if consis_start_step < 0:
            raise ValueError('consis_start_step must be nonnegative')
        self.motion_degradation = motion_degradation
        self.degrade_rate = degrade_rate
        self.l1_weight_x0, self.l2_weight_x0 = l1_weight_x0, l2_weight_x0
        self.consis_loss = consis_loss
        self.l1_weight_consis, self.l2_weight_consis = l1_weight_consis, l2_weight_consis
        self.consis_start_step = consis_start_step
        self.schedule_sampler = UniformSampler(self.diffusion)

    def prepare_batch(self, batch):
        device = next(self.bundle.parameters()).device
        clean = torch.as_tensor(batch['motion'], device=device, dtype=torch.float32)
        lengths = torch.as_tensor(batch['length'], device=device, dtype=torch.long).reshape(-1)
        if (clean.ndim != 3 or clean.shape[-1] != 201 or clean.shape[0] == 0
                or lengths.shape != (clean.shape[0],) or (lengths < 1).any()
                or (lengths > clean.shape[1]).any() or not torch.isfinite(clean).all()):
            raise ValueError('Expected finite (B,T,201) motion and B lengths in [1,T]')
        with torch.no_grad():
            timesteps, weights = self.schedule_sampler.sample(len(clean), device)
            keep = self.mask_scheduler.get_formulated_mask(clean, lengths, self.keyframe_modes)
            valid = self.mask_scheduler.get_length_mask_bool(clean, lengths)
            source = clean.clone()
            if self.motion_degradation and self.degrade_rate > 0:
                degradation = self.corruption.degradation
                degradation.degradation_probabilities = degradation.degradation_probabilities.to(device)
                indices = random_index(len(clean), self.degrade_rate)
                keep[indices] = 0
                ref_frames = random.randint(1, 10)
                keep[indices, :ref_frames] = 1
                keep *= valid
                # Native degradation must only see real frames: padding is not a skeleton.
                for i in indices:
                    length = int(lengths[i])
                    source[i:i+1, :length] = self.corruption.motion_degradation_batch(
                        clean[i:i+1, :length], keyframe_mask=keep[i:i+1, :length],
                        length=lengths[i:i+1], bool_length_mask=valid[i:i+1, :length])
            source = source.masked_fill(~valid, 0)
            noisy, _ = self.diffusion.q_sample(source, timesteps, obs_mask=keep, length_mask=valid)
        return dict(clean=clean, noisy=noisy, timesteps=timesteps, weights=weights,
                    keep_mask=keep, valid=valid, lengths=lengths)

    def compute_losses(self, prediction, clean, lengths, weights, *, step):
        valid = self.mask_scheduler.get_length_mask_bool(clean, lengths)
        losses = {}
        for name, channels in [('pose', slice(0,132)), ('joint', slice(132,198)), ('trans', slice(198,201))]:
            losses['loss_' + name] = weighted_masked_loss(
                prediction[...,channels], clean[...,channels], weights, valid[...,channels],
                self.l1_weight_x0, self.l2_weight_x0)
        shape = (*prediction.shape[:2],22,3)
        pred_global = prediction[...,132:198].reshape(shape) + prediction[...,198:201].unsqueeze(-2)
        gt_global = clean[...,132:198].reshape(shape) + clean[...,198:201].unsqueeze(-2)
        losses['loss_vel'] = weighted_masked_loss(
            (pred_global[:,1:] - pred_global[:,:-1]).flatten(2),
            (gt_global[:,1:] - gt_global[:,:-1]).flatten(2), weights, valid[:,1:,132:198],
            self.l1_weight_x0, self.l2_weight_x0)
        total = sum(losses.values())
        # The original activates this at iter > consis_start_step (strictly >).
        losses['loss_consis'] = prediction.new_zeros(())
        if self.consis_loss and step > self.consis_start_step:
            losses['loss_consis'] = self.kinematic_loss(
                prediction[...,:132], prediction[...,132:198], length=lengths,
                l1_weight=self.l1_weight_consis, l2_weight=self.l2_weight_consis)
            total = total + losses['loss_consis']
        return total, losses

    def train_step(self, batch):
        prepared = self.prepare_batch(batch)
        prediction = self.bundle(prepared['noisy'], prepared['timesteps'],
                                 keep_mask=prepared['keep_mask'],
                                 padding_mask=(~prepared['valid'][...,0]).float())
        total, losses = self.compute_losses(prediction, prepared['clean'], prepared['lengths'],
                                           prepared['weights'], step=self.get_global_step())
        return {'loss': total, 'log_vars': {k: float(v.detach()) for k,v in losses.items()}}

    @torch.no_grad()
    def val_step(self, batch):
        return self.train_step(batch)
