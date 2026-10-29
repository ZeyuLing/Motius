from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from motius.motion.conditioning.gentrack import _len_to_mask, pack_context
from .flow_grpo import clipped_grpo_loss, flow_dpo_pair_loss, reverse_kl_from_log_probs, sample_unique_timestep_indices
from motius.diffusion.gentrack import flow_grpo_transition

def grpo_loss_g1(self, text_vec: Tensor, text_ctxt: List[Tensor], ctxt_len: Tensor, lengths: Tensor, states: Tensor, old_log_probs: Tensor, sigmas: Tensor, advantages: Tensor, eta: float=0.7, clip_range: float=0.2, reference_kl_weight: float=0.01, timesteps_per_update: int=1, transition_microbatch_size: int=0, timestep_indices: Optional[Tensor]=None) -> Dict[str, Tensor]:
    """Replay one or more sampled transitions per candidate under the live policy."""
    device = self._device()
    dtype = torch.float32
    vtxt = text_vec.to(device, dtype)
    ctxt, ctxt_mask = pack_context(self, text_ctxt, ctxt_len, device, dtype)
    lengths = lengths.to(device)
    states = states.to(device, dtype)
    old_log_probs = old_log_probs.to(device, dtype)
    sigmas = sigmas.to(device, dtype)
    advantages = advantages.to(device, dtype)
    batch, num_transitions = old_log_probs.shape
    if states.shape[:2] != (batch, num_transitions + 1):
        raise ValueError(f'states {tuple(states.shape)} incompatible with log-probs {tuple(old_log_probs.shape)}')
    if advantages.shape != (batch,):
        raise ValueError(f'advantages must be ({batch},), got {tuple(advantages.shape)}')
    if timestep_indices is None:
        timestep_indices = sample_unique_timestep_indices(batch, num_transitions, timesteps_per_update, device)
    else:
        timestep_indices = timestep_indices.to(device=device, dtype=torch.long)
        if timestep_indices.ndim == 1:
            timestep_indices = timestep_indices.unsqueeze(1)
        if timestep_indices.ndim != 2 or timestep_indices.shape[0] != batch:
            raise ValueError(f'timestep_indices must have shape [batch] or [batch, K], got {tuple(timestep_indices.shape)}')
        if timestep_indices.numel() and (int(timestep_indices.min()) < 0 or int(timestep_indices.max()) >= num_transitions):
            raise ValueError(f'timestep_indices must be in [0, {num_transitions})')
    timestep_count = int(timestep_indices.shape[1])
    row = torch.arange(batch, device=device).unsqueeze(1).expand(batch, timestep_count)
    current_state = states[row, timestep_indices].flatten(0, 1)
    next_state = states[row, timestep_indices + 1].flatten(0, 1)
    sigma = sigmas[timestep_indices].reshape(-1)
    sigma_next = sigmas[timestep_indices + 1].reshape(-1)
    lengths = lengths.repeat_interleave(timestep_count)
    frame_mask = _len_to_mask(lengths, states.shape[2])
    vtxt = vtxt.repeat_interleave(timestep_count, dim=0)
    ctxt = ctxt.repeat_interleave(timestep_count, dim=0)
    ctxt_mask = ctxt_mask.repeat_interleave(timestep_count, dim=0)
    advantages = advantages.repeat_interleave(timestep_count)
    t = 1.0 - sigma
    transition_microbatch_size = int(transition_microbatch_size)
    if transition_microbatch_size < 0:
        raise ValueError('transition_microbatch_size must be non-negative')

    def predict_live(state_chunk: Tensor, ctxt_chunk: Tensor, vtxt_chunk: Tensor, timestep_chunk: Tensor, frame_mask_chunk: Tensor, ctxt_mask_chunk: Tensor) -> Tensor:
        return self.predict_flow(x_input=state_chunk, ctxt_input=ctxt_chunk, vtxt_input=vtxt_chunk, timesteps=timestep_chunk, x_mask_temporal=frame_mask_chunk, ctxt_mask_temporal=ctxt_mask_chunk)
    if 0 < transition_microbatch_size < current_state.shape[0]:
        velocity_chunks = []
        for start in range(0, current_state.shape[0], transition_microbatch_size):
            stop = min(start + transition_microbatch_size, current_state.shape[0])
            velocity_chunks.append(checkpoint(predict_live, current_state[start:stop], ctxt[start:stop], vtxt[start:stop], t[start:stop], frame_mask[start:stop], ctxt_mask[start:stop], use_reentrant=False))
        velocity = torch.cat(velocity_chunks, dim=0)
    else:
        velocity = predict_live(current_state, ctxt, vtxt, t, frame_mask, ctxt_mask)
    _, current_log_prob = flow_grpo_transition(model_output=-velocity, latents=current_state, sigma=sigma, sigma_next=sigma_next, sigma_first_next=sigmas[1], eta=eta, next_latents=next_state, frame_mask=frame_mask)
    behavior_log_prob = old_log_probs[row, timestep_indices].reshape(-1)
    log_ratio = current_log_prob - behavior_log_prob.detach()
    policy_loss, ratio_mean, clip_fraction = clipped_grpo_loss(current_log_prob, behavior_log_prob, advantages, clip_range=clip_range)
    detached_advantages = advantages.detach()
    detached_log_ratio = log_ratio.detach()
    advantage_log_ratio_cov = ((detached_advantages - detached_advantages.mean()) * (detached_log_ratio - detached_log_ratio.mean())).mean()
    advantage_log_ratio_alignment = (detached_advantages.sign() * detached_log_ratio).mean()
    positive_mask = detached_advantages > 0
    negative_mask = detached_advantages < 0
    positive_log_ratio_mean = detached_log_ratio[positive_mask].mean() if bool(positive_mask.any()) else detached_log_ratio.new_zeros(())
    negative_log_ratio_mean = detached_log_ratio[negative_mask].mean() if bool(negative_mask.any()) else detached_log_ratio.new_zeros(())
    reference_kl = current_log_prob.new_zeros(())
    if reference_kl_weight > 0:
        anchor = self._reference_transformer()
        if anchor is None:
            if self.require_immutable_anchor:
                raise RuntimeError('reference_kl_weight > 0 but immutable G0 anchor is unavailable')
        else:
            with torch.no_grad():
                if 0 < transition_microbatch_size < current_state.shape[0]:
                    reference_velocity = torch.cat([anchor(x=current_state[start:stop], ctxt_input=ctxt[start:stop], vtxt_input=vtxt[start:stop], timesteps=t[start:stop], x_mask_temporal=frame_mask[start:stop], ctxt_mask_temporal=ctxt_mask[start:stop], mask_density=None, task_emb=None) for start, stop in ((start, min(start + transition_microbatch_size, current_state.shape[0])) for start in range(0, current_state.shape[0], transition_microbatch_size))], dim=0)
                else:
                    reference_velocity = anchor(x=current_state, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t, x_mask_temporal=frame_mask, ctxt_mask_temporal=ctxt_mask, mask_density=None, task_emb=None)
                _, reference_log_prob = flow_grpo_transition(model_output=-reference_velocity, latents=current_state, sigma=sigma, sigma_next=sigma_next, sigma_first_next=sigmas[1], eta=eta, next_latents=next_state, frame_mask=frame_mask)
            reference_kl = reverse_kl_from_log_probs(current_log_prob, reference_log_prob)
    loss = policy_loss + float(reference_kl_weight) * reference_kl
    return {'loss': loss, 'policy_loss': policy_loss.detach(), 'reference_kl': reference_kl.detach(), 'ratio_mean': ratio_mean, 'ratio_abs_deviation_mean': (log_ratio.exp() - 1.0).abs().mean().detach(), 'log_ratio_abs_mean': log_ratio.abs().mean().detach(), 'advantage_log_ratio_cov': advantage_log_ratio_cov, 'advantage_log_ratio_alignment': advantage_log_ratio_alignment, 'positive_advantage_log_ratio_mean': positive_log_ratio_mean, 'negative_advantage_log_ratio_mean': negative_log_ratio_mean, 'clip_fraction': clip_fraction, 'current_log_prob': current_log_prob.detach().mean(), 'behavior_log_prob': behavior_log_prob.detach().mean(), 'timestep_mean': timestep_indices.float().mean().detach(), 'timesteps_per_update': current_log_prob.new_tensor(float(timestep_count)), 'transition_microbatch_size': current_log_prob.new_tensor(float(transition_microbatch_size))}

def flow_dpo_loss_g1(self, text_vec: Tensor, text_ctxt: List[Tensor], ctxt_len: Tensor, winner_motion: Tensor, loser_motion: Tensor, lengths: Tensor, *, beta: float=100.0, timesteps_per_pair: int=1) -> Dict[str, Tensor]:
    """Apply Flow-DPO to same-prompt physical-preference motion pairs.

        Winner and loser receive the same flow noise and timestep. The frozen
        immutable G0 predicts the same noised pair, making the objective the
        established Diffusion-DPO error difference rather than an ad-hoc
        reward-weighted motion loss.
        """
    if self.pred_type != 'velocity':
        raise ValueError('Flow-DPO currently requires velocity prediction')
    device = self._device()
    dtype = torch.float32
    winner = winner_motion.to(device=device, dtype=dtype).detach()
    loser = loser_motion.to(device=device, dtype=dtype).detach()
    if winner.shape != loser.shape or winner.ndim != 3:
        raise ValueError(f'Flow-DPO winner and loser must have matching (B,T,D) shapes, got {tuple(winner.shape)} and {tuple(loser.shape)}')
    batch, frames, motion_dim = winner.shape
    lengths = lengths.to(device=device, dtype=torch.long)
    if lengths.shape != (batch,):
        raise ValueError(f'Flow-DPO lengths must have shape {(batch,)}, got {tuple(lengths.shape)}')
    timesteps_per_pair = int(timesteps_per_pair)
    if timesteps_per_pair < 1:
        raise ValueError('timesteps_per_pair must be positive')
    vtxt = text_vec.to(device=device, dtype=dtype)
    ctxt, ctxt_mask = pack_context(self, text_ctxt, ctxt_len, device, dtype)
    if vtxt.shape[0] != batch or ctxt.shape[0] != batch:
        raise ValueError('Flow-DPO requires one text condition per motion pair')
    winner = winner.repeat_interleave(timesteps_per_pair, dim=0)
    loser = loser.repeat_interleave(timesteps_per_pair, dim=0)
    lengths = lengths.repeat_interleave(timesteps_per_pair, dim=0)
    vtxt = vtxt.repeat_interleave(timesteps_per_pair, dim=0)
    ctxt = ctxt.repeat_interleave(timesteps_per_pair, dim=0)
    ctxt_mask = ctxt_mask.repeat_interleave(timesteps_per_pair, dim=0)
    effective_batch = batch * timesteps_per_pair
    pair_noise = torch.randn_like(winner)
    pair_timestep = torch.rand(effective_batch, device=device, dtype=dtype)
    x1 = torch.cat((winner, loser), dim=0)
    x0 = torch.cat((pair_noise, pair_noise), dim=0)
    timesteps = torch.cat((pair_timestep, pair_timestep), dim=0)
    t = timesteps.view(2 * effective_batch, 1, 1)
    x_t = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    frame_mask = _len_to_mask(lengths, frames)
    pair_frame_mask = torch.cat((frame_mask, frame_mask), dim=0)
    pair_ctxt = torch.cat((ctxt, ctxt), dim=0)
    pair_vtxt = torch.cat((vtxt, vtxt), dim=0)
    pair_ctxt_mask = torch.cat((ctxt_mask, ctxt_mask), dim=0)
    prediction = self.predict_flow(x_input=x_t, ctxt_input=pair_ctxt, vtxt_input=pair_vtxt, timesteps=timesteps, x_mask_temporal=pair_frame_mask, ctxt_mask_temporal=pair_ctxt_mask)
    reference = self._reference_transformer()
    if reference is None:
        raise RuntimeError('Flow-DPO requires an immutable G0 reference')
    with torch.no_grad():
        reference_prediction = reference(x=x_t, ctxt_input=pair_ctxt, vtxt_input=pair_vtxt, timesteps=timesteps, x_mask_temporal=pair_frame_mask, ctxt_mask_temporal=pair_ctxt_mask, mask_density=None, task_emb=None)
    mask = pair_frame_mask.unsqueeze(-1).to(dtype)
    denominator = (mask.sum(dim=(1, 2)) * float(motion_dim)).clamp_min(1.0)
    model_mse = ((prediction.float() - target_velocity.float()).square() * mask).sum(dim=(1, 2)) / denominator
    reference_mse = ((reference_prediction.float() - target_velocity.float()).square() * mask).sum(dim=(1, 2)) / denominator
    model_winner_mse, model_loser_mse = model_mse.chunk(2)
    reference_winner_mse, reference_loser_mse = reference_mse.chunk(2)
    output = flow_dpo_pair_loss(model_winner_mse, model_loser_mse, reference_winner_mse, reference_loser_mse, beta=beta)
    output.update({'model_winner_mse': model_winner_mse.detach().mean(), 'model_loser_mse': model_loser_mse.detach().mean(), 'reference_winner_mse': reference_winner_mse.detach().mean(), 'reference_loser_mse': reference_loser_mse.detach().mean(), 'beta': model_mse.new_tensor(float(beta)), 'timestep_mean': pair_timestep.detach().mean(), 'timestep_std': pair_timestep.detach().std(unbiased=False), 'timesteps_per_pair': model_mse.new_tensor(float(timesteps_per_pair))})
    return output

def sft_loss_g1(self, text_vec: Tensor, text_ctxt: List[Tensor], ctxt_len: Tensor, target_motion: Tensor, lengths: Tensor, good_mask: Optional[Tensor]=None, sample_weights: Optional[Tensor]=None, anchor_weight: float=0.0, gt_target: Optional[Tensor]=None, gt_lengths: Optional[Tensor]=None, gt_weight: float=0.0) -> Dict[str, Tensor]:
    """Reward-filtered flow-matching velocity SFT toward the selected sample.

        x0 ~ N(0, I); t ~ U(0, 1); x_t = (1-t) x0 + t x1; the FM target velocity
        is (x1 - x0).  Loss = ||predict_flow(x_t, t) - (x1 - x0)||^2, masked to
        valid frames, reward-filtered by ``good_mask`` (rejected prompts give 0
        SFT gradient), plus optional anchor MSE to the frozen base generator.

        When ``gt_target`` / ``gt_weight`` are given a ground-truth supervised FM
        term is added.  Both the reward target and the GT target share the SAME
        text conditioning (the batch prompts), so they are concatenated along the
        batch dim and pushed through **one** ``predict_flow`` forward -- this is
        deliberate: under DDP a second grad-forward on the wrapped transformer
        would trip the reducer ("mark ready twice").  The total objective is
        ``reward_sft + gt_weight * gt_sft + anchor_weight * anchor`` (a sum of
        per-group means, not a blended average).
        """
    device = self._device()
    dtype = torch.float32
    vtxt = text_vec.to(device, dtype)
    ctxt, ctxt_mask = pack_context(self, text_ctxt, ctxt_len, device, dtype)
    x1 = target_motion.to(device, dtype).detach()
    lengths = lengths.to(device)
    B, Lr, D = x1.shape
    use_gt = gt_target is not None and gt_weight and (gt_weight > 0)
    Lg = int(gt_target.shape[1]) if use_gt else 0
    L = max(Lr, Lg)
    if Lr < L:
        x1 = F.pad(x1, (0, 0, 0, L - Lr))
    x_mask = _len_to_mask(lengths, L)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device, dtype=dtype)
    x_t = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
    v_target = x1 - x0
    if use_gt:
        x1g = gt_target.to(device, dtype).detach()
        gt_lengths = gt_lengths.to(device)
        if Lg < L:
            x1g = F.pad(x1g, (0, 0, 0, L - Lg))
        x_mask_g = _len_to_mask(gt_lengths, L)
        x0g = torch.randn_like(x1g)
        tg = torch.rand(B, device=device, dtype=dtype)
        x_tg = (1.0 - tg.view(B, 1, 1)) * x0g + tg.view(B, 1, 1) * x1g
        v_target_g = x1g - x0g
        pred_cat = self.predict_flow(x_input=torch.cat([x_t, x_tg], dim=0), ctxt_input=torch.cat([ctxt, ctxt], dim=0), vtxt_input=torch.cat([vtxt, vtxt], dim=0), timesteps=torch.cat([t, tg], dim=0), x_mask_temporal=torch.cat([x_mask, x_mask_g], dim=0), ctxt_mask_temporal=torch.cat([ctxt_mask, ctxt_mask], dim=0))
        pred_v, pred_v_g = (pred_cat[:B], pred_cat[B:])
    else:
        pred_v = self.predict_flow(x_input=x_t, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t, x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)
    frame_mask = x_mask.unsqueeze(-1).to(dtype)
    err = (pred_v - v_target) ** 2 * frame_mask
    denom = (frame_mask.sum(dim=(1, 2)) * D).clamp_min(1.0)
    per_sample = err.sum(dim=(1, 2)) / denom
    gm = good_mask.to(device, dtype) if good_mask is not None else torch.ones_like(per_sample)
    w = sample_weights.to(device, dtype) * gm if sample_weights is not None else gm
    wsum = w.sum()
    sft = (per_sample * w).sum() / wsum.clamp_min(1e-08) if float(wsum) > 0 else (per_sample * 0.0).sum()
    n_good = gm.sum()
    out: Dict[str, Tensor] = {'sft_mse': ((per_sample * gm).sum() / n_good.clamp_min(1.0)).detach(), 'n_good': n_good.detach()}
    loss = sft
    if use_gt:
        fmg = x_mask_g.unsqueeze(-1).to(dtype)
        errg = (pred_v_g - v_target_g) ** 2 * fmg
        denomg = (fmg.sum(dim=(1, 2)) * D).clamp_min(1.0)
        gt_sft = (errg.sum(dim=(1, 2)) / denomg).mean()
        loss = loss + float(gt_weight) * gt_sft
        out['gt_mse'] = gt_sft.detach()
    if anchor_weight and anchor_weight > 0:
        anc = self._reference_transformer()
        if anc is None:
            if self.require_immutable_anchor:
                raise RuntimeError('anchor_weight > 0 but immutable G0 anchor is unavailable')
        else:
            with torch.no_grad():
                base_v = anc(x=x_t, ctxt_input=ctxt, vtxt_input=vtxt, timesteps=t, x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask, mask_density=None, task_emb=None)
            anchor_err = (pred_v - base_v) ** 2 * frame_mask
            anchor = (anchor_err.sum(dim=(1, 2)) / denom).mean()
            loss = loss + float(anchor_weight) * anchor
            out['anchor_mse'] = anchor.detach()
    out['loss'] = loss
    return out
