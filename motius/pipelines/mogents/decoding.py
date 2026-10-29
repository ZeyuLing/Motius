"""MoGenTS token inference, including spatial and temporal refinement."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
import math
from einops import rearrange,repeat
from torch.distributions.categorical import Categorical
from motius.motion.conditioning.momask_tokens import lengths_to_mask,uniform,get_mask_subset_prob,cosine_schedule,q_schedule
from motius.pipelines.momask.token_sampling import eval_decorator,gumbel_sample,top_k

def aux_base_forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask, cond_scale=3, force_mask=False):
    if force_mask:
        return self(motion_ids, cond_vector, padding_mask, force_mask=True)
    logits = self(motion_ids, cond_vector, padding_mask)
    if cond_scale == 1:
        return logits
    aux_logits = self(motion_ids, cond_vector, padding_mask, force_mask=True)
    scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    return scaled_logits

@torch.no_grad()
@eval_decorator
def aux_base_generate(self, conds, m_lens, timesteps: int, cond_scale: int, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False):
    device = next(self.parameters()).device
    seq_len = max(m_lens)
    batch_size = len(m_lens)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    ids = torch.where(padding_mask, self.pad_id, self.mask_id)
    scores = torch.where(padding_mask, 100000.0, 0.0)
    starting_temperature = temperature
    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        rand_mask_prob = cosine_schedule(timestep)
        '\n            Maskout, and cope with variable length\n            '
        num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
        sorted_indices = scores.argsort(dim=1)
        ranks = sorted_indices.argsort(dim=1)
        is_mask = ranks < num_token_masked.unsqueeze(-1)
        ids = torch.where(is_mask, self.mask_id, ids)
        '\n            Preparing input\n            '
        logits = aux_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        '\n            Update ids\n            '
        temperature = starting_temperature
        if gsample:
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)
            pred_ids = Categorical(probs / temperature).sample()
        ids = torch.where(is_mask, pred_ids, ids)
        '\n            Updating scores\n            '
        probs_without_temperature = logits.softmax(dim=-1)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~is_mask, 100000.0)
    ids = torch.where(padding_mask, -1, ids)
    return ids

@torch.no_grad()
@eval_decorator
def aux_base_edit(self, conds, tokens, m_lens, timesteps: int, cond_scale: int, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False, edit_mask=None, padding_mask=None):
    assert edit_mask.shape == tokens.shape if edit_mask is not None else True
    device = next(self.parameters()).device
    seq_len = tokens.shape[1]
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    if padding_mask == None:
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
    if edit_mask == None:
        mask_free = True
        ids = torch.where(padding_mask, self.pad_id, tokens)
        edit_mask = torch.ones_like(padding_mask)
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        scores = torch.where(edit_mask, 0.0, 100000.0)
    else:
        mask_free = False
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        ids = torch.where(edit_mask, self.mask_id, tokens)
        scores = torch.where(edit_mask, 0.0, 100000.0)
    starting_temperature = temperature
    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
        '\n            Maskout, and cope with variable length\n            '
        num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
        sorted_indices = scores.argsort(dim=1)
        ranks = sorted_indices.argsort(dim=1)
        is_mask = ranks < num_token_masked.unsqueeze(-1)
        ids = torch.where(is_mask, self.mask_id, ids)
        '\n            Preparing input\n            '
        logits = aux_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        '\n            Update ids\n            '
        temperature = starting_temperature
        if gsample:
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)
            pred_ids = Categorical(probs / temperature).sample()
        ids = torch.where(is_mask, pred_ids, ids)
        '\n            Updating scores\n            '
        probs_without_temperature = logits.softmax(dim=-1)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~edit_mask, 100000.0) if mask_free else scores.masked_fill(~is_mask, 100000.0)
    ids = torch.where(padding_mask, -1, ids)
    return ids

@torch.no_grad()
@eval_decorator
def aux_base_edit_beta(self, conds, conds_og, tokens, m_lens, cond_scale: int, force_mask=False):
    device = next(self.parameters()).device
    seq_len = tokens.shape[1]
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
            if conds_og is not None:
                cond_vector_og = encode_clip_text(self, conds_og)
            else:
                cond_vector_og = None
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
        if conds_og is not None:
            cond_vector_og = self.enc_action(conds_og).to(device)
        else:
            cond_vector_og = None
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    ids = torch.where(padding_mask, self.pad_id, tokens)
    '\n        Preparing input\n        '
    logits = aux_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, cond_vector_neg=cond_vector_og, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
    logits = logits.permute(0, 2, 1)
    '\n        Updating scores\n        '
    probs_without_temperature = logits.softmax(dim=-1)
    tokens[tokens == -1] = 0
    og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))
    og_tokens_scores = og_tokens_scores.squeeze(-1)
    return og_tokens_scores

def aux_residual_forward_with_cond_scale(self, motion_codes, q_id, cond_vector, padding_mask, cond_scale=3, force_mask=False):
    bs = motion_codes.shape[0]
    qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
    if force_mask:
        logits = self(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        return logits
    logits = self(motion_codes, qids, cond_vector, padding_mask)
    if cond_scale == 1:
        return logits
    aux_logits = self(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
    scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    return scaled_logits

@torch.no_grad()
@eval_decorator
def aux_residual_generate(self, motion_ids, conds, m_lens, temperature=1, topk_filter_thres=0.9, cond_scale=2, num_res_layers=-1):
    self.process_embed_proj_weight()
    device = next(self.parameters()).device
    seq_len = motion_ids.shape[1]
    batch_size = len(conds)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0
    num_quant_layers = self.opt.num_quantizers if num_res_layers == -1 else num_res_layers + 1
    for i in range(1, num_quant_layers):
        token_embed = self.token_embed_weight[i - 1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)
        logits = aux_residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        ids = torch.where(padding_mask, self.pad_id, pred_ids)
        motion_ids = ids
        all_indices.append(ids)
    all_indices = torch.stack(all_indices, dim=-1)
    all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
    return all_indices

@torch.no_grad()
@eval_decorator
def aux_residual_edit(self, motion_ids, conds, m_lens, temperature=1, topk_filter_thres=0.9, cond_scale=2):
    self.process_embed_proj_weight()
    device = next(self.parameters()).device
    seq_len = motion_ids.shape[1]
    batch_size = len(conds)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0
    for i in range(1, self.opt.num_quantizers):
        token_embed = self.token_embed_weight[i - 1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)
        logits = aux_residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        ids = torch.where(padding_mask, self.pad_id, pred_ids)
        motion_ids = ids
        all_indices.append(ids)
    all_indices = torch.stack(all_indices, dim=-1)
    all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
    return all_indices

def ts_base_forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask, cond_scale=3, force_mask=False):
    if force_mask:
        return self(motion_ids, cond_vector, padding_mask, force_mask=True)
    logits = self(motion_ids, cond_vector, padding_mask)
    if cond_scale == 1:
        return logits
    aux_logits = self(motion_ids, cond_vector, padding_mask, force_mask=True)
    scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    return scaled_logits

@torch.no_grad()
@eval_decorator
def ts_base_generate(self, conds, m_lens, timesteps: int, cond_scale: int, n_j=1, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False):
    device = next(self.parameters()).device
    seq_len = max(m_lens)
    batch_size = len(m_lens)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    non_pad_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < m_lens.unsqueeze(1)
    padding_mask = ~non_pad_mask[..., None].repeat(1, 1, n_j)
    ids = torch.where(padding_mask, self.pad_id, self.mask_id)
    scores = torch.where(padding_mask, 100000.0, 0.0).reshape(batch_size, seq_len * n_j)
    starting_temperature = temperature
    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        rand_mask_prob = cosine_schedule(timestep)
        '\n            Maskout, and cope with variable length\n            '
        num_token_masked = torch.round(rand_mask_prob * m_lens * n_j).clamp(min=1)
        sorted_indices = scores.argsort(dim=1)
        ranks = sorted_indices.argsort(dim=1)
        is_mask = ranks < num_token_masked.unsqueeze(-1)
        ids = torch.where(is_mask, self.mask_id, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)
        '\n            Preparing input\n            '
        logits = ts_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        '\n            Update ids\n            '
        temperature = starting_temperature
        if gsample:
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)
            pred_ids = Categorical(probs / temperature).sample()
        ids = torch.where(is_mask, pred_ids, ids.reshape(batch_size, -1)).reshape(batch_size, seq_len, n_j)
        '\n            Updating scores\n            '
        probs_without_temperature = logits.softmax(dim=-1)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~is_mask, 100000.0)
        scores = scores.reshape(batch_size, seq_len * n_j)
    ids = torch.where(padding_mask, -1, ids)
    return ids

@torch.no_grad()
@eval_decorator
def ts_base_edit(self, conds, tokens, m_lens, timesteps: int, cond_scale: int, temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False, edit_mask=None, padding_mask=None):
    assert edit_mask.shape == tokens.shape if edit_mask is not None else True
    device = next(self.parameters()).device
    seq_len = tokens.shape[1]
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    if padding_mask == None:
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
    if edit_mask == None:
        mask_free = True
        ids = torch.where(padding_mask, self.pad_id, tokens)
        edit_mask = torch.ones_like(padding_mask)
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        scores = torch.where(edit_mask, 0.0, 100000.0)
    else:
        mask_free = False
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        ids = torch.where(edit_mask, self.mask_id, tokens)
        scores = torch.where(edit_mask, 0.0, 100000.0)
    starting_temperature = temperature
    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
        '\n            Maskout, and cope with variable length\n            '
        num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
        sorted_indices = scores.argsort(dim=1)
        ranks = sorted_indices.argsort(dim=1)
        is_mask = ranks < num_token_masked.unsqueeze(-1)
        ids = torch.where(is_mask, self.mask_id, ids)
        '\n            Preparing input\n            '
        logits = ts_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        '\n            Update ids\n            '
        temperature = starting_temperature
        if gsample:
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)
            pred_ids = Categorical(probs / temperature).sample()
        ids = torch.where(is_mask, pred_ids, ids)
        '\n            Updating scores\n            '
        probs_without_temperature = logits.softmax(dim=-1)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~edit_mask, 100000.0) if mask_free else scores.masked_fill(~is_mask, 100000.0)
    ids = torch.where(padding_mask, -1, ids)
    return ids

@torch.no_grad()
@eval_decorator
def ts_base_edit_beta(self, conds, conds_og, tokens, m_lens, cond_scale: int, force_mask=False):
    device = next(self.parameters()).device
    seq_len = tokens.shape[1]
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
            if conds_og is not None:
                cond_vector_og = encode_clip_text(self, conds_og)
            else:
                cond_vector_og = None
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
        if conds_og is not None:
            cond_vector_og = self.enc_action(conds_og).to(device)
        else:
            cond_vector_og = None
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    ids = torch.where(padding_mask, self.pad_id, tokens)
    '\n        Preparing input\n        '
    logits = ts_base_forward_with_cond_scale(self,ids, cond_vector=cond_vector, cond_vector_neg=cond_vector_og, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)
    logits = logits.permute(0, 2, 1)
    '\n        Updating scores\n        '
    probs_without_temperature = logits.softmax(dim=-1)
    tokens[tokens == -1] = 0
    og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))
    og_tokens_scores = og_tokens_scores.squeeze(-1)
    return og_tokens_scores

def ts_residual_forward_with_cond_scale(self, motion_codes, q_id, cond_vector, padding_mask, T, J, cond_scale=3, force_mask=False):
    bs = motion_codes.shape[0]
    qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
    if force_mask:
        logits = self(motion_codes, qids, cond_vector, padding_mask, T, J, force_mask=True)
        return logits
    logits = self(motion_codes, qids, cond_vector, padding_mask, T, J)
    if cond_scale == 1:
        return logits
    aux_logits = self(motion_codes, qids, cond_vector, padding_mask, T, J, force_mask=True)
    scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    return scaled_logits

@torch.no_grad()
@eval_decorator
def ts_residual_generate(self, motion_ids, conds, m_lens, temperature=1, topk_filter_thres=0.9, cond_scale=2, num_res_layers=-1):
    self.process_embed_proj_weight()
    device = next(self.parameters()).device
    B, T, J = motion_ids.shape
    motion_ids = motion_ids.reshape(B, T * J)
    seq_len = T
    batch_size = len(conds)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    non_pad_mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) < m_lens.unsqueeze(1)
    padding_mask = ~non_pad_mask[..., None].repeat(1, 1, J)
    padding_mask = padding_mask.reshape(B, T * J)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0
    num_quant_layers = self.opt.num_quantizers if num_res_layers == -1 else num_res_layers + 1
    for i in range(1, num_quant_layers):
        token_embed = self.token_embed_weight[i - 1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)
        logits = ts_residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, T, J, cond_scale=cond_scale)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        ids = torch.where(padding_mask, self.pad_id, pred_ids)
        motion_ids = ids
        all_indices.append(ids)
    all_indices = torch.stack(all_indices, dim=-1)
    all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
    if all_indices.max() == self.mask_id:
        print('!!! Warning: id exceeds mask_id !!!')
        all_indices = all_indices.clip(0, self.mask_id - 1)
    return all_indices.reshape(B, T, J, num_quant_layers)

@torch.no_grad()
@eval_decorator
def ts_residual_edit(self, motion_ids, conds, m_lens, temperature=1, topk_filter_thres=0.9, cond_scale=2):
    self.process_embed_proj_weight()
    device = next(self.parameters()).device
    seq_len = motion_ids.shape[1]
    batch_size = len(conds)
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0
    for i in range(1, self.opt.num_quantizers):
        token_embed = self.token_embed_weight[i - 1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)
        logits = ts_residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
        logits = logits.permute(0, 2, 1)
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
        ids = torch.where(padding_mask, self.pad_id, pred_ids)
        motion_ids = ids
        all_indices.append(ids)
    all_indices = torch.stack(all_indices, dim=-1)
    all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
    return all_indices
