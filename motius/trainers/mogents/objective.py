"""MoGenTS released token objectives and masking draws."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
from einops import repeat,rearrange
from motius.motion.conditioning.momask_tokens import lengths_to_mask,uniform,get_mask_subset_prob,cosine_schedule,q_schedule
from motius.trainers.momask.objective import cal_performance

def compute_aux_base_objective(self, predict, ids, y, m_lens):
    """
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        """
    bs, ntokens = ids.shape
    device = ids.device
    non_pad_mask = lengths_to_mask(m_lens, ntokens)
    ids = torch.where(non_pad_mask, ids, self.pad_id)
    force_mask = False
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, y)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(y).to(device).float()
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        force_mask = True
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    '\n        Prepare mask\n        '
    rand_time = uniform((bs,), device=device)
    rand_mask_probs = cosine_schedule(rand_time)
    num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask &= non_pad_mask
    labels = torch.where(mask, ids, self.mask_id)
    x_ids = ids.clone()
    mask_rid = get_mask_subset_prob(mask, 0.1)
    rand_id = torch.randint_like(x_ids, high=self.mask_id)
    x_ids = torch.where(mask_rid, rand_id, x_ids)
    mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
    x_ids = torch.where(mask_mid, self.mask_id, x_ids)
    logits = predict(x_ids, cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
    return (ce_loss, pred_id, acc)

def compute_aux_residual_objective(self, predict, all_indices, y, m_lens):
    """
        :param all_indices: (b, n, q)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        """
    self.process_embed_proj_weight()
    bs, ntokens, num_quant_layers = all_indices.shape
    device = all_indices.device
    non_pad_mask = lengths_to_mask(m_lens, ntokens)
    q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
    all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)
    active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)
    token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
    gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
    all_codes = token_embed.gather(1, gather_indices)
    cumsum_codes = torch.cumsum(all_codes, dim=-1)
    active_indices = all_indices[torch.arange(bs), :, active_q_layers]
    history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]
    force_mask = False
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, y)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(y).to(device).float()
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        force_mask = True
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    logits = predict(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
    return (ce_loss, pred_id, acc)

def compute_ts_base_objective(self, predict, ids_j, y, m_lens):
    """
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        """
    device = ids_j.device
    bs, xtokens, ytokens = ids_j.shape
    non_pad_mask = torch.arange(xtokens, device=device).expand(bs, xtokens) < m_lens.unsqueeze(1)
    non_pad_mask = non_pad_mask[..., None].repeat(1, 1, ytokens)
    ids_j = torch.where(non_pad_mask, ids_j, self.pad_id)
    force_mask = False
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, y)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(y).to(device).float()
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        force_mask = True
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    '\n        Prepare mask\n        '
    ntokens = xtokens * ytokens
    rand_time = uniform((bs,), device=device)
    rand_mask_probs = cosine_schedule(rand_time)
    num_token_masked = (xtokens * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((bs, xtokens), device=device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask & non_pad_mask[..., 0]
    labels = torch.where(mask[..., None].repeat(1, 1, ytokens), ids_j, self.mask_id)
    x_ids_j = ids_j.clone()
    mask_rid = get_mask_subset_prob(mask, 0.1)
    rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
    x_ids_j = torch.where(mask_rid[..., None].repeat(1, 1, ytokens), rand_id, x_ids_j)
    mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
    x_ids_j = torch.where(mask_mid[..., None].repeat(1, 1, ytokens), self.mask_id, x_ids_j)
    mask_time = mask
    mask_time = mask_time[..., None].repeat(1, 1, ytokens)
    num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
    batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask & non_pad_mask.reshape(bs, -1)
    mask = mask & ~mask_time.reshape(bs, -1)
    labels = torch.where(mask, x_ids_j.reshape(bs, -1), labels.reshape(bs, -1))
    x_ids_j = x_ids_j.reshape(bs, -1)
    mask_rid = get_mask_subset_prob(mask, 0.1)
    rand_id = torch.randint_like(x_ids_j, high=self.opt.num_tokens2d)
    x_ids_j = torch.where(mask_rid, rand_id, x_ids_j)
    mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
    x_ids_j = torch.where(mask_mid, self.mask_id, x_ids_j)
    logits = predict(x_ids_j.reshape(bs, xtokens, ytokens), cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
    return (ce_loss, pred_id, acc)

def compute_ts_residual_objective(self, predict, all_indices, y, m_lens):
    """
        :param all_indices: (b, n, q)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        """
    self.process_embed_proj_weight()
    bs, xtokens, ytokens, num_quant_layers = all_indices.shape
    ntokens = xtokens * ytokens
    all_indices = all_indices.reshape(bs, ntokens, num_quant_layers)
    device = all_indices.device
    non_pad_mask = torch.arange(xtokens, device=device).expand(bs, xtokens) < m_lens.unsqueeze(1)
    non_pad_mask = non_pad_mask[..., None].repeat(1, 1, ytokens)
    non_pad_mask = non_pad_mask.reshape(bs, xtokens * ytokens)
    q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
    all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)
    active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)
    token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
    gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
    all_codes = token_embed.gather(1, gather_indices)
    cumsum_codes = torch.cumsum(all_codes, dim=-1)
    active_indices = all_indices[torch.arange(bs), :, active_q_layers]
    history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]
    force_mask = False
    if self.cond_mode == 'text':
        with torch.no_grad():
            cond_vector = encode_clip_text(self, y)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(y).to(device).float()
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
        force_mask = True
    else:
        raise NotImplementedError('Unsupported condition mode!!!')
    logits = predict(history_sum, active_q_layers, cond_vector, ~non_pad_mask, xtokens, ytokens, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
    return (ce_loss, pred_id, acc)
