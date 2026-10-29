"""MoMask base-token refinement and residual-quantizer decoding."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical
from motius.motion.conditioning.momask_tokens import *
from .token_sampling import *
def base_forward_with_cond_scale(self,
                            motion_ids,
                            cond_vector,
                            padding_mask,
                            cond_scale=3,
                            force_mask=False):
    # bs = motion_ids.shape[0]
    # if cond_scale == 1:
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
def base_generate(self,
             conds,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False
             ):
    # print(self.opt.num_quantizers)
    # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

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
        raise NotImplementedError("Unsupported condition mode!!!")

    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    # print(padding_mask.shape, )

    # Start from all tokens being masked
    ids = torch.where(padding_mask, self.pad_id, self.mask_id)
    scores = torch.where(padding_mask, 1e5, 0.)
    starting_temperature = temperature

    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        # 0 < timestep < 1
        rand_mask_prob = cosine_schedule(timestep)  # Tensor

        '''
        Maskout, and cope with variable length
        '''
        # fix: the ratio regarding lengths, instead of seq_len
        num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

        # select num_token_masked tokens with lowest scores to be masked
        sorted_indices = scores.argsort(
            dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        is_mask = (ranks < num_token_masked.unsqueeze(-1))
        ids = torch.where(is_mask, self.mask_id, ids)

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = base_forward_with_cond_scale(self, ids, cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
        # print(logits.shape, self.opt.num_tokens)
        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        '''
        Update ids
        '''
        # if force_mask:
        temperature = starting_temperature
        # else:
        # temperature = starting_temperature * (steps_until_x0 / timesteps)
        # temperature = max(temperature, 1e-4)
        # print(filtered_logits.shape)
        # temperature is annealed, gradually reducing temperature as well as randomness
        if gsample:  # use gumbel_softmax sampling
            # print("1111")
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
        else:  # use multinomial sampling
            # print("2222")
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
            # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # print(probs / temperature)
            pred_ids = Categorical(probs).sample()  # (b, seqlen)

        # print(pred_ids.max(), pred_ids.min())
        # if pred_ids.
        ids = torch.where(is_mask, pred_ids, ids)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
        scores = scores.squeeze(-1)  # (b, seqlen)

        # We do not want to re-mask the previously kept tokens, or pad tokens
        scores = scores.masked_fill(~is_mask, 1e5)

    ids = torch.where(padding_mask, -1, ids)
    # print("Final", ids.max(), ids.min())
    return ids


@torch.no_grad()
@eval_decorator
def base_edit(self,
         conds,
         tokens,
         m_lens,
         timesteps: int,
         cond_scale: int,
         temperature=1,
         topk_filter_thres=0.9,
         gsample=False,
         force_mask=False,
         edit_mask=None,
         padding_mask=None,
         ):

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
        raise NotImplementedError("Unsupported condition mode!!!")

    if padding_mask == None:
        padding_mask = ~lengths_to_mask(m_lens, seq_len)

    # Start from all tokens being masked
    if edit_mask == None:
        mask_free = True
        ids = torch.where(padding_mask, self.pad_id, tokens)
        edit_mask = torch.ones_like(padding_mask)
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        scores = torch.where(edit_mask, 0., 1e5)
    else:
        mask_free = False
        edit_mask = edit_mask & ~padding_mask
        edit_len = edit_mask.sum(dim=-1)
        ids = torch.where(edit_mask, self.mask_id, tokens)
        scores = torch.where(edit_mask, 0., 1e5)
    starting_temperature = temperature

    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
        # 0 < timestep < 1
        rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)  # Tensor

        '''
        Maskout, and cope with variable length
        '''
        # fix: the ratio regarding lengths, instead of seq_len
        num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

        # select num_token_masked tokens with lowest scores to be masked
        sorted_indices = scores.argsort(
            dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        is_mask = (ranks < num_token_masked.unsqueeze(-1))
        # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
        ids = torch.where(is_mask, self.mask_id, ids)

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = base_forward_with_cond_scale(self, ids, cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
        # print(logits.shape, self.opt.num_tokens)
        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        '''
        Update ids
        '''
        # if force_mask:
        temperature = starting_temperature
        # else:
        # temperature = starting_temperature * (steps_until_x0 / timesteps)
        # temperature = max(temperature, 1e-4)
        # print(filtered_logits.shape)
        # temperature is annealed, gradually reducing temperature as well as randomness
        if gsample:  # use gumbel_softmax sampling
            # print("1111")
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
        else:  # use multinomial sampling
            # print("2222")
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
            # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # print(probs / temperature)
            pred_ids = Categorical(probs).sample()  # (b, seqlen)

        # print(pred_ids.max(), pred_ids.min())
        # if pred_ids.
        ids = torch.where(is_mask, pred_ids, ids)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
        scores = scores.squeeze(-1)  # (b, seqlen)

        # We do not want to re-mask the previously kept tokens, or pad tokens
        scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

    ids = torch.where(padding_mask, -1, ids)
    # print("Final", ids.max(), ids.min())
    return ids


@torch.no_grad()
@eval_decorator
def base_edit_beta(self,
              conds,
              conds_og,
              tokens,
              m_lens,
              cond_scale: int,
              force_mask=False,
              ):

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
        raise NotImplementedError("Unsupported condition mode!!!")

    padding_mask = ~lengths_to_mask(m_lens, seq_len)

    # Start from all tokens being masked
    ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

    '''
    Preparing input
    '''
    # (b, num_token, seqlen)
    logits = base_forward_with_cond_scale(self, ids,
                                          cond_vector=cond_vector,
                                          cond_vector_neg=cond_vector_og,
                                          padding_mask=padding_mask,
                                          cond_scale=cond_scale,
                                          force_mask=force_mask)

    logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

    '''
    Updating scores
    '''
    probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
    tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
    og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
    og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

    return og_tokens_scores


def residual_forward_with_cond_scale(self,
                            motion_codes,
                            q_id,
                            cond_vector,
                            padding_mask,
                            cond_scale=3,
                            force_mask=False):
    bs = motion_codes.shape[0]
    # if cond_scale == 1:
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
def residual_generate(self,
             motion_ids,
             conds,
             m_lens,
             temperature=1,
             topk_filter_thres=0.9,
             cond_scale=2,
             num_res_layers=-1, # If it's -1, use all.
             ):

    # print(self.opt.num_quantizers)
    # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
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
        raise NotImplementedError("Unsupported condition mode!!!")

    # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
    # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
    # history_sum = token_embed.gather(1, gathered_ids)

    # print(pa, seq_len)
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    # print(padding_mask.shape, motion_ids.shape)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0
    num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

    for i in range(1, num_quant_layers):
        # print(f"--> Working on {i}-th quantizer")
        # Start from all tokens being masked
        # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
        token_embed = self.token_embed_weight[i-1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)

        logits = residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
        # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

        # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
        # # print(temperature, starting_temperature, steps_until_x0, timesteps)
        # # print(probs / temperature)
        # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

        ids = torch.where(padding_mask, self.pad_id, pred_ids)

        motion_ids = ids
        all_indices.append(ids)

    all_indices = torch.stack(all_indices, dim=-1)
    # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
    # all_indices = torch.where(padding_mask, -1, all_indices)
    all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
    # all_indices = all_indices.masked_fill()
    return all_indices


@torch.no_grad()
@eval_decorator
def residual_edit(self,
        motion_ids,
        conds,
        m_lens,
        temperature=1,
        topk_filter_thres=0.9,
        cond_scale=2
        ):

    # print(self.opt.num_quantizers)
    # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
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
        raise NotImplementedError("Unsupported condition mode!!!")

    # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
    # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
    # history_sum = token_embed.gather(1, gathered_ids)

    # print(pa, seq_len)
    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    # print(padding_mask.shape, motion_ids.shape)
    motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
    all_indices = [motion_ids]
    history_sum = 0

    for i in range(1, self.opt.num_quantizers):
        # print(f"--> Working on {i}-th quantizer")
        # Start from all tokens being masked
        # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
        token_embed = self.token_embed_weight[i-1]
        token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
        gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        history_sum += token_embed.gather(1, gathered_ids)

        logits = residual_forward_with_cond_scale(self, history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
        # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

        # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
        # # print(temperature, starting_temperature, steps_until_x0, timesteps)
        # # print(probs / temperature)
        # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

        ids = torch.where(padding_mask, self.pad_id, pred_ids)

        motion_ids = ids
        all_indices.append(ids)

    all_indices = torch.stack(all_indices, dim=-1)
    # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
    # all_indices = torch.where(padding_mask, -1, all_indices)
    all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
    # all_indices = all_indices.masked_fill()
    return all_indices
