"""InterMask iterative token generation and reaction conditioning."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
import math
from torch.distributions.categorical import Categorical
from motius.motion.conditioning.intermask_tokens import lengths_to_mask, cosine_schedule
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

def forward_with_cond_scale(self,
                            motion_ids,
                            cond_vector,
                            padding_mask,
                            cond_scale=3,
                            force_mask=False):
    # bs = motion_ids.shape[0]
    # if cond_scale == 1:
    if force_mask:
        return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

    logits = self.trans_forward(motion_ids, cond_vector, padding_mask)
    if cond_scale == 1:
        return logits

    aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)

    scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
    return scaled_logits


@torch.no_grad()
@eval_decorator
def generate(self,
             conds,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False
             ):

    device = next(self.parameters()).device
    seq_len = max(m_lens)
    batch_size = len(m_lens)
    token_lengths = m_lens*2

    if self.cond_mode == 'text':
        cond_vector = encode_clip_text(self, conds)
    elif self.cond_mode == 'action':
        cond_vector = self.enc_action(conds).to(device)
    elif self.cond_mode == 'uncond':
        cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
    else:
        raise NotImplementedError("Unsupported condition mode!!!")

    padding_mask = ~lengths_to_mask(m_lens, seq_len)
    padding_mask = padding_mask.repeat(1, self.nbp)
    token_lengths = token_lengths*self.nbp

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
        num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

        # select num_token_masked tokens with lowest scores to be masked
        sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        is_mask = (ranks < num_token_masked.unsqueeze(-1))
        ids = torch.where(is_mask, self.mask_id, ids)

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = forward_with_cond_scale(self, ids, cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        '''
        Update ids
        '''
        temperature = starting_temperature
        # temperature is annealed, gradually reducing temperature hence randomness
        if gsample:  # use gumbel_softmax sampling
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
        else:  # use multinomial sampling
            probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

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
    return ids


@torch.no_grad()
@eval_decorator
def generate_reaction(self,
                        conds,
                        motion1_ids,
                        m_lens,
                        timesteps: int,
                        cond_scale: int,
                        temperature=1,
                        topk_filter_thres=0.9,
                        gsample=False,
                        force_mask=False
                        ):

    device = next(self.parameters()).device
    seq_len = max(m_lens)
    batch_size = len(m_lens)
    token_lengths = m_lens

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
    padding_mask = padding_mask.repeat(1, self.nbp)
    token_lengths = token_lengths*self.nbp

    # Start from all tokens being masked
    ids1 = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, motion1_ids)
    ids = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, self.mask_id)
    scores = torch.where(ids == self.mask_id, 0., 1e5)

    starting_temperature = temperature
    for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps//2, device=device), reversed(range(timesteps))):
        # 0 < timestep < 1
        rand_mask_prob = cosine_schedule(timestep)  # Tensor

        '''
        Maskout, and cope with variable length
        '''
        # fix: the ratio regarding lengths, instead of seq_len
        num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

        # select num_token_masked tokens with lowest scores to be masked
        sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        is_mask = (ranks < num_token_masked.unsqueeze(-1))
        ids = torch.where(is_mask, self.mask_id, ids)

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = forward_with_cond_scale(self, torch.cat((ids1,ids), dim=-1), cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)
        logits = logits[:, :, padding_mask.shape[1]//2:]

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
        # print(logits.shape, self.opt.num_tokens)
        # clean low prob token
        filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

        '''
        Update ids
        '''
        # if force_mask:
        temperature = starting_temperature
        if gsample:  # use gumbel_softmax sampling
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
        else:  # use multinomial sampling
            probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

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

    ids = torch.cat((ids1, ids), dim=-1)
    ids = torch.where(padding_mask, -1, ids)

    return ids
