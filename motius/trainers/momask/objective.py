"""MoMask native base and residual token objectives."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.distributions.categorical import Categorical
from motius.motion.conditioning.momask_tokens import *
def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc

def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss
def compute_base_objective(self, predict, ids, y, m_lens):
    '''
    :param ids: (b, n)
    :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
    :m_lens: (b,)
    :return:
    '''

    bs, ntokens = ids.shape
    device = ids.device

    # Positions that are PADDED are ALL FALSE
    non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
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
        raise NotImplementedError("Unsupported condition mode!!!")


    '''
    Prepare mask
    '''
    rand_time = uniform((bs,), device=device)
    rand_mask_probs = cosine_schedule(rand_time)
    num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

    batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
    # Positions to be MASKED are ALL TRUE
    mask = batch_randperm < num_token_masked.unsqueeze(-1)

    # Positions to be MASKED must also be NON-PADDED
    mask &= non_pad_mask

    # Note this is our training target, not input
    labels = torch.where(mask, ids, self.mask_id)

    x_ids = ids.clone()

    # Further Apply Bert Masking Scheme
    # Step 1: 10% replace with an incorrect token
    mask_rid = get_mask_subset_prob(mask, 0.1)
    rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
    x_ids = torch.where(mask_rid, rand_id, x_ids)
    # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
    mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

    # mask_mid = mask

    x_ids = torch.where(mask_mid, self.mask_id, x_ids)

    logits = predict(x_ids, cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

    return ce_loss, pred_id, acc


def compute_residual_objective(self, predict, all_indices, y, m_lens):
    '''
    :param all_indices: (b, n, q)
    :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
    :m_lens: (b,)
    :return:
    '''

    self.process_embed_proj_weight()

    bs, ntokens, num_quant_layers = all_indices.shape
    device = all_indices.device

    # Positions that are PADDED are ALL FALSE
    non_pad_mask = lengths_to_mask(m_lens, ntokens)  # (b, n)

    q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
    all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

    # randomly sample quantization layers to work on, [1, num_q)
    active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

    # print(self.token_embed_weight.shape, all_indices.shape)
    token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
    gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
    # print(token_embed.shape, gather_indices.shape)
    all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)

    cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)

    active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
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
        raise NotImplementedError("Unsupported condition mode!!!")

    logits = predict(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

    return ce_loss, pred_id, acc
