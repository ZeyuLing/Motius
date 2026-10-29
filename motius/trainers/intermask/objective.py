"""Masking curriculum and step-unroll objective from InterMask."""
from motius.motion.conditioning.clip_text import encode_clip_text
import torch
import torch.nn.functional as F
from motius.motion.conditioning.intermask_tokens import (lengths_to_mask, uniform, get_mask_subset_prob, cosine_schedule, cosine_schedule_backward)
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

def compute_objective(self, predict, ids, y, m_lens):
    '''
    :param ids: (b, n)
    :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
    :m_lens: (b,)
    :return:
    '''

    bs, ntokens = ids.shape
    device = ids.device

    # Positions that are PADDED are ALL FALSE
    max_len = ntokens/2
    max_len = max_len/self.nbp

    non_pad_mask = lengths_to_mask(m_lens, max_len) #(b, n)
    non_pad_mask = non_pad_mask.repeat(1, self.nbp)
    # print(f">>> Pad mask: {non_pad_mask.reshape(2,2,-1).reshape(2,2,5,-1)}")

    ids = torch.where(non_pad_mask, ids, self.pad_id)
    # print(f">>> Padded ids: {ids.reshape(2,2,-1).reshape(2,2,5,-1)}")

    force_mask = False
    if self.cond_mode == 'text':
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
    interaction_mask = torch.bernoulli(torch.tensor(self.opt.interaction_mask_prob)).bool().item()
    # print(f"Interaction: {interaction_mask}")
    if interaction_mask:
        mask_person_one = torch.bernoulli(torch.tensor(0.5)).bool().item()
        # print(f"zero first: {zeros_first}")
        if mask_person_one:
            half_batch_mask = torch.cat((torch.zeros((bs//2,ntokens//2)), torch.ones((bs//2,ntokens//2))), dim=-1).bool()
        else:
            half_batch_mask = torch.cat((torch.ones((bs//2,ntokens//2)), torch.zeros((bs//2,ntokens//2))), dim=-1).bool()
        mask = torch.cat((half_batch_mask, ~half_batch_mask), dim=0).to(device)
        rand_mask_probs = torch.ones((bs,), device=device) * 0.5
        rand_time = cosine_schedule_backward(rand_mask_probs)

    else:
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

    x_ids = torch.where(mask_mid, self.mask_id, x_ids)

    logits = predict(x_ids, cond_vector, ~non_pad_mask, force_mask)
    ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

    logits = logits.permute(0,2,1) # B,ntokens,T -> B,T,ntokens

    if self.opt.step_unroll:
        su_ce_loss, su_pred_id, su_acc = step_unroll_forward(self, predict, x_ids, mask_mid, labels, logits,
                                                            cond_vector, non_pad_mask, force_mask)

        return ce_loss + (self.opt.step_unroll * su_ce_loss), (acc + self.opt.step_unroll*su_acc)/2, pred_id, su_pred_id, logits
    else:
        return ce_loss, acc, pred_id, None, logits


def step_unroll_forward(self, predict, prev_masked_ids, prev_mask, prev_labels, logits, cond_vector, non_pad_mask, force_mask):
    # print(f">>>>>>>>>>>> Step unroll >>>>>>>>>>>>>>>")
    total_timesteps = 20
    prev_rand_mask_probs = prev_mask.count_nonzero(dim = -1).float() / prev_mask.shape[-1]
    prev_rand_time = cosine_schedule_backward(prev_rand_mask_probs)

    rand_time = (prev_rand_time + (1/(total_timesteps+1))).clamp(max=1)
    rand_mask_probs = cosine_schedule(rand_time)

    probs = logits.softmax(dim=-1)
    scores, pred_ids = probs.max(dim=-1)
    scores = scores.masked_fill(~prev_mask, 1e5)

    sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
    ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
    num_token_masked = torch.round(rand_mask_probs * (scores.shape[-1])).clamp(min=1)

    mask = (ranks < num_token_masked.unsqueeze(-1))

    retained_preds = torch.logical_and(prev_mask == True,  mask == False)
    labels = torch.where(retained_preds, self.mask_id, prev_labels)

    x_ids = torch.where(retained_preds, pred_ids, prev_masked_ids)

    step_unroll_logits = predict(x_ids, cond_vector, ~non_pad_mask, force_mask)
    return cal_performance(step_unroll_logits, labels, ignore_index=self.mask_id)
