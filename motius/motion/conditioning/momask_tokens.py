"""Released MoMask token masking and schedules."""
import torch
import math
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low
