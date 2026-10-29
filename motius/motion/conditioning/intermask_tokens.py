"""Released InterMask token masks and schedules."""
import torch
import math
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).repeat(len(lengths), 2) < lengths.unsqueeze(1)
    # mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def uniform(shape, device=None, min=0, max=1):
    return torch.zeros(shape, device=device).float().uniform_(min, max)

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def cosine_schedule_backward(x):
    return torch.acos(x) / (math.pi * 0.5)
