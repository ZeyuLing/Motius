"""Bailando greedy autoregressive token sampling."""
import torch
from torch.nn import functional as F

def sample_codes(self, xs, cond, shift=None):
    shift = None
    block_size = self.get_block_size() - 1
    if shift is not None:
        block_shift = min(shift, block_size)
    else:
        block_shift = block_size
    x_up, x_down = xs
    for k in range(cond.size(1)):
        x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
        x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed

        cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]
        # print(x_cond.size())
        # print(cond_input.size())

        logits = self((x_cond_up, x_cond_down), cond_input)
        # jj += 1
        # pluck the logits at the final step and scale by temperature
        logit_up, logit_down = logits
        logit_up = logit_up[:, -1, :]
        logit_down = logit_down[:, -1, :]

        probs_up = F.softmax(logit_up, dim=-1)
        probs_down = F.softmax(logit_down, dim=-1)

        _, ix_up = torch.topk(probs_up, k=1, dim=-1)
        _, ix_down = torch.topk(probs_down, k=1, dim=-1)

        # append to the sequence and continue
        x_up = torch.cat((x_up, ix_up), dim=1)
        x_down = torch.cat((x_down, ix_down), dim=1)

    return ([x_up], [x_down])
