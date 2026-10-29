"""Original autoregressive selection with the official eval EOS fallback."""

import torch
from torch.nn import functional as F
from torch.distributions import Categorical


@torch.no_grad()
def sample_tokens(self, clip_feature, if_categorial=False):
    xs = None
    for k in range(self.block_size):
        if k == 0:
            x = []
        else:
            x = xs
        logits = self.forward(x, clip_feature)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        if if_categorial:
            dist = Categorical(probs)
            idx = dist.sample()
            if idx == self.num_vq:
                break
            idx = idx.unsqueeze(-1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
            if idx[0] == self.num_vq:
                break
        # append to the sequence and continue
        if k == 0:
            xs = idx
        else:
            xs = torch.cat((xs, idx), dim=1)

        if k == self.block_size - 1:
            return xs[:, :-1]
    return (
        xs
        if xs is not None
        else torch.ones((1, 1), device=clip_feature.device, dtype=torch.long)
    )
