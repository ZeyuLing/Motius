"""Autoregressive token selection outside the neural transformer."""

import torch
from torch.nn import functional as F
from torch.distributions import Categorical


@torch.no_grad()
def sample_tokens(
    self, clip_feature, y_mask, if_categorial=False, max_sample_steps: int = 50
):
    """Greedy (or categorical) AR decoding. Returns token indices ``(1, L)``.

    Matches the released sampler: at most ``max_sample_steps`` motion tokens are
    produced, with early-stop on the EOS token (``vocab_size - 2``). The released
    default is 50.
    """
    xs = None
    for k in range(max_sample_steps + 1):
        x = [] if k == 0 else xs
        logits = self.forward_sample(x, clip_feature, y_mask)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        if if_categorial:
            dist = Categorical(probs)
            idx = dist.sample()
            if idx == self.config.vocab_size - 2:
                break
            idx = idx.unsqueeze(-1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
            if idx[0] == self.config.vocab_size - 2:
                break
        xs = idx if k == 0 else torch.cat((xs, idx), dim=1)
        if k == max_sample_steps:
            return xs[:, :-1]
    if xs is None:
        return torch.ones(1, 1, device=clip_feature.device).long()
    return xs


@torch.no_grad()
def sample_tokens_cached(
    self, clip_feature, y_mask, if_categorial=False, max_sample_steps: int = 150
):
    """KV-cached equivalent of :meth:`sample` (O(n) instead of O(n^2)).

    Produces the same greedy/categorical token stream as :meth:`sample` but
    reuses cached keys/values: a text *prefill* fills the cache, then each motion
    token is decoded with a single-token forward. Batch size must be 1.
    """
    text_len = int(y_mask[0].sum())
    caches = [{"k": None, "v": None} for _ in range(self.config.n_layer)]

    # --- prefill: encode the (bidirectional) text prefix into the cache ---
    x = self.llama_proj(clip_feature)[:, :text_len, :]
    for block, c in zip(self.transformer.h, caches):
        x = block.forward_cached(x, c, input_pos=0)
    logits = self.lm_head(self.transformer.ln_f(x)[:, -1, :])

    xs = None
    for k in range(max_sample_steps + 1):
        probs = F.softmax(logits, dim=-1)
        if if_categorial:
            idx = Categorical(probs).sample()
            if idx == self.config.vocab_size - 2:
                break
            idx = idx.unsqueeze(-1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
            if idx[0] == self.config.vocab_size - 2:
                break
        xs = idx if k == 0 else torch.cat((xs, idx), dim=1)
        if k == max_sample_steps:
            return xs[:, :-1]
        # --- decode: feed the just-produced token (position text_len + k) ---
        x = self.transformer.wte(idx)
        for block, c in zip(self.transformer.h, caches):
            x = block.forward_cached(x, c, input_pos=text_len + k)
        logits = self.lm_head(self.transformer.ln_f(x)[:, -1, :])
    if xs is None:
        return torch.ones(1, 1, device=clip_feature.device).long()
    return xs
