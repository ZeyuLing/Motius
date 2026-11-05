"""TM2D autoregressive token sampling."""

import torch
import torch.nn.functional as F


def generate_from_encoded(
    self,
    encoded,
    source_mask,
    initial_tokens,
    steps,
    *,
    sample=False,
    top_k=None,
    forbidden_ids=(),
    generator=None,
):
    tokens = initial_tokens
    for _ in range(int(steps)):
        logits = self.decode_logits(tokens, encoded, source_mask, mask_padding=False)[
            :, -1
        ]
        if top_k is not None and 0 < int(top_k) < logits.shape[-1]:
            threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        probabilities = F.softmax(logits, dim=-1)
        if sample:
            next_token = torch.multinomial(probabilities, 1, generator=generator)
            if forbidden_ids:
                invalid = torch.tensor(forbidden_ids, device=tokens.device)
                # The author samples from the original distribution and redraws
                # special tokens. Pre-masking changes the seeded token stream.
                valid_mass = probabilities.clone()
                valid_mass[:, list(forbidden_ids)] = 0
                if torch.any(valid_mass.sum(-1) == 0):
                    raise ValueError("TM2D top_k contains only special tokens")
                while torch.isin(next_token, invalid).any():
                    next_token = torch.multinomial(
                        probabilities, 1, generator=generator
                    )
        else:
            next_token = torch.topk(probabilities, 1, dim=-1).indices
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
