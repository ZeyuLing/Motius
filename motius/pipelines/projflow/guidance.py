"""Classifier-free guidance composition for the ACMDM flow prior."""

import torch


def guided_velocity(model, x, t, conds, attention_mask, cfg=1.0, **kwargs):
    if cfg != 1.0:
        half = x[: len(x) // 2]
        x = torch.cat([half, half], dim=0)
    x = model(x, t, conds, attention_mask)
    if cfg != 1.0:
        conditional, unconditional = torch.split(x, len(x) // 2, dim=0)
        half = unconditional + cfg * (conditional - unconditional)
        x = torch.cat([half, half], dim=0)
    return x
