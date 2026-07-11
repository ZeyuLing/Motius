"""Minimal inference helpers used by the MDM runtime."""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedSum(nn.Module):
    """Learnable convex-ish combination over rows (used by multi-target cond)."""

    def __init__(self, num_rows):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_rows))

    def forward(self, x):
        normalized_weights = self.weights / self.weights.sum()
        return torch.matmul(normalized_weights, x)


def wrapped_getattr(self, name, default=None, wrapped_member_name="model"):
    """Attribute delegation for model wrappers (e.g. the CFG sampler / spaced
    diffusion ``_WrappedModel``)."""
    if isinstance(self, torch.nn.Module):
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return attr
