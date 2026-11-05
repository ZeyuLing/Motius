# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 maths components; see licenses/sonic."""
import torch


@torch.jit.script
def normalize(x, eps: float = 1e-09):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)
