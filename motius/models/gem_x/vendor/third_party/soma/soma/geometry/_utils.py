# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def require_torch_tensors(*tensors, name="inputs"):
    """Validate that all inputs are torch.Tensors with matching dtype and device."""
    if not tensors:
        raise ValueError(f"{name} must not be empty.")
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError(f"All {name} must be torch.Tensor.")
    dtypes = {x.dtype for x in tensors}
    devices = {x.device for x in tensors}
    if len(dtypes) != 1:
        raise TypeError(f"All {name} must share dtype; got {dtypes}.")
    if len(devices) != 1:
        raise TypeError(f"All {name} must be on the same device; got {devices}.")
    return dtypes.pop(), devices.pop()


def one_hot_1d(L, idx, *, dtype, device):
    """(L,) with 1 at idx."""
    return torch.eye(L, dtype=dtype, device=device)[idx]


def mask_1d(L, indices, *, dtype, device):
    """(L,) with 1 at all 'indices' (e.g., [0,2] for x/z)."""
    if not indices:
        return torch.zeros((L,), dtype=dtype, device=device)
    ohs = torch.stack([one_hot_1d(L, i, dtype=dtype, device=device) for i in indices], dim=0)
    return ohs.sum(dim=0)


def one_hot_2d(R, C, r, c, *, dtype, device):
    """(R,C) with 1 at (r,c)."""
    return (
        one_hot_1d(R, r, dtype=dtype, device=device)[:, None]
        * one_hot_1d(C, c, dtype=dtype, device=device)[None, :]
    )
