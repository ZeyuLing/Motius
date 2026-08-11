"""Small tensor utilities (adapted from MotionLCM ``mld/utils``).

``lengths_to_mask`` / ``remove_padding`` come from ``mld/utils/temos_utils.py``
and ``get_guidance_scale_embedding`` from ``mld/utils/utils.py``. The latter is
the LCM "distilled CFG" embedding: instead of running an extra unconditional
forward pass at inference, the guidance scale is folded into the timestep
conditioning. Self-contained;.
"""

from typing import Optional

import torch


def lengths_to_mask(lengths, device: torch.device, max_len: Optional[int] = None) -> torch.Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def remove_padding(tensors: torch.Tensor, lengths) -> list:
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def get_guidance_scale_embedding(w: torch.Tensor, embedding_dim: int = 512,
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert len(w.shape) == 1
    w = w * 1000.0
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb
