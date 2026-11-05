# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared codebook lookup for the SOMA and G1 inference models."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


# The original MotionBricks quantizer credits MotionGPT.
# https://github.com/OpenMotionLab/MotionGPT
# See motius/licenses/motionbricks/MotionGPT-LICENSE.txt for its MIT notice.
# The original MotionBricks quantizer used vector-quantize-pytorch.
# https://github.com/lucidrains/vector-quantize-pytorch
# See motius/licenses/motionbricks/vector-quantize-pytorch-LICENSE.txt for its MIT notice.
# NVIDIA additions retain the copyright and Apache-2.0 terms above.
class MultiHeadCodebook(nn.Module):
    """Load checkpoint codebooks and look up predicted tokens using PyTorch."""

    def __init__(self, heads, entries, width):
        super().__init__()
        self._codebook = nn.Module()
        # Preserve the checkpoint structure for strict loading. Inference only
        # reads embed, and the other buffers are statistics recorded during training.
        self._codebook.register_buffer("initted", torch.zeros(1, dtype=torch.bool))
        self._codebook.register_buffer("cluster_size", torch.zeros(heads, entries))
        self._codebook.register_buffer("embed_avg", torch.zeros(heads, entries, width))
        self._codebook.register_buffer("embed", torch.zeros(heads, entries, width))

    @property
    def codebook(self):
        """Codebook vectors, laid out as [head, entry, width]."""
        return self._codebook.embed

    def get_codes_from_indices(self, indices):
        """Look up each head's code vector, and concatenate the heads."""
        batch, frames, _ = indices.shape
        index = indices.transpose(1, 2).unsqueeze(-1)
        index = index.expand(-1, -1, -1, self.codebook.shape[-1])
        codebook = self.codebook.unsqueeze(0).expand(batch, -1, -1, -1)
        codes = codebook.gather(2, index)
        return codes.transpose(1, 2).reshape(batch, frames, -1)


class QuantizeEMAResetMultiHead(nn.Module):
    """Multi-head vector quantizer, used here only to look up codes."""

    def __init__(self, nb_code: int, code_dim: int, num_heads: int):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.num_heads = num_heads
        assert self.code_dim % self.num_heads == 0, (
            "code dim cannot be divided by the number of heads."
        )
        self.nb_code_per_head = int(
            round(2 ** (np.log2(self.nb_code) / self.num_heads))
        )
        assert self.nb_code_per_head**self.num_heads == self.nb_code, (
            "the specified number of code is not compatible with the number of heads."
        )
        self.vq = MultiHeadCodebook(
            self.num_heads, self.nb_code_per_head, self.code_dim // self.num_heads
        )

    def from_overall_indices_to_mh_indices(self, overall_indices):
        """Split overall code indices into one index per head."""
        exponential_mul = self.nb_code_per_head ** torch.arange(0, self.num_heads)[
            None, None, :
        ].to(overall_indices.device)
        mh_indices = (
            overall_indices[:, :, None]
            % (exponential_mul * self.nb_code_per_head)
            // exponential_mul
        )
        return mh_indices

    def dequantize(self, code_idx, use_overall_indices=True):
        """Return the code vectors for token indices.

        With use_overall_indices, overall indices are split per head first.
        """
        if use_overall_indices:
            mh_indices = self.from_overall_indices_to_mh_indices(code_idx)
        else:
            mh_indices = code_idx
        batch_size, numToken = (code_idx.shape[0], code_idx.shape[1])
        x = self.vq.get_codes_from_indices(mh_indices).view([batch_size, numToken, -1])
        return x


@dataclass(frozen=True)
class MultiHeadVQDimensions:
    """Head count, codes per head, and code width per head of a multi-head quantizer."""

    num_heads: int
    codes_per_head: int
    code_dim_per_head: int

    @property
    def combined_code_space(self) -> int:
        """Number of distinct overall codes, codes_per_head ** num_heads."""
        return self.codes_per_head**self.num_heads

    @property
    def latent_dim(self) -> int:
        """Width of a full code vector across all heads."""
        return self.code_dim_per_head * self.num_heads


def _exact_integer_root(value: int, degree: int, field_name: str) -> int:
    candidate = int(round(value ** (1.0 / degree)))
    for possible in range(max(1, candidate - 1), candidate + 2):
        if possible**degree == value:
            return possible
    raise ValueError(
        f"{field_name}={value} is not an exact {degree}-head product code space."
    )


def resolve_multihead_vq_dimensions(
    *,
    num_heads: int,
    codes_per_head: Optional[int] = None,
    code_dim_per_head: Optional[int] = None,
    nb_code: Optional[int] = None,
    code_dim: Optional[int] = None,
) -> MultiHeadVQDimensions:
    """Resolve and check a multi-head quantizer's dimensions from checkpoint settings.

    Missing values come from the others. Conflicting values raise ValueError.
    """
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}.")
    if codes_per_head is None:
        if nb_code is None:
            raise ValueError("codes_per_head is required when nb_code is not provided.")
        codes_per_head = _exact_integer_root(nb_code, num_heads, "nb_code")
    elif not isinstance(codes_per_head, int) or codes_per_head <= 0:
        raise ValueError(
            f"codes_per_head must be a positive integer, got {codes_per_head!r}."
        )
    combined_code_space = codes_per_head**num_heads
    if nb_code is not None and nb_code != combined_code_space:
        raise ValueError(
            f"nb_code={nb_code} conflicts with {num_heads} heads x {codes_per_head} codes "
            f"(expected {combined_code_space})."
        )
    if code_dim_per_head is None:
        if code_dim is None:
            raise ValueError(
                "code_dim_per_head is required when code_dim is not provided."
            )
        if code_dim % num_heads:
            raise ValueError(
                f"code_dim={code_dim} must be divisible by num_heads={num_heads}."
            )
        code_dim_per_head = code_dim // num_heads
    elif not isinstance(code_dim_per_head, int) or code_dim_per_head <= 0:
        raise ValueError(
            f"code_dim_per_head must be a positive integer, got {code_dim_per_head!r}."
        )
    latent_dim = code_dim_per_head * num_heads
    if code_dim is not None and code_dim != latent_dim:
        raise ValueError(
            f"code_dim={code_dim} conflicts with {num_heads} heads x {code_dim_per_head} dimensions "
            f"(expected {latent_dim})."
        )
    return MultiHeadVQDimensions(
        num_heads=num_heads,
        codes_per_head=codes_per_head,
        code_dim_per_head=code_dim_per_head,
    )
