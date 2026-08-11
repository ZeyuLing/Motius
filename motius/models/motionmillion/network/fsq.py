"""Finite Scalar Quantization (FSQ) for MotionMillion / "Go to Zero".

Self-contained (torch + einops). Used as the motion tokenizer for the Go-to-Zero
discrete-AR text-to-motion model. For T2M *inference* we only need
:meth:`FSQ.dequantize` (token indices -> continuous codes via ``project_out``).

Original: https://github.com/VankouF/MotionMillion-Codes (models/FSQ.py),
itself adapted from "Finite Scalar Quantization: VQ-VAE Made Simple"
(https://arxiv.org/abs/2309.15505).
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial, wraps
from typing import List, Tuple

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks=1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32=True,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        assert exists(indices)
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        codes = self._indices_to_codes(indices)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")
        codes = self.project_out(codes)
        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")
        return codes

    def dequantize(self, indices):
        """Token indices -> continuous codes (the T2M decoder input)."""
        codes = self._indices_to_codes(indices)
        out = self.project_out(codes)
        return out

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.codebook_size, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        activate = torch.sum(code_count > 0).float() / self.codebook_size
        return perplexity, activate

    def forward(self, z):
        need_move_channel_last = True
        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():
            orig_dtype = z.dtype
            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()
            codes = self.quantize(z)
            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)
            codes = rearrange(codes, "b n c d -> b n (c d)")
            codes = codes.type(orig_dtype)

        out = self.project_out(codes)

        if need_move_channel_last:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = maybe(unpack_one)(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, "... 1 -> ...")

        perplexity, activate = self.compute_perplexity(indices.reshape(-1).to(torch.int64))
        dummy_loss = torch.tensor(0.0, device=indices.device)
        return out, indices, dummy_loss, perplexity, activate, indices
