from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from motius.registry import MODELS


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


@MODELS.register_module(force=True)
class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: VQ-VAE Made Simple — https://arxiv.org/abs/2309.15505

    Ported from the versatilemotion FSQ implementation.  Each dimension of the
    low-dimensional projection is independently rounded to one of a fixed
    number of ``levels``.  No learned codebook — quantization is purely
    geometric rounding, so the only trainable parameters are ``project_in``
    and ``project_out``.

    External contract (consumed by ``VQVAEVermo2DTK``):
      * ``forward(z)`` → ``(quantized, indices, dummy_loss, None)``
        - ``z``:         ``[B, dim, N]``
        - ``quantized``: ``[B, dim, N]`` (straight-through quantized)
        - ``indices``:   ``[B, N]``      (codebook indices, int32)
        - ``dummy_loss``: zero tensor (FSQ has no commitment loss)
      * ``dequantize(indices)`` → ``[B, dim, ...]`` (channel-first)
      * ``indices_to_codes(indices)`` — alias for ``dequantize``
      * ``codebook_size`` — product of levels
    """

    def __init__(
        self,
        levels: Sequence[int] | list[int] = (8, 5, 5, 5),
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        levels = list(levels)
        if not levels:
            raise ValueError("FSQuantizer.levels must be a non-empty sequence.")

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        if keep_num_codebooks_dim is None:
            keep_num_codebooks_dim = num_codebooks > 1
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = dim if dim is not None else len(levels) * num_codebooks

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = int(self._levels.prod().item())

        # Pre-compute the implicit codebook (no grad, just a lookup table)
        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    # ------------------------------------------------------------------
    # Core quantization helpers
    # ------------------------------------------------------------------

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound ``z`` (..., d) into the valid range per level via tanh."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z (..., d) → quantized codes in [-1, 1]."""
        quantized = _round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    # ------------------------------------------------------------------
    # Index <-> code conversion
    # ------------------------------------------------------------------

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes (..., d) to flat codebook indices (...)."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(
        self, indices: torch.Tensor, project_out: bool = True
    ) -> torch.Tensor:
        """Convert flat codebook indices back to projected feature vectors.

        When ``project_out=True`` the codes are mapped back to ``dim``
        via the learned ``project_out`` linear layer.
        """
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            dtype = next(self.project_out.parameters()).dtype if self.has_projections else codes.dtype
            codes = self.project_out(codes.to(dtype))
        return codes

    # ------------------------------------------------------------------
    # Public API consumed by VQVAEVermo
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor):
        """Forward pass — input/output are ``[B, dim, N]`` (channel-first).

        Returns ``(quantized, indices, dummy_loss, None)``.
        """
        z = rearrange(z, "b d n -> b n d")
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self._quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)").to(z.dtype)
        out = self.project_out(codes)

        # FSQ has no commitment loss — return a zero tensor for API compat
        dummy_loss = torch.zeros_like(out.mean(dim=[1, 2], keepdim=True)).unsqueeze(1)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        out = rearrange(out, "b n d -> b d n")
        return (out.to(z.dtype), indices, dummy_loss, None)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, dim, N]`` → indices ``[B, N]``."""
        z = rearrange(z, "b d n -> b n d")
        assert z.shape[-1] == self.dim
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)
        codes = self._quantize(z)
        indices = self.codes_to_indices(codes)
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        return indices

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices → feature tensor (channel-first).

        Supports:
          - 1-D ``[N]`` → ``[N, dim]``
          - 2-D ``[B, N]`` → ``[B, dim, N]``
          - 3-D ``[B, T, K]`` → ``[B, dim, T, K]``
        """
        codes = self.indices_to_codes(indices)  # project_out=True by default
        # Move channel dim to position 1 (channel-first)
        if codes.ndim >= 2:
            # codes: (..., dim) → (B, dim, ...) or (dim, ...)
            codes = rearrange(codes, "b ... c -> b c ...")
        return codes
