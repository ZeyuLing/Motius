import torch
import numpy as np
from typing import List, Optional, Tuple
from diffusers.models.autoencoders.autoencoder_kl import DiagonalGaussianDistribution

import math


class DiagonalGaussianDistributionNd:
    """
    Diagonal Gaussian distribution for arbitrary-shaped tensors.

    - `parameters` is a tensor where the channel dimension (dim=1) contains
      concatenated [mean, logvar], i.e. parameters.shape = (B, 2*C, ...).
    - The class will split along dim=1 into mean and logvar:
         mean.shape = (B, C, ...)
         logvar.shape = (B, C, ...)
    - All KL / NLL reductions sum over all dimensions except the batch dim (dim=0),
      returning a (B,) tensor.
    - If `deterministic=True`, sampling returns the mean and kl/nll return zeros.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.dim() < 2:
            raise ValueError(
                "parameters must have at least 2 dimensions (B, 2*C, ...)."
            )
        self.parameters = parameters
        # split along dim=1 into two equal parts
        self.mean, logvar = torch.chunk(parameters, 2, dim=1)
        # clamp for numerical stability
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            # keep device/dtype consistent
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def _non_batch_dims(self, t: torch.Tensor) -> Tuple[int, ...]:
        # dims to reduce (all dims except batch dim 0)
        return tuple(range(1, t.dim()))

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample z ~ N(mean, var) with same shape as mean.
        If deterministic, return mean.
        """
        if self.deterministic:
            return self.mean
        # create normal noise with same shape, device and dtype
        eps = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + self.std * eps

    def kl(
        self,
        other: "DiagonalGaussianDistributionNd" = None,
        valid_lengths: Optional[torch.Tensor] = None,
        time_dim: int = 2,
    ) -> torch.Tensor:
        """
        KL divergence per batch element.
        - If other is None: KL( q || N(0,I) ).
        - If other is given: KL( q || other ).
        Returns a tensor of shape (B,), sum over non-batch dims. When
        ``valid_lengths`` is provided, latent positions at or beyond each
        sample's valid temporal length are excluded before reduction.
        """
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype
            )

        if other is None:
            # KL(q || N(0, I)) = 0.5 * sum( mu^2 + var - 1 - logvar )
            elem = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        else:
            # handle possible other.deterministic: treat its var as ones (and logvar=0)
            if not isinstance(other, DiagonalGaussianDistributionNd):
                raise ValueError("other must be a DiagonalGaussianDistributionNd")
            if other.deterministic:
                other_var = torch.ones_like(self.var)
                other_logvar = torch.zeros_like(self.logvar)
                other_mean = other.mean.expand_as(self.mean)  # if shapes broadcast
            else:
                other_var = other.var
                other_logvar = other.logvar
                other_mean = other.mean

            # KL(q||p) = 0.5 * sum( (mu - mu2)^2 / var2 + var / var2 - 1 - logvar + logvar2 )
            num = (self.mean - other_mean).pow(2) / (other_var + 1e-12)
            elem = 0.5 * (
                num + self.var / (other_var + 1e-12) - 1.0 - self.logvar + other_logvar
            )

        if valid_lengths is not None:
            if time_dim <= 0 or time_dim >= elem.dim():
                raise ValueError(
                    f"time_dim must identify a non-batch dimension, got {time_dim} "
                    f"for shape={tuple(elem.shape)}"
                )
            lengths = torch.as_tensor(valid_lengths, device=elem.device).long()
            if lengths.numel() != elem.shape[0]:
                raise ValueError(
                    f"valid_lengths has {lengths.numel()} elements for batch "
                    f"size {elem.shape[0]}"
                )
            steps = torch.arange(elem.shape[time_dim], device=elem.device)
            mask = steps.unsqueeze(0) < lengths.clamp_min(0).unsqueeze(1)
            mask_shape = [elem.shape[0]] + [1] * (elem.dim() - 1)
            mask_shape[time_dim] = elem.shape[time_dim]
            elem = elem * mask.reshape(mask_shape).to(dtype=elem.dtype)

        return torch.sum(elem, dim=self._non_batch_dims(elem))

    def nll(
        self, sample: torch.Tensor, dims: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Negative log-likelihood per batch element:
        0.5 * sum( log(2*pi) + logvar + (x - mean)^2 / var )

        If dims is None, sum over all dims except batch dim (dim=0).
        """
        if self.deterministic:
            return torch.zeros(
                self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype
            )

        if dims is None:
            dims = self._non_batch_dims(self.mean)

        # ensure shapes broadcastable
        diff2 = (sample - self.mean).pow(2)
        logtwopi = math.log(2.0 * math.pi)
        term = 0.5 * (logtwopi + self.logvar + diff2 / (self.var + 1e-12))
        return torch.sum(term, dim=dims)

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) with original shape."""
        return self.mean


class DiagonalGaussianDistribution1D(DiagonalGaussianDistribution):
    def __init__(
        self,
        parameters: torch.Tensor,
        deterministic: bool = False,
        valid_lengths: List[int] = None,
    ):
        # parameters: [B, 2*C, T]
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            # zero-out variance and std for deterministic mode
            zeros = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )
            self.var = self.std = zeros

        # convert valid_lengths list to tensor
        if valid_lengths is not None:
            self.valid_lengths = torch.tensor(valid_lengths, device=self.mean.device)
        else:
            self.valid_lengths = None

    def _get_mask(self) -> Optional[torch.Tensor]:
        """
        Create a mask of shape [B, 1, T] with 1s for valid frames and 0s for padding.
        """
        if self.valid_lengths is None:
            return None
        B, C, T = self.mean.shape
        # [T] index vector
        idx = torch.arange(T, device=self.mean.device)
        # [B, T] mask where idx < valid_length
        frame_mask = (idx.unsqueeze(0) < self.valid_lengths.unsqueeze(1)).float()
        # [B, 1, T] for broadcasting across channels
        return frame_mask.unsqueeze(1)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        eps = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * eps

    def kl(self, other=None) -> torch.Tensor:
        if self.deterministic:
            # no KL divergence when deterministic
            return torch.zeros(self.mean.size(0), device=self.mean.device)

        # element-wise KL terms: [B, C, T]
        if other is None:
            kl_elems = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        else:
            kl_elems = 0.5 * (
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
        # apply mask to ignore padding frames
        mask = self._get_mask()
        if mask is not None:
            kl_elems = kl_elems * mask

        # sum over channels and time -> [B]
        return kl_elems.sum(dim=[1, 2])

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            # no NLL when deterministic
            return torch.zeros(self.mean.size(0), device=self.mean.device)

        log2pi = np.log(2.0 * np.pi)
        # element-wise NLL terms: [B, C, T]
        nll_elems = 0.5 * (
            log2pi + self.logvar + (sample - self.mean).pow(2) / self.var
        )
        # apply mask to ignore padding frames
        mask = self._get_mask()
        if mask is not None:
            nll_elems = nll_elems * mask

        # sum over channels and time -> [B]
        return nll_elems.sum(dim=[1, 2])
