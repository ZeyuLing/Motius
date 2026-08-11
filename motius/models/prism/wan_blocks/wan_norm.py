import torch
from torch import nn


class WanRMSNorm(nn.Module):
    """
    Generalized RMSNorm compatible with arbitrary tensor shapes.

    Args:
        normalized_dim: number of channels (int) along the normalization axis.
        channel_dim: which axis to normalize over (can be negative, default -1, i.e. last dim).
        eps: epsilon for numerical stability.
        elementwise_affine: if True, learnable weight (gamma) and bias are used.
        use_scale_sqrt: if True, multiply output by sqrt(normalized_dim) for backward compatibility
                        with implementations that scale by sqrt(D). Default False (matches paper/PyTorch).
    """

    def __init__(
        self,
        normalized_dim: int,
        channel_dim: int = 1,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        use_scale_sqrt: bool = False,
    ) -> None:
        super().__init__()
        self.normalized_dim = int(normalized_dim)
        self.channel_dim = int(channel_dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.use_scale_sqrt = bool(use_scale_sqrt)

        if elementwise_affine:
            # store gamma and beta as 1-D of length normalized_dim; we'll broadcast in forward
            self.gamma = nn.Parameter(torch.ones(self.normalized_dim))
            self.beta = nn.Parameter(torch.zeros(self.normalized_dim))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # optional legacy scale (if you want to replicate scale = sqrt(dim) behaviour)
        if self.use_scale_sqrt:
            self._legacy_scale = float(self.normalized_dim**0.5)
        else:
            self._legacy_scale = 1.0

    def _view_for_broadcast(self, x: torch.Tensor):
        """
        Return a view shape for gamma/beta so they broadcast over x when placed at channel_dim.
        For example, if x.ndim == 4 and channel_dim == 1 and normalized_dim==C,
        we want gamma.view(1, C, 1, 1) so multiplication broadcasts correctly.
        """
        ndim = x.ndim
        # normalize channel_dim to positive index
        ch = self.channel_dim % ndim
        shape = [1] * ndim
        shape[ch] = self.normalized_dim
        return shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: arbitrary tensor; normalize over axis `channel_dim` with RMS (root-mean-square).
        Output: same shape as x, scaled by gamma (and shifted by beta if enabled).
        """
        if x.shape[self.channel_dim] != self.normalized_dim:
            # It's possible the user passed a different channel size at runtime;
            # raise informative error rather than silently broadcasting wrong shape.
            raise ValueError(
                f"Input has size {x.shape[self.channel_dim]} on channel_dim={self.channel_dim}, "
                f"but WanRMSNorm was initialized with normalized_dim={self.normalized_dim}."
            )

        # compute RMS across channel dimension: sqrt(mean(x^2) + eps)
        # keepdim=True so shape aligns for broadcasting during division
        sq_mean = x.pow(2).mean(dim=self.channel_dim, keepdim=True)
        rms = torch.sqrt(sq_mean + self.eps)

        # normalize
        x_normed = x / rms

        # apply gamma & beta if present (broadcast to input shape)
        if self.elementwise_affine:
            view_shape = self._view_for_broadcast(x)
            gamma = self.gamma.view(*view_shape)
            beta = self.beta.view(*view_shape)
            out = x_normed * gamma + beta
        else:
            out = x_normed

        # optional legacy scale multiplier (keeps original sqrt(dim) behaviour if needed)
        if self._legacy_scale != 1.0:
            out = out * self._legacy_scale

        return out


if __name__ == "__main__":
    motion = torch.randn(2, 32, 17, 22)
    motion_2 = torch.randn(2, 32, 17)
    norm = WanRMSNorm(32, channel_dim=1)
    out = norm(motion)
    out_2 = norm(motion_2)
    print(out.shape)
    print(out_2.shape)
