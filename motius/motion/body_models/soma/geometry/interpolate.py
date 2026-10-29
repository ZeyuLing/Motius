# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

import torch


def _pairwise_dist(A, B):
    """(Na,D) vs (Nb,D) -> (Na,Nb) Euclidean distances (broadcasted)."""
    diff = A[:, None, :] - B[None, :, :]
    return torch.sqrt((diff * diff).sum(dim=-1))


def _lu_solve(lu_fac, b):
    lu, piv = lu_fac
    if b.ndim > 2:  # batched: (B, N, D)
        B, N, D = b.shape
        b_2d = b.permute(1, 0, 2).reshape(N, B * D)  # (N, BD)
        x_2d = torch.linalg.lu_solve(lu, piv, b_2d)  # (N, BD)
        return x_2d.reshape(N, B, D).permute(1, 0, 2)  # (B, N, D)
    elif b.ndim == 1:
        result = torch.linalg.lu_solve(lu, piv, b.unsqueeze(1))
        return result.squeeze(1)
    else:
        return torch.linalg.lu_solve(lu, piv, b)


class RadialBasisFunction:
    """RBF interpolator with precomputed system matrix (PyTorch)."""

    @staticmethod
    def _tps(r, eps=1e-10):
        return (r * r) * torch.log(r + eps)

    @staticmethod
    def _gaussian(r, eps=0.1):
        return torch.exp(-((r / eps) ** 2))

    @staticmethod
    def _multiquadric(r, eps=0.1):
        return torch.sqrt(1.0 + (r / eps) ** 2)

    @staticmethod
    def _inverse_multiquadric(r, eps=0.1):
        return 1.0 / torch.sqrt(1.0 + (r / eps) ** 2)

    @staticmethod
    def _inverse_quadratic(r, eps=0.1):
        return 1.0 / (1.0 + (r / eps) ** 2)

    @staticmethod
    def _linear(r, eps=1e-10):
        return r

    @staticmethod
    def _cubic(r, eps=1e-10):
        return r**3

    @staticmethod
    def _quintic(r, eps=1e-10):
        return r**5

    KERNELS = {
        "thin_plate_spline": _tps.__func__,
        "gaussian": _gaussian.__func__,
        "multiquadric": _multiquadric.__func__,
        "inverse_multiquadric": _inverse_multiquadric.__func__,
        "inverse_quadratic": _inverse_quadratic.__func__,
        "linear": _linear.__func__,
        "cubic": _cubic.__func__,
        "quintic": _quintic.__func__,
    }

    def __init__(
        self,
        source_control_points,
        kernel: str = "thin_plate_spline",
        kernel_params: Optional[Dict[str, Any]] = None,
        include_polynomial: bool = True,
    ):
        if kernel not in self.KERNELS:
            raise ValueError(f"Unknown kernel '{kernel}'. Available: {list(self.KERNELS.keys())}")

        scp = source_control_points
        if scp.ndim != 2:
            raise ValueError("source_control_points must be (N, D)")

        self.dtype = scp.dtype
        self.device = scp.device
        self.source_control_points = scp
        self.n_control, self.dim = scp.shape

        self.kernel_name = kernel
        self.kernel_params = kernel_params or {}
        self.include_polynomial = include_polynomial
        self._rbf_func = self.KERNELS[kernel]

        self._precompute_system_matrix()

    def _rbf(self, r):
        return self._rbf_func(r, **self.kernel_params)

    def _precompute_system_matrix(self):
        scp = self.source_control_points
        N, D = self.n_control, self.dim

        K = self._rbf(_pairwise_dist(scp, scp))
        K = K.to(self.dtype)

        eps = 1e-8 if self.dtype == torch.float32 or self.dtype == torch.float64 else 1e-4
        K = K + torch.eye(N, dtype=self.dtype, device=self.device) * eps

        if self.include_polynomial:
            ones = torch.ones((N, 1), dtype=self.dtype, device=self.device)
            P = torch.cat([ones, scp], dim=1)  # (N, D+1)
            Z = torch.zeros((D + 1, D + 1), dtype=self.dtype, device=self.device)

            top = torch.cat([K, P], dim=1)  # (N, N+D+1)
            PT = P.transpose(0, 1)  # (D+1, N)
            bottom = torch.cat([PT, Z], dim=1)  # (D+1, N+D+1)

            A = torch.cat([top, bottom], dim=0)  # (N+D+1, N+D+1)
        else:
            A = K

        self.A = A
        self._lu = torch.linalg.lu_factor(A)

    def get_basis_weights(self, query_point):
        """
        Computes the linear weights w such that:
        interpolated_pos = sum(w_i * source_point_i)

        Args:
        query_point: (D,) Vector representing the static template position
                        of the joint (or point) being interpolated.

        Returns:
        weights: (N,) Vector of weights corresponding to the source_control_points.
        """
        if query_point.ndim == 2:
            query_point = query_point.flatten()

        dists = self._rbf_func(
            torch.norm(self.source_control_points - query_point[None, :], dim=1),
            **self.kernel_params,
        )
        dists = dists.to(self.dtype)

        if self.include_polynomial:
            ones = torch.ones((1,), dtype=self.dtype, device=self.device)
            rhs = torch.cat([dists, ones, query_point], dim=0)
        else:
            rhs = dists

        w_full = _lu_solve(self._lu, rhs)

        return w_full[: self.n_control]

    def interpolate(self, target_control_positions, query_points):
        """
        target_control_positions: (N, D) or (B, N, D)
        query_points: (M, D)
        Returns:
            (M, D) if input was (N, D)
            (B, M, D) if input was (B, N, D)
        """
        N, D = self.n_control, self.dim

        single = False
        if target_control_positions.ndim == 2:
            target_control_positions = target_control_positions[None, ...]
            single = True
        if target_control_positions.ndim != 3 or target_control_positions.shape[1:] != (N, D):
            raise ValueError(
                f"target_control_positions must be (N,D) or (B,N,D), got {target_control_positions.shape}"
            )
        B = target_control_positions.shape[0]

        if query_points.ndim != 2 or query_points.shape[1] != D:
            raise ValueError(f"query_points must be (M, {D})")

        if self.include_polynomial:
            zeros_tail = torch.zeros((B, D + 1, D), dtype=self.dtype, device=self.device)
            b = torch.cat([target_control_positions, zeros_tail], dim=1)
        else:
            b = target_control_positions

        coeffs = _lu_solve(self._lu, b)

        Phi = self._rbf(_pairwise_dist(query_points, self.source_control_points))
        Phi = Phi.to(self.dtype)
        Phi_b = Phi.unsqueeze(0).expand(B, *Phi.shape)  # (B, M, N)

        rbf_contrib = Phi_b @ coeffs[:, :N, :]

        if not self.include_polynomial:
            return rbf_contrib[0] if single else rbf_contrib

        ones = torch.ones((query_points.shape[0], 1), dtype=self.dtype, device=self.device)
        query_aug = torch.cat([ones, query_points], dim=1)  # (M, D+1)
        QA_b = query_aug.unsqueeze(0).expand(B, *query_aug.shape)  # (B, M, D+1)

        affine_contrib = QA_b @ coeffs[:, N:, :]  # (B, M, D)
        out = rbf_contrib + affine_contrib

        return out[0] if single else out
