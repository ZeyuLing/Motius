# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

# ============================================================================
# Modular Rotation Estimation Functions
# ============================================================================


def compute_covariance(A, B, virtual_normal=True, eps=1e-8):
    """Compute covariance matrix H = A^T @ B for rotation estimation.

    Args:
        A: Target vectors (..., N, 3)
        B: Source vectors (..., N, 3)
        virtual_normal: If True, add synthetic normal correspondence for conditioning
        eps: Small constant for numerical stability

    Returns:
        H: Covariance matrix (..., 3, 3)
    """
    # Basic covariance: H = A^T @ B
    H = torch.einsum("...ni,...nj->...ij", A, B)

    # Virtual normal fix: add synthetic correspondence from cross product
    if virtual_normal and A.shape[-2] >= 2:
        p0, p1 = A[..., 0, :], A[..., 1, :]
        q0, q1 = B[..., 0, :], B[..., 1, :]

        # Compute normal direction (cross product)
        n_src = torch.cross(p0, p1, dim=-1)
        n_dst = torch.cross(q0, q1, dim=-1)

        # Normalize and scale by point cloud radius
        len_n_src = torch.linalg.norm(n_src, dim=-1, keepdim=True)
        len_n_dst = torch.linalg.norm(n_dst, dim=-1, keepdim=True)
        scale_src = torch.linalg.norm(p0, dim=-1, keepdim=True) / (len_n_src + eps)
        scale_dst = torch.linalg.norm(q0, dim=-1, keepdim=True) / (len_n_dst + eps)

        # Check for collinearity
        valid_normal = (len_n_src[..., 0] > 1e-9) & (len_n_dst[..., 0] > 1e-9)

        # Virtual normal vectors
        v_src = n_src * scale_src
        v_dst = n_dst * scale_dst

        # Add virtual correspondence (only for valid normals)
        if torch.any(valid_normal):
            virtual_contrib = torch.einsum("...i,...j->...ij", v_src, v_dst)
            mask = valid_normal[..., None, None].expand(H.shape)
            virtual_contrib = torch.where(mask, virtual_contrib, 0.0)
            H = H + virtual_contrib

    return H


def kabsch(H):
    """Compute rotation matrix from covariance using Kabsch algorithm (SVD).

    Args:
        H: Covariance matrix (..., 3, 3)

    Returns:
        R: Rotation matrix (..., 3, 3) with det(R) = 1
    """
    U, S, Vh = torch.linalg.svd(H)
    I3 = torch.eye(3, dtype=H.dtype, device=H.device)

    # Compute correction for determinant
    UVt = U @ Vh.swapaxes(-2, -1)
    det_sign = torch.where(torch.linalg.det(UVt) < 0, -1.0, 1.0)

    # Apply correction
    Dcorr = I3.expand(H.shape).clone()
    Dcorr[..., -1, -1] = det_sign
    R = U @ Dcorr @ Vh

    return R


def newton_schulz(H, num_iters=20, eps=1e-8):
    """Compute rotation matrix from covariance using Newton-Schulz iteration.

    This is primarily a reference implementation for testing and comparing against
    the Warp-accelerated Newton-Schulz kernel. For production use, prefer `kabsch()`
    which is more stable and has better-defined gradients through SVD.

    Args:
        H: Covariance matrix (..., 3, 3)
        num_iters: Number of iterations (default 20)
        eps: Small constant for numerical stability

    Returns:
        R: Rotation matrix (..., 3, 3) with det(R) = 1

    Note:
        Convergence depends on conditioning of H. Ill-conditioned matrices may
        require more iterations or may not converge to high precision.
    """
    # Scale by infinity norm (max absolute row sum) for guaranteed convergence
    row_sums = torch.abs(H).sum(dim=-1)
    max_row_sum = row_sums.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    R = H / (max_row_sum + eps)

    I3 = torch.eye(3, dtype=H.dtype, device=H.device)
    I3_batch = I3.expand(H.shape)

    # Newton-Schulz iteration: R_{k+1} = R_k * (3*I - R_k^T * R_k) / 2
    for _ in range(num_iters):
        RT_R = R.swapaxes(-2, -1) @ R
        term = 3.0 * I3_batch - RT_R
        R = R @ term * 0.5

    # Differentiable determinant correction
    det_R = torch.linalg.det(R)
    sign_factor = torch.where(det_R < 0, -1.0, 1.0)

    # Apply sign correction to last column
    R_corrected = R.clone()
    R_corrected[..., :, 2] = R[..., :, 2] * sign_factor[..., None]

    return R_corrected


def rodrigues_rotation(a, b, eps=1e-8):
    """Compute rotation matrix that aligns vector b to vector a.

    Uses the shortest arc rotation approach similar to SciPy's align_vectors.

    Args:
        a: Target vector (..., 3)
        b: Source vector (..., 3)
        eps: Small constant for numerical stability

    Returns:
        R: Rotation matrix (..., 3, 3) such that R @ b ≈ a
    """
    dtype, device = a.dtype, a.device

    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True)

    a_u = a / torch.clamp(a_norm, min=eps)
    b_u = b / torch.clamp(b_norm, min=eps)

    dot = torch.clamp((a_u * b_u).sum(dim=-1, keepdim=True), -1.0, 1.0)
    v = torch.cross(b_u, a_u, dim=-1)

    zeros = torch.zeros_like(v[..., 0])
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]

    skew_v = torch.stack(
        [
            torch.stack([zeros, -vz, vy], dim=-1),
            torch.stack([vz, zeros, -vx], dim=-1),
            torch.stack([-vy, vx, zeros], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=dtype, device=device).expand(a.shape[:-1] + (3, 3))

    factor = 1.0 / (1.0 + dot[..., None])
    R = eye + skew_v + factor * (skew_v @ skew_v)

    # Handle Antiparallel Case (180 degree rotation)
    antiparallel_mask = dot[..., 0] < -1.0 + 1e-6

    if torch.any(antiparallel_mask):
        b_anti = b_u[antiparallel_mask]

        basis_shape = b_anti.shape[:-1] + (3,)
        y_vec = torch.zeros(basis_shape, dtype=dtype, device=device)
        y_vec[..., 1] = 1.0
        x_vec = torch.zeros(basis_shape, dtype=dtype, device=device)
        x_vec[..., 0] = 1.0

        w = torch.where((torch.abs(b_anti[..., 0]) > 0.6)[..., None], y_vec, x_vec)

        axis_180 = torch.cross(b_anti, w, dim=-1)
        axis_180 = axis_180 / torch.linalg.norm(axis_180, dim=-1, keepdim=True)

        u_mat = axis_180[..., :, None] * axis_180[..., None, :]
        eye_3 = torch.eye(3, dtype=dtype, device=device)
        R_180 = 2.0 * u_mat - eye_3

        R[antiparallel_mask] = R_180

    return R


# ============================================================================
# High-Level Alignment Function
# ============================================================================


def align_vectors(A, B, eps=1e-8, method="kabsch"):
    """
    SciPy-compatible: return rotation C such that C @ b ≈ a.
    Supports broadcasting across leading batch dims. Inputs: (..., N, 3).

    Args:
        A: Target vectors (..., N, 3)
        B: Source vectors (..., N, 3)
        eps: Small constant for numerical stability
        method: 'kabsch' (SVD-based) or 'newton-schulz' (iterative)
    """
    if A.shape[-1] != 3 or B.shape[-1] != 3:
        raise NotImplementedError("Only 3D vectors are supported (last dim must be 3).")
    if A.shape[-2] != B.shape[-2]:
        raise ValueError(f"N must match, got {A.shape[-2]} vs {B.shape[-2]}.")

    N = A.shape[-2]

    if N == 1:
        return rodrigues_rotation(A[..., 0, :], B[..., 0, :], eps=eps)

    H = compute_covariance(A, B, virtual_normal=True, eps=eps)

    if method == "newton-schulz":
        return newton_schulz(H, num_iters=20, eps=eps)
    elif method == "kabsch":
        return kabsch(H)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kabsch' or 'newton-schulz'.")


def SE3_from_Rt(R, t):
    """
    autograd-safe SE(3) transform construction from rotation R and translation t.
    R: (..., 3, 3)
    t: (..., 3)
    Returns: T (..., 4, 4)
    """
    dtype, device = R.dtype, R.device
    upper = torch.cat([R, t[..., None]], dim=-1)  # (..., 3, 4)
    last_row = torch.cat(
        [
            torch.zeros((*upper.shape[:-2], 1, 3), dtype=dtype, device=device),
            torch.ones((*upper.shape[:-2], 1, 1), dtype=dtype, device=device),
        ],
        dim=-1,
    )  # (..., 1, 4)
    return torch.cat([upper, last_row], dim=-2)  # (..., 4, 4)


def SE3_inverse(T):
    """
    Invert SE(3) transform(s) in homogeneous coordinates.

    Args:
        T: (..., 4, 4) torch.Tensor
    Returns:
        Tinv: (..., 4, 4)
    """
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3:4]  # (..., 3, 1)
    R_T = R.swapaxes(-2, -1)  # (..., 3, 3)
    t_new = -(R_T @ t)  # (..., 3, 1)

    Tinv = SE3_from_Rt(R_T, t_new[..., 0])  # (..., 4, 4)
    return Tinv


# --- SO(3) conversions --------------------------------------------------------


def matrix_to_rotvec(R, eps=1e-6):
    """
    (...,3,3) rotation matrices -> (...,3) rotation vectors (axis * angle).
    Robust for small angles and near-pi.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")

    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    S = R - R.swapaxes(-2, -1)
    v = torch.stack(
        [
            S[..., 2, 1] - S[..., 1, 2],
            S[..., 0, 2] - S[..., 2, 0],
            S[..., 1, 0] - S[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.norm(v, dim=-1)

    # Regions
    small = theta <= 1e-3
    near_pi = theta >= (torch.pi - 1e-3)

    # Small-angle series
    theta2_approx = torch.clamp(3.0 - tr, min=0.0)
    factor_small = 0.5 + theta2_approx / 12.0
    w_small = v * factor_small[..., None]

    # Generic
    denom = torch.where(sin_theta < eps, eps, 2.0 * sin_theta)
    factor_gen = theta / denom
    w_gen = v * factor_gen[..., None]

    # Near-pi: axis from diagonals + sign from v
    R00, R11, R22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    u0 = torch.sqrt(torch.clamp((R00 - R11 - R22 + 1.0) * 0.5, min=0.0))
    u1 = torch.sqrt(torch.clamp((-R00 + R11 - R22 + 1.0) * 0.5, min=0.0))
    u2 = torch.sqrt(torch.clamp((-R00 - R11 + R22 + 1.0) * 0.5, min=0.0))
    u = torch.stack([u0, u1, u2], dim=-1)

    sx, sy, sz = torch.sign(v[..., 0]), torch.sign(v[..., 1]), torch.sign(v[..., 2])
    sx = torch.where(sx == 0, 1, sx)
    sy = torch.where(sy == 0, 1, sy)
    sz = torch.where(sz == 0, 1, sz)
    u = torch.stack([u[..., 0] * sx, u[..., 1] * sy, u[..., 2] * sz], dim=-1)

    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)
    u_norm = torch.where(u_norm < eps, eps, u_norm)
    axis_pi = u / u_norm
    w_pi = axis_pi * theta[..., None]

    return torch.where(near_pi[..., None], w_pi, torch.where(small[..., None], w_small, w_gen))


def rotvec_to_matrix(rotvec, eps=1e-8):
    """
    (...,3) rotation vectors -> (...,3,3) rotation matrices.
    Robust near zero.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {rotvec.shape}")

    theta = torch.linalg.norm(rotvec, dim=-1)
    denom = torch.where(theta < eps, eps, theta)[..., None]
    axis = rotvec / denom

    K = torch.zeros(rotvec.shape[:-1] + (3, 3), dtype=rotvec.dtype, device=rotvec.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    A = sin_t / torch.where(theta < eps, 1.0, theta)
    B = (1.0 - cos_t) / torch.where(theta < eps, 1.0, theta * theta)

    R = eye + A[..., None, None] * K + B[..., None, None] * (K @ K)

    small = theta < 1e-6
    return torch.where(small[..., None, None], eye + K, R)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
