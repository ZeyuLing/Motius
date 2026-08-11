"""Kinematics-aware projection primitives from the official ProjFlow release."""

import torch as th
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# ============================================================
# Metric: kinematics-aware R and efficient R^{-1} application
# ============================================================

@dataclass(frozen=True)
class KinematicMetric:
    """
    Helpers for the kinematics-aware metric R and its inverse.

    We model:
        R = w_kin * (I_D ⊗ I_L ⊗ L_kin) + ridge * I
    and apply R^{-1} efficiently along the joint axis using an eigendecomposition of L_kin.
    """
    apply_Rinv: Callable[[th.Tensor], th.Tensor]   # (B,D,L,J) -> (B,D,L,J)
    diag_Rinv_joint: th.Tensor                     # (J,)  diag of joint-only block of R^{-1}
    joint_weights_q: th.Tensor                     # (J,)  q_j ∝ 1 / ||col_j(R_J^{-1})||^2
    L_kin: th.Tensor                               # (J,J)


def build_skeleton_laplacian(J: int, device, dtype) -> th.Tensor:
    if J == 22:
        chains = (
            [0, 2, 5, 8, 11],
            [0, 1, 4, 7, 10],
            [0, 3, 6, 9, 12, 15],
            [9, 14, 17, 19, 21],
            [9, 13, 16, 18, 20],
        )
    elif J == 21:
        chains = (
            [0, 11, 12, 13, 14, 15],
            [0, 16, 17, 18, 19, 20],
            [0, 1, 2, 3, 4],
            [3, 5, 6, 7],
            [3, 8, 9, 10],
        )
    else:
        raise ValueError(f"Unsupported joint count J={J}")

    A = th.zeros(J, J, device=device, dtype=dtype)
    for ch in chains:
        for u, v in zip(ch[:-1], ch[1:]):
            A[u, v] = 1.0
            A[v, u] = 1.0

    deg = A.sum(dim=1)
    L = th.diag(deg) - A
    return L


def build_kinematic_metric(
    *,
    J: int,
    w_kin: float,
    ridge: float,
    L_kin: th.Tensor,
    ) -> KinematicMetric:
    """
    Build a kinematics-aware metric R and helpers to apply R^{-1}.

    We use an eigendecomposition of L_kin:
        L_kin = U diag(lam) U^T
    so that the joint-only block is:
        R_J = w_kin * L_kin + ridge * I
        R_J^{-1} = U diag(1 / (w_kin*lam + ridge)) U^T

    Returns:
        KinematicMetric with:
          - apply_Rinv(b): applies (I ⊗ R_J^{-1}) along joints for each (B,D,L)
          - diag_Rinv_joint: diag(R_J^{-1}) (needed for variance conversion)
          - joint_weights_q: q_j = 1 / ||col_j(R_J^{-1})||^2 (supp Eq. (51))
    """
    device, dtype = L_kin.device, L_kin.dtype

    lam, U = th.linalg.eigh(L_kin)  # (J,), (J,J)
    denom = (w_kin * lam + float(ridge)).to(device=device, dtype=dtype)  # (J,)

    U_T = U.mT

    def apply_Rinv(b: th.Tensor) -> th.Tensor:
        # b: (B,D,L,J)
        bhat = th.einsum("bdlj,jk->bdlk", b, U)               # transform joints -> eigenbasis
        xhat = bhat / denom.view(1, 1, 1, -1)               # divide eigenvalues
        x    = th.einsum("bdlk,kj->bdlj", xhat, U_T)
        return x.contiguous()

    # diag(R_J^{-1}) = sum_k U[j,k]^2 / denom[k]
    diag_Rinv_joint = (U**2) @ (1.0 / denom)  # (J,)

    # Column norms of R_J^{-1}: since symmetric, col norm == row norm.
    # ||col_j||^2 = sum_k U[j,k]^2 / denom[k]^2
    col_norm_sq = (U**2) @ (1.0 / (denom**2))
    joint_weights_q = (1.0 / (col_norm_sq + 1e-12)).clamp_min(1e-12)  # (J,)

    return KinematicMetric(
        apply_Rinv=apply_Rinv,
        diag_Rinv_joint=diag_Rinv_joint,
        joint_weights_q=joint_weights_q,
        L_kin=L_kin,
    )


# ============================================================
# Inpainting observation model: pseudo-observations + variance
# ============================================================

def halo_radius_linear(step: int, num_steps: int, ell_min: float, ell_max: float) -> float:
    """
    Dynamic masking radius ℓ(t): linearly shrink from ell_max to ell_min over sampling.
    (Matches the description in the paper/supplement.)
    """
    if num_steps <= 1:
        return float(ell_min)
    u = float(step) / float(num_steps - 1)
    return float(ell_max - (ell_max - ell_min) * u)


def gather_time(Y: th.Tensor, idx: th.Tensor) -> th.Tensor:
    """
    Gather along time dim=2 for Y shaped (B,D,L,J).

    idx: (B,L,J) with -1 meaning "missing"; we clamp -1 to 0 for gather and
    later mask with availability.
    """
    B, D, L, J = Y.shape
    idx_safe = idx.clamp(min=0).to(th.long).view(B, 1, L, J).expand(B, D, L, J)
    return th.gather(Y, dim=2, index=idx_safe)


def nearest_anchor_indices(mask_any: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    For each (b,t,j), find nearest observed anchor before and after t.
    mask_any: (B,L,J) boolean indicating *any* channel observed for that (t,j).

    Returns:
        prev_idx, next_idx: both (B,L,J) int indices in [0..L-1], or -1 if none exists.
    """
    device = mask_any.device
    B, L, J = mask_any.shape

    t_grid = th.arange(L, device=device).view(1, L, 1).expand(B, L, J)

    prev_idx = th.where(mask_any, t_grid, th.full_like(t_grid, -1)).cummax(dim=1).values

    # reverse scan for "next"
    mask_rev = mask_any.flip(1)
    t_rev = th.arange(L, device=device).view(1, L, 1).expand(B, L, J)
    next_rev = th.where(mask_rev, t_rev, th.full_like(t_rev, -1)).cummax(dim=1).values
    next_idx = (L - 1 - next_rev).flip(1)

    next_exist = next_rev.flip(1) >= 0
    next_idx = th.where(next_exist, next_idx, th.full_like(next_idx, -1))
    return prev_idx, next_idx


def build_pseudo_observations(
    *,
    hard_mask: th.Tensor,   # (B,D,L,J) 0/1
    hard_value: th.Tensor,  # (B,D,L,J)
    halo_radius: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
    """
    Create per-joint pseudo-observations y_src by interpolation/extrapolation
    and a halo selector M_halo indicating which entries are activated as soft guidance.

    Returns:
        y_src:  (B,D,L,J) pseudo targets
        M_halo: (B,D,L,J) 0/1 mask: soft pseudo-observation rows (excluding hard rows)
    """
    device, dtype = hard_value.device, hard_value.dtype
    B, D, L, J = hard_value.shape

    # Consider a joint observed at time t if ANY spatial channel is observed.
    mask_any = (hard_mask > 0.5).amax(dim=1)  # (B,L,J) bool

    prev_idx, next_idx = nearest_anchor_indices(mask_any)  # (B,L,J), (B,L,J)

    y_prev = gather_time(hard_value, prev_idx)  # (B,D,L,J)
    y_next = gather_time(hard_value, next_idx)  # (B,D,L,J)

    prev_ok = (prev_idx >= 0).view(B, 1, L, J)
    next_ok = (next_idx >= 0).view(B, 1, L, J)
    both_ok = prev_ok & next_ok

    # Linear interpolation coefficient along time.
    t = th.arange(L, device=device, dtype=dtype).view(1, 1, L, 1).expand(B, 1, L, J)
    denom = (next_idx - prev_idx).clamp(min=1).to(dtype=dtype).view(B, 1, L, J)
    alpha = ((t - prev_idx.to(dtype=dtype).view(B, 1, L, J)) / denom).clamp(0, 1)

    y_src = th.zeros_like(hard_value)
    y_src = th.where(both_ok, (1.0 - alpha) * y_prev + alpha * y_next, y_src)
    # Extrapolation: if only one side exists, copy that anchor.
    y_src = th.where(prev_ok & ~both_ok, y_prev, y_src)
    y_src = th.where(next_ok & ~both_ok, y_next, y_src)

    # Halo support: within halo_radius frames of nearest anchor.
    t_grid = th.arange(L, device=device).view(1, L, 1).expand(B, L, J)
    thr = max(float(halo_radius), 1e-6)

    near_prev = (prev_idx >= 0) & ((t_grid - prev_idx).to(dtype) < thr)
    near_next = (next_idx >= 0) & ((next_idx - t_grid).to(dtype) < thr)
    halo_support = (~mask_any) & (near_prev | near_next)   # (B,L,J) bool

    # Expand halo mask to (B,D,L,J), and explicitly exclude hard rows.
    M_halo = halo_support.to(dtype).view(B, 1, L, J).expand(B, D, L, J)
    M_halo = M_halo * (1.0 - (hard_mask > 0.5).to(dtype))

    return y_src.contiguous(), M_halo.contiguous()


def curvature_per_frame(
    *,
    x1_hat: th.Tensor,         # (B,D,L,J)
    metric: KinematicMetric,   # uses L_kin
    w_kin: float,
    ridge: float,
    ) -> th.Tensor:
    """
    Curvature score s_n(x̂_1) from the paper (discrete acceleration in time),
    measured using the *joint* metric block.

    We compute per-frame:
        acc_n = x_{n+1} - 2 x_n + x_{n-1}
        s_n = sqrt( acc_n^T (w_kin L_kin + ridge I) acc_n )
    summed over spatial channels.
    Returns:
        s: (B,L)
    """
    device, dtype = x1_hat.device, x1_hat.dtype
    B, D, L, J = x1_hat.shape

    # replicate boundary for second difference
    x_prev = th.cat([x1_hat[:, :, :1],  x1_hat[:, :, :-1]], dim=2)
    x_next = th.cat([x1_hat[:, :, 1:],  x1_hat[:, :, -1:]], dim=2)
    acc = x_next - 2.0 * x1_hat + x_prev  # (B,D,L,J)

    A = (w_kin * metric.L_kin + ridge * th.eye(J, device=device, dtype=dtype))  # (J,J)
    # energy per frame: sum_{d} acc[d]^T A acc[d]
    energy = th.einsum("bdlj,jk,bdlk->bl", acc, A, acc).clamp_min(0.0)
    return th.sqrt(energy + 1e-12)  # (B,L)


def frame_trust_schedule(
    *,
    t_scalar: float,
    curvature: th.Tensor,      # (B,L)
    pi_min: float,
    pi_max: float,
    tau_min: float,
    c0: float,
    lambda_s: float,
    p: float,
    ) -> th.Tensor:
    """
    Frame-wise trust π_n^(t) in [pi_min, pi_max].

    Paper design:
        τ(t) = τ_min + (1-τ_min)(1-t)
        π̃_n^(t) = τ(t) * c0 / (1 + λ_s (s_n / s_med)^p)
        π_n^(t) = clip(π̃_n^(t), π_min, π_max)

    Returns:
        pi_frame: (B,L)
    """
    # robust per-sample median scale (B,1)
    scale = th.quantile(curvature, 0.5, dim=1, keepdim=True, interpolation="lower").clamp_min(1e-6)

    tau_t = float(tau_min) + (1.0 - float(tau_min)) * (1.0 - float(t_scalar))
    pi = tau_t * float(c0) / (1.0 + float(lambda_s) * (curvature / scale).pow(float(p)))
    return pi.clamp(float(pi_min), float(pi_max))


def distribute_trust_over_halo_joints(
    *,
    pi_frame: th.Tensor,          # (B,L)
    M_halo_any: th.Tensor,        # (B,L,J) bool or 0/1
    q_joint: th.Tensor,           # (J,) weights (supp Eq. (52))
    pi_min: float,
    pi_max: float,
    ) -> th.Tensor:
    """
    Convert frame-wise trust π_n^(t) into per-joint trust π_{n,j}^(t) for halo joints.

    Supplement idea:
      - For each frame n, distribute trust across halo joints H_n proportional to q_j.
      - If no halo joints active, trust isn't used.

    Returns:
        pi_joint: (B,1,L,J) in [pi_min, pi_max] for halo joints,
                  and 1.0 elsewhere (so variance becomes 0 outside halo after masking).
    """
    device, dtype = pi_frame.device, pi_frame.dtype
    B, L = pi_frame.shape
    J = q_joint.numel()

    halo = (M_halo_any > 0.5)  # (B,L,J) bool
    halo_f = halo.to(dtype)

    q = q_joint.view(1, 1, J).to(device=device, dtype=dtype)  # (1,1,J)

    # Normalize weights over active halo joints: sum_{j in H_n} q_j
    denom = (halo_f * q).sum(dim=2, keepdim=True).clamp_min(1e-12)  # (B,L,1)

    # Allocate per-joint trust; if no halo active => values unused anyway.
    pi_nj = (pi_frame.view(B, L, 1) * q) / denom  # (B,L,J)
    pi_nj = pi_nj.clamp(float(pi_min), float(pi_max))

    return pi_nj.view(B, 1, L, J)


def trust_to_variance(
    *,
    pi_joint: th.Tensor,          # (B,1,L,J)
    metric: KinematicMetric,      # uses diag_Rinv_joint
    hard_mask: th.Tensor,         # (B,D,L,J)
    M_all: th.Tensor,             # (B,D,L,J) union mask (hard ∪ halo)
    ) -> th.Tensor:
    """
    Convert trust π -> diagonal variances Σ^2 for the active pseudo-observations.

    For halo rows (pseudo-observations), paper relation:
        σ_i^2(t) = r_i * (1/π_i - 1)
    where r_i = diag(R^{-1})_i.

    We apply the joint-only diagonal r_jj to all frames and channels.
    Hard observations always keep variance 0.
    Non-selected positions are irrelevant and set to 0.

    Returns:
        sigma2: (B,D,L,J)
    """
    device, dtype = hard_mask.device, hard_mask.dtype
    B, D, L, J = hard_mask.shape

    hard = (hard_mask > 0.5)
    # Halo = selected rows minus hard rows
    halo = (M_all > 0.5) & (~hard)

    # Map per-joint trust to all channels; π only matters on halo.
    pi_map = pi_joint.to(device=device, dtype=dtype).view(B, 1, L, J).expand(B, D, L, J)

    # r_jj broadcast
    rjj = metric.diag_Rinv_joint.view(1, 1, 1, J).to(device=device, dtype=dtype)

    sigma2 = rjj * (1.0 / pi_map.clamp_min(1e-12) - 1.0)  # (B,D,L,J)
    sigma2 = th.where(halo, sigma2, th.zeros_like(sigma2))
    sigma2 = sigma2 * (M_all > 0.5).to(dtype)
    return sigma2.contiguous()


# ============================================================
# Projection: x1_proj = x1_hat + R^{-1} A^T (A R^{-1} A^T + Σ)^{-1} (y - A x1_hat)
# ============================================================

@dataclass
class SchurCholesky:
    """
    Cholesky factorization of the observation-space system for one sample.

    We exploit that in inpainting, A is a diagonal selector mask. The selected
    indices correspond to (L,J) positions (shared across D channels), so we can
    factor one K0 and reuse it for all D channels.
    """
    idx_lj: th.Tensor   # (m,) flattened indices in [0, L*J)
    chol: Optional[th.Tensor]  # (m,m) lower-tri Cholesky


def build_observation_cholesky(
    *,
    selector_any: th.Tensor,     # (B,L,J) bool: which (t,j) are selected (hard ∪ halo)
    apply_Rinv: Callable[[th.Tensor], th.Tensor],
    sigma2_any: Optional[th.Tensor],  # (B,L,J) float: diagonal variances for channel-0 pattern
    block_size: int = 1024,
    ) -> List[SchurCholesky]:
    """
    For each batch element b, build Cholesky of:
        K = A R^{-1} A^T + Σ
    where A selects the active (L,J) entries (same for all D).

    This is the "observation-space" system we solve for λ in:
        K λ = (y - x1_hat) on selected indices.

    NOTE: This builds K by applying R^{-1} to unit vectors in blocks.
    It's not the absolute fastest approach, but it's simple and faithful to the math.
    """
    device = selector_any.device
    B, L, J = selector_any.shape

    infos: List[SchurCholesky] = []

    for b in range(B):
        sel_flat = selector_any[b].reshape(-1)  # (L*J,)
        idx = th.nonzero(sel_flat, as_tuple=False).flatten()  # (m,)
        m = int(idx.numel())

        if m == 0:
            infos.append(SchurCholesky(idx_lj=idx, chol=None))
            continue

        # Build K = A R^{-1} A^T + diag(sigma2) in blocks.
        K = th.empty((m, m), device=device, dtype=selector_any.dtype)
        start = 0
        while start < m:
            end = min(start + block_size, m)
            bs = end - start

            # E selects bs basis vectors at the active indices.
            E = th.zeros((bs, 1, L, J), device=device, dtype=selector_any.dtype)
            E_flat = E.reshape(bs, -1)  # (bs, L*J)
            E_flat[th.arange(bs, device=device), idx[start:end]] = 1.0

            RE = apply_Rinv(E).reshape(bs, -1)  # (bs, L*J)
            K_block = RE[:, idx]               # (bs, m)
            K[:, start:end] = K_block.mT       # fill columns
            start = end

        if sigma2_any is not None:
            sig = sigma2_any[b].reshape(-1)[idx]  # (m,)
            K = K + th.diag(sig)

        # Symmetrize + stabilize, then factor.
        K = 0.5 * (K + K.mT) + 1e-8 * th.eye(m, device=device, dtype=K.dtype)
        chol = th.linalg.cholesky(K)

        infos.append(SchurCholesky(idx_lj=idx, chol=chol))

    return infos


def metric_project_clean_endpoint(
    *,
    x1_hat: th.Tensor,             # (B,D,L,J)
    selector: th.Tensor,           # (B,D,L,J) 0/1 (hard ∪ halo)
    targets: th.Tensor,            # (B,D,L,J)
    apply_Rinv: Callable[[th.Tensor], th.Tensor],
    sigma2: Optional[th.Tensor],   # (B,D,L,J) diagonal variances, nonzero only on halo
    block_size: int = 1024,
    ) -> th.Tensor:
    """
    Compute the ProjFlow projection update in the clean-endpoint space.

    Inpainting case uses A as a diagonal selector (mask). We solve:
        x1_proj = x1_hat + R^{-1} A^T (A R^{-1} A^T + Σ)^{-1} (y - A x1_hat)

    Implementation strategy:
      - Build observation-space Cholesky (one per batch element, shared across D channels).
      - Solve for λ on active (L,J) indices for each D.
      - Lift back via R^{-1} and add to x1_hat.
    """
    B, D, L, J = x1_hat.shape
    device, dtype = x1_hat.device, x1_hat.dtype

    sel_any = (selector[:, :1] > 0.5).squeeze(1)  # (B,L,J) bool
    sigma2_any = None
    if sigma2 is not None:
        sigma2_any = sigma2[:, 0]  # (B,L,J)

    schur = build_observation_cholesky(
        selector_any=sel_any.to(dtype),
        apply_Rinv=apply_Rinv,
        sigma2_any=sigma2_any.to(dtype) if sigma2_any is not None else None,
        block_size=block_size,
    )

    lam_full = th.zeros_like(x1_hat)  # (B,D,L,J)

    # Solve per-sample, per-channel system (reusing the same chol across D).
    for b in range(B):
        info = schur[b]
        if info.chol is None or info.idx_lj.numel() == 0:
            continue

        idx = info.idx_lj  # (m,)
        l_idx = (idx // J).to(th.long)
        j_idx = (idx %  J).to(th.long)

        # residual on active indices: (D, m)
        resid = targets[b, :, l_idx, j_idx] - x1_hat[b, :, l_idx, j_idx]  # (D,m)
        rhs = resid.T.contiguous()  # (m,D)

        sol = th.cholesky_solve(rhs, info.chol, upper=False)  # (m,D)
        lam_full[b, :, l_idx, j_idx] = sol.T  # back to (D,m) scatter

    delta = apply_Rinv(lam_full)
    return x1_hat + delta


# ============================================================
# Flow endpoints + recomposition
# ============================================================

def expand_t_like_x(t: th.Tensor, x: th.Tensor) -> th.Tensor:
    """(B,) -> (B,1,1,1) to broadcast with x."""
    return t.view(t.size(0), *([1] * (x.dim() - 1)))


def tweedie_endpoints_from_velocity(
    *,
    x_t: th.Tensor,     # (B,D,L,J)
    t: th.Tensor,       # (B,)
    v: th.Tensor,       # (B,D,L,J)
    path_sampler,
    ) -> Tuple[th.Tensor, th.Tensor]:
    """
    Recover (x1_hat, x0_hat) from the flow field v and the path coefficients.
    This version works for general (alpha_t, sigma_t) paths by using derivatives:

        det    = α_t σ̇_t − σ_t α̇_t
        x1_hat = (σ̇_t x_t − σ_t v) / det
        x0_hat = (−α̇_t x_t + α_t v) / det

    For the linear rectified-flow path α=t, σ=1−t, this reduces to:
        x1_hat = x_t + (1-t) v
        x0_hat = x_t - t v
    """
    a_t, da_t = path_sampler.compute_alpha_t(expand_t_like_x(t, x_t))
    s_t, ds_t = path_sampler.compute_sigma_t(expand_t_like_x(t, x_t))

    det = a_t * ds_t - s_t * da_t
    det = th.where(det.abs() < 1e-12, det + 1e-12, det)

    x1_hat = (ds_t * x_t - s_t * v) / det
    x0_hat = (-da_t * x_t + a_t * v) / det
    return x1_hat, x0_hat


def flowdps_eta(sigma_next: th.Tensor) -> th.Tensor:
    """
    FlowDPS-style noise mixing schedule used in your current code:
        η_t = 1 - σ(t_{k+1})

    sigma_next is already broadcastable (B,1,1,1) or (B,D,L,J).
    """
    return (1.0 - sigma_next).detach()


def noise_mix(
    x0_hat: th.Tensor,
    eta: th.Tensor,
    generator: Optional[th.Generator],
    ) -> th.Tensor:
    """
    Eq. (9): x0_tilde = sqrt(1-η) x0_hat + sqrt(η) eps
    """
    if generator is not None:
        try:
            eps = th.randn_like(x0_hat, generator=generator)
        except TypeError:
            # Older PyTorch versions do not accept `generator` in randn_like.
            eps = th.randn(
                x0_hat.shape,
                device=x0_hat.device,
                dtype=x0_hat.dtype,
                generator=generator,
            )
    else:
        eps = th.randn_like(x0_hat)
    a = th.sqrt(th.clamp(1.0 - eta, min=0.0))
    b = th.sqrt(th.clamp(eta, min=0.0))
    return a * x0_hat + b * eps
