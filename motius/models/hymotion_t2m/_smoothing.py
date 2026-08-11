"""Native inference-time temporal smoothing matching the official HY-Motion-1.0
inference path.

The official ``MotionFlowMatching.generate`` calls
``decode_motion_from_latent(sampled, should_apply_smooothing=True)``, which
applies:

  * **SLERP/quaternion Gaussian smoothing** (``sigma=1.0``) to the first 22
    joints' rot6d (``smooth_with_slerp``), and
  * **Savitzky-Golay smoothing** (``window_length=11, polyorder=5``) to the
    root translation (``smooth_with_savgol``).

Our reimplementation previously skipped both, so the raw flow-matching output
was rendered without any temporal filtering -> severe high-frequency jitter.
This module reproduces the official smoothing bit-for-bit so the Motius
decode matches the official inference.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from torch import Tensor

# ---------------------------------------------------------------------------
# Rotation conversions (pytorch3d-style, official convention)
# ---------------------------------------------------------------------------


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rot6d_to_rotation_matrix(rot6d: Tensor) -> Tensor:
    """Zhou et al. 2019 6D -> 3x3 (first two columns + Gram-Schmidt)."""
    x = rot6d.view(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation_matrix_to_rot6d(rotation_matrix: Tensor) -> Tensor:
    v1 = rotation_matrix[..., 0:1]
    v2 = rotation_matrix[..., 1:2]
    return torch.cat([v1, v2], dim=-1).reshape(*v1.shape[:-2], 6)


def quaternion_fix_continuity(q: Tensor) -> Tensor:
    """Force quaternion continuity across time by flipping sign (q vs -q) so
    consecutive frames keep a positive dot product."""
    assert q.ndim in (2, 3), f"Expected (L,J,4) or (L,4), got {q.shape}"
    assert q.shape[-1] == 4, f"Last dim must be 4, got {q.shape[-1]}"
    if q.shape[0] <= 1:
        return q.clone()

    result = q.clone()
    dot_products = torch.sum(q[1:] * q[:-1], dim=-1)
    flip_mask = dot_products < 0
    flip_mask = (torch.cumsum(flip_mask.int(), dim=0) % 2).bool()
    result[1:][flip_mask] *= -1
    return result


# ---------------------------------------------------------------------------
# Gaussian quaternion averaging (numpy)
# ---------------------------------------------------------------------------


def gaussian_kernel1d(sigma: float, order: int, radius: int) -> np.ndarray:
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    q = np.zeros(order + 1)
    q[0] = 1
    D = np.diag(exponent_range[1:], 1)
    P = np.diag(np.ones(order) / -sigma2, -1)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return q * phi_x


def slice_seq_with_padding(whole_seq: np.ndarray, middle_idx: int, length: int) -> np.ndarray:
    whole_seq_padded = whole_seq.copy()
    if middle_idx - length // 2 < 0:
        l_pad_len = length // 2 - middle_idx
        whole_seq_padded = np.concatenate(
            [np.stack([whole_seq_padded[0]] * l_pad_len), whole_seq_padded], axis=0
        )
    else:
        l_pad_len = 0
    if middle_idx + length - length // 2 > len(whole_seq):
        r_pad_len = middle_idx + length - length // 2 - len(whole_seq)
        whole_seq_padded = np.concatenate(
            [whole_seq_padded, np.stack([whole_seq_padded[-1]] * r_pad_len)], axis=0
        )
    else:
        r_pad_len = 0
    assert len(whole_seq_padded) == len(whole_seq) + l_pad_len + r_pad_len
    middle_idx_padded = middle_idx + l_pad_len
    return whole_seq_padded[
        middle_idx_padded - length // 2 : middle_idx_padded - length // 2 + length
    ]


def wavg_quaternion_markley(Q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted quaternion average (Markley 2007 eigenvector method)."""
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0
    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        if q[0] < 0:
            q = -q
        A += w_i * (np.outer(q, q))
        wSum += w_i
    A /= wSum
    return np.linalg.eigh(A)[1][:, -1]


def smooth_quats(quats: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if len(quats) == 0 or sigma <= 0:
        return quats.copy()

    q_all = quaternion_fix_continuity(torch.from_numpy(quats)).numpy()

    results = q_all.copy()
    truncate = 4.0
    order = 0
    lw = int(truncate * float(sigma) + 0.5)
    weights = gaussian_kernel1d(sigma=sigma, order=order, radius=lw)[::-1]
    kernel_len = len(weights)

    for fr in range(len(q_all)):
        cur_quats = slice_seq_with_padding(q_all, fr, kernel_len)  # (K,4)
        ref = cur_quats[kernel_len // 2 : kernel_len // 2 + 1]  # (1,4)
        dots = (cur_quats * ref).sum(axis=-1, keepdims=True)  # (K,1)
        cur_quats = np.where(dots < 0.0, -cur_quats, cur_quats)
        results[fr, :] = wavg_quaternion_markley(cur_quats, weights)

    return results.copy()


def smooth_rotation(quats: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if quats.ndim == 4:
        is_batch = True
    else:
        is_batch = False
        quats = quats[None, ...]
    for b in range(quats.shape[0]):
        for j_idx in range(quats.shape[2]):
            cur_quats = quats[b, :, j_idx].copy()
            cur_quats_t = quaternion_fix_continuity(torch.from_numpy(cur_quats)).numpy()
            quats[b, :, j_idx] = smooth_quats(cur_quats_t, sigma=sigma)
    if not is_batch:
        quats = quats.squeeze(0)
    return quats


# ---------------------------------------------------------------------------
# Public smoothing entry points (mirror MotionFlowMatching.smooth_with_*)
# ---------------------------------------------------------------------------


def smooth_with_savgol(
    input: Tensor, window_length: int = 11, polyorder: int = 5
) -> Tensor:
    if len(input.shape) == 2:
        is_batch = False
        input = input.unsqueeze(0)
    else:
        is_batch = True
    if input.shape[1] <= window_length:
        return input if is_batch else input.squeeze(0)
    input_np = input.cpu().numpy()
    input_smooth_np = np.empty_like(input_np, dtype=np.float32)
    for b in range(input_np.shape[0]):
        for j in range(input_np.shape[2]):
            input_smooth_np[b, :, j] = savgol_filter(input_np[b, :, j], window_length, polyorder)
    input_smooth = torch.from_numpy(input_smooth_np).to(input)
    if not is_batch:
        input_smooth = input_smooth.squeeze(0)
    return input_smooth


def smooth_with_slerp(input: Tensor, sigma: float = 1.0) -> Tensor:
    """Quaternion Gaussian smoothing of rot6d ``(B, L, J, 6)``."""

    def fix_time_continuity(q: Tensor, time_dim: int = -3):
        shape = q.shape
        qv = q.moveaxis(time_dim, 0).contiguous().view(shape[time_dim], -1, 4)
        qv = quaternion_fix_continuity(qv)
        return qv.view(
            shape[time_dim], *shape[:time_dim], *shape[time_dim + 1 :]
        ).moveaxis(0, time_dim)

    RR = rot6d_to_rotation_matrix(input)
    qq = matrix_to_quaternion(RR)
    qq_np = fix_time_continuity(qq, time_dim=1).cpu().numpy()
    qq_s_np = smooth_rotation(qq_np, sigma=sigma)
    input_smooth = rotation_matrix_to_rot6d(
        quaternion_to_matrix(torch.from_numpy(qq_s_np))
    )
    return input_smooth.to(input.device)
