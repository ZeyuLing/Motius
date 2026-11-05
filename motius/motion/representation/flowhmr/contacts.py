import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

def _tv1d_denoise_l2(y: Tensor, lam: float = 0.02, n_iter: int = 60) -> Tensor:
    # 1D ROF: min_x 0.5*||x - y||_2^2 + lam * TV(x)
    assert y.dim() == 1, "y must be 1D tensor"
    L = y.shape[0]
    if L <= 1:
        return y.clone()
    device = y.device
    dtype = y.dtype

    x = y.clone()
    x_bar = x.clone()
    p = torch.zeros(L - 1, device=device, dtype=dtype)

    tau = 0.5
    sigma = 0.5
    theta = 1.0

    for _ in range(n_iter):
        # p^{k+1} = proj_{|p| <= lam}(p^k + sigma * D x_bar)
        g = x_bar[1:] - x_bar[:-1]
        p = p + sigma * g
        p = torch.clamp(p, -lam, lam)

        # x^{k+1} = (x^k + tau*(-D^T p) + tau*y) / (1+tau)
        div_p = torch.zeros_like(x)
        div_p[0] = -p[0]
        if L > 2:
            div_p[1:-1] = p[:-1] - p[1:]
        div_p[-1] = p[-1]

        x_prev = x
        x = (x + tau * div_p + tau * y) / (1.0 + tau)
        x_bar = x + theta * (x - x_prev)

    return x

def _estimate_floor_height_series(
    y_all: Tensor,
    spd_xz: Tensor,
    floor_vel_thr_per_frame: float,
    floor_height_offset_m: float,
    win: int = 15,
) -> Tensor:
    L, K = y_all.shape
    device = y_all.device
    dtype = y_all.dtype

    if spd_xz.numel() == 0:
        slow_L = torch.ones((L, K), dtype=torch.bool, device=device)
    else:
        if spd_xz.shape[0] == L:
            slow_L = spd_xz < floor_vel_thr_per_frame
        else:
            slow = spd_xz < floor_vel_thr_per_frame
            slow_L = torch.cat([slow[:1], slow], dim=0)
    if slow_L.shape[1] != K:
        if slow_L.shape[1] > K:
            slow_L = slow_L[:, :K]
        else:
            pad = torch.ones((L, K - slow_L.shape[1]), dtype=slow_L.dtype, device=device)
            slow_L = torch.cat([slow_L, pad], dim=1)

    y_obs = torch.empty(L, device=device, dtype=dtype)
    q = 0.2
    for t in range(L):
        mask = slow_L[t]
        if mask.any():
            vals = y_all[t, mask]
            y_obs[t] = torch.quantile(vals, q)
        else:
            t0, t1 = max(0, t - win), min(L, t + 1)
            mask_win = slow_L[t0:t1].any(dim=1)
            if mask_win.any():
                vals = y_all[t0:t1][slow_L[t0:t1]]
                y_obs[t] = torch.quantile(vals, q)
            else:
                y_obs[t] = y_obs[t - 1] if t > 0 else torch.quantile(y_all[t], 0.1)

    if L >= 3:
        x = y_obs.view(1, 1, L)
        y_obs = torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1).view(-1)

    lambda_tv = 0.02
    floor_h_t = _tv1d_denoise_l2(y_obs, lam=lambda_tv, n_iter=60)

    return floor_h_t - floor_height_offset_m

def _hysteresis_and_morph(
    prob: Tensor,
    on_thr: float = 0.7,
    off_thr: float = 0.5,
    morph_min_len: int = 3,
    morph_max_gap: int = 2,
) -> Tensor:
    L, K = prob.shape
    device = prob.device
    contact = torch.zeros_like(prob, dtype=torch.bool)
    prev = torch.zeros((K,), dtype=torch.bool, device=device)
    for t in range(L):
        on = prob[t] > on_thr
        off = prob[t] < off_thr
        prev = torch.where(on, torch.ones_like(prev, dtype=torch.bool), prev)
        prev = torch.where(off, torch.zeros_like(prev, dtype=torch.bool), prev)
        contact[t] = prev

    def morph_clean(x: Tensor, min_len: int = 3, max_gap: int = 2) -> Tensor:
        x = x.clone()
        cnt = 0
        for tt in range(L):
            if x[tt]:
                cnt += 1
            if (not x[tt]) or tt == L - 1:
                if 0 < cnt < min_len:
                    x[tt - cnt : tt] = False
                cnt = 0
        gap = 0
        last_on = -1
        for tt in range(L):
            if x[tt]:
                if 0 < gap <= max_gap and last_on >= 0:
                    x[last_on + 1 : tt] = True
                last_on = tt
                gap = 0
            else:
                gap += 1
        return x

    return torch.stack(
        [morph_clean(contact[:, j], morph_min_len, morph_max_gap) for j in range(K)],
        dim=1,
    )

def get_contact_prob(
    kpts: Tensor,
    joint_ids: tuple,
    vxz_thr: float,
    height_thr_per_joint: torch.Tensor,
    use_height: bool = True,
    floor_vel_thr_per_frame: float = 0.005,
    floor_height_offset_m: float = 0.0,
    use_dbscan: bool = False,
    cluster_eps_m: float = 0.005,
    cluster_min_samples: int = 3,
    floor_h_override: Optional[Tensor] = None,
    on_thr: float = 0.7,
    off_thr: float = 0.5,
    morph_min_len: int = 3,
    morph_max_gap: int = 2,
    use_vertical_cues: bool = False,
    vy_ext_thr: float = 1e-4,
    dvy_thr: float = 0.030,
    vy_static_thr: float = 5e-4,
):
    def _smooth1d_avg(x: Tensor, win: int = 3) -> Tensor:
        if win <= 1 or x.shape[0] < 3:
            return x
        pad_total = win - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        xT = x.permute(1, 0).unsqueeze(1)
        x_pad = F.pad(xT, (pad_left, pad_right), mode="replicate")
        y = F.avg_pool1d(x_pad, kernel_size=win, stride=1, padding=0)
        return y.squeeze(1).permute(1, 0)

    L = kpts.shape[0]
    device = kpts.device
    idx = torch.tensor(joint_ids, device=device)
    pts = kpts[:, idx, :]
    K = pts.shape[1]
    y_all = pts[..., 1]
    if L >= 2:
        disp = pts[1:] - pts[:-1]
        spd_xz = torch.linalg.norm(disp[..., [0, 2]], dim=-1)
        spd_xz_L = torch.empty((L, K), device=device, dtype=pts.dtype)
        spd_xz_L[:-1] = spd_xz
        spd_xz_L[-1] = spd_xz_L[-2]
    else:
        disp = torch.empty((0, K, 3), device=device, dtype=pts.dtype)
        spd_xz = torch.zeros((L, K), device=device, dtype=pts.dtype)
        spd_xz_L = torch.zeros((L, K), device=device, dtype=pts.dtype)

    if floor_h_override is None:
        floor_h = _estimate_floor_height_series(
            y_all,
            (spd_xz if L >= 2 else torch.empty((0, K), device=device, dtype=pts.dtype)),
            floor_vel_thr_per_frame,
            floor_height_offset_m,
        )
    else:
        floor_h = floor_h_override.clone().detach()
    y_rel = y_all - floor_h[:, None]
    h_thr = height_thr_per_joint.to(device).view(1, -1).expand_as(y_rel)

    eps = torch.tensor(1e-6, device=device, dtype=pts.dtype)
    if use_height:
        height_band = torch.tensor(0.02, device=device, dtype=pts.dtype)
        s_height = torch.clamp((h_thr + height_band - y_rel) / height_band, 0.0, 1.0)
    else:
        s_height = torch.ones_like(y_rel, dtype=pts.dtype)
    vxz = torch.tensor(vxz_thr, device=device, dtype=pts.dtype)
    vel_band = torch.clamp(vxz * 0.5, min=eps)
    s_hspd = torch.clamp((vxz + vel_band - spd_xz_L) / vel_band, 0.0, 1.0)

    if use_vertical_cues:
        y_s = _smooth1d_avg(y_all, win=3)
        vy = torch.zeros((L, K), device=device, dtype=pts.dtype)
        if L >= 2:
            vy[1:] = y_s[1:] - y_s[:-1]
            vy[0] = vy[1]
        dvy = torch.zeros((L, K), device=device, dtype=pts.dtype)
        if L >= 2:
            dvy[1:] = vy[1:] - vy[:-1]
            dvy[0] = dvy[1]
        plate_win = 5
        plate_eps = 5e-5
        plate_dwell = 3
        x = y_s.permute(1, 0).unsqueeze(1)
        roll_min = (-F.max_pool1d(-x, kernel_size=plate_win, stride=1, padding=plate_win // 2)).squeeze(1).permute(1, 0)
        near_min = y_s <= (roll_min + plate_eps)
        run = torch.zeros((L, K), dtype=torch.int32, device=y_s.device)
        run[0] = near_min[0].int()
        for t in range(1, L):
            run[t] = torch.where(
                near_min[t],
                run[t - 1] + 1,
                torch.tensor(0, device=y_s.device, dtype=torch.int32),
            )
        y_minima = run >= plate_dwell
        vy_prev = torch.zeros_like(vy)
        if L >= 2:
            vy_prev[1:] = vy[:-1]
            vy_prev[0] = vy[0]
        not_rising = (vy_prev < vy_ext_thr) & (vy <= vy_ext_thr)
        vertical_on = not_rising & (y_minima | (dvy > dvy_thr))
        vertical_on = vertical_on & (y_rel < (h_thr + 0.01))
        sustain_mem = 5
        if sustain_mem > 1:
            x = vertical_on.float().permute(1, 0).unsqueeze(1)
            y = F.max_pool1d(x, kernel_size=sustain_mem, stride=1, padding=0)
            y = F.pad(y, (sustain_mem - 1, 0))
            vertical_on_recent = y.squeeze(1).permute(1, 0) > 0
        else:
            vertical_on_recent = vertical_on
        s_vertical = vertical_on_recent.float()
        boost = torch.tensor(0.7, device=device, dtype=pts.dtype)
    else:
        s_vertical = torch.zeros_like(y_rel, dtype=pts.dtype)
        boost = torch.tensor(0.0, device=device, dtype=pts.dtype)

    score_base = s_hspd * s_height
    prob = torch.maximum(score_base, (boost * s_vertical) * s_height)
    prob = _smooth1d_avg(prob, win=5)
    contact_hys = _hysteresis_and_morph(prob, on_thr, off_thr, morph_min_len, morph_max_gap)
    return prob, contact_hys

def get_foot_detect(
    kpts: Tensor,
    joint_ids: tuple = (7, 10, 8, 11), # [L_ankle, L_toe, R_ankle, R_toe]
    vel_thr_per_frame: float = 0.005,
    use_height: bool = True,
    toe_height_thr_m: float = 0.04,
    ankle_height_thr_m: float = 0.08,
    floor_h_override: Optional[Tensor] = None,
):
    height_thr = torch.tensor(
        [ankle_height_thr_m, toe_height_thr_m, ankle_height_thr_m, toe_height_thr_m],
        dtype=torch.float32,
    )
    return get_contact_prob(
        kpts=kpts,
        joint_ids=joint_ids,
        vxz_thr=vel_thr_per_frame,
        height_thr_per_joint=height_thr,
        use_height=use_height,
        floor_h_override=floor_h_override,
        on_thr=0.6,
        off_thr=0.4,
        morph_min_len=2,
        morph_max_gap=3,
    )
