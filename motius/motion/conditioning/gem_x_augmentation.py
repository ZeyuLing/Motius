# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

# SOMA77 hierarchy
# SOMA finger/feet groups from old repo.
NVSKEL_LFINGERS_IDX = [
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
]
NVSKEL_RFINGERS_IDX = [
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
]
NVSKEL_LFEET_IDX = [70, 71]
NVSKEL_RFEET_IDX = [75, 76]

SOMA77_AUG = {
    "jittering": torch.tensor(
        [
            # 0-10
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # 11-14
            0.3,
            0.3,
            0.2,
            0.1,
            # 15-23
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # 24-32
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # 33-38
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # 39-42
            0.3,
            0.3,
            0.2,
            0.1,
            # 43-66
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # 67-76
            0.4,
            0.2,
            0.1,
            0.1,
            0.1,
            0.4,
            0.2,
            0.1,
            0.1,
            0.1,
        ]
    ),
    "bias": torch.tensor(
        [
            0.15,
            0.15,
            0.15,
            0.15,
            0.15,
            0.15,
            0.1,
            0.1,
            0.10,
            0.10,
            0.10,
            0.15,
            0.15,
            0.25,
            0.35,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.15,
            0.15,
            0.25,
            0.35,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.10,
            0.15,
            0.25,
            0.05,
            0.05,
            0.10,
            0.15,
            0.25,
            0.05,
            0.05,
        ]
    ),
    "peak": torch.tensor(
        [
            0.15,
            0.15,
            0.15,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.15,
            0.15,
            0.25,
            0.40,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.15,
            0.15,
            0.25,
            0.40,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.08,
            0.30,
            0.40,
            0.40,
            0.40,
            0.08,
            0.30,
            0.40,
            0.40,
            0.40,
        ]
    ),
    "pmask": torch.tensor(
        [
            0.10,
            0.10,
            0.10,
            0.10,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.10,
            0.10,
            0.20,
            0.30,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.10,
            0.10,
            0.20,
            0.30,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.10,
            0.20,
            0.20,
            0.30,
            0.30,
            0.10,
            0.20,
            0.20,
            0.30,
            0.30,
        ]
    ),
}
SOMA77_AUG_CUDA = {}
NVHUMAN77_TREE = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
]


def get_bias_cuda(shape=(8, 120), s_bias=1e-1, num_J=77):
    if num_J != 77:
        raise ValueError(f"num_J: {num_J} is not supported (only 77)")
    if "bias" not in SOMA77_AUG_CUDA:
        SOMA77_AUG_CUDA["bias"] = SOMA77_AUG["bias"].cuda().reshape(1, num_J, 1)

    bias = SOMA77_AUG_CUDA["bias"]
    bias_noise = torch.randn((shape[0], num_J, 3), device="cuda") * bias * s_bias
    bias_noise = bias_noise[:, None].expand(-1, shape[1], -1, -1).clone()
    # use the same bias for all fingertips and feet
    bias_noise[:, :, NVSKEL_LFINGERS_IDX, :] += bias_noise[:, :, [14], :]
    bias_noise[:, :, NVSKEL_RFINGERS_IDX, :] += bias_noise[:, :, [42], :]
    bias_noise[:, :, NVSKEL_LFEET_IDX, :] += bias_noise[:, :, [69], :]
    bias_noise[:, :, NVSKEL_RFEET_IDX, :] += bias_noise[:, :, [74], :]
    return bias_noise


def get_lfhp_cuda(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3, num_J=77):
    if num_J != 77:
        raise ValueError(f"num_J: {num_J} is not supported (only 77)")
    if "peak" not in SOMA77_AUG_CUDA:
        SOMA77_AUG_CUDA["pmask"] = SOMA77_AUG["pmask"].cuda()
        SOMA77_AUG_CUDA["peak"] = SOMA77_AUG["peak"].cuda().reshape(num_J, 1)

    pmask = SOMA77_AUG_CUDA["pmask"]
    peak = SOMA77_AUG_CUDA["peak"]
    peak_noise_mask = torch.rand(*shape, num_J, device="cuda") * pmask < s_peak_mask
    peak_noise = (
        peak_noise_mask.float().unsqueeze(-1).expand(-1, -1, -1, 3)
        * torch.randn(3, device="cuda")
        * peak
        * s_peak
    )
    return peak_noise


def get_jitter_cuda(shape=(8, 120), s_jittering=5e-2, num_J=77):
    if num_J != 77:
        raise ValueError(f"num_J: {num_J} is not supported (only 77)")
    if "jittering" not in SOMA77_AUG_CUDA:
        SOMA77_AUG_CUDA["jittering"] = SOMA77_AUG["jittering"].cuda().reshape(1, 1, num_J, 1)
    jittering = SOMA77_AUG_CUDA["jittering"]
    jittering_noise = torch.randn((*shape, num_J, 3), device="cuda") * jittering * s_jittering
    return jittering_noise


def get_wham_aug_kp3d(shape=(8, 120), num_J=77, device=None):
    aug = (
        get_bias_cuda(shape, num_J=num_J)
        + get_lfhp_cuda(shape, num_J=num_J)
        + get_jitter_cuda(shape, num_J=num_J)
    )
    return aug


def get_visible_mask(shape=(8, 120), num_J=77, s_mask=0.03, device=None):
    if num_J != 77:
        raise ValueError(f"num_J: {num_J} is not supported (only 77)")
    mask = torch.rand(*shape, num_J, device=device) < s_mask
    visible = (~mask).clone().reshape(-1, num_J)

    for child in range(num_J):
        parent = NVHUMAN77_TREE[child]
        if parent == -1:
            continue
        if isinstance(parent, list):
            visible[:, child] = visible[:, child] & visible[:, parent[0]] & visible[:, parent[1]]
        else:
            visible[:, child] = visible[:, child] & visible[:, parent]
    return visible.reshape(*shape, num_J)


def get_invisible_legs_mask(shape, num_J=77, s_mask=0.03, device=None):
    if num_J != 77:
        raise ValueError(f"num_J: {num_J} is not supported (only 77)")
    B, L = shape
    starts = torch.randint(0, max(L - 90, 1), (B,), device=device)
    ends = starts + torch.randint(30, 90, (B,), device=device)
    mask_range = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, num_J).clone()
    mask_to_apply[:, :, 67:] = False  # only legs are invisible
    return mask_to_apply & (torch.rand(B, 1, 1, device=device) < s_mask)


def randomly_modify_hands_legs(j3d, num_J=77):
    if num_J != 77:
        return j3d
    lhand, rhand, lleg, rleg = 14, 42, 67, 72

    B, L = j3d.shape[:2]
    out = j3d.clone()
    mask = torch.rand(B, L, device=j3d.device) < 0.001
    out[mask][:, [lhand, rhand]] = out[mask][:, [rhand, lhand]]
    mask = torch.rand(B, L, device=j3d.device) < 0.001
    out[mask][:, [lleg, rleg]] = out[mask][:, [rleg, lleg]]
    return out
