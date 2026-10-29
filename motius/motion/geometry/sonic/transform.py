# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 transform components; see licenses/sonic."""
from motius.motion.geometry.sonic.conversions import angle_axis_to_quaternion
from motius.motion.geometry.sonic.conversions import quaternion_to_angle_axis


def normalize(x, eps: float = 1e-09):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def get_heading_q(q):
    q_new = q.clone()
    q_new[..., 1] = 0
    q_new[..., 2] = 0
    q_new = normalize(q_new)
    return q_new
