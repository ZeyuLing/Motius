# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 actions components; see licenses/sonic."""
from __future__ import annotations
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = None
