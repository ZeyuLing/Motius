# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 curriculum components; see licenses/sonic."""
from __future__ import annotations
from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    force_push_curriculum = None
    force_push_linear_curriculum = None
