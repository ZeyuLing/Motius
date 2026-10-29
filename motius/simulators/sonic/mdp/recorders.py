# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 recorders components; see licenses/sonic."""
from __future__ import annotations
from isaaclab.managers import recorder_manager
from isaaclab.utils import configclass


@configclass
class RecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    render_envs = None
    running_ref_root_height = None
    trajectory = None
