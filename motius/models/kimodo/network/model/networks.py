# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neural components shared by the native KIMODO trainer and pipeline."""
import torch
from torch import nn


class KimodoNetworks(nn.Module):
    def __init__(self, denoiser, num_base_steps, text_encoder=None, device=None, cfg_type="separated"):
        super().__init__()
        self.denoiser = denoiser.eval()
        self.num_base_steps = int(num_base_steps)
        self.cfg_type = "nocfg" if cfg_type is None else cfg_type
        self.to(device)

    @property
    def motion_rep(self):
        return self.denoiser.motion_rep

    @property
    def skeleton(self):
        return self.motion_rep.skeleton

    @property
    def fps(self):
        return self.motion_rep.fps

    @property
    def device(self):
        return next(self.denoiser.parameters()).device

    def forward(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs)
