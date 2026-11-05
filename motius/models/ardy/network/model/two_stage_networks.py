from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Root/body Transformer projections; physical representation stays in the Pipeline."""
from torch import nn
from .backbone import TransformerEncoderBlock


class TwoStageNetworks(nn.Module):
    def __init__(
        self,
        local_root_dim,
        motion_rep_dim,
        body_dim,
        nbjoints,
        motion_mask_mode,
        num_frames_per_token,
        nframe_root_dim,
        latent_embedding_dim,
        **kwargs,
    ):
        super().__init__()
        self.num_frames_per_token = num_frames_per_token
        self.nframe_root_dim = nframe_root_dim
        self.latent_embedding_dim = latent_embedding_dim
        self.latent_dim = kwargs["latent_dim"]
        self.nframe_local_root_dim = local_root_dim * self.num_frames_per_token
        global_root_hybrid_token_dim = self.nframe_root_dim + self.latent_embedding_dim

        local_root_hybrid_token_dim = (
            self.nframe_local_root_dim + self.latent_embedding_dim
        )  # body stage always takes in local root info for motion (but still the global mask)

        if motion_mask_mode == "concat":
            # stage 1: root pred only
            self.global_root_hybrid_proj = nn.Linear(
                global_root_hybrid_token_dim, self.latent_dim
            )
            global_root_hybrid_constraints_dim = (
                global_root_hybrid_token_dim
                + (motion_rep_dim + body_dim) * self.num_frames_per_token
            )
            self.global_root_hybrid_constraints_proj = nn.Linear(
                global_root_hybrid_constraints_dim, self.latent_dim
            )  # generation tokens with constraints infilling and concat
            # stage 2: body pred only
            self.local_root_hybrid_proj = nn.Linear(
                local_root_hybrid_token_dim, self.latent_dim
            )
            local_root_hybrid_constraints_dim = (
                local_root_hybrid_token_dim
                + (motion_rep_dim + body_dim) * self.num_frames_per_token
            )  # body stage always takes in local root info for motion (but still the global mask)
            self.local_root_hybrid_constraints_proj = nn.Linear(
                local_root_hybrid_constraints_dim, self.latent_dim
            )  # generation tokens with constraints infilling and concat
            # explicit future constraints projection for both stages
            self.future_constraints_proj = nn.Linear(
                motion_rep_dim * 2 * self.num_frames_per_token,
                self.latent_dim,
            )
        else:
            assert motion_mask_mode is None
            self.global_root_hybrid_proj = nn.Linear(
                global_root_hybrid_token_dim, self.latent_dim
            )
            self.local_root_hybrid_proj = nn.Linear(
                local_root_hybrid_token_dim, self.latent_dim
            )

        root_output_dim = self.nframe_root_dim
        self.root_model = TransformerEncoderBlock(
            add_input_proj=False,
            input_dim=-1,  # for compatibility
            output_dim=root_output_dim,
            nbjoints=nbjoints,
            **kwargs,
        )

        body_output_dim = self.latent_embedding_dim
        self.body_model = TransformerEncoderBlock(
            add_input_proj=False,
            input_dim=-1,  # for compatibility
            output_dim=body_output_dim,
            nbjoints=nbjoints,
            **kwargs,
        )

    def predict_root(self, *args, **kwargs):
        return self.root_model(*args, **kwargs)

    def predict_body(self, *args, **kwargs):
        return self.body_model(*args, **kwargs)
