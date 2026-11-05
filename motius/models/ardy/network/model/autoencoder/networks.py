from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Neural encoder, scalar quantizer and conditional decoder for ARDY."""
import torch
from torch import nn
from vector_quantize_pytorch import FSQ
from .transformer import EncoderTransformer, DoubleCondDecoderTransformer


class FSQNetworks(nn.Module):
    def __init__(
        self,
        encoder_state_dim,
        decoder_state_dim,
        decoder_target_cond_dim,
        decoder_external_cond_dim,
        num_fsq_levels=5,
        fsq_level_list=(16, 16, 16, 16, 16),
        num_frames_per_token=4,
        latent_dim=512,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation="gelu",
        norm_first=False,
        num_layers=8,
        pe_dropout=0.1,
        causal_encoder=False,
        causal_decoder=False,
    ):
        super().__init__()
        latent_embedding_dim = num_fsq_levels
        self.encoder = EncoderTransformer(
            encoder_state_dim,
            latent_embedding_dim,
            num_frames_per_token,
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            norm_first,
            num_layers,
            pe_dropout,
            causal_encoder,
        )
        self.decoder = DoubleCondDecoderTransformer(
            latent_embedding_dim,
            decoder_state_dim,
            num_frames_per_token,
            latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            norm_first,
            num_layers,
            pe_dropout,
            causal_decoder,
            target_cond_dim=decoder_target_cond_dim,
            external_cond_dim=decoder_external_cond_dim,
        )

        if isinstance(fsq_level_list, int):
            fsq_level_list = [fsq_level_list] * num_fsq_levels
        assert (
            len(fsq_level_list) == num_fsq_levels
        ), f"fsq_level_list must have num_fsq_levels={num_fsq_levels} entries, got {len(fsq_level_list)}"
        self.quantizer = FSQ(levels=fsq_level_list, return_indices=False)

    def encode(self, features, motion_pad_mask):
        return self.encoder(features, motion_pad_mask)

    def quantize(self, latent):
        return self.quantizer(latent)

    def decode(self, latent, external_cond=None, motion_pad_mask=None):
        return self.decoder(
            latent, external_cond=external_cond, motion_pad_mask=motion_pad_mask
        )
