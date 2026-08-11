from __future__ import annotations
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TMR model: encoder, and text-to-motion retrieval head."""

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from motius.models.kimodo.network.model import load_checkpoint_state_dict
from motius.models.kimodo.network.motion_rep.feature_utils import length_to_mask
from motius.models.kimodo.network.sanitize import sanitize_texts
from motius.models.kimodo.network.skeleton import SkeletonBase, build_skeleton
from motius.models.kimodo.network.tools import ensure_batched


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences (batch_first optional)."""

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Note: have to replace torch.exp() and math.log() with torch.pow()
        # due to MKL exp() and ln() throws floating point exceptions on certain CPUs
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        # )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


def load_ckpt(self, ckpt_path):
    """Load model weights from checkpoint path."""
    state_dict = load_checkpoint_state_dict(ckpt_path)
    self.load_state_dict(state_dict)


class ACTORStyleEncoder(nn.Module):
    """Motion encoder in ACTOR style: optional motion_rep projection, VAE/MLP tokens, transformer."""

    def __init__(
        self,
        motion_rep: Optional[nn.Module],
        llm_shape: Optional[Tuple],
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.motion_rep = motion_rep
        if motion_rep is not None and llm_shape is None:
            nfeats = motion_rep.motion_rep_dim
        elif motion_rep is None and llm_shape is not None:
            nfeats = llm_shape[-1]
        else:
            raise ValueError

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        if ckpt_path is not None:
            load_ckpt(self, ckpt_path)

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]


class TMR(nn.Module):
    r"""TMR: Text-to-Motion Retrieval inference code (no decoder)
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr
    """

    @classmethod
    def from_args(
        cls,
        motion_rep: nn.Module,
        llm_shape: tuple | list,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        ckpt_folder: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        motion_encoder, top_text_encoder = None, None

        motion_encoder = ACTORStyleEncoder(
            motion_rep=motion_rep,
            llm_shape=None,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            ckpt_path=Path(ckpt_folder) / "motion_encoder.pt",
        ).to(device)

        top_text_encoder = ACTORStyleEncoder(
            motion_rep=None,
            llm_shape=llm_shape,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            ckpt_path=Path(ckpt_folder) / "text_encoder.pt",
        ).to(device)
        return cls(
            motion_encoder,
            top_text_encoder,
            vae,
            device=device,
            **kwargs,
        )

    def __init__(
        self,
        motion_encoder: nn.Module,
        top_text_encoder: nn.Module,
        vae: bool,
        text_encoder: Optional = None,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = True,
        unit_vector: Optional[bool] = False,
        compute_grads: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.motion_encoder = motion_encoder
        self.text_encoder = top_text_encoder
        self.raw_text_encoder = text_encoder

        self.motion_rep = None
        self.skeleton = None
        if self.motion_encoder is not None:
            self.motion_rep = self.motion_encoder.motion_rep
        if self.motion_rep is not None:
            self.skeleton = self.motion_rep.skeleton

        self.compute_grads = compute_grads

        self.device = device

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean
        self.unit_vector = unit_vector

    def full_text_encoder(self, texts: list[str]):
        assert isinstance(texts, list), "The input should be batched."
        # sanitize the texts first
        # then encode the text, and then use the top text encoder
        texts = sanitize_texts(texts)
        text_feat, text_length = self.raw_text_encoder(texts)
        if isinstance(text_length, list):
            text_length = torch.tensor(text_length, device=self.device)
        else:
            text_length = text_length.to(self.device)
        inputs = {
            "x": text_feat.to(self.device),
            "mask": length_to_mask(text_length, device=self.device),
        }
        return self.text_encoder(inputs)

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "raw_text", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "motion":
            return self.motion_encoder
        elif modality == "raw_text":
            return self.full_text_encoder

        if isinstance(inputs[0], str):
            return self.full_text_encoder

        m_nfeats = self.motion_encoder.nfeats
        t_nfeats = self.text_encoder.nfeats

        if m_nfeats == t_nfeats:
            raise ValueError("Cannot automatically find the encoder, as they share the same input space.")

        nfeats = inputs["x"].shape[-1]
        if nfeats == m_nfeats:
            return self.motion_encoder
        elif nfeats == t_nfeats:
            return self.text_encoder
        else:
            raise ValueError("The inputs is not recognized.")

    def _encode(
        self,
        inputs,
        modality: str = "auto",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
        unit_vector: Optional[bool] = None,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact
        unit_vector = self.unit_vector if unit_vector is None else unit_vector

        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs)

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        if unit_vector:
            latent_vectors = torch.nn.functional.normalize(latent_vectors, dim=-1)

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors

    @ensure_batched(posed_joints=4, lengths=1)
    def encode_motion(
        self,
        posed_joints: torch.Tensor,
        original_skeleton: Optional[SkeletonBase] = None,
        lengths: Optional[torch.Tensor] = None,
        unit_vector: Optional[bool] = None,
    ):
        # TODO here.
        convert_ctx = torch.no_grad() if not self.compute_grads else contextlib.nullcontext()

        if original_skeleton is None:
            original_skeleton = build_skeleton(posed_joints.shape[-2])

        if lengths is None:
            nbatch, nbframes = posed_joints.shape[:2]
            device = posed_joints.device
            assert nbatch == 1, "If lenghts is not provided, the input should not be batched."
            lengths = torch.tensor([nbframes], device=device)

        # slice the posed joints if we use less joints
        skel_slice = self.motion_rep.skeleton.get_skel_slice(original_skeleton)
        posed_joints = posed_joints[..., skel_slice, :]

        with convert_ctx:
            features = self.motion_rep(
                posed_joints=posed_joints,
                to_normalize=True,
                lengths=lengths,
            )
            mask = length_to_mask(lengths, device=features.device)
            x_dict = {"x": features, "mask": mask}
            latent_vectors = self._encode(
                x_dict,
                modality="motion",
                unit_vector=unit_vector,
            )
        return latent_vectors

    def encode_text(
        self,
        x_dict: Dict,
        unit_vector: Optional[bool] = None,
    ):
        # TODO: make it ensure batched
        convert_ctx = torch.no_grad() if not self.compute_grads else contextlib.nullcontext()

        with convert_ctx:
            latent_vectors = self._encode(
                x_dict,
                modality="text",
                unit_vector=unit_vector,
            )
        return latent_vectors

    def encode_raw_text(
        self,
        texts: List[str],
        unit_vector: Optional[bool] = None,
    ):
        is_batched = True
        if isinstance(texts, str):
            is_batched = False
            texts = [texts]

        convert_ctx = torch.no_grad() if not self.compute_grads else contextlib.nullcontext()

        with convert_ctx:
            latent_vectors = self._encode(
                texts,
                modality="raw_text",
                unit_vector=unit_vector,
            )
        if not is_batched:
            latent_vectors = latent_vectors[0]
        return latent_vectors
