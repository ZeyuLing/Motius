"""Motius-native TMR architecture used by the released evaluator artifacts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


_DEFAULT_ARCH = {
    "latent_dim": 256,
    "ff_size": 1024,
    "num_layers": 6,
    "num_heads": 4,
    "dropout": 0.1,
    "activation": "gelu",
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(inputs + self.pe[:, : inputs.shape[1]])


class ACTORStyleEncoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.nfeats = int(nfeats)
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = bool(vae)
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        features = self.projection(inputs["x"])
        mask = inputs["mask"].bool()
        tokens = self.tokens.unsqueeze(0).expand(len(features), -1, -1)
        sequence = torch.cat((tokens, features), dim=1)
        token_mask = torch.ones(
            (len(features), self.nbtokens), dtype=torch.bool, device=features.device
        )
        mask = torch.cat((token_mask, mask), dim=1)
        sequence = self.sequence_pos_encoding(sequence)
        encoded = self.seqTransEncoder(sequence, src_key_padding_mask=~mask)
        return encoded[:, : self.nbtokens]


class ACTORStyleDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, latent: Tensor, mask: Tensor) -> Tensor:
        queries = torch.zeros(
            len(latent), mask.shape[1], latent.shape[-1], device=latent.device
        )
        queries = self.sequence_pos_encoding(queries)
        output = self.seqTransDecoder(
            tgt=queries,
            memory=latent[:, None],
            tgt_key_padding_mask=~mask,
        )
        output = self.final_layer(output)
        output[~mask] = 0
        return output


class TMRCore(nn.Module):
    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        sample_mean: bool = True,
        fact: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder
        self.motion_decoder = motion_decoder
        self.vae = bool(vae)
        self.sample_mean = bool(sample_mean)
        self.fact = 1.0 if fact is None else float(fact)

    def encode(
        self,
        inputs: Dict[str, Tensor],
        modality: str,
        sample_mean: Optional[bool] = None,
    ) -> Tensor:
        encoder = self.text_encoder if modality == "text" else self.motion_encoder
        encoded = encoder(inputs)
        if not self.vae:
            return encoded[:, 0]
        mean, log_variance = encoded.unbind(1)
        use_mean = self.sample_mean if sample_mean is None else bool(sample_mean)
        if use_mean:
            return mean
        return mean + self.fact * torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, latent: Tensor, mask: Tensor) -> Tensor:
        return self.motion_decoder(latent, mask)


@MODEL_BUNDLES.register_module()
class TMRBundle(ModelBundle):
    """Self-contained TMR bundle shared by SMPL-22 and G1 evaluators."""

    def __init__(
        self,
        motion_nfeats: int = 38,
        text_nfeats: int = 768,
        vae: bool = True,
        arch: Optional[Dict[str, Any]] = None,
        sample_mean: bool = True,
        fact: Optional[float] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        architecture = deepcopy(_DEFAULT_ARCH)
        if arch:
            architecture.update(deepcopy(arch))
        motion_encoder = ACTORStyleEncoder(motion_nfeats, vae, **architecture)
        text_encoder = ACTORStyleEncoder(text_nfeats, vae, **architecture)
        motion_decoder = ACTORStyleDecoder(motion_nfeats, **architecture)
        self.tmr = TMRCore(
            motion_encoder,
            text_encoder,
            motion_decoder,
            vae=vae,
            sample_mean=sample_mean,
            fact=fact,
        )
        self._save_ckpt_modules.append("tmr")

    @torch.inference_mode()
    def encode_text(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self.tmr.encode(inputs, modality="text", sample_mean=sample_mean)

    @torch.inference_mode()
    def encode_motion(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self.tmr.encode(inputs, modality="motion", sample_mean=sample_mean)


__all__ = ["TMRBundle"]
