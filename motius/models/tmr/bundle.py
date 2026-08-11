"""Motius-native TMR architecture used by the released evaluator artifacts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def encode_with_distribution(
        self,
        inputs: Dict[str, Tensor],
        modality: str,
        sample_mean: Optional[bool] = None,
    ):
        """Return a latent together with its Gaussian parameters."""
        encoder = self.text_encoder if modality == "text" else self.motion_encoder
        encoded = encoder(inputs)
        if not self.vae:
            return encoded[:, 0], None
        mean, log_variance = encoded.unbind(1)
        use_mean = self.sample_mean if sample_mean is None else bool(sample_mean)
        latent = mean
        if not use_mean:
            latent = mean + self.fact * torch.randn_like(mean) * torch.exp(
                0.5 * log_variance
            )
        return latent, (mean, log_variance)

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
        lmd: Optional[Dict[str, float]] = None,
        temperature: float = 0.1,
        threshold_selfsim: float = 0.8,
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
        self._trainable_modules.append("tmr")
        self._module_checkpoint_formats["tmr"] = "full"
        self.lmd = deepcopy(lmd) if lmd is not None else {
            "recons": 1.0,
            "latent": 1.0e-5,
            "kl": 1.0e-5,
            "contrastive": 0.1,
        }
        self.temperature = float(temperature)
        self.threshold_selfsim = float(threshold_selfsim)

    @staticmethod
    def _unwrap_tmr(module: nn.Module) -> TMRCore:
        while hasattr(module, "module") and isinstance(module.module, nn.Module):
            module = module.module
        return module

    @staticmethod
    def _kl_loss(q, p) -> Tensor:
        mu_q, logvar_q = q
        mu_p, logvar_p = p
        return 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1
        ).mean()

    def _contrastive_loss(
        self,
        text_latents: Tensor,
        motion_latents: Tensor,
        sentence_embeddings: Optional[Tensor],
    ) -> Tensor:
        text_latents = F.normalize(text_latents, dim=-1)
        motion_latents = F.normalize(motion_latents, dim=-1)
        logits = text_latents @ motion_latents.T / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        if sentence_embeddings is not None and self.threshold_selfsim:
            similarities = sentence_embeddings @ sentence_embeddings.T
            similarities = similarities - torch.diag_embed(similarities.diag())
            filtered = similarities > (2 * self.threshold_selfsim - 1)
            logits = logits.masked_fill(filtered, -torch.inf)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Compute the released TMR reconstruction and retrieval objective."""
        core = self._unwrap_tmr(self.tmr)
        text_inputs = batch["text_x_dict"]
        motion_inputs = batch["motion_x_dict"]
        mask = motion_inputs["mask"]
        reference = motion_inputs["x"]

        text_latents, text_dist = core.encode_with_distribution(
            text_inputs, modality="text"
        )
        motion_latents, motion_dist = core.encode_with_distribution(
            motion_inputs, modality="motion"
        )
        text_reconstruction = core.decode(text_latents, mask)
        motion_reconstruction = core.decode(motion_latents, mask)

        losses: Dict[str, Tensor] = {}
        losses["recons_text"] = F.smooth_l1_loss(
            text_reconstruction, reference
        )
        losses["recons_motion"] = F.smooth_l1_loss(
            motion_reconstruction, reference
        )
        losses["recons"] = losses["recons_text"] + losses["recons_motion"]
        if core.vae:
            zero_dist = (
                torch.zeros_like(motion_dist[0]),
                torch.zeros_like(motion_dist[1]),
            )
            losses["kl"] = (
                self._kl_loss(text_dist, motion_dist)
                + self._kl_loss(motion_dist, text_dist)
                + self._kl_loss(motion_dist, zero_dist)
                + self._kl_loss(text_dist, zero_dist)
            )
        losses["latent"] = F.smooth_l1_loss(text_latents, motion_latents)
        losses["contrastive"] = self._contrastive_loss(
            text_latents,
            motion_latents,
            batch.get("sent_emb"),
        )
        losses["loss"] = sum(
            self.lmd[name] * value
            for name, value in losses.items()
            if name in self.lmd
        )
        return losses

    @torch.inference_mode()
    def encode_text(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self._unwrap_tmr(self.tmr).encode(
            inputs, modality="text", sample_mean=sample_mean
        )

    @torch.inference_mode()
    def encode_motion(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self._unwrap_tmr(self.tmr).encode(
            inputs, modality="motion", sample_mean=sample_mean
        )


__all__ = ["TMRBundle"]
