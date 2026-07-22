"""MotionStreamer-272 text-motion evaluator.

This is a plain-PyTorch port of the ACTOR/TEMOS encoders bundled with the
MotionStreamer evaluator. Parameter names intentionally match the public
checkpoint converted by ``tools/export_evaluator_hf.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        values = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        values[:, 0::2] = torch.sin(positions * divisor)
        values[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("pe", values.unsqueeze(1))

    def forward(self, values: Tensor) -> Tensor:
        if self.batch_first:
            values = values + self.pe[: values.shape[1]].permute(1, 0, 2)
        else:
            values = values + self.pe[: values.shape[0]]
        return self.dropout(values)


class DistilbertActorAgnosticEncoder(nn.Module):
    def __init__(
        self,
        model_config,
        tokenizer,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.tokenizer = tokenizer
        self.text_model = AutoModel.from_config(model_config)
        self.projection = nn.Sequential(
            nn.ReLU(), nn.Linear(model_config.hidden_size, latent_dim)
        )
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()
        return self

    def forward(self, texts: Sequence[str]):
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        encoded = encoded.to(self.mu_token.device)
        hidden = self.text_model(**encoded).last_hidden_state
        values = self.projection(hidden).permute(1, 0, 2)
        batch_size = values.shape[1]
        mu = self.mu_token.expand(batch_size, -1)
        logvar = self.logvar_token.expand(batch_size, -1)
        values = torch.cat((mu[None], logvar[None], values), dim=0)
        token_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=values.device)
        mask = torch.cat((token_mask, encoded.attention_mask.bool()), dim=1)
        final = self.seqTransEncoder(
            self.sequence_pos_encoding(values), src_key_padding_mask=~mask
        )
        return torch.distributions.Normal(final[0], final[1].exp().sqrt())


class ActorAgnosticEncoder(nn.Module):
    def __init__(
        self,
        nfeats: int = 272,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_len: int = 300,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.skel_embedding = nn.Linear(nfeats, latent_dim)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, features: Tensor, lengths: Tensor):
        batch_size = features.shape[0]
        mask = torch.arange(self.max_len, device=features.device).expand(
            batch_size, self.max_len
        ) < lengths.unsqueeze(1)
        values = self.skel_embedding(features).permute(1, 0, 2)
        mu = self.mu_token.expand(batch_size, -1)
        logvar = self.logvar_token.expand(batch_size, -1)
        values = torch.cat((mu[None], logvar[None], values), dim=0)
        token_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=features.device)
        final = self.seqTransEncoder(
            self.sequence_pos_encoding(values),
            src_key_padding_mask=~torch.cat((token_mask, mask), dim=1),
        )
        return torch.distributions.Normal(final[0], final[1].exp().sqrt())


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


@MODEL_BUNDLES.register_module()
class MotionStreamer272Evaluator(ModelBundle):
    """Self-contained MotionStreamer evaluator for raw MotionStreamer-272 data."""

    def __init__(self, artifact_dir: str, *, device: str = "cuda") -> None:
        super().__init__()
        artifact = Path(artifact_dir).expanduser().resolve()
        config = json.loads((artifact / "config.json").read_text())
        preprocessor = json.loads((artifact / "preprocessor_config.json").read_text())
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer

        tokenizer_dir = artifact / preprocessor.get("tokenizer", "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        model_config = AutoConfig.from_pretrained(tokenizer_dir, local_files_only=True)
        latent_dim = int(config.get("latent_dim", 256))
        num_layers = int(config.get("num_layers", 4))
        num_heads = int(config.get("num_heads", 4))
        self.text_encoder = DistilbertActorAgnosticEncoder(
            model_config,
            tokenizer,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.motion_encoder = ActorAgnosticEncoder(
            nfeats=int(config.get("motion_nfeats", 272)),
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=int(preprocessor.get("max_motion_length", 300)),
        )
        state = load_file(artifact / config.get("weights_file", "model.safetensors"))
        self.text_encoder.load_state_dict(
            {
                key.removeprefix("text_encoder."): value
                for key, value in state.items()
                if key.startswith("text_encoder.")
            },
            strict=True,
        )
        self.motion_encoder.load_state_dict(
            {
                key.removeprefix("motion_encoder."): value
                for key, value in state.items()
                if key.startswith("motion_encoder.")
            },
            strict=True,
        )
        mean_path = artifact / preprocessor.get("mean", "stats/mean.npy")
        std_path = artifact / preprocessor.get("std", "stats/std.npy")
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)).float())
        self.register_buffer("std", torch.from_numpy(np.load(std_path)).float())
        self.max_motion_length = int(preprocessor.get("max_motion_length", 300))
        self.unit_length = int(preprocessor.get("unit_length", 4))
        self.artifact_dir = artifact
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(resolved_device).eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(str(_download_artifact(pretrained_model_name_or_path)), **kwargs)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), batch_size):
            rows.append(self.text_encoder(texts[start : start + batch_size]).loc.cpu())
        return torch.cat(rows).numpy() if rows else np.empty((0, 256), np.float32)

    @torch.no_grad()
    def encode_motions(
        self,
        motions: Sequence[np.ndarray],
        *,
        lengths: Sequence[int] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        if lengths is None:
            lengths = [len(motion) for motion in motions]
        rows = np.zeros(
            (len(motions), self.max_motion_length, len(self.mean)), dtype=np.float32
        )
        clipped_lengths = []
        mean = self.mean.cpu().numpy()
        std = self.std.cpu().numpy()
        for index, (motion, requested_length) in enumerate(zip(motions, lengths)):
            value = np.asarray(motion, dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != len(mean):
                raise ValueError(f"Expected MotionStreamer-272, got {value.shape}.")
            length = min(len(value), int(requested_length), self.max_motion_length)
            length = length // self.unit_length * self.unit_length
            if length < self.unit_length:
                raise ValueError("Motion is too short for the MotionStreamer evaluator.")
            rows[index, :length] = (value[:length] - mean) / np.maximum(std, 1e-8)
            clipped_lengths.append(length)
        embeddings = []
        for start in range(0, len(rows), batch_size):
            batch = torch.from_numpy(rows[start : start + batch_size]).to(self.device)
            batch_lengths = torch.tensor(
                clipped_lengths[start : start + batch_size],
                device=self.device,
                dtype=torch.long,
            )
            embeddings.append(self.motion_encoder(batch, batch_lengths).loc.cpu())
        return torch.cat(embeddings).numpy() if embeddings else np.empty((0, 256), np.float32)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use encode_texts() or encode_motions().")


__all__ = [
    "ActorAgnosticEncoder",
    "DistilbertActorAgnosticEncoder",
    "MotionStreamer272Evaluator",
    "PositionalEncoding",
]
