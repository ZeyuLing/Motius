"""MotionStreamer matching neural architectures."""
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn

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

    def forward(self, encoded):
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
        mask = torch.arange(features.shape[1], device=features.device).expand(
            batch_size, features.shape[1]
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
