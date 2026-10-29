"""TMR encoders, decoder and shared reconstruction forward graph."""
from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor, nn
from einops import repeat

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
        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=len(features))
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
        standard_deviation = log_variance.exp().pow(0.5)
        noise = standard_deviation.data.new(standard_deviation.size()).normal_()
        return mean + self.fact * noise * standard_deviation

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
            standard_deviation = log_variance.exp().pow(0.5)
            noise = standard_deviation.data.new(standard_deviation.size()).normal_()
            latent = mean + self.fact * noise * standard_deviation
        return latent, (mean, log_variance)

    def decode(self, latent: Tensor, mask: Tensor) -> Tensor:
        return self.motion_decoder(latent, mask)

    def forward(self, text_inputs, motion_inputs):
        text_latents, text_dist = self.encode_with_distribution(text_inputs, modality="text")
        text_reconstruction = self.decode(text_latents, motion_inputs["mask"])
        motion_latents, motion_dist = self.encode_with_distribution(motion_inputs, modality="motion")
        motion_reconstruction = self.decode(motion_latents, motion_inputs["mask"])
        return dict(text_latents=text_latents, text_dist=text_dist,
                    motion_latents=motion_latents, motion_dist=motion_dist,
                    text_reconstruction=text_reconstruction,
                    motion_reconstruction=motion_reconstruction)
