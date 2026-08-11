"""Transformer and VQ modules used by the TM2D release."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def length_mask(lengths: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Return a broadcastable ``(B, 1, T)`` validity mask."""

    positions = torch.arange(sequence_length, device=lengths.device)
    return (positions[None] < lengths[:, None]).unsqueeze(1)


def token_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (tokens != pad_id).unsqueeze(1)


def causal_mask(tokens: torch.Tensor) -> torch.Tensor:
    length = tokens.shape[1]
    return torch.ones((1, length, length), dtype=torch.bool, device=tokens.device).tril()


class PositionalEncoding(nn.Module):
    def __init__(self, width: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32)[:, None]
        frequencies = torch.arange(0, width, 2, dtype=torch.float32)
        frequencies = 10000 ** (-frequencies / width)
        encoding = torch.zeros(max_length, width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positional_encoding", encoding)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.positional_encoding[: values.shape[1]]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, dropout: float):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attention = query @ key.transpose(2, 3) / self.temperature
        if mask is not None:
            attention = attention.masked_fill(~mask, -1e9)
        attention = self.dropout(F.softmax(attention, dim=-1))
        return attention @ value, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        batch, query_length = query.shape[:2]
        key_length, value_length = key.shape[1], value.shape[1]
        residual = query
        query = self.w_qs(query).view(batch, query_length, self.n_head, self.d_k)
        key = self.w_ks(key).view(batch, key_length, self.n_head, self.d_k)
        value = self.w_vs(value).view(batch, value_length, self.n_head, self.d_v)
        query, key, value = (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, attention = self.attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        output = self.layer_norm(self.dropout(self.fc(output)) + residual)
        return output, attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        residual = values
        values = self.dropout(self.w_2(F.relu(self.w_1(values))))
        return self.layer_norm(values + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, values, mask):
        values, attention = self.slf_attn(values, values, values, mask)
        return self.pos_ffn(values), attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, values, encoded, self_mask, source_mask):
        values, self_attention = self.slf_attn(values, values, values, self_mask)
        values, cross_attention = self.enc_attn(values, encoded, encoded, source_mask)
        return self.pos_ffn(values), self_attention, cross_attention


class VectorEncoder(nn.Module):
    def __init__(self, input_width, config):
        super().__init__()
        width = config["d_model"]
        self.position_enc = PositionalEncoding(width, config["max_source_length"])
        self.emb = nn.Linear(input_width, width, bias=False)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    width,
                    config["d_inner"],
                    config["n_head"],
                    config["d_k"],
                    config["d_v"],
                    config["dropout"],
                )
                for _ in range(config["n_encoder_layers"])
            ]
        )
        self.d_model = width

    def forward(self, values, mask):
        values = self.position_enc(self.emb(values) * self.d_model**0.5)
        for layer in self.layer_stack:
            values, _ = layer(values, mask)
        return values


class TokenEncoder(nn.Module):
    def __init__(self, vocabulary_size, pad_id, config):
        super().__init__()
        width = config["d_model"]
        self.position_enc = PositionalEncoding(width, config["max_source_length"])
        self.src_word_emb = nn.Embedding(vocabulary_size, width, padding_idx=pad_id)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    width,
                    config["d_inner"],
                    config["n_head"],
                    config["d_k"],
                    config["d_v"],
                    config["dropout"],
                )
                for _ in range(config["n_encoder_layers"])
            ]
        )
        self.d_model = width

    def forward(self, tokens, mask):
        values = self.position_enc(self.src_word_emb(tokens) * self.d_model**0.5)
        for layer in self.layer_stack:
            values, _ = layer(values, mask)
        return values


class TokenDecoder(nn.Module):
    def __init__(self, vocabulary_size, pad_id, config):
        super().__init__()
        width = config["d_model"]
        self.trg_word_emb = nn.Embedding(vocabulary_size, width, padding_idx=pad_id)
        self.position_enc = PositionalEncoding(width, config["max_target_length"])
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    width,
                    config["d_inner"],
                    config["n_head"],
                    config["d_k"],
                    config["d_v"],
                    config["dropout"],
                )
                for _ in range(config["n_decoder_layers"])
            ]
        )
        self.d_model = width

    def forward(self, tokens, self_mask, encoded, source_mask):
        values = self.position_enc(self.trg_word_emb(tokens) * self.d_model**0.5)
        for layer in self.layer_stack:
            values, _, _ = layer(values, encoded, self_mask, source_mask)
        return values


class _MotionTokenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trg_pad_idx = config["motion_pad_id"]
        self.decoder = TokenDecoder(config["motion_vocabulary_size"], self.trg_pad_idx, config)
        self.trg_word_prj = nn.Linear(
            config["d_model"], config["motion_vocabulary_size"], bias=False
        )
        self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

    def decode_logits(self, tokens, encoded, source_mask):
        mask = token_mask(tokens, self.trg_pad_idx) & causal_mask(tokens)
        values = self.decoder(tokens, mask, encoded, source_mask)
        return self.trg_word_prj(values)

    def generate_from_encoded(
        self,
        encoded,
        source_mask,
        initial_tokens,
        steps,
        *,
        sample=False,
        top_k=None,
        forbidden_ids=(),
        generator=None,
    ):
        tokens = initial_tokens
        for _ in range(int(steps)):
            logits = self.decode_logits(tokens, encoded, source_mask)[:, -1]
            if top_k is not None and 0 < int(top_k) < logits.shape[-1]:
                threshold = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            if forbidden_ids:
                logits[:, list(forbidden_ids)] = -torch.inf
            if sample:
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, 1, generator=generator
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


class AudioMotionTransformer(_MotionTokenTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VectorEncoder(config["audio_feature_dim"], config)

    def encode(self, features, lengths):
        mask = length_mask(lengths, features.shape[1])
        return self.encoder(features, mask), mask


class TextMotionTransformer(_MotionTokenTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.src_pad_idx = config["text_pad_id"]
        self.encoder = TokenEncoder(
            config["text_vocabulary_size"], self.src_pad_idx, config
        )

    def encode(self, tokens):
        mask = token_mask(tokens, self.src_pad_idx)
        return self.encoder(tokens, mask), mask


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(width, width, 3, 1, 1),
        )

    def forward(self, values):
        return values + self.model(values)


class VQEncoder(nn.Module):
    """TM2D encoder: 283 motion channels to one token per eight frames."""

    def __init__(self, input_width=283, latent_width=1024, n_down=3):
        super().__init__()
        layers = []
        for index in range(n_down):
            source = input_width if index == 0 else latent_width
            layers.extend(
                [
                    nn.Conv1d(source, latent_width, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResBlock(latent_width),
                ]
            )
        self.main = nn.Sequential(*layers)

    def forward(self, motion):
        return self.main(motion.transpose(1, 2)).transpose(1, 2)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_width=1024):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_width)

    def indices(self, latent):
        flat = latent.reshape(-1, latent.shape[-1])
        distance = (
            flat.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * flat @ self.embedding.weight.t()
        )
        return distance.argmin(dim=1).reshape(latent.shape[:-1])

    def lookup(self, indices):
        return self.embedding(indices)


class VQDecoder(nn.Module):
    def __init__(self, latent_width=1024, output_width=287, n_resblocks=3, n_up=3):
        super().__init__()
        channels = [latent_width] * n_up + [output_width]
        layers = [ResBlock(latent_width) for _ in range(n_resblocks)]
        for index in range(n_up):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(channels[index], channels[index + 1], 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv1d(output_width, output_width, 3, 1, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, latent):
        return self.main(latent.transpose(1, 2)).transpose(1, 2)


__all__ = [
    "AudioMotionTransformer",
    "TextMotionTransformer",
    "VQDecoder",
    "VQEncoder",
    "VectorQuantizer",
]
