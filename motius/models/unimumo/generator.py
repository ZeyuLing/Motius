"""Inference-only dual-stream autoregressive model for UniMuMo."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

def sinusoidal_embedding(length: int, dimension: int, *, device: torch.device, dtype: torch.dtype, max_period: float=10000.0) -> torch.Tensor:
    if dimension % 2:
        raise ValueError('Sinusoidal embedding dimension must be even')
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.arange(dimension // 2, device=device, dtype=dtype)[None]
    denominator = torch.as_tensor(max_period, device=device, dtype=dtype) ** (frequencies / (dimension // 2 - 1))
    phases = positions / denominator
    return torch.cat((phases.cos(), phases.sin()), dim=-1)

class MultiheadAttention(nn.Module):
    """Batch-first attention with additive masks and checkpoint-stable names."""

    def __init__(self, dimension: int, num_heads: int, *, dropout: float=0.0, bias: bool=False):
        super().__init__()
        if dimension % num_heads:
            raise ValueError('dimension must be divisible by num_heads')
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dimension = self.dimension // self.num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(dimension, dimension, bias=bias)
        self.k_proj = nn.Linear(dimension, dimension, bias=bias)
        self.v_proj = nn.Linear(dimension, dimension, bias=bias)
        self.out_proj = nn.Linear(dimension, dimension, bias=bias)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_dimension).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        if query is key and key is value:
            bias=None if self.q_proj.bias is None else torch.cat((self.q_proj.bias,self.k_proj.bias,self.v_proj.bias))
            projected=F.linear(query,torch.cat((self.q_proj.weight,self.k_proj.weight,self.v_proj.weight)),bias)
            batch,length,_=projected.shape
            query,key,value=projected.reshape(batch,length,3,self.num_heads,self.head_dimension).permute(0,3,2,1,4).unbind(2)
        else:
            query = self._split(self.q_proj(query))
            key = self._split(self.k_proj(key))
            value = self._split(self.v_proj(value))
        scores = torch.matmul(query, key.transpose(-1, -2)) * (1 / math.sqrt(self.head_dimension))
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[None, None]
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None]
            elif attention_mask.ndim != 4:
                raise ValueError('attention_mask must have 2, 3, or 4 dimensions')
            scores = scores + attention_mask.to(scores)
        probabilities = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(query.shape[0], query.shape[2], self.dimension)
        return self.out_proj(output)

class DualStreamTransformerLayer(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        dimension = int(config['dimension'])
        hidden = int(config['hidden_dimension'])
        heads = int(config['num_heads'])
        dropout = float(config.get('dropout', 0.0))
        attention_dropout = float(config.get('attention_dropout', 0.0))
        bias_attention = bool(config.get('bias_attention', False))
        bias_ffn = bool(config.get('bias_ffn', False))
        self.norm_first = bool(config.get('norm_first', True))
        self.dropout = dropout
        self.activation = nn.GELU()
        self.self_attention = MultiheadAttention(dimension, heads, dropout=attention_dropout, bias=bias_attention)
        self.caption_attention = MultiheadAttention(dimension, heads, dropout=attention_dropout, bias=bias_attention)
        self.cross_attention = MultiheadAttention(dimension, heads, dropout=attention_dropout, bias=bias_attention)
        self.music_ffn_in = nn.Linear(dimension, hidden, bias=bias_ffn)
        self.music_ffn_out = nn.Linear(hidden, dimension, bias=bias_ffn)
        self.motion_ffn_in = nn.Linear(dimension, hidden, bias=bias_ffn)
        self.motion_ffn_out = nn.Linear(hidden, dimension, bias=bias_ffn)
        self.self_norm = nn.LayerNorm(dimension, eps=1e-05)
        self.motion_self_norm = nn.LayerNorm(dimension, eps=1e-05)
        self.music_ffn_norm = nn.LayerNorm(dimension, eps=1e-05)
        self.motion_ffn_norm = nn.LayerNorm(dimension, eps=1e-05)
        self.cross_norm = nn.LayerNorm(dimension, eps=1e-05)

    def _attention_block(self, module: MultiheadAttention, value: torch.Tensor, *, key_value: torch.Tensor | None=None, mask: torch.Tensor | None=None) -> torch.Tensor:
        key_value = value if key_value is None else key_value
        output = module(value, key_value, key_value, attention_mask=mask)
        return F.dropout(output, p=self.dropout, training=self.training)

    def _ffn(self, value: torch.Tensor, input_projection: nn.Linear, output_projection: nn.Linear) -> torch.Tensor:
        value = self.activation(input_projection(value))
        value = F.dropout(value, p=self.dropout, training=self.training)
        value = output_projection(value)
        return F.dropout(value, p=self.dropout, training=self.training)

    def _cross_mask(self, condition_mask: torch.Tensor, stream_length: int, dtype: torch.dtype) -> torch.Tensor:
        if condition_mask.ndim != 3 or condition_mask.shape[1] != 2:
            raise ValueError('condition_mask must have shape (B,2,L)')
        valid = torch.cat((condition_mask[:, :1].expand(-1, stream_length, -1), condition_mask[:, 1:].expand(-1, stream_length, -1)), dim=1)
        return torch.where(valid, torch.zeros((), dtype=dtype, device=valid.device), torch.full((), float('-inf'), dtype=dtype, device=valid.device))

    def forward(self, value: torch.Tensor, *, self_attention_mask: torch.Tensor, condition: torch.Tensor | None, condition_mask: torch.Tensor | None, caption_mode: bool=False) -> torch.Tensor:
        stream_length = value.shape[1] // 2
        attention = self.caption_attention if caption_mode else self.self_attention
        if self.norm_first:
            value = value + self._attention_block(attention, self.self_norm(value), mask=self_attention_mask)
            if condition is not None:
                if condition_mask is None:
                    raise ValueError('condition_mask is required with condition')
                value = value + self._attention_block(self.cross_attention, self.cross_norm(value), key_value=condition, mask=self._cross_mask(condition_mask, stream_length, value.dtype))
            music, motion = (value[:, :stream_length], value[:, stream_length:])
            music = music + self._ffn(self.music_ffn_norm(music), self.music_ffn_in, self.music_ffn_out)
            motion = motion + self._ffn(self.motion_ffn_norm(motion), self.motion_ffn_in, self.motion_ffn_out)
            return torch.cat((music, motion), dim=1)
        value = self.self_norm(value + self._attention_block(attention, value, mask=self_attention_mask))
        if condition is not None:
            value = self.cross_norm(value + self._attention_block(self.cross_attention, value, key_value=condition, mask=self._cross_mask(condition_mask, stream_length, value.dtype)))
        music, motion = (value[:, :stream_length], value[:, stream_length:])
        music = self.music_ffn_norm(music + self._ffn(music, self.music_ffn_in, self.music_ffn_out))
        motion = self.motion_ffn_norm(motion + self._ffn(motion, self.motion_ffn_in, self.motion_ffn_out))
        return torch.cat((music, motion), dim=1)

class UniMuMoGenerator(nn.Module):
    """MusicGen-compatible transformer with parallel music and motion streams."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.num_codebooks = int(config.get('num_codebooks', 4))
        self.codebook_size = int(config.get('codebook_size', 2048))
        self.dimension = int(config.get('dimension', 1024))
        vocabulary_size = self.codebook_size + 1
        self.music_embeddings = nn.ModuleList([nn.Embedding(vocabulary_size, self.dimension) for _ in range(self.num_codebooks)])
        self.motion_embeddings = nn.ModuleList([nn.Embedding(vocabulary_size, self.dimension) for _ in range(self.num_codebooks)])
        self.layers = nn.ModuleList([DualStreamTransformerLayer(config) for _ in range(int(config.get('num_layers', 24)))])
        self.output_norm = nn.LayerNorm(self.dimension, eps=1e-05) if bool(config.get('norm_first', True)) else nn.Identity()
        output_bias = bool(config.get('output_bias', False))
        self.music_heads = nn.ModuleList([nn.Linear(self.dimension, self.codebook_size, bias=output_bias) for _ in range(self.num_codebooks)])
        self.motion_heads = nn.ModuleList([nn.Linear(self.dimension, self.codebook_size, bias=output_bias) for _ in range(self.num_codebooks)])

    @property
    def special_token_id(self) -> int:
        return self.codebook_size

    def self_attention_mask(self, stream_length: int, *, device: torch.device, dtype: torch.dtype, caption_mode: bool=False) -> torch.Tensor:
        if caption_mode:
            valid = torch.zeros((stream_length * 2, stream_length * 2), dtype=torch.bool, device=device)
            valid[:stream_length, :stream_length] = True
            valid[stream_length:, stream_length:] = True
        else:
            causal = torch.ones((stream_length, stream_length), dtype=torch.bool, device=device).tril()
            valid = torch.cat((torch.cat((causal, causal), dim=1),) * 2, dim=0)
        return torch.where(valid, torch.zeros((), dtype=dtype, device=device), torch.full((), float('-inf'), dtype=dtype, device=device))

    def forward_features(self, music_sequence: torch.Tensor, motion_sequence: torch.Tensor, *, condition: torch.Tensor | None, condition_mask: torch.Tensor | None, caption_mode: bool=False) -> torch.Tensor:
        if music_sequence.shape != motion_sequence.shape:
            raise ValueError('music and motion sequences must share shape')
        if music_sequence.ndim != 3:
            raise ValueError('token sequences must have shape (B,K,S)')
        if music_sequence.shape[1] != self.num_codebooks:
            raise ValueError('token sequence has the wrong number of codebooks')
        stream_length = music_sequence.shape[-1]
        music = sum((embedding(music_sequence[:, index]) for index, embedding in enumerate(self.music_embeddings)))
        motion = sum((embedding(motion_sequence[:, index]) for index, embedding in enumerate(self.motion_embeddings)))
        position = sinusoidal_embedding(stream_length, self.dimension, device=music.device, dtype=music.dtype, max_period=float(self.config.get('max_period', 10000.0)))[None]
        position_scale = float(self.config.get('position_scale', 1.0))
        value = torch.cat((music + position_scale * position, motion + position_scale * position), dim=1)
        self_mask = self.self_attention_mask(stream_length, device=value.device, dtype=value.dtype, caption_mode=caption_mode)
        for layer in self.layers:
            value = layer(value, self_attention_mask=self_mask, condition=condition, condition_mask=condition_mask, caption_mode=caption_mode)
        return self.output_norm(value)

    def forward(self, music_sequence: torch.Tensor, motion_sequence: torch.Tensor, *, condition: torch.Tensor | None, condition_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        stream_length = music_sequence.shape[-1]
        value = self.forward_features(music_sequence, motion_sequence, condition=condition, condition_mask=condition_mask)
        music = value[:, :stream_length]
        motion = value[:, stream_length:]
        music_logits = torch.stack([head(music) for head in self.music_heads], dim=1)
        motion_logits = torch.stack([head(motion) for head in self.motion_heads], dim=1)
        return (music_logits, motion_logits)
__all__ = ['DualStreamTransformerLayer', 'MultiheadAttention', 'UniMuMoGenerator', 'sinusoidal_embedding']
