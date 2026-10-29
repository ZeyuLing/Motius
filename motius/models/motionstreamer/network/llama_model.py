import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self
from typing import Optional
import torch.nn.functional as F


@dataclass
class LLaMAHFConfig:
    block_size: int = 78
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    T5_xxl_dim: int = 768

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {"Normal_size": dict(n_layer=12, n_head=12, n_embd=768)}


class LLaMAHF(nn.Module):

    def __init__(
        self,
        config: LLaMAHFConfig,
        num_diffusion_head_layers=9,
        input_token_dim=16,
        device=torch.device("cuda"),
        width=1792,
        use_out_proj=True,
    ) -> None:
        super().__init__()
        assert config.block_size is not None
        self.config = config
        cond_dim = config.T5_xxl_dim
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_token_dim, config.n_embd),
                cond_embed=nn.Linear(cond_dim, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        target_channels = input_token_dim
        from .diffusion_head import DiffusionHead

        self.diff_loss = DiffusionHead(
            target_channels=target_channels,
            z_channels=config.n_embd,
            width=width,
            depth=num_diffusion_head_layers,
            num_sampling_steps="50",
            grad_checkpointing=False,
        )
        self.diff_loss = self.diff_loss.to(device)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.use_out_proj = bool(use_out_proj)

    def forward(self, idx: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if len(idx) == 0:
            token_embeddings = self.transformer.cond_embed(feature).unsqueeze(0)
        else:
            b, t, c = idx.size()
            idx = idx.float()
            assert (
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            token_embeddings = self.transformer.wte(idx)
            text_embeddings = self.transformer.cond_embed(feature).unsqueeze(1)
            token_embeddings = torch.cat([text_embeddings, token_embeddings], dim=1)
        x = token_embeddings
        for i, block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.out_proj(x)
        return logits

    def forward_babel_eval(self, x, return_attention=False) -> torch.Tensor:
        layer_attentions = []
        for block in self.transformer.h:
            if return_attention:
                x, att = block(x, return_attention=True)
                layer_attentions.append(att)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        if self.use_out_proj:
            logits = self.out_proj(x)
        else:
            logits = x
        if return_attention:
            return (logits, layer_attentions)
        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAHFConfig.from_name(name))


class Block(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, last_past=None, use_cache=False, return_attention=False
    ) -> torch.Tensor:
        if use_cache:
            if return_attention:
                a, attn = self.attn.forward_attn(self.rms_1(x), last_past, use_cache)
            else:
                a, present = self.attn(self.rms_1(x), last_past, use_cache)
            x = x + a
        else:
            if return_attention:
                a, attn = self.attn.forward_attn(self.rms_1(x))
            else:
                a = self.attn(self.rms_1(x))
            x = x + a
        x = x + self.mlp(self.rms_2(x))
        if use_cache:
            if return_attention:
                return (x, present, attn)
            else:
                return (x, present)
        elif return_attention:
            return (x, attn)
        else:
            return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

        def scaling_factor(sequence_threshold):
            return np.log2(sequence_threshold**2 - sequence_threshold)

        scale_init = scaling_factor(self.block_size)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, x: torch.Tensor, last_past=None, use_cache=False) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        if use_cache:
            if last_past is not None:
                past_key, past_value = last_past
                k = torch.cat([past_key, k], dim=-2)
                v = torch.cat([past_value, v], dim=-2)
        if use_cache:
            present = (k, v)
        else:
            present = None
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head,
                dtype=x.dtype,
                device=x.device,
            )
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale.item(),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if use_cache:
            return (y, present)
        return y

    def forward_attn(
        self, x: torch.Tensor, last_past=None, use_cache=False
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        if use_cache:
            if last_past is not None:
                past_key, past_value = last_past
                k = torch.cat([past_key, k], dim=-2)
                v = torch.cat([past_value, v], dim=-2)
        if use_cache:
            present = (k, v)
        else:
            present = None
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head,
                dtype=x.dtype,
                device=x.device,
            )
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return (y, att)


class MLP(nn.Module):

    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        n_hidden = (n_hidden - 1) // N * N + N
        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-05) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    theta = 1.0 / base ** (
        torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem
    )
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    idx_theta = torch.outer(seq_idx, theta)
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = torch.float32 if dtype in dtypes_requiring_casting else dtype
    complex_dtype = (
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)
    T = x.size(1)
    rope_cache = rope_cache[:T]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
