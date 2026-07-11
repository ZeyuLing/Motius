"""LLaMA autoregressive motion transformer for MotionMillion / "Go to Zero".

Self-contained (torch only). Faithful port of
``models/lit_llama/model_hf.py`` (the released Go-to-Zero T2M LLaMA), with the
``args``-derived config fields (``clip_dim``, ``tie_weights``) promoted to explicit
:class:`LLaMAHFConfig` fields and a configurable ``max_sample_steps`` exposed on
:meth:`LLaMAHF.sample`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# (n_layer, n_head, n_embd) presets, identical to the official ``llama_configs``.
_LLAMA_CONFIGS = {
    "44M": dict(n_layer=8, n_head=8, n_embd=512),
    "111M": dict(n_layer=12, n_head=12, n_embd=768),
    "343M": dict(n_layer=24, n_head=16, n_embd=1024),
    "775M": dict(n_layer=36, n_head=20, n_embd=1280),
    "1B": dict(n_layer=48, n_head=24, n_embd=1536),
    "3B": dict(n_layer=24, n_head=32, n_embd=3200),
    "5B": dict(n_layer=24, n_head=32, n_embd=4096),
    "6B": dict(n_layer=28, n_head=32, n_embd=4096),
    "7B": dict(n_layer=36, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


@dataclass
class LLaMAHFConfig:
    block_size: int = 301
    vocab_size: int = 65538  # nb_code + 2 (PAD/EOS)
    n_layer: int = 36
    n_head: int = 32
    n_embd: int = 4096
    clip_dim: int = 2048  # flan-t5-xl hidden size
    tie_weights: bool = False

    @classmethod
    def from_name(cls, name: str, **overrides) -> "LLaMAHFConfig":
        cfg = dict(_LLAMA_CONFIGS[name])
        cfg.update(overrides)
        return cls(**cfg)


class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len, n_elem, dtype, device, base: int = 10000):
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    idx_theta = torch.outer(seq_idx, theta)
    casting = [torch.float16, torch.bfloat16, torch.int8]
    working_dtype = torch.float32 if dtype in casting else dtype
    complex_dtype = torch.complex32 if dtype in casting else torch.complex64
    return torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor, start: int = 0) -> torch.Tensor:
    x = x.transpose(1, 2)
    T = x.size(1)
    rope_cache = rope_cache[start : start + T]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)


class LengthCausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rope_cache = None

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size, n_elem=self.n_embd // self.n_head,
                dtype=x.dtype, device=x.device,
            )
        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        attn_mask = torch.tril(attn_mask)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
        text_mask = y_mask.unsqueeze(2) * y_mask.unsqueeze(1)
        text_mask = F.pad(
            text_mask, (0, T - y_mask.shape[1], 0, T - y_mask.shape[1]), mode="constant", value=0
        )
        attn_mask = torch.logical_or(attn_mask, text_mask)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask.unsqueeze(1), dropout_p=0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def forward_cached(self, x: torch.Tensor, cache: dict, input_pos: int) -> torch.Tensor:
        """Incremental attention with a per-layer KV cache.

        Used only for AR decoding. ``x`` is the new chunk (text prefill or a single
        motion token); ``input_pos`` is its absolute start position (for RoPE).
        The attention is fully visible over the cached prefix — equivalent to the
        ``forward`` mask (text bidirectional + motion causal), since each new motion
        token only ever attends to already-generated prefix positions.
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size, n_elem=self.n_embd // self.n_head,
                dtype=x.dtype, device=x.device,
            )
        q = apply_rope(q, self.rope_cache, start=input_pos)
        k = apply_rope(k, self.rope_cache, start=input_pos)
        if cache.get("k") is not None:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
        cache["k"] = k
        cache["v"] = v
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        n_hidden = ((n_hidden - 1) // N) * N + N
        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.c_fc1(x)) * self.c_fc2(x))


class Block(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = LengthCausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), y_mask)
        x = x + self.mlp(self.rms_2(x))
        return x

    def forward_cached(self, x: torch.Tensor, cache: dict, input_pos: int) -> torch.Tensor:
        x = x + self.attn.forward_cached(self.rms_1(x), cache, input_pos)
        x = x + self.mlp(self.rms_2(x))
        return x


class LLaMAHF(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.llama_proj = nn.Linear(config.clip_dim, config.n_embd)
        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

    @torch.no_grad()
    def sample(self, clip_feature, y_mask, if_categorial=False, max_sample_steps: int = 50):
        """Greedy (or categorical) AR decoding. Returns token indices ``(1, L)``.

        Matches the released sampler: at most ``max_sample_steps`` motion tokens are
        produced, with early-stop on the EOS token (``vocab_size - 2``). The released
        default is 50.
        """
        xs = None
        for k in range(max_sample_steps + 1):
            x = [] if k == 0 else xs
            logits = self.forward_sample(x, clip_feature, y_mask)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.config.vocab_size - 2:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.config.vocab_size - 2:
                    break
            xs = idx if k == 0 else torch.cat((xs, idx), dim=1)
            if k == max_sample_steps:
                return xs[:, :-1]
        if xs is None:
            return torch.ones(1, 1, device=clip_feature.device).long()
        return xs

    @torch.no_grad()
    def sample_cached(self, clip_feature, y_mask, if_categorial=False, max_sample_steps: int = 150):
        """KV-cached equivalent of :meth:`sample` (O(n) instead of O(n^2)).

        Produces the same greedy/categorical token stream as :meth:`sample` but
        reuses cached keys/values: a text *prefill* fills the cache, then each motion
        token is decoded with a single-token forward. Batch size must be 1.
        """
        text_len = int(y_mask[0].sum())
        caches = [{"k": None, "v": None} for _ in range(self.config.n_layer)]

        # --- prefill: encode the (bidirectional) text prefix into the cache ---
        x = self.llama_proj(clip_feature)[:, :text_len, :]
        for block, c in zip(self.transformer.h, caches):
            x = block.forward_cached(x, c, input_pos=0)
        logits = self.lm_head(self.transformer.ln_f(x)[:, -1, :])

        xs = None
        for k in range(max_sample_steps + 1):
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                idx = Categorical(probs).sample()
                if idx == self.config.vocab_size - 2:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.config.vocab_size - 2:
                    break
            xs = idx if k == 0 else torch.cat((xs, idx), dim=1)
            if k == max_sample_steps:
                return xs[:, :-1]
            # --- decode: feed the just-produced token (position text_len + k) ---
            x = self.transformer.wte(idx)
            for block, c in zip(self.transformer.h, caches):
                x = block.forward_cached(x, c, input_pos=text_len + k)
            logits = self.lm_head(self.transformer.ln_f(x)[:, -1, :])
        if xs is None:
            return torch.ones(1, 1, device=clip_feature.device).long()
        return xs

    def forward_sample(self, idx, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:
        if len(idx) == 0:
            x = self.llama_proj(clip_feature)[:, : int(y_mask[0].sum()), :]
        else:
            _, t = idx.size()
            assert t <= self.config.block_size, (
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )
            x = self.transformer.wte(idx)
            x = torch.cat((self.llama_proj(clip_feature)[:, : int(y_mask[0].sum()), :], x), dim=1)
        for block in self.transformer.h:
            x = block(x, y_mask)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def forward(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:
        text_length = clip_feature.shape[1]
        if len(idx) == 0:
            x = self.llama_proj(clip_feature)[:, : int(y_mask[0].sum()), :]
        else:
            _, t = idx.size()
            assert t <= self.config.block_size
            x = self.transformer.wte(idx)
            expanded_mask = y_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            result = torch.where(expanded_mask == 1, self.llama_proj(clip_feature), x[:, :text_length, :])
            result = torch.cat((result, x[:, text_length:, :]), dim=1)
            x = result
        for block in self.transformer.h:
            x = block(x, y_mask)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_name(cls, name: str, **overrides) -> "LLaMAHF":
        return cls(LLaMAHFConfig.from_name(name, **overrides))
