"""SDPA-free Wan attention processor for PRISM.

Some Taiji A100 images ship a diffusers/PyTorch combination where
``F.scaled_dot_product_attention`` raises ``CUDA driver error: invalid
argument`` for PRISM's motion-token shapes.  This processor keeps the public
Wan processor API, but computes attention with chunked matmul + softmax so it
does not depend on any SDPA backend.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

try:
    from diffusers.models.transformers.transformer_wan import (
        WanAttnProcessor,
        _get_added_kv_projections,
        _get_qkv_projections,
    )

    _WAN_PROCESSOR_API = "wan"
except ImportError:
    from diffusers.models.transformers.transformer_wan import (
        WanAttnProcessor2_0 as WanAttnProcessor,
    )

    _get_added_kv_projections = None
    _get_qkv_projections = None
    _WAN_PROCESSOR_API = "attention"


def _normalise_attention_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    query_len: int,
    start: int,
    end: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]

    if mask.shape[-2] == query_len:
        mask = mask[..., start:end, :]
    return mask


def _manual_attention_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention for tensors shaped ``[B, H, S, D]`` without SDPA."""

    original_dtype = query.dtype
    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()

    query_len = query_fp32.shape[-2]
    dim = query_fp32.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    key_t = key_fp32.transpose(-2, -1)
    chunk = int(os.environ.get("PRISM_MANUAL_ATTN_CHUNK", "128"))
    chunk = max(1, chunk)

    outputs = []
    for start in range(0, query_len, chunk):
        end = min(start + chunk, query_len)
        scores = torch.matmul(query_fp32[:, :, start:end, :], key_t) * scale
        mask = _normalise_attention_mask(
            attention_mask,
            query_len=query_len,
            start=start,
            end=end,
            dtype=scores.dtype,
            device=scores.device,
        )
        if mask is not None:
            scores = scores + mask
        max_scores = scores.amax(dim=-1, keepdim=True)
        scores = scores - torch.where(
            torch.isfinite(max_scores),
            max_scores,
            torch.zeros_like(max_scores),
        )
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        outputs.append(torch.matmul(probs, value_fp32))

    return torch.cat(outputs, dim=-2).to(original_dtype)


def _manual_attention_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention for tensors shaped ``[B, S, H, D]`` without SDPA."""

    out = _manual_attention_bhsd(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention_mask,
    )
    return out.transpose(1, 2)


class WanAttnProcessorFP32Upcast(WanAttnProcessor):
    """Wan attention processor that avoids SDPA for both old and new APIs."""

    _use_fp32_upcast = True

    def __init__(self, use_fp32_upcast: bool = True):
        super().__init__()
        self._use_fp32_upcast = use_fp32_upcast

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if _WAN_PROCESSOR_API == "attention":
            return self._call_old_attention_api(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
            )
        return self._call_wan_api(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
        )

    def _call_old_attention_api(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        rotary_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query.float()).to(query.dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key.float()).to(key.dtype)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if states.device.type == "mps" else torch.float64
                rotated = torch.view_as_complex(states.to(dtype).unflatten(3, (-1, 2)))
                out = torch.view_as_real(rotated * freqs).flatten(3, 4)
                return out.type_as(states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            if attn.norm_added_k is not None:
                key_img = attn.norm_added_k(key_img.float()).to(key_img.dtype)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = _manual_attention_bhsd(
                query,
                key_img,
                value_img,
                attention_mask=None,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = _manual_attention_bhsd(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _call_wan_api(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(
            attn,
            hidden_states,
            encoder_hidden_states,
        )

        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(
                attn,
                encoder_hidden_states_img,
            )
            key_img = attn.norm_added_k(key_img.float()).to(key_img.dtype)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = _manual_attention_bshd(
                query,
                key_img,
                value_img,
                attention_mask=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = _manual_attention_bshd(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @classmethod
    def enable_fp32_upcast(cls, enable: bool = True) -> None:
        cls._use_fp32_upcast = enable

    @classmethod
    def get_fp32_upcast_enabled(cls) -> bool:
        return cls._use_fp32_upcast
