"""Exact attention visibility without sequence-square masks or score buffers.

Queries are evaluated in bounded batches. Local motion/video keys still share
one softmax; the token refiner retains every global key and its padding rule.
No weights, RoPE positions, sequence windows or precision settings are changed.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class QueryAttentionMask:
    chunk_size: int
    valid_motion: torch.Tensor
    valid_condition: torch.Tensor | None = None
    # -1 = all keys, -2 = no keys; nonnegative values are radii in frames.
    radii: tuple[int, int, int, int] = (-1, -1, -2, -1)
    causal_motion: bool = False
    all_keys_valid: bool = False

    @classmethod
    def from_network(cls, network, motion_valid, condition_valid):
        mode = network.mask_mode
        mm, mc, cm, cc = -1, -1, -2, -1
        if mode in ('narrowband', 'narrowband_v2m'):
            radius = int(round(network.narrowband_length))
            mm = radius if radius >= 0 else -2
        if mode == 'narrowband_v2m':
            mc = int(round(network.narrowband_v2m_length)) if network.narrowband_v2m_length > 0 else -1
            tm = network.narrowband_tm_length
            cm = -2 if tm == 0 else int(round(tm)) if tm > 0 else -1
            tt = network.narrowband_tt_length
            tt = network.narrowband_v2m_length if tt is None else tt
            cc = int(round(tt)) if tt > 0 else -1
        elif mode not in (None, 'causal', 'narrowband'):
            raise ValueError(f'Unsupported attention mode: {mode}')
        all_valid = bool((motion_valid.all() & condition_valid.all()).item())
        return cls(network.attention_query_chunk_size, motion_valid, condition_valid,
                   (mm, mc, cm, cc), mode == 'causal', all_valid)


def query_batched_attention(q, k, v, policy):
    """Input/output layouts match the existing FlowHMR attention function."""
    if policy.chunk_size < 1:
        raise ValueError('attention query chunk size must be positive')
    Q, K, V = (tensor.transpose(1, 2) for tensor in (q, k, v))
    output = torch.empty_like(Q)
    device = q.device

    def sdpa(query, keys, values, allowed=None):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(query, keys, values,
                       attn_mask=None if allowed is None else allowed[:, None],
                       dropout_p=0.0, is_causal=False)

    if policy.valid_condition is None:
        # Refiner attention stays GLOBAL. Padded queries see their own diagonal,
        # matching IndividualTokenRefiner rather than the main-block fallback.
        valid = policy.valid_motion
        length = q.shape[1]
        keys = torch.arange(length, device=device)
        for start in range(0, length, policy.chunk_size):
            end = min(start + policy.chunk_size, length)
            allowed = valid[:, start:end, None] & valid[:, None, :]
            diagonal = torch.arange(start, end, device=device)[:, None] == keys[None]
            allowed = torch.where(allowed.any(-1, keepdim=True), allowed, diagonal[None])
            output[:, :, start:end] = sdpa(Q[:, :, start:end], K, V, allowed)
    else:
        lengths = (policy.valid_motion.shape[1], policy.valid_condition.shape[1])
        offsets = (0, lengths[0])
        valid = torch.cat((policy.valid_motion, policy.valid_condition), dim=1)
        for stream, (length, offset) in enumerate(zip(lengths, offsets)):
            radii = policy.radii[2 * stream:2 * stream + 2]
            for start in range(0, length, policy.chunk_size):
                end = min(start + policy.chunk_size, length)
                ids, times, kinds = [], [], []
                for kind, (radius, total, base) in enumerate(zip(radii, lengths, offsets)):
                    if radius == -2:
                        continue
                    lo = 0 if radius == -1 else min(total, max(0, start - radius))
                    hi = total if radius == -1 else max(lo, min(total, end + radius))
                    positions = torch.arange(lo, hi, device=device)
                    ids.append(positions + base)
                    times.append(positions)
                    kinds.append(torch.full_like(positions, kind))
                query = Q[:, :, offset + start:offset + end]
                if not ids or sum(index.numel() for index in ids) == 0:
                    output[:, :, offset + start:offset + end] = sdpa(query, K, V)
                    continue
                indices, key_times, key_kinds = map(torch.cat, (ids, times, kinds))
                query_times = torch.arange(start, end, device=device)[:, None]
                distance = (query_times - key_times[None]).abs()
                visible = torch.zeros_like(distance, dtype=torch.bool)
                for kind, radius in enumerate(radii):
                    if radius != -2:
                        visible |= (key_kinds[None] == kind) & (
                            True if radius == -1 else distance <= radius)
                if stream == 0 and policy.causal_motion:
                    visible &= (key_kinds[None] != 0) | (key_times[None] <= query_times)
                allowed = visible[None] & valid[:, None, indices]
                bad = ~allowed.any(-1)
                value = sdpa(query, K.index_select(2, indices), V.index_select(2, indices), allowed)
                self_visible = radii[stream] != -2
                if not (policy.all_keys_valid and self_visible) and bad.any():
                    # Main attention unmasks ALL original keys for an otherwise
                    # fully masked row, including keys outside the local union.
                    fallback = sdpa(query, K, V)
                    value = torch.where(bad[:, None, :, None], fallback, value)
                output[:, :, offset + start:offset + end] = value
    return output.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)
