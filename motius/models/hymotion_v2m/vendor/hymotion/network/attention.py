from __future__ import annotations
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import flash_attn  # pyright: ignore
    from flash_attn.bert_padding import pad_input, unpad_input  # pyright: ignore
    from flash_attn.flash_attn_interface import flash_attn_varlen_func  # pyright: ignore
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    unpad_input = None
    pad_input = None
    _flash_attn_forward = None


MEMORY_LAYOUT = {
    "flash": (lambda x: x, lambda x: x),
    "torch": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
    "vanilla": (lambda x: x.transpose(1, 2), lambda x: x.transpose(1, 2)),
}


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mode: str = "flash",
    drop_rate: float = 0.0,
    attn_mask: Optional[Tensor] = None,
    causal: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    batch_size: int = 1,
    training: bool = True,
    return_attn: bool = False,
    gate: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Perform QKV self attention.

    Args:
        q (Tensor): Query tensor with shape [b, s, h, d], where h is the number of heads.
        k (Tensor): Key tensor with shape [b, s1, h, d]
        v (Tensor): Value tensor with shape [b, s1, h, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, h, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        Tensor: Output tensor after self attention with shape [b, s, hd]
    """
    if return_attn and mode == "flash":
        # 如果需要返回 attention map 用于可视化，强制回退到 torch 模式
        mode = "torch"

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["vanilla" if return_attn else mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)
    orig_dtype = q.dtype

    if gate is not None:
        assert mode in ["torch", "vanilla"], "gate is only supported for 'torch' or 'vanilla' attention mode"
        gate = pre_attn_layout(gate)

    if not return_attn:
        if mode == "torch":
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(q.dtype)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=drop_rate if training else 0.0,
                is_causal=causal,
            )

        elif mode == "flash":
            assert flash_attn_varlen_func is not None, "flash_attn is not installed or not supported"
            batch_size, seqlen_q, num_heads, head_dim = q.shape

            if attn_mask is None:
                # 如果没有 Mask，假设全是有效的
                q_unpad = q.flatten(0, 1)
                k_unpad = k.flatten(0, 1)
                v_unpad = v.flatten(0, 1)
                cu_seqlens_q = torch.arange(
                    0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device
                )
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = seqlen_q
                max_seqlen_kv = seqlen_q
                indices_q = None
            else:
                # NOTE: unpad_input 期望的 attention_mask 是 (batch, seqlen)，且 1/True 为保留，0/False 为 Padding
                assert unpad_input is not None, "flash_attn is not installed or not supported"
                q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, attn_mask)
                k_unpad, _, cu_seqlens_kv, max_seqlen_kv, _ = unpad_input(k, attn_mask)
                v_unpad, _, _, _, _ = unpad_input(v, attn_mask)

            # Call Flash Attention VarLen
            x_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=drop_rate if training else 0.0,
                softmax_scale=None,
                causal=False,  #  Full Attention
                window_size=(-1, -1),  #  No Window
            )

            # Pad Output
            if indices_q is None:
                x = x_unpad.view(batch_size, seqlen_q, num_heads, head_dim)
            else:
                assert pad_input is not None, "flash_attn is not installed or not supported"
                x = pad_input(x_unpad, indices_q, batch_size, seqlen_q)

        elif mode == "vanilla":
            use_fp32 = orig_dtype in (torch.float16, torch.bfloat16)
            q_attn = q.float() if use_fp32 else q
            k_attn = k.float() if use_fp32 else k
            v_attn = v.float() if use_fp32 else v
            scale_factor = 1.0 / math.sqrt(q.size(-1))
            b, a, s_q, _ = q.shape
            s_k = k.size(2)
            attn_bias = torch.zeros(b, a, s_q, s_k, dtype=q_attn.dtype, device=q_attn.device)
            if causal:
                # Only applied to self attention
                assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
                temp_mask = torch.ones(b, a, s_q, s_q, dtype=torch.bool, device=q.device).tril(diagonal=0)
                attn_bias.masked_fill_(~temp_mask, float("-inf"))
                attn_bias = attn_bias.to(q_attn.dtype)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(~attn_mask, float("-inf"))
                else:
                    attn_bias = attn_bias + (attn_mask.float() if use_fp32 else attn_mask.to(q_attn.dtype))

            attn = (q_attn @ k_attn.transpose(-2, -1)) * scale_factor
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = torch.dropout(attn, p=drop_rate, train=training)
            x = attn @ v_attn
            if use_fp32:
                x = x.to(dtype=orig_dtype)
        else:
            raise NotImplementedError(f"Unsupported attention mode: {mode}")

        if gate is not None:
            x = x * torch.sigmoid(gate)

        x = post_attn_layout(x)
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out
    else:
        use_fp32 = orig_dtype in (torch.float16, torch.bfloat16)
        scale = 1.0 / math.sqrt(q.size(-1))
        b, h, s_q, d = q.shape
        s_k = k.size(2)
        attn_bias = torch.zeros(b, h, s_q, s_k, dtype=(torch.float32 if use_fp32 else q.dtype), device=q.device)
        if causal:
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, h, s_q, s_q, dtype=torch.bool, device=q.device).tril(0)
            attn_bias.masked_fill_(~temp_mask, float("-inf"))
            if use_fp32:
                attn_bias = attn_bias.float()
            else:
                attn_bias = attn_bias.to(q.dtype)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_bias = attn_bias + (attn_mask.float() if use_fp32 else attn_mask.to(q.dtype))

        q_attn = q.float() if use_fp32 else q
        k_attn = k.float() if use_fp32 else k
        v_attn = v.float() if use_fp32 else v
        attn = (q_attn @ k_attn.transpose(-2, -1)) * scale
        attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=training)
        x = attn @ v_attn
        if use_fp32:
            x = x.to(dtype=orig_dtype)

        if gate is not None:
            x = x * torch.sigmoid(gate)

        x = post_attn_layout(x)
        b, s, h, d = x.shape
        out = x.reshape(b, s, -1)
        return out, attn
