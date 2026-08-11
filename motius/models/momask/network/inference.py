"""MoMask masked iterative decoding (inference-only, self-contained).

These helpers replicate the released MoMask generation path
(``gen_t2m.py`` / ``eval_t2m_trans_res.py``):

1. ``MaskTransformer.generate`` — confidence-based **masked iterative decoding**
   of the base (q=0) token map, with classifier-free guidance (``cond_scale``)
   over ``time_steps`` refinement iterations and a cosine mask schedule.
2. ``ResidualTransformer.generate`` — autoregressively fills quantizers
   ``q=1..Q-1`` conditioned on the lower layers.
3. ``RVQVAE.forward_decoder`` — de-quantizes the ``(b, T, Q)`` token grid and
   decodes back to the **normalized** 263-dim HumanML3D feature.

The returned motion is still in the RVQ-VAE *normalized* space; callers
(``MoMaskBundle.denormalize`` / ``MoMaskPipeline``) un-standardize with the
training ``Mean`` / ``Std``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


@torch.no_grad()
def estimate_token_lengths(
    t2m_transformer,
    length_estimator,
    captions: Sequence[str],
) -> torch.Tensor:
    """Sample per-caption token lengths from the length estimator.

    Mirrors ``gen_t2m.py``'s ``est_length`` branch: encode text with CLIP,
    predict a categorical distribution over length bins, sample one bin per
    caption. Returns a ``(B,)`` LongTensor of **token** counts (frames = 4x).
    """
    text_embedding = t2m_transformer.encode_text(list(captions))
    pred_dis = length_estimator(text_embedding)
    probs = F.softmax(pred_dis, dim=-1)
    token_lens = Categorical(probs).sample()
    return token_lens.long()


@torch.no_grad()
def generate_motion(
    t2m_transformer,
    res_transformer,
    vq_model,
    captions: Sequence[str],
    token_lens: torch.Tensor,
    *,
    cond_scale: float = 4.0,
    time_steps: int = 10,
    temperature: float = 1.0,
    topk_filter_thres: float = 0.9,
    gsample: bool = False,
    res_cond_scale: float = 5.0,
    res_temperature: float = 1.0,
) -> torch.Tensor:
    """Run the full MoMask T2M decoding stack.

    Args:
        t2m_transformer: the masked generative transformer (base tokens).
        res_transformer: the residual transformer (quantizers 1..Q-1).
        vq_model: the RVQ-VAE (de-quantize + decode).
        captions: list of B text prompts.
        token_lens: ``(B,)`` LongTensor of token counts (= frames // 4).
        cond_scale / time_steps / temperature / topk_filter_thres / gsample:
            base masked-decoding params (parity defaults cond_scale=4,
            time_steps=10, topkr=0.9, temperature=1.0).
        res_cond_scale / res_temperature: residual-transformer params
            (parity defaults cond_scale=5, temperature=1.0).

    Returns:
        ``(B, T, 263)`` normalized motion (T = max(token_lens) * 4).
    """
    captions = list(captions)
    mids = t2m_transformer.generate(
        captions,
        token_lens,
        timesteps=time_steps,
        cond_scale=cond_scale,
        temperature=temperature,
        topk_filter_thres=topk_filter_thres,
        gsample=gsample,
    )
    mids = res_transformer.generate(
        mids,
        captions,
        token_lens,
        temperature=res_temperature,
        cond_scale=res_cond_scale,
    )
    pred_motions = vq_model.forward_decoder(mids)  # (B, T, 263), normalized
    return pred_motions
