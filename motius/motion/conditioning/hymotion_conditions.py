"""HY-Motion/MotionCanvas padding, condition dropout and source-token policy outside neural modules."""

from __future__ import annotations
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor


def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert length list to boolean mask. (B,) -> (B, max_len)."""
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    return (
        torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        < lengths
    )


def mask_text_cond(
    bundle,
    vtxt: Tensor,
    ctxt: Tensor,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
    return_text_available: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Apply classifier-free guidance masking to text conditions.

    Args:
        vtxt: Sentence-level text embeddings, shape (B, 1, D_v).
        ctxt: Token-level text embeddings, shape (B, L_c, D_c).
        force_mask: If True, return null embeddings for all samples.
        cond_mask_prob: Probability of masking text (CFG dropout rate).
        return_text_available: If True, also return boolean mask indicating
            which samples have real text (not masked). Shape (B,).

    Returns:
        - If return_text_available=False: (vtxt_masked, ctxt_masked)
        - If return_text_available=True: (vtxt_masked, ctxt_masked, text_available)
          where text_available[b]=True means sample b has real text,
          text_available[b]=False means sample b was masked to null.
    """
    bs = vtxt.shape[0]
    text_available = torch.ones(bs, dtype=torch.bool, device=vtxt.device)
    if force_mask:
        text_available.fill_(False)
        result = (
            bundle.null_vtxt_feat.expand(*vtxt.shape),
            bundle.null_ctxt_input.expand(*ctxt.shape),
        )
        if return_text_available:
            return result + (text_available,)
        return result
    if bundle.training and cond_mask_prob > 0.0:
        mask = (
            torch.bernoulli(torch.ones(bs, device=vtxt.device) * cond_mask_prob)
            .view(bs, 1)
            .bool()
        )
        text_available = ~mask.squeeze(-1)
        mask_vtxt = mask
        while mask_vtxt.ndim < vtxt.ndim:
            mask_vtxt = mask_vtxt.unsqueeze(-1)
        vtxt = torch.where(mask_vtxt, bundle.null_vtxt_feat.expand_as(vtxt), vtxt)
        mask_ctxt = mask
        while mask_ctxt.ndim < ctxt.ndim:
            mask_ctxt = mask_ctxt.unsqueeze(-1)
        ctxt = torch.where(mask_ctxt, bundle.null_ctxt_input.expand_as(ctxt), ctxt)
    result = (vtxt, ctxt)
    if return_text_available:
        return result + (text_available,)
    return result


def prepare_padding(
    src_motion: Tensor,
    tgt_motion: Optional[Tensor],
    tgt_length: List[int],
    src_mask: Optional[Tensor] = None,
    src_length: Optional[List[int]] = None,
    ref_pose: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, List[int], List[int], Tensor]:
    """Pad src/tgt motions to the same length and build tgt_padding_mask.

    Returns:
        (src_motion, src_mask, tgt_motion, src_length, tgt_length,
         tgt_padding_mask)
    """
    device = src_motion.device
    B, L_s, D = src_motion.shape
    L_t = tgt_motion.shape[1] if tgt_motion is not None else L_s
    L_r = ref_pose.shape[1] if ref_pose is not None else 0
    if src_length is None:
        src_length = tgt_length
    max_len = max(L_s, L_t)
    if src_mask is None:
        src_mask = torch.ones_like(src_motion)
    if L_s < max_len:
        pad = max_len - L_s
        src_motion = F.pad(src_motion, (0, 0, 0, pad))
        src_mask = F.pad(src_mask, (0, 0, 0, pad))
    if tgt_motion is not None and L_t < max_len:
        pad = max_len - L_t
        tgt_motion = F.pad(tgt_motion, (0, 0, 0, pad))
    elif tgt_motion is None:
        tgt_motion = torch.zeros(B, max_len, D, dtype=src_motion.dtype, device=device)
    if L_r > 0:
        ref_mask = torch.ones(B, L_r, dtype=torch.bool, device=device)
    else:
        ref_mask = torch.empty(B, 0, dtype=torch.bool, device=device)
    tgt_mask = _length_to_mask(
        torch.tensor(tgt_length, dtype=torch.long, device=device), max_len
    )
    tgt_padding_mask = torch.cat([ref_mask, tgt_mask], dim=1)
    return (src_motion, src_mask, tgt_motion, src_length, tgt_length, tgt_padding_mask)


def prepare_condition_context(
    src_motion: Tensor,
    ref_pose: Optional[Tensor] = None,
    src_mask: Optional[Tensor] = None,
) -> Tensor:
    """Build the fixed motion-conditioning context.

    The transformer always receives ``[x_t, edit_context, target_mask]``.
    Hard motion observations are already imputed into ``x_t`` and are
    re-imputed after every inference step, so they are not duplicated in
    this context. ``edit_context`` contains the source motion only where
    the model must generate (``target_mask == 1``); it is therefore zero
    for completion tasks and populated for motion editing.

    Returns a tensor of shape ``(B, L, 2 * D)``. Concatenating it with
    ``x_t`` gives the transformer's fixed ``3 * D`` motion input.
    """
    B, L_src, D = src_motion.shape
    if src_mask is None:
        src_mask = torch.ones_like(src_motion)
    edit_context = src_motion * src_mask
    if ref_pose is not None:
        _, L_ref, _ = ref_pose.shape
        src_mask = torch.cat(
            [
                torch.zeros(B, L_ref, D, dtype=src_mask.dtype, device=src_mask.device),
                src_mask,
            ],
            dim=1,
        )
        edit_context = torch.cat([ref_pose, edit_context], dim=1)
    return torch.cat([edit_context, src_mask], dim=-1)


def maybe_inject_source_token(
    bundle,
    vtxt_input: Tensor,
    ctxt_input: Tensor,
    ctxt_mask_temporal: Tensor,
    sources: Optional[List[str]],
    trigger_sources: Optional[Set[str]] = None,
    prob: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Inject the official HYMotion special source token for selected sources."""
    if (
        sources is None or trigger_sources is None
    ) or not bundle.enable_special_game_feat:
        return (vtxt_input, ctxt_input, ctxt_mask_temporal)
    B, Lc, Dc = ctxt_input.shape
    if not isinstance(sources, (list, tuple)) or len(sources) != B:
        raise ValueError(f"sources length must equal batch size: {len(sources)} vs {B}")
    trig = {str(s).lower() for s in trigger_sources}
    src_mask = torch.tensor(
        [str(s).lower() in trig for s in sources],
        dtype=torch.bool,
        device=ctxt_input.device,
    )
    if not src_mask.any():
        return (vtxt_input, ctxt_input, ctxt_mask_temporal)
    if bundle.training:
        rand_mask = torch.rand(B, device=ctxt_input.device) < prob
    else:
        rand_mask = torch.ones(B, dtype=torch.bool, device=ctxt_input.device)
    apply_mask = src_mask & rand_mask
    if not apply_mask.any():
        return (vtxt_input, ctxt_input, ctxt_mask_temporal)
    vtxt_token = bundle.special_game_vtxt_feat.to(vtxt_input).expand(B, 1, -1)
    vtxt_input = vtxt_input + vtxt_token * apply_mask.view(B, 1, 1).to(vtxt_input.dtype)
    if ctxt_mask_temporal.dtype == torch.bool:
        cur_len = ctxt_mask_temporal.sum(dim=1).long()
    else:
        cur_len = (ctxt_mask_temporal > 0).sum(dim=1).long()
    can_inplace = apply_mask & (cur_len < Lc)
    b_inplace = torch.nonzero(can_inplace, as_tuple=False).squeeze(1)
    if b_inplace.numel() > 0:
        pos = cur_len[b_inplace]
        token = bundle.special_game_ctxt_feat.squeeze(0).squeeze(0).to(ctxt_input)
        ctxt_input = ctxt_input.clone()
        ctxt_mask_temporal = ctxt_mask_temporal.clone()
        ctxt_input[b_inplace, pos, :] = token.unsqueeze(0).expand(b_inplace.numel(), Dc)
        if ctxt_mask_temporal.dtype == torch.bool:
            ctxt_mask_temporal[b_inplace, pos] = True
        else:
            ctxt_mask_temporal[b_inplace, pos] = 1
    need_expand = (apply_mask & (cur_len >= Lc)).any()
    if need_expand:
        suffix = torch.zeros(
            (B, 1, Dc), dtype=ctxt_input.dtype, device=ctxt_input.device
        )
        full_hit = apply_mask & (cur_len >= Lc)
        b_full = torch.nonzero(full_hit, as_tuple=False).squeeze(1)
        if b_full.numel() > 0:
            suffix[b_full, 0, :] = (
                bundle.special_game_ctxt_feat.expand(b_full.numel(), 1, -1)
                .to(ctxt_input)
                .squeeze(1)
            )
        ctxt_input = torch.cat([ctxt_input, suffix], dim=1)
        if ctxt_mask_temporal.dtype == torch.bool:
            suffix_mask = torch.zeros(
                (B, 1), dtype=torch.bool, device=ctxt_input.device
            )
            suffix_mask[b_full, 0] = True
        else:
            suffix_mask = torch.zeros(
                (B, 1), dtype=ctxt_mask_temporal.dtype, device=ctxt_input.device
            )
            suffix_mask[b_full, 0] = 1
        ctxt_mask_temporal = torch.cat([ctxt_mask_temporal, suffix_mask], dim=1)
    return (vtxt_input, ctxt_input, ctxt_mask_temporal)


def predict_conditioned_flow(
    bundle,
    *,
    x_input,
    ctxt_input,
    vtxt_input,
    timesteps,
    x_mask_temporal=None,
    ctxt_mask_temporal=None,
    sources=None,
    trigger_sources=None,
    special_game_prob=0.5,
    **network_options,
):
    if ctxt_mask_temporal is not None:
        vtxt_input, ctxt_input, ctxt_mask_temporal = maybe_inject_source_token(
            bundle,
            vtxt_input=vtxt_input,
            ctxt_input=ctxt_input,
            ctxt_mask_temporal=ctxt_mask_temporal,
            sources=sources,
            trigger_sources=trigger_sources,
            prob=special_game_prob,
        )
    return bundle.predict_flow(
        x_input=x_input,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
        **network_options,
    )
