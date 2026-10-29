"""OpenAI CLIP token preparation, separate from neural model ownership."""
from __future__ import annotations

from typing import Sequence
import torch


def tokenize_clip(
    captions: Sequence[str] | torch.Tensor,
    *,
    device: str | torch.device | None = None,
    context_length: int = 77,
) -> torch.Tensor:
    """Preserve source truncation, EOT and optional short-context zero padding.

    HumanML/Kit MDM variants truncate to 20 CLIP tokens plus SOT/EOT, then pad to
    the neural tower's 77 positions. Other CLIP methods truncate at 77.
    Prepared token tensors can pass through a data pipeline unchanged.
    """
    if isinstance(captions, torch.Tensor):
        return captions.to(device=device) if device is not None else captions
    import clip

    tokens = clip.tokenize(list(captions), context_length=context_length, truncate=True)
    if context_length < 77:
        tokens = torch.cat(
            [tokens, tokens.new_zeros((tokens.shape[0], 77 - context_length))], dim=1
        )
    return tokens.to(device=device) if device is not None else tokens


def encode_clip_text(network, captions, *, context_length: int = 77):
    """Prepare raw captions for a tensor-only neural ``encode_text`` method."""
    return network.encode_text(tokenize_clip(captions, context_length=context_length))
