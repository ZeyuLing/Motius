"""Prepare single-caption or temporal multi-caption neural conditions."""
from __future__ import annotations
import torch
from .clip_text import encode_clip_text


def prepare_text_conditions(network, conditions):
    result = dict(conditions)
    multi = "all_texts" in result or not isinstance(result["text"][0], str)
    result["multi_text"] = multi
    if "text_embeddings" in result:
        return result
    context = 22 if network.dataset == "humanml" else 77
    if multi:
        rows = result["all_texts"] if "all_texts" in result else result["text"]
        size = max(len(row) for row in rows)
        features = [encode_clip_text(network, list(row) + [""] * (size - len(row)),
                                     context_length=context) for row in rows]
        result["text_embeddings"] = torch.stack(features, dim=1)
    else:
        result["text_embeddings"] = encode_clip_text(network, result["text"], context_length=context)
    return result
