"""Original text truncation, batching and sorting outside MotionLCM/MLD networks."""

from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch


@lru_cache(maxsize=8)
def _tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


@torch.no_grad()
def encode_text(network, captions):
    captions = list(captions)
    tokenizer = _tokenizer(network.modelpath)
    if network.name == "t5":
        config = json.loads(
            (Path(network.modelpath) / "sentence_bert_config.json").read_text()
        )
        order = np.argsort([-len(text) for text in captions])
        sentences = [str(captions[index]).strip() for index in order]
        if config.get("do_lower_case", False):
            sentences = [text.lower() for text in sentences]
        features = tokenizer(
            sentences,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=config["max_seq_length"],
        )
        embeddings = network(features)
        return embeddings[np.argsort(order)].clone()
    if network.name in {"clip", "clip_hidden"}:
        features = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        features["input_ids"] = features["input_ids"][:, : tokenizer.model_max_length]
    else:
        features = tokenizer(captions, return_tensors="pt", padding=True)
    return network(features)
