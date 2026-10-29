# SPDX-License-Identifier: Apache-2.0
"""Zero text encoder for unconditional KIMODO batch inference."""

import torch


class DummyTextEncoder:
    """Return zero text features without loading the heavy LLM2Vec model."""

    def __init__(self, llm_dim: int = 4096):
        self.llm_dim = llm_dim
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        feats = torch.zeros(
            len(texts),
            1,
            self.llm_dim,
            device=self.device,
            dtype=self.dtype,
        )
        return feats, [0 for _ in texts]
