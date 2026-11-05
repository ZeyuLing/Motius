from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(
                os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path
            )
            peft_model_name_or_path = os.path.join(
                os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path
            )

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @classmethod
    def from_network(cls, network, directory=None, metadata=None):
        from pathlib import Path
        import json
        from transformers import AutoTokenizer

        if directory is not None:
            directory = Path(directory)
            metadata = json.loads((directory / "architecture.json").read_text())
            tokenizer_source = directory / "tokenizer"
        else:
            tokenizer_source = metadata["tokenizer_source"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        frontend = cls.__new__(cls)
        frontend.llm_dim = network.config.hidden_size
        frontend.model = LLM2Vec(
            model=network, tokenizer=tokenizer, **metadata["frontend_options"]
        )
        return frontend.eval()

    def to(self, device: torch.device):
        pinned_device = os.environ.get("KIMODO_TEXT_ENCODER_DEVICE")
        if pinned_device:
            device = torch.device(pinned_device)
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        encode_device = os.environ.get("KIMODO_TEXT_ENCODER_DEVICE")
        with torch.no_grad():
            encoded_text = self.model.encode(
                text,
                batch_size=1,
                show_progress_bar=False,
                device=encode_device or str(self.get_device()),
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths
