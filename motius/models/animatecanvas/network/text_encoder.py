import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

from .text_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION


def get_module_device(module):
    return next(module.parameters()).device

LLM_ENCODER_LAYOUT = {
    "qwen3_embedding": {
        "module_path": "checkpoints/Qwen3-Embedding-8B",
        "template": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}\n{{}}",
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
    "qwen3": {
        "module_path": "checkpoints/Qwen3-8B",
        "template": [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ],
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModelForCausalLM,
    },
    "t5": {
        "module_path": "checkpoints/t5-v1_1-xxl",
        "template": "{}",
        "crop_start": 0,
        "tokenizer_class": T5Tokenizer,
        "text_encoder_class": T5EncoderModel,
    },
    "distilbert": {
        "module_path": "checkpoints/distilbert-base-uncased",
        "template": "{}",
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
}

SENTENCE_EMB_LAYOUT = {
    "clipl": {
        "module_path": "checkpoints/clip-vit-large-patch14",
        "tokenizer_class": CLIPTokenizer,
        "text_encoder_class": CLIPTextModel,
    },
    "sentence_transformer": {
        "module_path": "checkpoints/all-mpnet-base-v2",
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
}


class HYTextModel(nn.Module):
    def __init__(
        self,
        llm_type: str = "qwen3",
        max_length_llm: int = 512,
        sentence_emb_type: str = "clipl",
        max_length_sentence_emb: int = 77,
        enable_llm_padding: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        llm_model_path: Optional[str] = None,
        llm_tokenizer_path: Optional[str] = None,
        sentence_emb_model_path: Optional[str] = None,
        sentence_emb_tokenizer_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.text_encoder_type = "hy_text_model"
        assert llm_type in LLM_ENCODER_LAYOUT, f"Unsupported LLM type: {llm_type}"
        assert sentence_emb_type in SENTENCE_EMB_LAYOUT, f"Unsupported sentence embedding type: {sentence_emb_type}"

        self.sentence_emb_type = sentence_emb_type
        self.max_length_sentence_emb = max_length_sentence_emb
        sentence_layout = SENTENCE_EMB_LAYOUT[sentence_emb_type]
        sentence_emb_model_path = sentence_emb_model_path or sentence_layout["module_path"]
        sentence_emb_tokenizer_path = (
            sentence_emb_tokenizer_path
            or sentence_emb_model_path
            or sentence_layout["module_path"]
        )
        self.sentence_emb_tokenizer = sentence_layout["tokenizer_class"].from_pretrained(
            sentence_emb_tokenizer_path,
            max_length=max_length_sentence_emb,
        )
        self.sentence_emb_text_encoder = sentence_layout["text_encoder_class"].from_pretrained(
            sentence_emb_model_path,
            torch_dtype=torch_dtype,
        )
        self.sentence_emb_text_encoder = self.sentence_emb_text_encoder.eval().requires_grad_(False)
        self.vtxt_dim = self.sentence_emb_text_encoder.config.hidden_size

        self.llm_type = llm_type
        self._orig_max_length_llm = max_length_llm
        self.enable_llm_padding = enable_llm_padding
        llm_layout = LLM_ENCODER_LAYOUT[llm_type]
        llm_model_path = llm_model_path or llm_layout["module_path"]
        llm_tokenizer_path = llm_tokenizer_path or llm_model_path or llm_layout["module_path"]
        self.llm_tokenizer = llm_layout["tokenizer_class"].from_pretrained(
            llm_tokenizer_path,
            padding_side="right",
        )
        self.crop_start = self._compute_crop_start()
        self.max_length_llm = self._orig_max_length_llm + self.crop_start

        self.llm_text_encoder = llm_layout["text_encoder_class"].from_pretrained(
            llm_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
        self.llm_text_encoder = self.llm_text_encoder.eval().requires_grad_(False)
        self.ctxt_dim = self.llm_text_encoder.config.hidden_size

    @torch.no_grad()
    def _encode_llm(self, text: List[str]) -> Tuple[Tensor, Tensor]:
        device = get_module_device(self)
        llm_text = [
            (
                self.llm_tokenizer.apply_chat_template(
                    self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"]),
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if self.llm_type == "qwen3"
                else self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
            )
            for one_text in text
        ]
        padding_mode = "max_length" if self.enable_llm_padding else False
        llm_batch_encoding = self.llm_tokenizer(
            llm_text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_llm,  # = crop_start + _orig_max_length_llm
            padding=padding_mode,
            return_tensors="pt",
        )
        llm_outputs = (
            self.llm_text_encoder(
                input_ids=llm_batch_encoding["input_ids"].to(device),
                attention_mask=llm_batch_encoding["attention_mask"].to(device),
                output_hidden_states=True,
            )
            if self.llm_type == "qwen3"
            else self.llm_text_encoder(
                input_ids=llm_batch_encoding["input_ids"].to(device),
                attention_mask=llm_batch_encoding["attention_mask"].to(device),
            )
        )
        if self.llm_type == "qwen3":
            ctxt_raw = llm_outputs.hidden_states[-1]
        else:
            ctxt_raw = llm_outputs.last_hidden_state

        start = self.crop_start
        end = start + self._orig_max_length_llm
        ctxt_raw = ctxt_raw[:, start:end].contiguous()  # [bs, _orig_max_length_llm, hidden]
        ctxt_length = (llm_batch_encoding["attention_mask"].sum(dim=-1).to(device) - start).clamp(
            min=0, max=self._orig_max_length_llm
        )
        return ctxt_raw, ctxt_length

    @torch.no_grad()
    def _encode_sentence_emb(self, text: List[str]) -> Tensor:
        device = get_module_device(self)
        enc = self.sentence_emb_tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_sentence_emb,
            padding="max_length",
            return_tensors="pt",
        )
        out = self.sentence_emb_text_encoder(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        if hasattr(out, "pooler_output"):
            vtxt_raw = out["pooler_output"].unsqueeze(1)
        else:
            vtxt_raw = self.encode_pooling(
                enc["attention_mask"].to(device),
                out["last_hidden_state"],
            )
        return vtxt_raw

    def encode(self, text: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        ctxt_raw, ctxt_length = self._encode_llm(text=text)
        vtxt_raw = self._encode_sentence_emb(text=text)
        return vtxt_raw, ctxt_raw, ctxt_length

    @staticmethod
    def apply_text_to_template(text: str, template: Union[str, list]) -> Union[str, list]:
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            return [
                {"role": "system", "content": f"{template[0]['content']}"},
                {"role": "user", "content": f"{text}"},
            ]
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def _compute_crop_start(self) -> int:
        def _find_subseq(a: str, b: str) -> int:
            for i in range(0, len(a) - len(b) + 1):
                if a[i : i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        if self.llm_type == "qwen3":
            msgs = self.apply_text_to_template(marker, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
            s = self.llm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        else:
            s = self.apply_text_to_template(marker, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
        full_ids = self.llm_tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = self.llm_tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        if pos >= 0:
            return pos
        else:
            return max(0, len(full_ids) - 1)

    def _pad_or_truncate_tensor(self, tensor: Tensor, target_length: int, dim: int = 0) -> Tensor:
        current_length = tensor.shape[dim]
        if current_length > target_length:
            return tensor.narrow(dim, 0, target_length)
        elif current_length < target_length:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = target_length - current_length
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device) + tensor.narrow(dim, -1, 1)
            return torch.cat([tensor, padding], dim=dim)
        return tensor

    def encode_pooling(self, attention_mask: Tensor, token_embeddings: Tensor) -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        # Normalize embeddings
        vtxt_raw = nn.functional.normalize(sentence_embeddings, p=2, dim=1).unsqueeze(1)  # shape of [bs, 1, 768]
        return vtxt_raw


if __name__ == "__main__":
    # python -m hymotion.network.text_encoders.text_encoder
    text_encoder = HYTextModel(llm_type="qwen3_embedding", max_length_llm=5)
    vtxt_raw, ctxt_raw, ctxt_length = text_encoder.encode(["Hello, world!"])
    print(vtxt_raw.shape, ctxt_raw.shape, ctxt_length)

    crop_start = text_encoder._compute_crop_start()
    print(f"crop_start: {crop_start} when using {text_encoder.llm_type}")

    assert (
        vtxt_raw.shape[1:] == (1, text_encoder.vtxt_dim)
        and ctxt_raw.shape[1:] == (text_encoder._orig_max_length_llm, text_encoder.ctxt_dim)
        and torch.all((ctxt_length >= 0) & (ctxt_length <= text_encoder._orig_max_length_llm))
    ), f"Got unexpected output shape: {vtxt_raw.shape}, {ctxt_raw.shape}, {ctxt_length}"
