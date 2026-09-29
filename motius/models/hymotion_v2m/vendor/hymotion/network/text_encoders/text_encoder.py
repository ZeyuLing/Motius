from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen3VLForConditionalGeneration,
    T5EncoderModel,
    T5Tokenizer,
)

from ...utils.type_converter import get_module_device
from .model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

LLM_ENCODER_LAYOUT = {
    "qwen3_embedding": {
        "module_path": "ckpts/Qwen3-Embedding-8B",
        "template": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}\n{{}}",
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
    "qwen3": {
        "module_path": "ckpts/Qwen3-8B",
        "template": [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ],
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModelForCausalLM,
    },
    "qwen3_vl": {
        "module_path": "ckpts/Qwen3-VL-8B-Instruct",
        "template": [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": [{"type": "text", "text": "{}"}]},
        ],
        "crop_start": 0,
        "tokenizer_class": AutoProcessor,
        "text_encoder_class": Qwen3VLForConditionalGeneration,
    },
    "t5": {
        "module_path": "ckpts/t5-v1_1-xxl",
        "template": "{}",
        "crop_start": 0,
        "tokenizer_class": T5Tokenizer,
        "text_encoder_class": T5EncoderModel,
    },
    "distilbert": {
        "module_path": "ckpts/distilbert-base-uncased",
        "template": "{}",
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
}

SENTENCE_EMB_LAYOUT = {
    "clipl": {
        "module_path": "ckpts/clip-vit-large-patch14",
        "tokenizer_class": CLIPTokenizer,
        "text_encoder_class": CLIPTextModel,
        "pooling_mode": "pooler_output",
        "max_length": 77,
    },
    "sentence_transformer": {
        "module_path": "ckpts/all-mpnet-base-v2",
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
        "pooling_mode": "mean",
        "max_length": 512,
    },
    "qwen3emb": {
        "module_path": "ckpts/Qwen3-Embedding-8B",
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
        "pooling_mode": "last_token",
        "max_length": 8192,
        "tokenizer_kwargs": {"padding_side": "left"},
        # NOTE: 暂时不涉及instruction注入，后续可以考虑
    },
}


class HYTextModel(nn.Module):
    def __init__(
        self,
        llm_type: Optional[str] = "qwen3_embedding",
        max_length_llm: int = 512,
        sentence_emb_type: Optional[str] = "clipl",
        max_length_sentence_emb: int = 77,
        enable_llm_padding: bool = True,
        llm_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.text_encoder_type = "hy_text_model"

        # --- Sentence Embedding Initialization ---
        self.sentence_emb_type = sentence_emb_type
        self.sentence_emb_text_encoder = None
        self.sentence_emb_tokenizer = None
        self.vtxt_dim = 0
        if sentence_emb_type is not None:
            assert sentence_emb_type in SENTENCE_EMB_LAYOUT, f"Unsupported sentence embedding type: {sentence_emb_type}"
            self.max_length_sentence_emb = max_length_sentence_emb or SENTENCE_EMB_LAYOUT[sentence_emb_type].get(
                "max_length", 77
            )
            self._sentence_emb_pooling_mode = SENTENCE_EMB_LAYOUT[sentence_emb_type].get(
                "pooling_mode", "pooler_output"
            )
            tokenizer_kwargs = SENTENCE_EMB_LAYOUT[sentence_emb_type].get("tokenizer_kwargs", {})

            self.sentence_emb_tokenizer = SENTENCE_EMB_LAYOUT[sentence_emb_type]["tokenizer_class"].from_pretrained(
                SENTENCE_EMB_LAYOUT[sentence_emb_type]["module_path"],
                max_length=self.max_length_sentence_emb,
                **tokenizer_kwargs,
            )
            self.sentence_emb_text_encoder = SENTENCE_EMB_LAYOUT[sentence_emb_type][
                "text_encoder_class"
            ].from_pretrained(SENTENCE_EMB_LAYOUT[sentence_emb_type]["module_path"])
            self.sentence_emb_text_encoder = self.sentence_emb_text_encoder.eval().requires_grad_(False)
            self.vtxt_dim = self.sentence_emb_text_encoder.config.hidden_size

        # --- LLM / VL Initialization ---
        self.llm_type = llm_type
        self.llm_text_encoder = None
        self.llm_tokenizer = None
        self.ctxt_dim = 0
        self.crop_start = 0
        self.max_length_llm = max_length_llm
        if llm_type is not None:
            assert llm_type in LLM_ENCODER_LAYOUT, f"Unsupported LLM type: {llm_type}"
            self._orig_max_length_llm = max_length_llm
            self.enable_llm_padding = enable_llm_padding

            config = LLM_ENCODER_LAYOUT[llm_type]
            self.model_category = config.get("type", "text")

            self.llm_tokenizer = config["tokenizer_class"].from_pretrained(
                config["module_path"],
                padding_side="right",
            )
            self.llm_text_encoder = config["text_encoder_class"].from_pretrained(
                config["module_path"],
                low_cpu_mem_usage=True,
                torch_dtype=llm_dtype,
            )

            self.llm_text_encoder = self.llm_text_encoder.eval().requires_grad_(False)
            if self.llm_type == "qwen3_vl":
                self.ctxt_dim = self.llm_text_encoder.config.text_config.hidden_size
                self.ctxt_dim_vl = self.llm_text_encoder.config.vision_config.hidden_size
            else:
                self.ctxt_dim = self.llm_text_encoder.config.hidden_size

            self.crop_start = self._compute_crop_start()
            self.max_length_llm = self._orig_max_length_llm + self.crop_start

    @torch.no_grad()
    def encode_llm(self, text: List[str]) -> Tuple[Tensor, Tensor]:
        if self.llm_type is None or self.llm_text_encoder is None or self.llm_tokenizer is None:
            raise ValueError("LLM model not initialized")

        device = get_module_device(self)
        llm_tokenizer = self.llm_tokenizer.tokenizer if self.llm_type == "qwen3_vl" else self.llm_tokenizer
        need_apply_chat_template = self.llm_type in ["qwen3_vl", "qwen3"]
        padding_mode = "max_length" if self.enable_llm_padding else False

        llm_text = [
            (
                self.llm_tokenizer.apply_chat_template(
                    self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"]),
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if need_apply_chat_template
                else self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
            )
            for one_text in text
        ]
        llm_batch_encoding = llm_tokenizer(
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
            if need_apply_chat_template
            else self.llm_text_encoder(
                input_ids=llm_batch_encoding["input_ids"].to(device),
                attention_mask=llm_batch_encoding["attention_mask"].to(device),
            )
        )

        if need_apply_chat_template:
            ctxt_raw = llm_outputs.hidden_states[-1]
        else:
            ctxt_raw = llm_outputs.last_hidden_state

        start = self.crop_start
        end = start + self._orig_max_length_llm
        ctxt_raw = ctxt_raw[:, start:end].contiguous()  # [bs, _orig_max_length_llm, hidden]
        ctxt_length = (llm_batch_encoding["attention_mask"].sum(dim=-1).to(device) - start).clamp(
            min=0, max=self._orig_max_length_llm
        )

        # --- for debug ---
        # full_ids = llm_batch_encoding["input_ids"][0]
        # cropped_ids = full_ids[start : start + ctxt_length]
        # print(llm_tokenizer.decode(cropped_ids))
        return ctxt_raw, ctxt_length

    @torch.no_grad()
    def encode_sentence_emb(self, text: List[str]) -> Tensor:
        if (
            self.sentence_emb_type is None
            or self.sentence_emb_text_encoder is None
            or self.sentence_emb_tokenizer is None
        ):
            raise ValueError("Sentence embedding model not initialized")

        device = get_module_device(self)
        enc = self.sentence_emb_tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_sentence_emb,
            padding=True,
            return_tensors="pt",
        )
        out = self.sentence_emb_text_encoder(
            input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)
        )
        if self._sentence_emb_pooling_mode == "pooler_output":
            # Pooler output pooling (clip-vit-large-patch14 等)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vtxt_raw = out.pooler_output.unsqueeze(1)
            else:
                vtxt_raw = self._encode_pooling(enc["attention_mask"].to(device), out.last_hidden_state)
        elif self._sentence_emb_pooling_mode == "mean":
            # Mean pooling (sentence_transformer 等)
            vtxt_raw = self._encode_pooling(enc["attention_mask"].to(device), out.last_hidden_state)
        elif self._sentence_emb_pooling_mode == "last_token":
            # Last token pooling (Qwen3-Embedding)
            vtxt_raw = self._last_token_pool(out.last_hidden_state, enc["attention_mask"].to(device))
        else:
            raise ValueError(f"Unknown pooling mode: {self._sentence_emb_pooling_mode}")

        return vtxt_raw

    def encode(self, text: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        ctxt_raw, ctxt_length = self.encode_llm(text=text)
        vtxt_raw = self.encode_sentence_emb(text=text)
        return vtxt_raw, ctxt_raw, ctxt_length

    @staticmethod
    def apply_text_to_template(text: str, template: Union[str, list]) -> Any:
        import copy

        template_obj = copy.deepcopy(template)

        def recursive_format(obj: Any, fill_text: str) -> Any:
            if isinstance(obj, str):
                if "{}" in obj:
                    return obj.format(fill_text)
                return obj
            elif isinstance(obj, list):
                return [recursive_format(item, fill_text) for item in obj]
            elif isinstance(obj, dict):
                return {k: recursive_format(v, fill_text) for k, v in obj.items()}
            else:
                return obj

        return recursive_format(template_obj, text)

    def _compute_crop_start(self) -> int:
        if self.llm_type is None or self.llm_text_encoder is None or self.llm_tokenizer is None:
            raise ValueError("LLM model not initialized")

        def _find_subseq(a: str, b: str) -> int:
            for i in range(0, len(a) - len(b) + 1):
                if a[i : i + len(b)] == b:
                    return i
            return -1

        llm_tokenizer = self.llm_tokenizer.tokenizer if self.llm_type == "qwen3_vl" else self.llm_tokenizer
        need_apply_chat_template = self.llm_type in ["qwen3_vl", "qwen3"]

        marker = "<BOC>"
        msgs = self.apply_text_to_template(marker, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
        if need_apply_chat_template:
            full_str = llm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        else:
            full_str = msgs

        full_ids = llm_tokenizer(full_str, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = llm_tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

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

    def _encode_pooling(self, attention_mask: Tensor, token_embeddings: Tensor) -> Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(dtype=token_embeddings.dtype)
        )
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        vtxt_raw = nn.functional.normalize(sentence_embeddings, p=2, dim=1).unsqueeze(1)  # shape of [bs, 1, D]
        return vtxt_raw

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Last token pooling (用于 Qwen3-Embedding 等 left-padding 模型)"""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            vtxt_raw = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            vtxt_raw = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]
        vtxt_raw = nn.functional.normalize(vtxt_raw, p=2, dim=-1).unsqueeze(1)  # shape of [bs, 1, D]
        return vtxt_raw


if __name__ == "__main__":
    # python -m hymotion.network.text_encoders.text_encoder

    # --------- test qwen3_embedding ---------
    # encoder = HYTextModel(llm_type="qwen3_embedding", max_length_llm=128)
    # vtxt_raw, ctxt_raw, ctxt_length = encoder.encode(["Hello, world!"])
    # print(vtxt_raw.shape, ctxt_raw.shape, ctxt_length)

    # crop_start = encoder._compute_crop_start()
    # print(f"crop_start: {crop_start} when using {encoder.llm_type}")

    # assert (
    #     vtxt_raw.shape[1:] == (1, encoder.vtxt_dim)
    #     and ctxt_raw.shape[1:] == (encoder._orig_max_length_llm, encoder.ctxt_dim)
    #     and torch.all((ctxt_length >= 0) & (ctxt_length <= encoder._orig_max_length_llm))
    # ), f"Got unexpected output shape: {vtxt_raw.shape}, {ctxt_raw.shape}, {ctxt_length}"

    # --------- test qwen3 ---------
    # encoder = HYTextModel(llm_type="qwen3", max_length_llm=128)
    # ctxt_raw, ctxt_length = encoder.encode_llm(text=["Hello, world!"])
    # print(ctxt_raw.shape, ctxt_length)

    # --------- test qwen3_vl ---------
    encoder = HYTextModel(llm_type="qwen3_vl", max_length_llm=128)
    ctxt_raw, ctxt_length = encoder.encode_llm(text=["Hello, world!"])
    print(ctxt_raw.shape, ctxt_length)
