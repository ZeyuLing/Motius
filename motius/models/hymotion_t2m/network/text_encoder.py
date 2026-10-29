"""Tensor-only frozen HY-Motion/MotionCanvas text networks."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from motius.configuration.hymotion_text import (
    LLM_ENCODER_LAYOUT,
    SENTENCE_EMB_LAYOUT,
    resolve_text_dtype,
)


def get_module_device(module):
    return next(module.parameters()).device


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
        llm_torch_dtype=None,
        sentence_torch_dtype=None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_text_dtype(torch_dtype)
        llm_torch_dtype = resolve_text_dtype(llm_torch_dtype)
        sentence_torch_dtype = resolve_text_dtype(sentence_torch_dtype)
        self.text_encoder_type = "hy_text_model"
        assert llm_type in LLM_ENCODER_LAYOUT, f"Unsupported LLM type: {llm_type}"
        assert (
            sentence_emb_type in SENTENCE_EMB_LAYOUT
        ), f"Unsupported sentence embedding type: {sentence_emb_type}"

        self.sentence_emb_type = sentence_emb_type
        self.max_length_sentence_emb = max_length_sentence_emb
        sentence_layout = SENTENCE_EMB_LAYOUT[sentence_emb_type]
        sentence_emb_model_path = (
            sentence_emb_model_path or sentence_layout["module_path"]
        )
        self.sentence_emb_text_encoder = sentence_layout[
            "text_encoder_class"
        ].from_pretrained(
            sentence_emb_model_path,
            torch_dtype=(
                sentence_torch_dtype
                if sentence_torch_dtype is not None
                else torch_dtype
            ),
        )
        self.sentence_emb_text_encoder = (
            self.sentence_emb_text_encoder.eval().requires_grad_(False)
        )
        self.vtxt_dim = self.sentence_emb_text_encoder.config.hidden_size

        self.llm_type = llm_type
        self._orig_max_length_llm = max_length_llm
        self.enable_llm_padding = enable_llm_padding
        llm_layout = LLM_ENCODER_LAYOUT[llm_type]
        llm_model_path = llm_model_path or llm_layout["module_path"]
        self.llm_text_encoder = llm_layout["text_encoder_class"].from_pretrained(
            llm_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=(
                llm_torch_dtype
                if llm_torch_dtype is not None
                else (
                    torch_dtype
                    if torch_dtype is not None
                    else (torch.bfloat16 if llm_type == "qwen3" else None)
                )
            ),
        )
        self.llm_text_encoder = self.llm_text_encoder.eval().requires_grad_(False)
        self.ctxt_dim = self.llm_text_encoder.config.hidden_size

    @torch.no_grad()
    def _encode_llm(self, llm_batch_encoding, crop_start: int) -> Tuple[Tensor, Tensor]:
        device = get_module_device(self)
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

        start = crop_start
        end = start + self._orig_max_length_llm
        ctxt_raw = ctxt_raw[
            :, start:end
        ].contiguous()  # [bs, _orig_max_length_llm, hidden]
        ctxt_length = (
            llm_batch_encoding["attention_mask"].sum(dim=-1).to(device) - start
        ).clamp(min=0, max=self._orig_max_length_llm)
        return ctxt_raw, ctxt_length

    @torch.no_grad()
    def _encode_sentence_emb(self, enc) -> Tensor:
        device = get_module_device(self)
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

    def encode(self, prepared) -> Tuple[Tensor, Tensor, Tensor]:
        ctxt_raw, ctxt_length = self._encode_llm(
            prepared["llm"], prepared["crop_start"]
        )
        vtxt_raw = self._encode_sentence_emb(prepared["sentence"])
        return vtxt_raw, ctxt_raw, ctxt_length

    def encode_pooling(
        self, attention_mask: Tensor, token_embeddings: Tensor
    ) -> Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        vtxt_raw = nn.functional.normalize(sentence_embeddings, p=2, dim=1).unsqueeze(
            1
        )  # shape of [bs, 1, 768]
        return vtxt_raw
