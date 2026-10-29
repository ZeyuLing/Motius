"""HY-Motion and MotionCanvas prompt templates, tokenization and cropping."""

from functools import lru_cache
import json
from typing import Union
from motius.configuration.hymotion_text import LLM_ENCODER_LAYOUT, SENTENCE_EMB_LAYOUT


class HYTextProcessor:
    def __init__(
        self,
        llm_type="qwen3",
        max_length_llm=512,
        sentence_emb_type="clipl",
        max_length_sentence_emb=77,
        enable_llm_padding=True,
        llm_model_path=None,
        llm_tokenizer_path=None,
        sentence_emb_model_path=None,
        sentence_emb_tokenizer_path=None,
        prompt_template=None,
        sentence_padding=True,
    ):
        llm = LLM_ENCODER_LAYOUT[llm_type]
        sentence = SENTENCE_EMB_LAYOUT[sentence_emb_type]
        self.llm_type = llm_type
        self.template = llm["template"]
        if prompt_template is not None and llm_type == "qwen3":
            self.template = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": "{}"},
            ]
        self.sentence_padding = sentence_padding
        self.enable_llm_padding = enable_llm_padding
        self.max_length_sentence_emb = max_length_sentence_emb
        self.llm_tokenizer = llm["tokenizer_class"].from_pretrained(
            llm_tokenizer_path or llm_model_path or llm["module_path"],
            padding_side="right",
        )
        self.sentence_emb_tokenizer = sentence["tokenizer_class"].from_pretrained(
            sentence_emb_tokenizer_path
            or sentence_emb_model_path
            or sentence["module_path"],
            max_length=max_length_sentence_emb,
        )
        self.crop_start = self._compute_crop_start()
        self.max_length_llm = max_length_llm + self.crop_start

    @staticmethod
    def apply_text_to_template(
        text: str, template: Union[str, list]
    ) -> Union[str, list]:
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
            msgs = self.apply_text_to_template(marker, self.template)
            s = self.llm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        else:
            s = self.apply_text_to_template(marker, self.template)
        full_ids = self.llm_tokenizer(s, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ][0].tolist()
        marker_ids = self.llm_tokenizer(
            marker, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        if pos >= 0:
            return pos
        else:
            return max(0, len(full_ids) - 1)

    def __call__(self, text):
        llm_text = [
            (
                self.llm_tokenizer.apply_chat_template(
                    self.apply_text_to_template(one_text, self.template),
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if self.llm_type == "qwen3"
                else self.apply_text_to_template(one_text, self.template)
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
        enc = self.sentence_emb_tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_sentence_emb,
            padding=self.sentence_padding,
            return_tensors="pt",
        )
        return {
            "llm": llm_batch_encoding,
            "sentence": enc,
            "crop_start": self.crop_start,
        }


@lru_cache(maxsize=8)
def _processor(config_json):
    return HYTextProcessor(**json.loads(config_json))


def prepare_hymotion_text(config, captions):
    if config is None:
        raise RuntimeError("No text_encoder config provided; cannot encode text.")
    options = {
        key: value
        for key, value in config.items()
        if key not in {"type", "torch_dtype", "llm_torch_dtype", "sentence_torch_dtype"}
    }
    return _processor(json.dumps(options, sort_keys=True))(list(captions))


def encode_hymotion_text(bundle, captions):
    return bundle.encode_text(prepare_hymotion_text(bundle._text_encoder_cfg, captions))


def encode_hymotion_instruction(bundle, instructions):
    return bundle.encode_task_instruction(
        prepare_hymotion_text(bundle._text_encoder_cfg, instructions)
    )
