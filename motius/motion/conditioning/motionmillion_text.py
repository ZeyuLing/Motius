"""Flan-T5 token preparation for MotionMillion inference and training."""
from functools import lru_cache


@lru_cache(maxsize=8)
def text_tokenizer(model_name):
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained(model_name)


def prepare_motionmillion_text(model_name, captions):
    return text_tokenizer(model_name)(
        list(captions), padding=True, truncation=True, return_tensors="pt"
    )
