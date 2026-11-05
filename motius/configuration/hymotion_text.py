"""Published text backbone layouts and prompt templates."""


def resolve_text_dtype(value):
    """Resolve serialized component precision before Transformers construction."""
    import torch

    if isinstance(value, str) and value != "auto":
        return getattr(
            torch,
            {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
                value, value.removeprefix("torch.")
            ),
        )
    return value


def hymotion_text_precision(config, motion_type):
    """Restore the original human model's mixed text-tower precision.

    Legacy human conversions declared a uniform BF16 text dtype, losing the
    source's FP32 CLIP tower. Explicit per-component fields and G1 artifacts
    retain their declared precision.
    """
    from copy import deepcopy

    result = deepcopy(config)
    if result and motion_type not in {"g1_29dof", "g1_38", "unitree_g1"}:
        result.setdefault("llm_torch_dtype", "bf16")
        result.setdefault("sentence_torch_dtype", "fp32")
    return result


def copy_hymotion_notices(directory, *, wooden=False):
    """Keep HY-Motion code/geometry provenance with exported neural artifacts."""
    from pathlib import Path
    import shutil

    source = Path(__file__).resolve().parents[1] / "licenses" / "hymotion_t2m"
    target = Path(directory)
    shutil.copy2(source / "UPSTREAM_LICENSE.md", target / "HY_MOTION_LICENSE.md")
    if wooden:
        shutil.copy2(source / "WOODEN_ASSETS.json", target / "WOODEN_ASSETS.json")


from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

MOTIONCANVAS_PROMPT_TEMPLATE = "\n    Summarize human motion only from the user text for representation: action categories, key body-part movements, order/transitions, trajectory/direction, posture; include style/emotion/speed only if present. Explicitly capture laterality (left/right) when mentioned; do not guess. If multiple actions are described, indicate the count of distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one concise paragraph.\n    "

PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION = "\n    Summarize human motion only from the user text for representation: action categories, key body-part movements, order/transitions, trajectory/direction, posture; include style/emotion/speed only if present. Explicitly capture laterality (left/right) when mentioned; do not guess. If multiple actions are described, indicate the count of distinct actions (e.g., actions=3) and their order. Do not invent missing info. Keep one concise paragraph.\n"

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
        "text_encoder_class": AutoModel,
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


def export_text_networks(
    bundle,
    directory,
    config,
    text_encoder_subdir="text_encoder",
    *,
    fallback_dtype=None,
):
    """Save current selected text towers and their original tokenizers.

    Each component records its own dtype, so decoder BF16 and sentence FP32
    remain distinct during offline loading. Unused vision/vocabulary heads are
    absent from these native encoders. A never-used encoder is materialized on
    CPU from the declared configuration instead of copying a source application.
    """
    from copy import deepcopy
    from pathlib import Path
    import shutil
    import torch
    from motius.models.hymotion_t2m.network.text_encoder import HYTextModel

    cfg = deepcopy(config)
    network = getattr(bundle, "_text_encoder", None)
    if network is None:
        constructor = {
            key: value
            for key, value in cfg.items()
            if key not in {"type", "prompt_template", "sentence_padding"}
        }
        dtype = getattr(bundle, "_text_dtype", None)
        if dtype is not None:
            constructor["llm_torch_dtype"] = dtype
            constructor["sentence_torch_dtype"] = dtype
        if dtype is None:
            dtype = constructor.get("torch_dtype", fallback_dtype)
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(
                torch,
                {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
                    dtype, dtype.removeprefix("torch.")
                ),
            )
        constructor["torch_dtype"] = dtype
        network = HYTextModel(**constructor).eval()
    while hasattr(network, "module"):
        network = network.module
    destination = Path(directory)
    for component, model, layout, name, source_key, tokenizer_key in [
        (
            "llm",
            network.llm_text_encoder,
            LLM_ENCODER_LAYOUT,
            cfg.get("llm_type", "qwen3"),
            "llm_model_path",
            "llm_tokenizer_path",
        ),
        (
            "sentence",
            network.sentence_emb_text_encoder,
            SENTENCE_EMB_LAYOUT,
            cfg.get("sentence_emb_type", "clipl"),
            "sentence_emb_model_path",
            "sentence_emb_tokenizer_path",
        ),
    ]:
        relative = f"{text_encoder_subdir}/{component}"
        target = destination / relative
        target.mkdir(parents=True, exist_ok=True)
        # Remove stale duplicate weights/indexes from an existing export.
        for pattern in [
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
            "pytorch_model*.bin.index.json",
            "tf_model.h5",
            "flax_model.msgpack",
        ]:
            for path in target.glob(pattern):
                path.unlink()
        model.save_pretrained(target, safe_serialization=True)
        source = (
            cfg.get(tokenizer_key) or cfg.get(source_key) or layout[name]["module_path"]
        )
        layout[name]["tokenizer_class"].from_pretrained(source).save_pretrained(target)
        source_path = Path(source)
        if source_path.is_dir():
            for pattern in ["LICENSE*", "NOTICE*"]:
                for path in source_path.glob(pattern):
                    if (
                        path.is_file()
                        and path.resolve() != (target / path.name).resolve()
                    ):
                        shutil.copy2(path, target / path.name)
        cfg[source_key] = cfg[tokenizer_key] = relative
    cfg["llm_torch_dtype"] = "auto"
    cfg["sentence_torch_dtype"] = "auto"
    return cfg
