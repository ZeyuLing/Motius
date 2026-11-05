"""Build and serialize actual SentenceT5 tensors without a text-processing model wrapper."""

from pathlib import Path
import json
import shutil
import torch


def load_text_network(source, device):
    from transformers import T5EncoderModel
    from safetensors.torch import load_file
    from motius.models.motionstreamer.network.text import SentenceT5Encoder

    source = Path(source)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                str(source),
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.txt",
                    "README.md",
                ],
            )
        )
    modules = json.loads((source / "modules.json").read_text())
    if [item["type"].rsplit(".", 1)[-1] for item in modules] != [
        "Transformer",
        "Pooling",
        "Dense",
        "Normalize",
    ]:
        raise ValueError(
            "Expected the released SentenceT5 Transformer/Pooling/Dense/Normalize architecture"
        )
    pooling = json.loads((source / "1_Pooling/config.json").read_text())
    if not pooling.get("pooling_mode_mean_tokens") or any(
        value
        for key, value in pooling.items()
        if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens"
    ):
        raise ValueError("The released SentenceT5 encoder requires mean pooling only")
    dense = json.loads((source / "2_Dense/config.json").read_text())
    if dense["activation_function"].rsplit(".", 1)[-1] != "Identity":
        raise ValueError("Unsupported SentenceT5 projection activation")
    # Original SentenceTransformer construction uses FP32; checkpoint storage
    # dtype does not select the numerical inference policy.
    backbone = T5EncoderModel.from_pretrained(
        source, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    )
    projection = torch.nn.Linear(
        dense["in_features"], dense["out_features"], bias=dense["bias"]
    )
    path = source / "2_Dense/model.safetensors"
    state = (
        load_file(str(path))
        if path.is_file()
        else torch.load(
            source / "2_Dense/pytorch_model.bin", map_location="cpu", weights_only=True
        )
    )
    projection.load_state_dict(
        {name.removeprefix("linear."): value for name, value in state.items()},
        strict=True,
    )
    model = (
        SentenceT5Encoder(backbone, projection).to(device).eval().requires_grad_(False)
    )
    return model, source


def save_text_network(model, source, destination):
    from safetensors.torch import save_file

    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(destination, safe_serialization=True)
    dense = destination / "2_Dense"
    dense.mkdir(exist_ok=True)
    save_file(
        {
            "linear." + name: value.detach().cpu().contiguous()
            for name, value in model.projection.state_dict().items()
        },
        str(dense / "model.safetensors"),
    )
    files = [
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",
        "tokenizer.model",
        "1_Pooling/config.json",
        "2_Dense/config.json",
        "README.md",
        "LICENSE",
        "NOTICE",
    ]
    for name in files:
        path = source / name
        target = destination / name
        if path.is_file() and path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
