"""Portable current-weight Llama/LoRA architecture and tensor serialization."""

from pathlib import Path
import json
import torch


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError("Unsupported neural architecture metadata: " + type(value).__name__)


def save_text_network(network, directory, *, tokenizer_source, frontend_options=None):
    from safetensors.torch import save_file
    import shutil

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = network.config.to_dict()
    config["_name_or_path"] = network.config._name_or_path
    adapters = {name: value.to_dict() for name, value in network.peft_config.items()}
    if set(adapters) != {"default"}:
        raise ValueError("The original ARDY text frontend uses one supervised adapter")
    adapters["default"]["base_model_name_or_path"] = network.config._name_or_path
    metadata = dict(
        config=config,
        adapter=adapters["default"],
        attention_implementation=network.config._attn_implementation,
        frontend_options=frontend_options or {},
        canonical_name=network.config._name_or_path,
    )
    (directory / "architecture.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n"
    )
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in network.state_dict().items()
        },
        str(directory / "networks.safetensors"),
    )
    persistent = set(network.state_dict())
    buffers = {
        name: value.detach().cpu()
        for name, value in network.named_buffers()
        if name not in persistent
    }
    torch.save(buffers, directory / "buffers.pt")
    source = Path(tokenizer_source)
    tokenizer = directory / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ]:
        if (source / name).is_file():
            shutil.copy2(source / name, tokenizer / name)
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError(
            "The actual tokenizer files are required for a portable text artifact"
        )


def load_text_network(directory, device, *, model_class=None):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import LlamaConfig
    from peft import LoraConfig, get_peft_model

    if model_class is None:
        from motius.models.ardy.network.model.llm2vec.models.bidirectional_llama import (
            LlamaBiModel,
        )

        model_class = LlamaBiModel

    directory = Path(directory)
    metadata = json.loads((directory / "architecture.json").read_text())
    config = LlamaConfig(**metadata["config"])
    config._attn_implementation = metadata["attention_implementation"]
    config._name_or_path = metadata["canonical_name"]
    with init_empty_weights(include_buffers=True):
        backbone = model_class(config)
        network = get_peft_model(backbone, LoraConfig(**metadata["adapter"]))
    state = load_file(str(directory / "networks.safetensors"), device=str(device))
    network.load_state_dict(state, strict=True, assign=True)
    buffers = torch.load(
        directory / "buffers.pt", map_location=device, weights_only=True
    )
    for name, value in buffers.items():
        parent, _, leaf = name.rpartition(".")
        module = network.get_submodule(parent)
        module._buffers[leaf] = value
        if leaf == "inv_freq" and hasattr(module, "original_inv_freq"):
            module.original_inv_freq = value
    network.config._name_or_path = metadata["canonical_name"]
    return network.eval().requires_grad_(False)
