"""Load and export the real KIMODO text network independently of its frontend."""

from pathlib import Path
import json


def load_original_text_network(bundle):
    from motius.models.kimodo.bundle import _patched_env
    from motius.utils.kimodo_loading import _build_local_text_encoder_conf
    from motius.utils.kimodo_checkpoint import instantiate_from_dict

    with _patched_env(bundle._kimodo_env()):
        frontend = instantiate_from_dict(_build_local_text_encoder_conf())
        frontend.to(bundle.network.device)
    wrapper = frontend.model
    options = {
        key: getattr(wrapper, key)
        for key in ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
    }
    metadata = {
        "frontend_options": options,
        "tokenizer_source": wrapper.tokenizer.name_or_path,
    }
    return wrapper.model, metadata


def load_packed_text_network(directory, device):
    from motius.configuration.ardy.text import load_text_network
    from motius.models.kimodo.network.model.llm2vec.models.bidirectional_llama import (
        LlamaBiModel,
    )

    return load_text_network(directory, device, model_class=LlamaBiModel)


def save_current_text_network(bundle, destination):
    from motius.configuration.ardy.text import save_text_network

    network = bundle.load_text_model()
    if bundle.packed_text_path is not None:
        metadata = json.loads(
            (Path(bundle.packed_text_path) / "architecture.json").read_text()
        )
        source = Path(bundle.packed_text_path) / "tokenizer"
    else:
        metadata = bundle.text_frontend_metadata
        source = metadata["tokenizer_source"]
    save_text_network(
        network,
        destination,
        tokenizer_source=source,
        frontend_options=metadata["frontend_options"],
    )
    copy_text_notices(bundle, destination)


def copy_text_notices(bundle, destination):
    """Retain source weight/tokenizer notices without copying application code."""
    import shutil

    target = Path(destination) / "notices"
    if bundle.packed_text_path is not None:
        source = Path(bundle.packed_text_path) / "notices"
        if source.is_dir() and source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
        return
    if not bundle.text_encoders_dir:
        return
    source = Path(bundle.text_encoders_dir)
    for relative in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    ]:
        for name in [
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "NOTICE",
            "README.md",
            "USE_POLICY.md",
        ]:
            path = source / relative / name
            if path.is_file():
                output = target / relative / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
