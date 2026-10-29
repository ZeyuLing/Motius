"""TM2T artifact resolution and vocabulary architecture metadata."""

from pathlib import Path


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _first_existing(root: Path, candidates) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"TM2T artifact is missing all expected paths: {', '.join(map(str, candidates))}"
    )


def vocabulary_metadata(root):
    import pickle

    with _first_existing(root, ["vocab/our_vab_idx.pkl", "glove/our_vab_idx.pkl"]).open(
        "rb"
    ) as handle:
        mapping = pickle.load(handle)
    return len(mapping), int(mapping["sos"]), int(mapping["eos"])


def save_bundle(bundle, directory):
    """Write actual current neural tensors and the selected vocabulary/statistics."""
    import json
    import shutil
    import numpy as np
    from safetensors.torch import save_file

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["vq_encoder", "quantizer", "transformer"]:
        module = getattr(bundle, name)
        while hasattr(module, "module"):
            module = module.module
        save_file(
            {
                key: value.detach().cpu().contiguous().clone()
                for key, value in module.state_dict().items()
            },
            str(destination / (name + ".safetensors")),
        )
    (destination / "vocab").mkdir(exist_ok=True)
    for name in ["our_vab_words.pkl", "our_vab_idx.pkl"]:
        source = _first_existing(
            bundle.artifact_dir, ["vocab/" + name, "glove/" + name]
        )
        target = destination / "vocab" / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    (destination / "stats").mkdir(exist_ok=True)
    for name in ["mean", "std"]:
        np.save(
            destination / "stats" / (name + ".npy"),
            getattr(bundle, name).detach().cpu().numpy(),
        )
    (destination / "tm2t_config.json").write_text(
        json.dumps(
            {
                "format": "motius-tm2t-neural-v2",
                "beam_size": bundle.beam_size,
                "max_text_length": bundle.max_text_length,
            },
            indent=2,
        )
        + "\n"
    )
    (destination / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TM2TPipeline",
                "pipeline_class": "motius.pipelines.tm2t.TM2TPipeline",
                "bundle_class": "motius.models.tm2t.TM2TBundle",
                "tasks": ["motion_to_text"],
            },
            indent=2,
        )
        + "\n"
    )
    return str(destination)
