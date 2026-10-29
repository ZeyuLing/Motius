"""Portable current-weight artifacts for native Gaussian motion bundles."""

from pathlib import Path
import json
import shutil
import numpy as np
import torch


def unwrap_network(network):
    while hasattr(network, "module") or hasattr(network, "_orig_mod"):
        network = network.module if hasattr(network, "module") else network._orig_mod
    return network


def save_gaussian_motion_artifact(
    bundle, directory, *, method, safe_serialization=True
):
    """Serialize the current denoiser, frozen CLIP and actual normalization."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    network = unwrap_network(bundle.net)
    class_name = {"condmdi": "CondMDI", "omnicontrol": "OmniControl"}[method]
    tasks = ["text_to_motion", "kinematic_motion_control"]
    stats = (
        {"mean_abs": "Mean_abs_3d.npy", "std_abs": "Std_abs_3d.npy"}
        if method == "condmdi"
        else {
            "mean": "Mean.npy",
            "std": "Std.npy",
            "raw_mean": "Mean_raw.npy",
            "raw_std": "Std_raw.npy",
        }
    )
    model_file = "model.safetensors" if safe_serialization else "model.pt"
    clip_file = "clip.safetensors" if safe_serialization else "clip.pt"
    metadata = dict(
        model_type=method,
        weights=model_file,
        config=bundle.config,
        guidance_param=bundle.guidance_param,
        respacing=bundle.respacing,
        clip_weights=clip_file,
    )
    (directory / (method + "_config.json")).write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in network.state_dict().items()
        if not name.startswith("clip_model.") and "sequence_pos_encoder.pe" not in name
    }
    clip = {
        name: value.detach().cpu().contiguous()
        for name, value in network.clip_model.state_dict().items()
    }
    if safe_serialization:
        from safetensors.torch import save_file

        save_file(state, str(directory / model_file))
        save_file(clip, str(directory / clip_file))
    else:
        torch.save(state, directory / model_file)
        torch.save(clip, directory / clip_file)
    for name, filename in stats.items():
        np.save(directory / filename, getattr(bundle, name).detach().cpu().numpy())
    manifest = dict(
        _class_name=class_name + "Pipeline",
        _library_name="motius",
        model_type=method,
        pipeline_class=f"motius.pipelines.{method}.{class_name}Pipeline",
        bundle_class=f"motius.models.{method}.{class_name}Bundle",
        tasks=tasks,
        required_files=[
            method + "_config.json",
            model_file,
            clip_file,
            *stats.values(),
        ],
        api=dict(
            loader="motius.Pipeline.from_pretrained",
            task_methods=["infer_text_to_motion", "infer_kinematic_motion_control"],
        ),
    )
    (directory / "model_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    license_dir = Path(__file__).resolve().parents[1] / "licenses" / method
    if license_dir.is_dir():
        for origin in license_dir.iterdir():
            if origin.is_file() and origin.suffix != ".py":
                shutil.copy2(origin, directory / origin.name)
    return str(directory)
