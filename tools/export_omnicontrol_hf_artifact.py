#!/usr/bin/env python3
"""Build a self-contained Motius OmniControl Hugging Face artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.omnicontrol.network.model_util import DEFAULT_CONFIG


PACKAGE_ASSETS = ROOT / "motius" / "models" / "omnicontrol" / "assets"
PACKAGE_LICENSE = ROOT / "motius" / "models" / "omnicontrol" / "LICENSE"
ARTIFACT_FILES = (
    "model.safetensors",
    "clip.safetensors",
    "Mean.npy",
    "Std.npy",
    "Mean_raw.npy",
    "Std_raw.npy",
    "omnicontrol_config.json",
    "LICENSE",
)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "model_avg" in value:
        value = value["model_avg"]
    elif isinstance(value, dict) and "model" in value:
        value = value["model"]
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported OmniControl checkpoint payload: {type(value)!r}")
    return {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in value.items()
        if torch.is_tensor(tensor) and "sequence_pos_encoder.pe" not in str(name)
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_artifact(
    checkpoint: Path,
    clip_weights: Path,
    output: Path,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    save_file(_load_state(checkpoint), str(output / "model.safetensors"))
    shutil.copy2(clip_weights, output / "clip.safetensors")
    for name in ("Mean.npy", "Std.npy", "Mean_raw.npy", "Std_raw.npy"):
        shutil.copy2(PACKAGE_ASSETS / name, output / name)
    shutil.copy2(PACKAGE_LICENSE, output / "LICENSE")

    config = {
        "format_version": 1,
        "config": dict(DEFAULT_CONFIG),
        "guidance_param": 2.5,
        "respacing": "",
        "clip_version": "ViT-B/32",
        "clip_weights": "clip.safetensors",
    }
    _write_json(output / "omnicontrol_config.json", config)
    _write_json(
        output / "model_index.json",
        {
            "_class_name": "OmniControlPipeline",
            "_library_name": "motius",
            "format_version": 1,
            "artifact_format": "motius-omnicontrol-v1",
            "pipeline_class": "motius.pipelines.omnicontrol.OmniControlPipeline",
            "bundle_class": "motius.models.omnicontrol.OmniControlBundle",
            "tasks": [
                "text_to_motion",
                "temporal_motion_completion",
                "kinematic_motion_control",
            ],
            "required_files": list(ARTIFACT_FILES),
            "artifacts": {
                "model": "model.safetensors",
                "clip": "clip.safetensors",
                "motion_statistics": ["Mean.npy", "Std.npy"],
                "spatial_statistics": ["Mean_raw.npy", "Std_raw.npy"],
            },
            "components": {
                "clip": {
                    "stored_in_artifact": True,
                    "path": "clip.safetensors",
                }
            },
            "api": {
                "loader": "motius.Pipeline.from_pretrained",
                "task_methods": [
                    "infer_text_to_motion",
                    "infer_temporal_motion_completion",
                    "infer_kinematic_motion_control",
                ],
            },
        },
    )
    (output / "README.md").write_text(
        """---
library_name: motius
license: mit
pipeline_tag: other
---

# Motius OmniControl HumanML3D

Self-contained OmniControl checkpoint with the HumanML3D and spatial
normalization statistics plus OpenAI CLIP ViT-B/32.

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-omnicontrol-humanml3d",
    bundle_kwargs={"device": "cuda"},
)
```

- Paper: https://arxiv.org/abs/2310.08580
- Official implementation: https://github.com/neu-vi/OmniControl
""",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "hf_artifacts" / "omnicontrol-humanml3d",
    )
    parser.add_argument("--repo-id")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_artifact(
        args.checkpoint.expanduser().resolve(),
        args.clip_weights.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    if args.upload:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --upload")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=artifact,
            commit_message="Release self-contained Motius OmniControl artifact",
        )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
