#!/usr/bin/env python3
"""Convert the official CondMDI checkpoint into a Motius Hub artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.models.condmdi.network.model_util import normalize_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Official model*.pt checkpoint")
    parser.add_argument("--args", required=True, help="Official args.json")
    parser.add_argument("--mean-abs", required=True, help="Mean_abs_3d.npy")
    parser.add_argument("--std-abs", required=True, help="Std_abs_3d.npy")
    parser.add_argument("--output", required=True, help="Output artifact directory")
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--revision", default="model_avg", choices=("model", "model_avg"))
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_config = json.loads(Path(args.args).read_text())
    config = normalize_config(raw_config)
    config.update(
        {
            "keyframe_conditioned": True,
            "keyframe_selection_scheme": "random_frames",
            "zero_keyframe_loss": False,
            "abs_3d": True,
        }
    )
    metadata = {
        "model_type": "condmdi",
        "library_name": "motius",
        "tasks": ["text-to-motion", "motion-control", "motion-in-betweening"],
        "native_motion_representation": "HumanML3D-263 absolute-root",
        "guidance_param": args.guidance,
        "respacing": "",
        "source_revision": args.revision,
        "config": config,
    }
    (output / "condmdi_config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state = checkpoint[args.revision] if args.revision in checkpoint else checkpoint
    state = {
        key: value.contiguous()
        for key, value in state.items()
        if not key.startswith("clip_model.") and "sequence_pos_encoder.pe" not in key
    }
    save_file(state, str(output / "model.safetensors"))

    mean = np.load(args.mean_abs).astype(np.float32)
    std = np.load(args.std_abs).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,) or np.any(std <= 0):
        raise ValueError(f"invalid CondMDI statistics: mean={mean.shape}, std={std.shape}")
    shutil.copyfile(args.mean_abs, output / "Mean_abs_3d.npy")
    shutil.copyfile(args.std_abs, output / "Std_abs_3d.npy")
    (output / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "CondMDIPipeline",
                "_motius_version": "0.1.0",
                "model_type": "condmdi",
                "motion_representation": "HumanML3D-263 absolute-root",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"exported {len(state)} tensors to {output}")


if __name__ == "__main__":
    main()
