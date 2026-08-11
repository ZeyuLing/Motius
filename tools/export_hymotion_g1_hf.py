#!/usr/bin/env python3
"""Export the released HYMotion-G1 checkpoint as a Motius Hub artifact.

The Accelerate checkpoint stores the transformer in an unprefixed safetensors
file and bundle-level CFG embeddings in ``model.pt``. This exporter preserves
both without loading or rewriting the 1.8 GB transformer tensor payload.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import io
import json
import os
import pickle
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np
import torch
from safetensors.torch import save_file


@dataclass(frozen=True)
class _TensorDescriptor:
    storage: tuple[Any, ...]
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor(storage, offset, size, stride, *unused):
    return _TensorDescriptor(storage, int(offset), tuple(size), tuple(stride))


class _MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, persistent_id):
        return tuple(persistent_id)

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return f"{module}.{name}"
        return super().find_class(module, name)


_STORAGE_DTYPES = {
    "torch.FloatStorage": torch.float32,
    "torch.HalfStorage": torch.float16,
    "torch.BFloat16Storage": torch.bfloat16,
}


def _read_checkpoint_tensor(
    archive: zipfile.ZipFile,
    archive_root: str,
    descriptor: _TensorDescriptor,
) -> torch.Tensor:
    kind, storage_type, storage_key, _device, _numel = descriptor.storage
    if kind != "storage" or storage_type not in _STORAGE_DTYPES:
        raise ValueError(f"Unsupported checkpoint storage: {descriptor.storage}")
    payload = bytearray(archive.read(f"{archive_root}/data/{storage_key}"))
    flat = torch.frombuffer(payload, dtype=_STORAGE_DTYPES[storage_type])
    tensor = torch.as_strided(
        flat[descriptor.offset :],
        descriptor.size,
        descriptor.stride,
    )
    return tensor.clone().contiguous()


def extract_bundle_params(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Extract small orphan tensors from an Accelerate ``model.pt`` archive."""
    with zipfile.ZipFile(checkpoint) as archive:
        pickle_name = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        archive_root = pickle_name.rsplit("/", 1)[0]
        metadata = _MetadataUnpickler(
            io.BytesIO(archive.read(pickle_name))
        ).load()
        descriptors = metadata.get("__bundle_params__", {})
        required = ("null_vtxt_feat", "null_ctxt_input")
        missing = [name for name in required if name not in descriptors]
        if missing:
            raise ValueError(f"Checkpoint is missing bundle params: {missing}")
        return {
            name: _read_checkpoint_tensor(archive, archive_root, descriptors[name])
            for name in required
        }


def _link_or_symlink(source: str, target: str) -> str:
    try:
        os.link(source, target)
        return target
    except OSError:
        os.symlink(os.path.abspath(source), target)
        return target


def _copy_tree(source: Path, target: Path, hardlink: bool = False) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        symlinks=False,
        copy_function=_link_or_symlink if hardlink else shutil.copy2,
    )


def _artifact_config() -> dict[str, Any]:
    text_encoder = {
        "type": "HYTextModel",
        "llm_type": "qwen3",
        "max_length_llm": 128,
        "sentence_emb_type": "clipl",
        "max_length_sentence_emb": 77,
        "enable_llm_padding": True,
        "torch_dtype": "bf16",
        "llm_model_path": "text_encoder/llm",
        "llm_tokenizer_path": "text_encoder/llm",
        "sentence_emb_model_path": "text_encoder/sentence",
        "sentence_emb_tokenizer_path": "text_encoder/sentence",
    }
    components = {
        "llm": {
            "type": "qwen3",
            "stored_in_artifact": True,
            "path": "text_encoder/llm",
        },
        "sentence": {
            "type": "clipl",
            "stored_in_artifact": True,
            "path": "text_encoder/sentence",
        },
    }
    return {
        "format": "motius-hymotion-t2m-v1",
        "motion_transformer": {
            "type": "HunyuanMotionMMDiT",
            "trainable": True,
            "input_dim": 38,
            "feat_dim": 1024,
            "output_dim": 38,
            "ctxt_input_dim": 4096,
            "vtxt_input_dim": 768,
            "num_layers": 18,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "mlp_act_type": "gelu_tanh",
            "norm_type": "layer",
            "qk_norm_type": "rms",
            "qkv_bias": True,
            "dropout": 0.0,
            "text_refiner_cfg": {"num_layers": 2},
            "final_layer_cfg": {"act_type": "silu"},
            "mask_mode": "narrowband",
            "apply_rope_to_single_branch": False,
            "insert_start_token": False,
            "with_long_skip_connection": False,
            "time_factor": 1000.0,
        },
        "text_encoder": text_encoder,
        "text_encoder_components": components,
        "external_text_encoder_components": components,
        "motion_type": "g1_29dof",
        "pred_type": "velocity",
        "uncondition_mode": False,
        "losses_cfg": {
            "loss_type": "smooth_l1",
            "velocity_weight": 1.0,
            "x1_weight": 0.0,
            "keypoints3d_weight": 0.0,
            "translation_weight": 0.0,
            "velocity_loss_reduction": "official_element_mean",
            "spike_downweight_enabled": False,
        },
        "noise_scheduler_cfg": {"method": "euler"},
        "infer_noise_scheduler_cfg": {"validation_steps": 50},
        "cond_mask_prob": 0.1,
        "enable_special_game_feat": False,
        "train_null_embeddings": True,
        "train_special_game_embeddings": False,
        "vtxt_input_dim": 768,
        "ctxt_input_dim": 4096,
        "body_model_path": None,
    }


def export(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    transformer_source = checkpoint / "model.safetensors"
    if not transformer_source.is_file():
        raise FileNotFoundError(transformer_source)
    copy_file = _link_or_symlink if args.hardlink else shutil.copy2
    copy_file(transformer_source, output / "motion_transformer.safetensors")

    params = extract_bundle_params(checkpoint / "model.pt")
    save_file(params, str(output / "bundle_params.safetensors"))
    shutil.copy2(Path(args.stats) / "Mean.npy", output / "Mean.npy")
    shutil.copy2(Path(args.stats) / "Std.npy", output / "Std.npy")
    _copy_tree(
        Path(args.qwen_dir),
        output / "text_encoder" / "llm",
        hardlink=args.hardlink,
    )
    _copy_tree(
        Path(args.clip_dir),
        output / "text_encoder" / "sentence",
        hardlink=args.hardlink,
    )

    weights = {
        "motion_transformer": "motion_transformer.safetensors",
        "bundle_params": "bundle_params.safetensors",
        "mean": "Mean.npy",
        "std": "Std.npy",
        "text_encoder": "text_encoder",
    }
    metadata = {
        "model_type": "hymotion_t2m",
        "format": "motius-hymotion-t2m-v1",
        "variant": "HYMotion-G1",
        "source_checkpoint": "checkpoint-iter_339000",
        "config": _artifact_config(),
        "pipeline_class": (
            "motius.pipelines.hymotion_t2m.hymotion_t2m_pipeline."
            "HyMotionT2MPipeline"
        ),
        "bundle_class": "motius.models.hymotion_t2m.bundle.HyMotionT2MBundle",
        "weights": weights,
        "components": {
            "text_encoder": _artifact_config()["text_encoder_components"]
        },
        "external_components": {},
    }
    (output / "hymotion_t2m_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    model_index = {
        "_class_name": "HyMotionT2MPipeline",
        "_library_name": "motius",
        "model_type": "hymotion_t2m",
        "format": "motius-hymotion-t2m-v1",
        "variant": "HYMotion-G1",
        "bundle_class": metadata["bundle_class"],
        "pipeline_class": metadata["pipeline_class"],
        "artifacts": weights,
        "tasks": ["text_to_motion"],
        "motion_representation": "g1_38",
        "fps": 30,
    }
    (output / "model_index.json").write_text(
        json.dumps(model_index, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        """---
library_name: motius
tags:
  - motius
  - text-to-motion
  - unitree-g1
  - robotics
---

# HYMotion-G1

This is the self-contained Motius release of the HYMotion-G1 339k checkpoint.
It includes the 0.46B motion transformer, G1-38 normalization statistics,
classifier-free guidance embeddings, Qwen3-8B, and CLIP-L.

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-HYMotion-G1",
    device="cuda",
)
result = pipe.infer_text_to_motion(
    "a robot walks forward and waves",
    num_frames=180,
)
g1_38 = result["g1_38"]
qpos_36 = result["g1_qpos"]
```

The native output is canonical `g1_38` at 30 fps. The `g1_qpos` decode is
exact and does not pass through SMPL or inverse kinematics. See the
[Motius Model Card](https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/hymotion_t2m.md)
for training, representation, and benchmark details.
""",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help=(
            "Hardlink files on the same filesystem and symlink cross-filesystem "
            "sources; use a symlink-aware uploader for the resulting folder."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    artifact = export(parse_args())
    print(artifact)
