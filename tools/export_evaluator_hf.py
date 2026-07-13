#!/usr/bin/env python3
"""Export Motius evaluator checkpoints as self-contained Hugging Face artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import types
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file


FORMAT_VERSION = 1


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_dict(state: Dict[str, Any], prefix: str = "") -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}{name}": value.detach().cpu().contiguous()
        for name, value in state.items()
        if torch.is_tensor(value)
    }


def _copy_files(source: Path, target: Path, names: Iterable[str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, target / name)


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _write_manifest(output_dir: Path, source_files: Iterable[Path]) -> None:
    files = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files.append({
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    sources = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in source_files
    ]
    _write_json(
        output_dir / "artifact_manifest.json",
        {"format_version": FORMAT_VERSION, "files": files, "source_files": sources},
    )


def _push(output_dir: Path, repo_id: str | None) -> None:
    if not repo_id:
        return
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Publish Motius evaluator artifact",
    )


HUMANML3D_README = """---
library_name: motius
pipeline_tag: feature-extraction
license: mit
tags:
- human-motion
- text-to-motion
- evaluator
- humanml3d
---

# HumanML3D Official Evaluator

Hugging Face artifact for the official HumanML3D text-motion matching evaluator
used by the T2M leaderboard protocol. Motius repackages the released inference
weights without retraining and removes optimizer state.

## Provenance

- Paper: [Generating Diverse and Natural 3D Human Motions From Text](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html)
- Original repository: [EricGuo5513/text-to-motion](https://github.com/EricGuo5513/text-to-motion)
- Motius implementation: `HumanML263Evaluator`

## Artifact layout

- `model.safetensors`: movement, text, and motion encoders
- `stats/`: HumanML3D-263 normalization statistics
- `glove/`: released `our_vab` word-vector lookup files
- `config.json` and `preprocessor_config.json`: architecture and protocol
- `artifact_manifest.json`: file and source-checkpoint hashes

The HumanML3D dataset and its annotations remain subject to their original terms.

## Download

```python
from huggingface_hub import snapshot_download

artifact = snapshot_download("ZeyuLing/motius-evaluator-humanml3d-official")
```
"""


MOTIONSTREAMER_README = """---
library_name: motius
pipeline_tag: feature-extraction
license: mit
tags:
- human-motion
- text-to-motion
- evaluator
- motionstreamer
---

# MotionStreamer Evaluator

Hugging Face artifact for the TMR-style evaluator released with MotionStreamer.
Motius extracts only the DistilBERT text encoder and ACTOR motion encoder used
for MotionStreamer-272 evaluation; Lightning trainer state is excluded.

## Provenance

- Paper: [MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](https://arxiv.org/abs/2503.15451)
- Original repository: [zju3dv/MotionStreamer](https://github.com/zju3dv/MotionStreamer)
- Motius implementation: `MotionStreamer272Evaluator`

## Artifact layout

- `model.safetensors`: text and motion evaluator encoders
- `stats/`: MotionStreamer-272 normalization statistics
- `tokenizer/`: DistilBERT tokenizer/config files
- `config.json` and `preprocessor_config.json`: architecture and protocol
- `artifact_manifest.json`: file and source-checkpoint hashes

## Download

```python
from huggingface_hub import snapshot_download

artifact = snapshot_download("ZeyuLing/motius-evaluator-motionstreamer-272")
```
"""


UNIVERSAL_TMR_README = """---
library_name: motius
pipeline_tag: feature-extraction
license: mit
tags:
- human-motion
- text-to-motion
- evaluator
- tmr
- smpl
- smpl-22
---

# Motius Universal SMPL-22 Joints66 Evaluator

Motius reproduction of the TMR architecture, trained from scratch on the full
HYMotion Data SFT set and the full single-person MotionHub training union. Motion
is represented as canonicalized SMPL-22 body-joint positions (22 joints in xyz)
at 30 fps. The tensors are materialized with a neutral SMPL-H model, with hand
articulation excluded. This is a Motius-trained checkpoint, not an official TMR
checkpoint.

## Provenance

- Architecture paper: [TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis](https://arxiv.org/abs/2305.00976)
- Original architecture repository: [Mathux/TMR](https://github.com/Mathux/TMR)
- Motius implementation: `TMRBundle` / `TMRG1Evaluator`

## Artifact layout

- `model.safetensors`: motion encoder, text encoder, and reconstruction decoder
- `stats/`: training-set joints66 normalization statistics
- `config.json` and `preprocessor_config.json`: architecture and protocol
- `artifact_manifest.json`: file and source-checkpoint hashes

The public checkpoint is epoch 248 from the corrected canonicalization, caption,
training-data, and normalization run used by the Motius T2M leaderboard.

## Download

```python
from huggingface_hub import snapshot_download

artifact = snapshot_download("ZeyuLing/motius-evaluator-universal-smplh-joints66")
```
"""


G1_TMR_README = """---
library_name: motius
pipeline_tag: feature-extraction
license: mit
tags:
- robotics
- human-motion
- text-to-motion
- evaluator
- tmr
- unitree-g1
---

# Motius TMR-G1 Evaluator

Motius reproduction of the TMR architecture trained from scratch for native
Unitree G1 motion. The motion input is the canonicalized G1-38D representation
at 30 fps: root XY velocity and height, root rotation 6D, and 29 joint angles.

## Provenance

- Architecture paper: [TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis](https://arxiv.org/abs/2305.00976)
- Original architecture repository: [Mathux/TMR](https://github.com/Mathux/TMR)
- Motius implementation: `TMRBundle` / `TMRG1Evaluator`

## Artifact layout

- `model.safetensors`: motion encoder, text encoder, and reconstruction decoder
- `stats/`: G1-38D training-set normalization statistics
- `config.json` and `preprocessor_config.json`: architecture and protocol
- `artifact_manifest.json`: file and source-checkpoint hashes

The public checkpoint is epoch 139, trained on the 344,966-motion training split
of the 359,153-clip HYMotion Data G1 materialization.

## Download

```python
from motius.evaluation import TMRG1Evaluator

evaluator = TMRG1Evaluator.from_pretrained(
    "ZeyuLing/motius-evaluator-g1-38d-tmr"
)
```
"""


INTERCLIP_README = """---
library_name: motius
pipeline_tag: feature-extraction
license: cc-by-nc-sa-4.0
tags:
- human-motion
- two-person-text-to-motion
- evaluator
- interhuman
- interclip
---

# InterCLIP InterHuman-262 Evaluator

Inference-only artifact for the InterCLIP evaluator released with InterGen.
It embeds a caption and two synchronized InterHuman-262 motion tracks, and is
the standard evaluator used by InterGen and InterMask on InterHuman.

## Provenance

- Paper: [InterGen: Diffusion-based Multi-human Motion Generation under Complex Interactions](https://arxiv.org/abs/2304.05684)
- Original repository: [tr3e/InterGen](https://github.com/tr3e/InterGen)
- Motius implementation: `InterHuman262Evaluator`
- License: CC BY-NC-SA 4.0, following the official repository

The artifact contains only `model.safetensors` and protocol metadata. Optimizer,
trainer, callback, and Lightning state are excluded. OpenAI CLIP is used only
for tokenization; all learned InterCLIP text and motion weights are contained in
the SafeTensors file.

```python
from motius.evaluation.evaluators import InterHuman262Evaluator

evaluator = InterHuman262Evaluator.from_pretrained(
    "ZeyuLing/motius-evaluator-interhuman-interclip",
    device="cuda",
)
metrics = evaluator.evaluate_npz("gt.npz", {"InterGen": "pred.npz"})
```
"""


def export_humanml3d(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _prepare_output(output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensors: Dict[str, torch.Tensor] = {}
    for key in ("movement_encoder", "text_encoder", "motion_encoder"):
        tensors.update(_tensor_dict(checkpoint[key], f"{key}."))
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    _copy_files(args.glove_dir, output_dir / "glove", (
        "our_vab_data.npy", "our_vab_idx.pkl", "our_vab_words.pkl",
    ))
    _copy_files(args.mean.parent, output_dir / "stats", (args.mean.name, args.std.name))
    (output_dir / "stats" / args.mean.name).rename(output_dir / "stats" / "mean.npy")
    (output_dir / "stats" / args.std.name).rename(output_dir / "stats" / "std.npy")
    config = {
        "architectures": ["HumanML263Evaluator"],
        "format_version": FORMAT_VERSION,
        "library_name": "motius",
        "model_type": "motius-humanml3d-263-evaluator",
        "weights_file": "model.safetensors",
        "motion_nfeats": 263,
        "movement_latent_dim": 512,
        "text_hidden_dim": 512,
        "motion_hidden_dim": 1024,
        "embedding_dim": 512,
        "source": {
            "paper": "https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html",
            "repository": "https://github.com/EricGuo5513/text-to-motion",
            "checkpoint_retrained": False,
        },
    }
    preprocessor = {
        "motion_representation": "HumanML3D-263",
        "fps": 20,
        "max_motion_length": 196,
        "unit_length": 4,
        "max_text_length": 20,
        "caption_protocol": "HumanML3D selected caption",
        "mean": "stats/mean.npy",
        "std": "stats/std.npy",
        "word_vectorizer": "glove/our_vab",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "preprocessor_config.json", preprocessor)
    (output_dir / "README.md").write_text(HUMANML3D_README)
    _write_manifest(output_dir, (args.checkpoint, args.mean, args.std))
    _push(output_dir, args.repo_id)


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _tolerant_pickle_module():
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    shim = types.ModuleType("motius_tolerant_pickle")
    shim.Unpickler = _Unpickler
    shim.load = lambda handle, **kwargs: _Unpickler(handle, **kwargs).load()
    shim.loads = pickle.loads
    shim.Pickler = pickle.Pickler
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    return shim


def export_motionstreamer(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _prepare_output(output_dir)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        pickle_module=_tolerant_pickle_module(),
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    tensors: Dict[str, torch.Tensor] = {}
    for source_prefix, target_prefix in (
        ("textencoder.", "text_encoder."),
        ("motionencoder.", "motion_encoder."),
    ):
        selected = {
            name[len(source_prefix):]: value
            for name, value in state.items()
            if name.startswith(source_prefix)
        }
        tensors.update(_tensor_dict(selected, target_prefix))
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    _copy_files(args.mean.parent, output_dir / "stats", (args.mean.name, args.std.name))
    (output_dir / "stats" / args.mean.name).rename(output_dir / "stats" / "mean.npy")
    (output_dir / "stats" / args.std.name).rename(output_dir / "stats" / "std.npy")
    tokenizer_files = [
        name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
        if (args.tokenizer_dir / name).is_file()
    ]
    _copy_files(args.tokenizer_dir, output_dir / "tokenizer", tokenizer_files)
    config = {
        "architectures": ["MotionStreamer272Evaluator"],
        "format_version": FORMAT_VERSION,
        "library_name": "motius",
        "model_type": "motius-motionstreamer-272-evaluator",
        "weights_file": "model.safetensors",
        "motion_nfeats": 272,
        "latent_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "text_backbone": "distilbert-base-uncased",
        "source": {
            "paper": "https://arxiv.org/abs/2503.15451",
            "repository": "https://github.com/zju3dv/MotionStreamer",
            "checkpoint_retrained": False,
        },
    }
    preprocessor = {
        "motion_representation": "MotionStreamer-272",
        "fps": 30,
        "max_motion_length": 300,
        "unit_length": 4,
        "caption_protocol": "HumanML3D selected caption",
        "mean": "stats/mean.npy",
        "std": "stats/std.npy",
        "tokenizer": "tokenizer",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "preprocessor_config.json", preprocessor)
    (output_dir / "README.md").write_text(MOTIONSTREAMER_README)
    _write_manifest(output_dir, (args.checkpoint, args.mean, args.std))
    _push(output_dir, args.repo_id)


def export_universal_tmr(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _prepare_output(output_dir)
    tensors = {name: value.contiguous() for name, value in load_file(args.checkpoint).items()}
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    _copy_files(args.mean.parent, output_dir / "stats", (args.mean.name, args.std.name))
    (output_dir / "stats" / args.mean.name).rename(output_dir / "stats" / "mean.npy")
    (output_dir / "stats" / args.std.name).rename(output_dir / "stats" / "std.npy")
    config = {
        "architectures": ["TMRBundle"],
        "format_version": FORMAT_VERSION,
        "library_name": "motius",
        "model_type": "motius-universal-smpl22-joints66-tmr",
        "weights_file": "model.safetensors",
        "motion_nfeats": 66,
        "text_nfeats": 768,
        "vae": True,
        "arch": {
            "latent_dim": 256,
            "ff_size": 1024,
            "num_layers": 6,
            "num_heads": 4,
            "dropout": 0.1,
            "activation": "gelu",
        },
        "training": {
            "checkpoint_epoch": 248,
            "data": "full HYMotion Data SFT + full single-person MotionHub training union",
            "precision": "fp32",
        },
        "source": {
            "paper": "https://arxiv.org/abs/2305.00976",
            "repository": "https://github.com/Mathux/TMR",
            "checkpoint_retrained": True,
        },
    }
    preprocessor = {
        "motion_representation": "canonicalized SMPL-22 joints66 (22 xyz joints)",
        "fk_implementation": "neutral SMPL-H with hand articulation excluded",
        "fps": 30,
        "min_seconds": 0.5,
        "max_seconds": 20.0,
        "token_model": "distilbert-base-uncased",
        "sentence_model": "sentence-transformers/all-mpnet-base-v2",
        "humanml3d_caption_protocol": "selected caption",
        "motionhub_caption_protocol": "official test annotations",
        "mean": "stats/mean.npy",
        "std": "stats/std.npy",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "preprocessor_config.json", preprocessor)
    (output_dir / "README.md").write_text(UNIVERSAL_TMR_README)
    _write_manifest(output_dir, (args.checkpoint, args.mean, args.std))
    _push(output_dir, args.repo_id)


def export_g1_tmr(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _prepare_output(output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "tmr" not in checkpoint:
        raise KeyError(f"Expected a 'tmr' state dict in {args.checkpoint}")
    tensors = _tensor_dict(checkpoint["tmr"], prefix="tmr.")
    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})
    _copy_files(args.mean.parent, output_dir / "stats", (args.mean.name, args.std.name))
    (output_dir / "stats" / args.mean.name).rename(output_dir / "stats" / "mean.npy")
    (output_dir / "stats" / args.std.name).rename(output_dir / "stats" / "std.npy")
    config = {
        "architectures": ["TMRBundle"],
        "format_version": FORMAT_VERSION,
        "library_name": "motius",
        "model_type": "motius-g1-38d-tmr",
        "weights_file": "model.safetensors",
        "motion_nfeats": 38,
        "text_nfeats": 768,
        "vae": True,
        "arch": {
            "latent_dim": 256,
            "ff_size": 1024,
            "num_layers": 6,
            "num_heads": 4,
            "dropout": 0.1,
            "activation": "gelu",
        },
        "training": {
            "checkpoint_epoch": 139,
            "data": "HYMotion Data G1 materialization",
            "total_clips": 359153,
            "train_clips": 344966,
            "validation_clips": 7083,
            "test_clips": 7104,
        },
        "source": {
            "paper": "https://arxiv.org/abs/2305.00976",
            "repository": "https://github.com/Mathux/TMR",
            "checkpoint_retrained": True,
        },
    }
    preprocessor = {
        "motion_representation": "canonicalized Unitree G1-38D",
        "layout": "root XY velocity, root Z height, root rotation 6D, 29 joint angles",
        "rotation_6d": "first two rotation-matrix columns flattened row-wise",
        "canonicalization": "frame-0 ground origin and zero heading",
        "root_velocity": True,
        "fps": 30,
        "min_seconds": 0.5,
        "max_seconds": 120.0,
        "token_model": "distilbert-base-uncased",
        "sentence_model": "sentence-transformers/all-mpnet-base-v2",
        "mean": "stats/mean.npy",
        "std": "stats/std.npy",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "preprocessor_config.json", preprocessor)
    (output_dir / "README.md").write_text(G1_TMR_README)
    _write_manifest(output_dir, (args.checkpoint, args.mean, args.std))
    _push(output_dir, args.repo_id)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mean", type=Path, required=True)
    parser.add_argument("--std", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=None)


def export_interclip(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _prepare_output(output_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {
        (name.replace("model.", "", 1) if name.startswith("model.") else name): value.detach().cpu().contiguous()
        for name, value in raw_state.items()
        if torch.is_tensor(value)
    }
    save_file(state, output_dir / "model.safetensors", metadata={"format": "pt"})
    config = {
        "model_type": "interclip",
        "library_name": "motius",
        "input_dim_per_person": 262,
        "encoded_dim_per_person": 258,
        "embedding_dim": 512,
        "motion_latent_dim": 1024,
        "motion_layers": 8,
        "motion_heads": 8,
        "text_width": 768,
        "text_layers": 8,
        "text_heads": 8,
    }
    protocol = {
        "motion_representation": "InterHuman-262 per person",
        "pair_shape": "(B, T, 2, 262)",
        "fps": 30,
        "retrieval_batch_size": 96,
        "retrieval_repeats": 20,
        "embedding_scale": 6.0,
        "metrics": ["R-Precision", "MM-Dist", "FID", "Diversity"],
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "preprocessor_config.json", protocol)
    (output_dir / "README.md").write_text(INTERCLIP_README)
    _write_manifest(output_dir, (args.checkpoint,))
    _push(output_dir, args.repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    humanml3d = subparsers.add_parser("humanml3d")
    _add_common(humanml3d)
    humanml3d.add_argument("--glove-dir", type=Path, required=True)
    humanml3d.set_defaults(func=export_humanml3d)

    motionstreamer = subparsers.add_parser("motionstreamer")
    _add_common(motionstreamer)
    motionstreamer.add_argument("--tokenizer-dir", type=Path, required=True)
    motionstreamer.set_defaults(func=export_motionstreamer)

    universal = subparsers.add_parser("universal-tmr")
    _add_common(universal)
    universal.set_defaults(func=export_universal_tmr)

    g1 = subparsers.add_parser("g1-tmr")
    _add_common(g1)
    g1.set_defaults(func=export_g1_tmr)

    interclip = subparsers.add_parser("interclip")
    interclip.add_argument("--checkpoint", type=Path, required=True)
    interclip.add_argument("--output-dir", type=Path, required=True)
    interclip.add_argument("--repo-id", default=None)
    interclip.set_defaults(func=export_interclip)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
