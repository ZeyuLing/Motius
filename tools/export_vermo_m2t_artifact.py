#!/usr/bin/env python3
"""Package a trained VerMo M2T checkpoint as a self-contained HF artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-safetensors", required=True)
    parser.add_argument("--base-lm", required=True)
    parser.add_argument("--motion-tokenizer", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--link-large-files",
        action="store_true",
        help="Symlink large weights for a local smoke artifact instead of copying them.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    return parser.parse_args()


def copy_files(source: Path, target: Path, names) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)


def transfer_large(source: Path, target: Path, link: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if link:
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def infer_vocab_size(checkpoint: Path) -> int:
    """Read the expanded VerMo vocabulary directly from the trained weights."""

    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        candidates = (
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
        )
        for name in candidates:
            if name in tensors.keys():
                return int(tensors.get_slice(name).get_shape()[0])
    raise KeyError(
        "Could not find an embedding matrix in the VerMo model checkpoint."
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    lm_dir = output / "lm"
    tokenizer_dir = output / "tokenizer"
    motion_dir = output / "motion_tokenizer"
    stats_dir = output / "stats"
    output.mkdir(parents=True, exist_ok=True)

    base_lm = Path(args.base_lm).expanduser().resolve()
    copy_files(
        base_lm,
        lm_dir,
        ("config.json", "generation_config.json"),
    )
    copy_files(
        base_lm,
        tokenizer_dir,
        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
    )
    model_safetensors = Path(args.model_safetensors).expanduser().resolve()
    transfer_large(
        model_safetensors,
        lm_dir / "model.safetensors",
        args.link_large_files,
    )
    config_path = lm_dir / "config.json"
    lm_config = json.loads(config_path.read_text(encoding="utf-8"))
    lm_config["vocab_size"] = infer_vocab_size(model_safetensors)
    config_path.write_text(
        json.dumps(lm_config, indent=2) + "\n", encoding="utf-8"
    )
    copy_files(
        Path(args.motion_tokenizer).expanduser().resolve(),
        motion_dir,
        ("config.json",),
    )
    transfer_large(
        Path(args.motion_tokenizer).expanduser().resolve()
        / "diffusion_pytorch_model.safetensors",
        motion_dir / "diffusion_pytorch_model.safetensors",
        args.link_large_files,
    )
    stats_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.stats), stats_dir / "smplh_universal_stats_aug.json")

    pose_processor = {
        "type": "VermoSMPL22Processor",
        "do_normalize": True,
        "stats_file": "stats/smplh_universal_stats_aug.json",
        "rot_type": "rotation_6d",
        "transl_type": "abs_rel",
        "smpl_type": "smpl_22",
    }
    bundle_config = {
        "processor": {
            "type": "VermoProcessor",
            "trainable": False,
            "save_ckpt": False,
            "module_dtype": "fp32",
            "pretrained_text_tokenizer": {
                "type": "AutoTokenizer",
                "from_pretrained": {
                    "pretrained_model_name_or_path": "tokenizer"
                },
            },
            "smpl_pose_processor": pose_processor,
            "multi_person_smpl_pose_processor": dict(pose_processor),
            "motion_tokenizer": {
                "type": "VQVAEWanMotion2DTK",
                "from_pretrained": {
                    "pretrained_model_name_or_path": "motion_tokenizer"
                },
            },
            "audio_tokenizer": None,
            "audio_codebook_size": 4096,
            "instruction_stage": True,
            "optional_input_modal_mode": "none",
            "task_template_mode": "first",
            "shuffle_condition_parts": False,
            "shuffle_output_parts": False,
            "max_seq_len": 0,
        },
        "lm": {
            "type": "VermoLlamaForCausalLM",
            "trainable": False,
            "save_ckpt": False,
            "module_dtype": args.torch_dtype,
            "from_pretrained": {
                "pretrained_model_name_or_path": "lm",
                "torch_dtype": args.torch_dtype,
                "attn_implementation": "sdpa",
            },
        },
        "mean_init_embeddings": False,
    }
    (output / "bundle_config.json").write_text(
        json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": "motius_vermo_bundle_v1",
        "task": "M2T",
        "motion_representation": "vermo_smpl22_abs_rel_rot6d_column_138",
        "model_safetensors": str(model_safetensors),
        "vocab_size": lm_config["vocab_size"],
        "torch_dtype": args.torch_dtype,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
