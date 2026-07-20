#!/usr/bin/env python3
"""Run any released Motius M2T pipeline on the official HumanML3D split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from motius.evaluation.m2t import (
    load_humanml3d_m2t_manifest,
    load_humanml3d_m2t_samples,
    write_humanml3d_m2t_manifest,
    write_prediction_records,
)


DEFAULT_MODELS = {
    "motiongpt": "ZeyuLing/Motius-MotionGPT-HumanML3D",
    "motiongpt3": "ZeyuLing/Motius-MotionGPT3-HumanML3D",
    "tm2t": "ZeyuLing/Motius-TM2T-HumanML3D",
    "vermo": "ZeyuLing/Motius-VerMo-HumanML3D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=tuple(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--model",
        default="",
        help="Hugging Face model id or local artifact; defaults per method.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", default="test.txt")
    parser.add_argument(
        "--protocol-manifest",
        default="",
        help="Shared HumanML3D M2T population manifest; defaults beside method outputs.",
    )
    parser.add_argument("--rebuild-protocol-manifest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--motiongpt-official-batch-padding",
        action="store_true",
        help=(
            "Reproduce MotionGPT's released batch-dependent VQ encoding. "
            "The default encodes each clip at its true length."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the protocol population into this many disjoint resumable shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard handled by this process.",
    )
    parser.add_argument(
        "--shard-mode",
        choices=("sample_stride", "batch_group"),
        default="sample_stride",
        help=(
            "Use batch_group when a model's outputs depend on official batch "
            "padding; complete batches stay intact across shards and resumes."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--smpl-model-dir",
        default="",
        help="SMPL model directory required when VerMo consumes HumanML3D-263.",
    )
    parser.add_argument("--full-motion-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_pipeline(method: str, model: str, args: argparse.Namespace):
    if method == "motiongpt":
        from motius.pipelines.motiongpt import MotionGPTPipeline

        return MotionGPTPipeline.from_pretrained(
            model,
            bundle_kwargs={
                "device": args.device,
                "local_files_only": args.local_files_only,
            },
        )
    if method == "motiongpt3":
        from motius.pipelines.motiongpt3 import MotionGPT3Pipeline

        return MotionGPT3Pipeline.from_pretrained(
            model,
            bundle_kwargs={"device": args.device},
        )
    if method == "tm2t":
        from motius.pipelines.tm2t import TM2TPipeline

        return TM2TPipeline.from_pretrained(
            model,
            bundle_kwargs={"device": args.device},
        )
    if method == "vermo":
        from motius.pipelines.vermo import VermoPipeline

        return VermoPipeline.from_pretrained(
            model,
            bundle_kwargs={"device": args.device},
            smpl_model_dir=args.smpl_model_dir or None,
        )
    raise ValueError(f"Unsupported M2T method: {method}")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    random.seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    torch.manual_seed(args.seed + args.shard_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.shard_index)

    model = args.model or DEFAULT_MODELS[args.method]
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    protocol_manifest = Path(args.protocol_manifest) if args.protocol_manifest else (
        output_dir.parent / "protocol_manifest.json"
    )
    use_manifest = not args.full_motion_only and protocol_manifest.exists()
    if use_manifest and not args.rebuild_protocol_manifest:
        samples = load_humanml3d_m2t_manifest(
            protocol_manifest, data_root=args.data_root
        )
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
    else:
        samples = load_humanml3d_m2t_samples(
            args.data_root,
            args.split_file,
            include_subclips=not args.full_motion_only,
            max_samples=args.max_samples,
        )
        if not args.full_motion_only and args.max_samples is None:
            write_humanml3d_m2t_manifest(
                samples,
                protocol_manifest,
                data_root=args.data_root,
                split_file=args.split_file,
            )
    protocol_num_samples = len(samples)
    if args.shard_mode == "batch_group":
        global_batches = [
            samples[start : start + args.batch_size]
            for start in range(0, len(samples), args.batch_size)
        ]
        selected_batches = global_batches[args.shard_index :: args.num_shards]
        samples = [sample for batch in selected_batches for sample in batch]
        batches = []
        completed = 0
        for batch in selected_batches:
            missing = sum(
                not (prediction_dir / f"{sample.sample_id}.json").exists()
                for sample in batch
            )
            if args.overwrite:
                batches.append((batch, len(batch)))
            elif missing:
                # Re-run the complete official batch so resume cannot change
                # the padding boundary after a partially written batch.
                batches.append((batch, missing))
                completed += len(batch) - missing
            else:
                completed += len(batch)
    else:
        samples = samples[args.shard_index :: args.num_shards]
        pending = [
            sample
            for sample in samples
            if args.overwrite
            or not (prediction_dir / f"{sample.sample_id}.json").exists()
        ]
        completed = len(samples) - len(pending)
        batches = [
            (pending[start : start + args.batch_size], len(pending[start : start + args.batch_size]))
            for start in range(0, len(pending), args.batch_size)
        ]
    if batches:
        pipeline = build_pipeline(args.method, model, args)
        for batch, newly_completed in batches:
            inference_kwargs = {}
            if args.method == "motiongpt":
                inference_kwargs["pad_to_batch_max"] = (
                    args.motiongpt_official_batch_padding
                )
            predictions = pipeline.infer_m2t(
                [sample.load_motion() for sample in batch],
                lengths=[sample.length for sample in batch],
                **inference_kwargs,
            )
            if len(predictions) != len(batch):
                raise RuntimeError(
                    f"{args.method} returned {len(predictions)} captions for "
                    f"{len(batch)} motions."
                )
            write_prediction_records(output_dir, zip(batch, predictions))
            completed += newly_completed
            print(
                f"[{args.method}:m2t shard "
                f"{args.shard_index + 1}/{args.num_shards}] "
                f"{completed}/{len(samples)}",
                flush=True,
            )

    manifest = {
        "task": "M2T",
        "method": args.method,
        "model": model,
        "dataset": "HumanML3D",
        "split_file": str(args.split_file),
        "include_subclips": not args.full_motion_only,
        "frame_policy": "accept_[40,200)_truncate_196",
        "reference_policy": "tm2t_first3_repeat_to3",
        "protocol_manifest": str(protocol_manifest.resolve()),
        "protocol_num_samples": protocol_num_samples,
        "num_samples": len(samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_mode": args.shard_mode,
        "motiongpt_official_batch_padding": (
            args.motiongpt_official_batch_padding
            if args.method == "motiongpt"
            else None
        ),
        "seed": args.seed,
        "predictions_dir": str(prediction_dir.resolve()),
    }
    manifest_name = (
        "manifest.json"
        if args.num_shards == 1
        else f"manifest.shard-{args.shard_index:05d}-of-{args.num_shards:05d}.json"
    )
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
