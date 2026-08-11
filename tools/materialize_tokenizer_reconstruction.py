#!/usr/bin/env python3
"""Reconstruct HumanML3D-263 clips with a Motius tokenizer pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


PIPELINES = {
    "t2mgpt": (
        "motius.pipelines.t2mgpt",
        "T2MGPTPipeline",
        "ZeyuLing/Motius-T2M-GPT-HumanML3D",
    ),
    "momask": (
        "motius.pipelines.momask",
        "MoMaskPipeline",
        "ZeyuLing/Motius-MoMask-HumanML3D",
    ),
    "motiongpt": (
        "motius.pipelines.motiongpt",
        "MotionGPTPipeline",
        "ZeyuLing/Motius-MotionGPT-HumanML3D",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(PIPELINES))
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", default="checkpoints/body_models/smpl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--io-workers", type=int, default=32)
    parser.add_argument("--io-chunk-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reconstruction-only", action="store_true")
    mode.add_argument("--smpl-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.io_chunk_size < 1:
        parser.error("io-chunk-size must be positive")
    if args.reconstruction_only and args.num_shards != 1:
        parser.error("reconstruction-only runs once; shard only --smpl-only")
    return args


def load_pipeline(method: str, checkpoint: str, device: str):
    import importlib

    module_name, class_name, _ = PIPELINES[method]
    pipeline_class = getattr(importlib.import_module(module_name), class_name)
    bundle_kwargs = {"device": device}
    if method == "t2mgpt":
        bundle_kwargs["load_clip"] = False
    return pipeline_class.from_pretrained(
        checkpoint,
        bundle_kwargs=bundle_kwargs,
        device=device,
    )


def main() -> None:
    args = parse_args()
    _, _, default_checkpoint = PIPELINES[args.method]
    checkpoint = args.checkpoint or default_checkpoint
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    hml_dir = output_dir / "hml263"
    smpl_dir = output_dir / "motion135"
    hml_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(source_dir.glob("*.npy"))
    if args.max_samples:
        paths = paths[: args.max_samples]
    started = time.time()
    workers = max(1, int(args.io_workers))
    if not args.smpl_only:
        reconstruction_paths = paths
        if args.skip_existing:
            reconstruction_paths = [
                path
                for path in reconstruction_paths
                if not (hml_dir / path.name).is_file()
            ]
        pipeline = load_pipeline(args.method, checkpoint, args.device)
        completed = 0
        for start in range(0, len(reconstruction_paths), args.io_chunk_size):
            chunk_paths = reconstruction_paths[start : start + args.io_chunk_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                motions = list(
                    executor.map(
                        lambda path: np.load(path).astype(np.float32), chunk_paths
                    )
                )
            reconstructed = pipeline.infer_motion_reconstruction(
                motions,
                batch_size=args.batch_size,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(
                    executor.map(
                        lambda item: np.save(hml_dir / item[0].name, item[1]),
                        zip(chunk_paths, reconstructed),
                    )
                )
            completed += len(chunk_paths)
            print(
                f"[{args.method}:vq] {completed}/{len(reconstruction_paths)} "
                f"elapsed={(time.time() - started) / 60:.1f}m",
                flush=True,
            )
        if args.reconstruction_only:
            (output_dir / "reconstruction_provenance.json").write_text(
                json.dumps(
                    {
                        "method": args.method,
                        "checkpoint": checkpoint,
                        "source_dir": str(source_dir),
                        "population": len(paths),
                        "motion_input": "HumanML3D-263 physical scale at 20 fps",
                        "reconstruction_api": "infer_motion_reconstruction",
                        "elapsed_seconds": time.time() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
            return

    paths = [path for path in paths if (hml_dir / path.name).is_file()]
    paths = [
        path
        for index, path in enumerate(paths)
        if index % args.num_shards == args.shard_index
    ]
    if args.skip_existing:
        paths = [
            path
            for path in paths
            if not (smpl_dir / f"{path.stem}.npz").is_file()
        ]
    if not paths:
        print("No pending SMPL conversion clips", flush=True)
        return

    smpl_rest = load_smpl_rest(args.model_dir, args.device)

    for index, path in enumerate(paths, start=1):
        motion = np.load(hml_dir / path.name).astype(np.float32)
        converted = retarget_hml263_clip(
            motion,
            smpl_rest=smpl_rest,
            device=args.device,
            source_fps=20.0,
            target_fps=30.0,
            floor_align=True,
            refine_iters=0,
            rotation_init="position_ik",
        )
        np.savez_compressed(smpl_dir / f"{path.stem}.npz", **converted)
        if index % 100 == 0 or index == len(paths):
            elapsed = time.time() - started
            print(
                f"[{args.method}:smpl {args.shard_index}/{args.num_shards}] "
                f"{index}/{len(paths)} "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    provenance_name = (
        f"smpl_provenance_shard_{args.shard_index:02d}.json"
        if args.num_shards > 1
        else "provenance.json"
    )
    (output_dir / provenance_name).write_text(
        json.dumps(
            {
                "method": args.method,
                "checkpoint": checkpoint,
                "source_dir": str(source_dir),
                "population": len(paths),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "motion_input": "HumanML3D-263 physical scale at 20 fps",
                "reconstruction_api": "infer_motion_reconstruction",
                "smpl_bridge": "Motius position IK, SMPL-22, 30 fps",
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
