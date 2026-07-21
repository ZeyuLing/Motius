#!/usr/bin/env python3
"""Run TM2D music-only inference on the AIST++ leaderboard package."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.datasets.aistpp_music_to_dance import AISTPPMusicDanceDataset
from motius.pipelines.tm2d import TM2DPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--music-feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = AISTPPMusicDanceDataset(
        args.data_root, args.music_feature_root, max_samples=args.max_samples
    )
    pipeline = TM2DPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": Path(args.checkpoint).exists()},
        device=args.device,
    )
    completed = []
    started = time.time()
    for index in range(len(dataset)):
        if index % args.num_shards != args.shard_index:
            continue
        sample = dataset[index]
        name = sample["name"]
        destination = args.output / f"{name}.npz"
        if destination.is_file() and not args.overwrite:
            completed.append(name)
            print(f"skip {name}", flush=True)
            continue
        case_seed = args.seed + index
        random_seed_token = int(
            np.random.default_rng(case_seed).integers(
                0, pipeline.bundle.config["codebook_size"]
            )
        )
        case_started = time.time()
        result = pipeline.infer_music_to_dance(
            music_features=sample["music_features"],
            initial_token=random_seed_token,
            sample=not args.greedy,
            top_k=args.top_k,
            seed=case_seed,
            max_frames=len(sample["gt_joints"]),
        )
        np.savez_compressed(
            destination,
            joints=result.joints,
            tm2d_motion=result.model_motion,
            motion_tokens=result.motion_tokens,
            fps=np.float32(result.fps),
            music_id=np.asarray(sample["music_id"]),
            initial_token=np.int64(random_seed_token),
            seed=np.int64(case_seed),
        )
        completed.append(name)
        print(
            f"[{index + 1}/{len(dataset)}] {name} frames={len(result.joints)} "
            f"seconds={time.time() - case_started:.2f}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "task": "music_to_dance",
        "method": "TM2D",
        "dataset": "AIST++ official crossmodal test+validation package",
        "checkpoint": args.checkpoint,
        "representation": "TM2D HumanML-24 287-D at 60 fps",
        "initialization": "deterministic reference-free random VQ token",
        "sampling": "greedy" if args.greedy else "categorical",
        "top_k": args.top_k,
        "seed": args.seed,
        "num_samples": len(completed),
        "samples": completed,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / f"manifest_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
