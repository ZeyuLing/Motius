#!/usr/bin/env python3
"""Run the Motius Bailando pipeline on the official AIST++ evaluation set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from motius.datasets.aistpp_music_to_dance import AISTPPMusicDanceDataset
from motius.models.bailando.bundle import BailandoBundle
from motius.pipelines.bailando import BailandoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--music-feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--vqvae", type=Path)
    parser.add_argument("--gpt", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.vqvae or args.gpt):
        parser.error("Use --checkpoint or both --vqvae and --gpt")
    if not args.checkpoint and (args.vqvae is None or args.gpt is None):
        parser.error("Both --vqvae and --gpt are required for official checkpoints")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = AISTPPMusicDanceDataset(
        args.data_root,
        args.music_feature_root,
        max_samples=args.max_samples,
    )
    if args.checkpoint:
        pipeline = BailandoPipeline.from_pretrained(args.checkpoint, device=args.device)
    else:
        bundle = BailandoBundle(
            vqvae_weights=str(args.vqvae),
            gpt_weights=str(args.gpt),
            strict=True,
        )
        pipeline = BailandoPipeline(bundle, device=args.device)

    completed = []
    started = time.time()
    for index, sample in enumerate(dataset):
        output_path = args.output / f"{sample['name']}.npz"
        if output_path.is_file() and not args.overwrite:
            completed.append(sample["name"])
            print(f"[{index + 1}/{len(dataset)}] skip {sample['name']}", flush=True)
            continue
        case_started = time.time()
        result = pipeline(
            music_features=sample["music_features"],
            initial_motion=sample["gt_joints"],
            max_frames=args.max_frames,
        )
        np.savez_compressed(
            output_path,
            joints=result.joints[0],
            model_motion=result.model_motion[0],
            codes_up=result.codes_up[0],
            codes_down=result.codes_down[0],
            fps=np.float32(result.fps),
            music_id=np.asarray(sample["music_id"]),
        )
        completed.append(sample["name"])
        print(
            f"[{index + 1}/{len(dataset)}] {sample['name']} "
            f"frames={len(result.joints[0])} seconds={time.time() - case_started:.2f}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "task": "music_to_dance",
        "method": "Bailando",
        "dataset": "AIST++ official crossmodal test+validation package",
        "checkpoint": args.checkpoint or "official .pt checkpoints",
        "initialization": "first GT VQ token",
        "motion_fps": 60.0,
        "music_feature_fps": 7.5,
        "num_samples": len(completed),
        "samples": completed,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
