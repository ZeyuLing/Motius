#!/usr/bin/env python3
"""Run the Motius EDGE pipeline on one or more music tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from motius.models.edge.audio import extract_edge_jukebox_features
from motius.pipelines.edge import EDGEPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", type=Path, nargs="+")
    parser.add_argument("--music-features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--jukebox-fp16", action="store_true")
    parser.add_argument("--jukebox-cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.audio) == bool(args.music_features):
        parser.error("Provide --audio or --music-features, not both")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "music_features"
    feature_dir.mkdir(exist_ok=True)
    pipeline = EDGEPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": Path(args.checkpoint).exists()},
        device=args.device,
    )
    inputs = args.audio or args.music_features
    completed = []
    started = time.time()
    for index, path in enumerate(inputs):
        path = path.expanduser().resolve()
        name = path.stem
        output_path = args.output / f"{name}.npz"
        if output_path.is_file() and not args.overwrite:
            completed.append(name)
            print(f"[{index + 1}/{len(inputs)}] skip {name}", flush=True)
            continue
        case_started = time.time()
        feature_path = feature_dir / f"{name}.npy"
        if args.audio:
            if feature_path.is_file() and not args.overwrite:
                features = np.load(feature_path, allow_pickle=False)
            else:
                features = extract_edge_jukebox_features(
                    path,
                    max_seconds=args.max_seconds,
                    fp16=args.jukebox_fp16,
                    cache_dir=args.jukebox_cache_dir,
                )
                np.save(feature_path, features)
        else:
            features = np.load(path, allow_pickle=False)
        result = pipeline(
            music_features=features,
            max_frames=args.max_frames,
            seed=args.seed + index,
            sampling_steps=args.sampling_steps,
            guidance_weight=args.guidance_weight,
            eta=args.eta,
        )
        np.savez_compressed(
            output_path,
            joints=result.joints,
            edge_motion=result.edge_motion,
            motion_135=result.motion_135,
            contacts=result.contacts,
            fps=np.float32(result.fps),
            audio=np.asarray(str(path) if args.audio else ""),
            music_features=np.asarray(str(feature_path if args.audio else path)),
            seed=np.int64(args.seed + index),
        )
        completed.append(name)
        print(
            f"[{index + 1}/{len(inputs)}] {name} frames={len(result.joints)} "
            f"seconds={time.time() - case_started:.2f}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "task": "music_to_dance",
        "method": "EDGE",
        "checkpoint": args.checkpoint,
        "representation": "EDGE-151 -> AIST++ SMPL-24 joints",
        "motion_fps": pipeline.bundle.fps,
        "music_features": "Jukebox layer 66, 4800-D at 30 fps",
        "seed": args.seed,
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
