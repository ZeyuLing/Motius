#!/usr/bin/env python3
"""Run TM2D text-only inference on the selected-caption HumanML3D protocol."""

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

from motius.pipelines.tm2d import TM2DPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--path-root",
        type=Path,
        help="Root used to resolve annotation caption paths.",
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        help="Optional JSON mapping motion ids to pretokenized caption words.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


def selected_caption(path: Path) -> str:
    payload = json.loads(path.read_text())
    for level in ("macro", "meso", "micro"):
        values = payload.get(level)
        if values:
            caption = str(values[0]).strip()
            if caption:
                return caption
    raise ValueError(f"No selected caption in {path}")


def main():
    args = parse_args()
    protocol = json.loads(args.annotation.read_text())
    token_cache = (
        json.loads(args.token_cache.read_text()) if args.token_cache is not None else {}
    )
    records = sorted(protocol["data_list"].items())
    records = [
        (index, name, record)
        for index, (name, record) in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    joints_dir = args.output / "joints66"
    native_dir = args.output / "tm2d287"
    token_dir = args.output / "tokens"
    for path in (joints_dir, native_dir, token_dir):
        path.mkdir(parents=True, exist_ok=True)
    pipeline = TM2DPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": Path(args.checkpoint).exists()},
        device=args.device,
    )

    completed = []
    started = time.time()
    for local_index, (global_index, name, record) in enumerate(records):
        destination = joints_dir / f"{name}.npy"
        if destination.is_file() and not args.overwrite:
            completed.append(name)
            print(f"[{local_index + 1}/{len(records)}] skip {name}", flush=True)
            continue
        caption_path = Path(record["hierarchical_caption_path"])
        if not caption_path.is_absolute():
            root = args.path_root or args.annotation.resolve().parents[1]
            caption_path = root / caption_path
        caption = selected_caption(caption_path)
        pretokenized = token_cache.get(name)
        if args.token_cache is not None and pretokenized is None:
            raise KeyError(f"Missing {name!r} in token cache {args.token_cache}")
        case_started = time.time()
        result = pipeline.infer_text_to_motion(
            caption,
            num_frames=int(record["num_frames"]),
            output_fps=float(record["fps"]),
            pretokenized=pretokenized,
            sample=not args.greedy,
            top_k=args.top_k,
            seed=args.seed + global_index,
        )
        np.save(destination, result.joints[:, :22].reshape(len(result.joints), 66))
        np.save(native_dir / f"{name}.npy", result.model_motion)
        np.save(token_dir / f"{name}.npy", result.motion_tokens)
        completed.append(name)
        print(
            f"[{local_index + 1}/{len(records)}] {name} "
            f"frames={len(result.joints)} seconds={time.time() - case_started:.2f}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "task": "text_to_motion",
        "method": "TM2D",
        "dataset": "HumanML3D official test, selected-caption protocol",
        "checkpoint": args.checkpoint,
        "representation": "TM2D HumanML-24 287-D -> SMPL-22 joints66",
        "sampling": "greedy" if args.greedy else "categorical",
        "top_k": args.top_k,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
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
