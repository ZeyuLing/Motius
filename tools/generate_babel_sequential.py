#!/usr/bin/env python3
"""Generate FlowMDM BABEL compositions with resumable sharding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import babel135_to_joints
from motius.pipelines.flowmdm import FlowMDMPipeline


SUPPORTED_PROTOCOLS = {
    "babel-official-val-shortmerge30-llm-joints66-v1",
    "babel-official-val-shortmerge30-llm-joints66-multipositive-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-param", type=float, default=1.5)
    parser.add_argument("--bpe-denoising-step", type=int, default=125)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards.")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Unsupported sequential manifest protocol {manifest.get('protocol')!r}."
        )
    offset_path = Path(manifest["smpl22_offsets"])
    if not offset_path.is_absolute():
        offset_path = manifest_path.parent / offset_path
    offsets = np.load(offset_path)

    output_dir = Path(args.output_dir).resolve()
    feature_dir = output_dir / "babel135"
    joints_dir = output_dir / "joints66"
    feature_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)
    pipeline = FlowMDMPipeline.from_pretrained(
        args.model,
        bundle_kwargs={
            "device": args.device,
            "seed": args.seed,
            "guidance_param": args.guidance_param,
            "bpe_denoising_step": args.bpe_denoising_step,
        },
        device=args.device,
    )

    generated = skipped = 0
    cases = manifest.get("cases", [])
    for case_index, case in enumerate(cases):
        if case_index % args.num_shards != args.shard_index:
            continue
        case_id = str(case["case_id"])
        feature_path = feature_dir / f"{case_id}.npy"
        joints_path = joints_dir / f"{case_id}.npy"
        if joints_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        captions = [item["caption"] for item in case["segments"]]
        lengths = [int(item["end_frame"]) - int(item["start_frame"]) for item in case["segments"]]
        features = pipeline.infer_sequential_t2m(
            [captions],
            [lengths],
            seed=args.seed + case_index,
        )[0]
        expected = int(case["total_frames"])
        if features.shape != (expected, 135):
            raise RuntimeError(
                f"Case {case_id} returned {features.shape}, expected ({expected}, 135)."
            )
        np.save(feature_path, features.astype(np.float32))
        joints = babel135_to_joints(features, bone_offsets=offsets).reshape(expected, 66)
        np.save(joints_path, joints.astype(np.float32))
        generated += 1
        print(f"[{case_index + 1}/{len(cases)}] generated {case_id}", flush=True)

    run = {
        "protocol": manifest["protocol"],
        "model": args.model,
        "seed": args.seed,
        "guidance_param": args.guidance_param,
        "bpe_denoising_step": args.bpe_denoising_step,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generated": generated,
        "skipped": skipped,
    }
    (output_dir / f"run_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(run, indent=2) + "\n"
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
