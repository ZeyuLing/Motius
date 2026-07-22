#!/usr/bin/env python3
"""Run UniMuMo motion-to-music on the shared AIST++ case package."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.representation.humanml import (
    joints_to_hml263,
    linear_resample_joints,
)
from motius.pipelines.unimumo import UniMuMoPipeline


DEFAULT_ASSET_BASE_URL = (
    "https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/"
    "cases/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-base-url", default=DEFAULT_ASSET_BASE_URL)
    parser.add_argument("--asset-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.shard_index = (
        int(os.environ.get("RANK", "0"))
        if args.shard_index is None
        else args.shard_index
    )
    args.num_shards = (
        int(os.environ.get("WORLD_SIZE", "1"))
        if args.num_shards is None
        else args.num_shards
    )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if not 0 < args.max_seconds <= 10:
        parser.error("--max-seconds must be in (0, 10]")
    if args.device is None:
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return args


def load_cases(path: Path, max_samples: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No AIST++ cases found in {path}")
    return cases if max_samples is None else cases[:max_samples]


def cached_asset(args: argparse.Namespace, relative_path: str) -> Path:
    relative = Path(relative_path)
    if args.asset_root is not None:
        source = args.asset_root / relative
        if source.is_file():
            return source
    destination = args.asset_cache / relative.name
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = args.asset_base_url.rstrip("/") + "/" + relative.as_posix()
        urllib.request.urlretrieve(url, destination)
    return destination


def decode_packed_joints(path: Path, descriptor: dict) -> np.ndarray:
    frames = int(descriptor["frames"])
    joint_count = int(descriptor["joint_count"])
    count = int(descriptor["position_count"])
    expected = frames * joint_count * 3
    if count != expected:
        raise ValueError(f"Packed joint count {count} does not match {expected}")
    with path.open("rb") as handle:
        handle.seek(int(descriptor["position_offset"]))
        quantized = np.fromfile(handle, dtype="<u2", count=count)
    if len(quantized) != count:
        raise ValueError(f"Truncated packed joint asset {path}")
    minimum = np.asarray(descriptor["position_minimum"], dtype=np.float32)
    scale = np.asarray(descriptor["position_scale"], dtype=np.float32)
    return quantized.reshape(frames, joint_count, 3) * scale + minimum


def main() -> None:
    args = parse_args()
    cases = load_cases(args.case_manifest, args.max_samples)
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = UniMuMoPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": args.local_files_only},
        device=args.device,
    )

    completed: list[str] = []
    started = time.time()
    for index, case in enumerate(cases):
        if index % args.num_shards != args.shard_index:
            continue
        case_id = str(case.get("case_id") or case.get("sample_id"))
        audio_path = args.output / f"{case_id}.wav"
        metadata_path = args.output / f"{case_id}.npz"
        if audio_path.is_file() and metadata_path.is_file() and not args.overwrite:
            completed.append(case_id)
            print(f"skip {case_id}", flush=True)
            continue

        descriptor = case["skeletons"]["gt"]
        packed_path = cached_asset(args, str(descriptor["asset"]))
        joints = decode_packed_joints(packed_path, descriptor)[:, :22]
        source_fps = float(descriptor["fps"])
        source_frames = min(len(joints), int(round(args.max_seconds * source_fps)))
        joints = joints[:source_frames]
        joints60 = linear_resample_joints(joints, source_fps, 60.0)
        motion = joints_to_hml263(joints60)
        references = [str(value) for value in case.get("references") or ()]
        genre = references[0].strip().lower() if references else ""
        music_prompt = f"The genre of the music is {genre}." if genre else ""

        case_started = time.time()
        result = pipeline.infer_motion_to_music(
            motion,
            input_fps=60.0,
            music_prompt=music_prompt,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + index,
        )
        sf.write(audio_path, result.waveform, int(result.sample_rate))
        np.savez_compressed(
            metadata_path,
            input_joints=joints,
            input_humanml3d_263=motion,
            generated_music_codes=result.music_codes,
            generated_motion_codes=result.motion_codes,
            input_fps=np.float32(source_fps),
            sample_rate=np.int64(result.sample_rate),
            music_prompt=np.asarray(music_prompt),
            seed=np.int64(args.seed + index),
        )
        completed.append(case_id)
        print(
            f"[{index + 1}/{len(cases)}] {case_id} "
            f"samples={len(result.waveform)} seconds={time.time() - case_started:.2f}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "task": "dance_to_music",
        "method": "UniMuMo",
        "dataset": "AIST++ shared crossmodal 40-case package",
        "checkpoint": args.checkpoint,
        "input_representation": (
            "AIST++ SMPL-24 joints at 30 fps -> SMPL-22 -> HumanML3D-263 "
            "at UniMuMo's native 60 fps"
        ),
        "generation_mode": "zero-shot motion-to-music",
        "guidance_scale": args.guidance_scale,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_samples": len(completed),
        "samples": completed,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / f"manifest_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
