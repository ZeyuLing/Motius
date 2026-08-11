#!/usr/bin/env python3
"""Run batched UniMuMo inference on the paper's D2M-GAN AIST++ split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.protocols import d2mgan_aistpp_test_segments
from motius.pipelines.unimumo import UniMuMoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.shard_index = int(
        os.environ.get("RANK", "0") if args.shard_index is None else args.shard_index
    )
    args.num_shards = int(
        os.environ.get("WORLD_SIZE", "1")
        if args.num_shards is None
        else args.num_shards
    )
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.device is None:
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return args


def find_motion(root: Path, motion_id: str) -> Path:
    candidates = [
        root / split / "joint_vecs" / f"{motion_id}.npy"
        for split in ("train", "val", "test")
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one UniMuMo motion for {motion_id}, found {matches}"
        )
    return matches[0]


def load_segment(root: Path, segment) -> tuple[Path, np.ndarray]:
    source = find_motion(root, segment.source_motion_id)
    motion = np.load(source)
    first = (segment.segment_index - 1) * 120
    clip = np.asarray(motion[first : first + 120], dtype=np.float32)
    if clip.ndim != 2 or clip.shape[1] != 263 or not 114 <= len(clip) <= 120:
        raise ValueError(
            f"{segment.case_id} expected 114-120 HML263 frames, "
            f"got {clip.shape} from {source}"
        )
    return source, clip


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_segments = d2mgan_aistpp_test_segments()
    indexed = [
        (index, segment)
        for index, segment in enumerate(all_segments)
        if index % args.num_shards == args.shard_index
    ]
    pipeline = UniMuMoPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": args.local_files_only},
        device=args.device,
    )

    rows = []
    pending_by_length: dict[int, list] = {}
    for global_index, segment in indexed:
        wav_path = args.output / f"{segment.case_id}.wav"
        npz_path = args.output / f"{segment.case_id}.npz"
        source, motion = load_segment(args.motion_root, segment)
        if wav_path.is_file() and npz_path.is_file() and not args.overwrite:
            rows.append(
                {
                    "index": global_index,
                    "case_id": segment.case_id,
                    "source_motion_id": segment.source_motion_id,
                    "input_frames": int(len(motion)),
                    "music_id": segment.music_id,
                    "reference_start_seconds": segment.start_seconds,
                    "generated_audio": wav_path.name,
                }
            )
            continue
        pending_by_length.setdefault(len(motion), []).append(
            (global_index, segment, source, motion)
        )

    started = time.time()
    for input_frames in sorted(pending_by_length, reverse=True):
        bucket = pending_by_length[input_frames]
        for batch_start in range(0, len(bucket), args.batch_size):
            pending = bucket[batch_start : batch_start + args.batch_size]
            batch_motion = np.stack([item[3] for item in pending])
            batch_seed = args.seed + pending[0][0]
            batch_started = time.time()
            outputs = pipeline.infer_motion_to_music_batch(
                batch_motion,
                input_fps=60.0,
                music_prompts=[""] * len(pending),
                guidance_scale=args.guidance_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=batch_seed,
            )
            if len(outputs) != len(pending):
                raise RuntimeError("UniMuMo returned an incomplete inference batch")
            for (global_index, segment, source, motion), result in zip(
                pending, outputs
            ):
                wav_path = args.output / f"{segment.case_id}.wav"
                npz_path = args.output / f"{segment.case_id}.npz"
                sf.write(wav_path, result.waveform, int(result.sample_rate))
                np.savez_compressed(
                    npz_path,
                    input_humanml3d_263=motion,
                    generated_music_codes=result.music_codes,
                    conditioned_motion_codes=result.motion_codes,
                    sample_rate=np.int64(result.sample_rate),
                    seed=np.int64(batch_seed),
                )
                rows.append(
                    {
                        "index": global_index,
                        "case_id": segment.case_id,
                        "source_motion_id": segment.source_motion_id,
                        "input_frames": int(len(motion)),
                        "music_id": segment.music_id,
                        "reference_start_seconds": segment.start_seconds,
                        "generated_audio": wav_path.name,
                    }
                )
            print(
                f"frames={input_frames} batch={len(pending)} "
                f"completed={len(rows)}/{len(indexed)} "
                f"seconds={time.time() - batch_started:.2f}",
                flush=True,
            )

    rows.sort(key=lambda item: item["index"])
    manifest = {
        "schema_version": 1,
        "task": "dance_to_music",
        "method": "UniMuMo",
        "dataset": "AIST++ D2M-GAN official 86x2-second test split",
        "protocol_source": "https://github.com/L-YeZhu/D2M-GAN",
        "checkpoint": args.checkpoint,
        "motion_representation": "official UniMuMo HumanML3D-263 at 60 fps",
        "duration_seconds": 2.0,
        "music_prompt": "",
        "guidance_scale": args.guidance_scale,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_samples": len(rows),
        "elapsed_seconds": time.time() - started,
        "cases": rows,
    }
    manifest_path = args.output / f"manifest_shard_{args.shard_index:02d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
