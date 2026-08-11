#!/usr/bin/env python3
"""Run UniMuMo zero-shot T2M on HumanML3D selected captions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.representation.humanml import (
    joints_to_hml263,
    linear_resample_joints,
)
from motius.pipelines.unimumo import UniMuMoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument(
        "--generation-duration-seconds",
        type=float,
        help=(
            "Generate every sample at this duration, then crop to the GT duration. "
            "By default generation directly uses the GT duration."
        ),
    )
    parser.add_argument(
        "--music-prompt",
        default="",
        help="Optional companion music description for UniMuMo joint generation.",
    )
    parser.add_argument(
        "--motion-prompt-template",
        choices=("raw", "motion", "dance"),
        default="raw",
        help="Match UniMuMo's HumanML3D caption templates when requested.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
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
    if args.device is None:
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if (
        args.generation_duration_seconds is not None
        and not 0 < args.generation_duration_seconds <= 10
    ):
        parser.error("--generation-duration-seconds must be in (0, 10]")
    return args


def selected_caption(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for level in ("macro", "meso", "micro"):
        values = payload.get(level)
        if values:
            caption = str(values[0]).strip()
            if caption:
                return caption
    raise ValueError(f"No selected caption in {path}")


def exact_resample(
    joints: np.ndarray,
    source_fps: float,
    target_fps: float,
    target_frames: int,
) -> np.ndarray:
    values = linear_resample_joints(joints, source_fps, target_fps)
    if len(values) == target_frames:
        return values
    source_time = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    output = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for channel in range(flat.shape[1]):
        output[:, channel] = np.interp(target_time, source_time, flat[:, channel])
    return output.reshape(target_frames, 22, 3).astype(np.float32)


def inverse_native_hml263(
    motion: np.ndarray, target_frames: int
) -> np.ndarray:
    """Invert UniMuMo's 20-to-60 fps linear HML263 interpolation."""

    values = np.asarray(motion, dtype=np.float32)[1::3]
    if len(values) < target_frames:
        raise ValueError(
            f"Native HML263 has {len(values)} inverse frames, expected "
            f"at least {target_frames}"
        )
    return values[:target_frames]


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.annotation.read_text(encoding="utf-8"))
    records = sorted(protocol["data_list"].items())
    if args.max_samples is not None:
        records = records[: args.max_samples]
    records = [
        (index, name, record)
        for index, (name, record) in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]

    native_dir = args.output / "native_hml263_60fps"
    hml_dir = args.output / "hml263_20fps"
    roundtrip_dir = args.output / "hml263_20fps_joints_roundtrip"
    joints_dir = args.output / "joints66"
    codes_dir = args.output / "motion_codes"
    for directory in (
        native_dir,
        hml_dir,
        roundtrip_dir,
        joints_dir,
        codes_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline = UniMuMoPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": args.local_files_only},
        device=args.device,
    )
    completed: list[str] = []
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
        if args.motion_prompt_template == "motion":
            motion_prompt = f"The motion is that {caption}."
        elif args.motion_prompt_template == "dance":
            motion_prompt = f"The dance is that {caption}."
        else:
            motion_prompt = caption
        output_fps = float(record["fps"])
        output_frames = int(record["num_frames"])
        duration = min(output_frames / output_fps, 10.0)
        output_frames = max(2, min(output_frames, int(round(duration * output_fps))))
        generation_duration = args.generation_duration_seconds or duration

        case_started = time.time()
        result = pipeline.infer_text_to_music_motion(
            music_prompt=args.music_prompt,
            motion_prompt=motion_prompt,
            duration_seconds=generation_duration,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + global_index,
        )
        native_frames = max(
            2,
            min(
                len(result.motion),
                int(round(duration * float(result.motion_fps))),
            ),
        )
        native_motion = np.asarray(result.motion[:native_frames], dtype=np.float32)
        native_joints = np.asarray(result.joints[:native_frames], dtype=np.float32)
        joints30 = exact_resample(
            native_joints,
            float(result.motion_fps),
            output_fps,
            output_frames,
        )
        hml_frames = max(2, int(round(duration * 20.0)))
        evaluator_frames = max(1, hml_frames - 1)
        joints20 = exact_resample(
            native_joints,
            float(result.motion_fps),
            20.0,
            hml_frames,
        )
        np.save(destination, joints30.reshape(len(joints30), 66))
        np.save(native_dir / f"{name}.npy", native_motion)
        np.save(
            hml_dir / f"{name}.npy",
            inverse_native_hml263(native_motion, evaluator_frames),
        )
        np.save(
            roundtrip_dir / f"{name}.npy",
            joints_to_hml263(joints20),
        )
        np.save(codes_dir / f"{name}.npy", result.motion_codes)
        completed.append(name)
        print(
            f"[{local_index + 1}/{len(records)}] {name} "
            f"frames={len(joints30)} seconds={time.time() - case_started:.2f}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "task": "text_to_motion",
        "method": "UniMuMo",
        "dataset": "HumanML3D official test, selected-caption protocol",
        "checkpoint": args.checkpoint,
        "representation": (
            "native HumanML3D-263 at 60 fps; canonical SMPL-22 joints66 at "
            "30 fps; evaluator HumanML3D-263 uses the phase-aligned native "
            "20 fps inverse; joints round-trip is diagnostic only"
        ),
        "generation_mode": "zero-shot joint music-motion token generation",
        "guidance_scale": args.guidance_scale,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "generation_duration_seconds": args.generation_duration_seconds,
        "music_prompt": args.music_prompt,
        "motion_prompt_template": args.motion_prompt_template,
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
