#!/usr/bin/env python3
"""Convert generated HML263 clips to the shared Motius evaluation views."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.representation import motion135_to_motion272
from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


def _annotation_entries(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    entries = payload.get("data_list", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, dict):
        raise ValueError("annotation must contain a data_list mapping")
    return {str(key): value for key, value in entries.items()}


def _relative_offsets(rest_joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    rest = np.asarray(rest_joints, dtype=np.float32)
    parent_ids = np.asarray(parents, dtype=np.int64)[: len(rest)]
    offsets = rest.copy()
    for joint, parent in enumerate(parent_ids):
        if parent >= 0:
            offsets[joint] = rest[joint] - rest[parent]
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--motion135-dir", required=True)
    parser.add_argument("--joints66-dir", required=True)
    parser.add_argument("--ms272-dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--model-dir", default="checkpoints/body_models/smpl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--refine-iters", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    input_dir = Path(args.in_dir).resolve()
    motion135_dir = Path(args.motion135_dir).resolve()
    joints66_dir = Path(args.joints66_dir).resolve()
    ms272_dir = Path(args.ms272_dir).resolve()
    for directory in (motion135_dir, joints66_dir, ms272_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries = _annotation_entries(Path(args.annotation).resolve())
    available_ids = sorted(
        path.stem for path in input_dir.glob("*.npy") if path.stem in entries
    )
    motion_ids = [
        motion_id
        for index, motion_id in enumerate(available_ids)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_samples:
        motion_ids = motion_ids[: args.max_samples]
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} jobs={len(motion_ids)} ",
        f"device={args.device}",
        flush=True,
    )
    smpl_rest = load_smpl_rest(args.model_dir, args.device)
    smpl_offsets = _relative_offsets(smpl_rest[1], smpl_rest[2])
    started = time.time()
    written = skipped = failed = 0
    failures: list[dict[str, str]] = []
    for index, motion_id in enumerate(motion_ids):
        source = input_dir / f"{motion_id}.npy"
        motion135_path = motion135_dir / f"{motion_id}.npz"
        joints66_path = joints66_dir / f"{motion_id}.npy"
        ms272_path = ms272_dir / f"{motion_id}.npy"
        if args.skip_existing and all(
            path.is_file() for path in (motion135_path, joints66_path, ms272_path)
        ):
            skipped += 1
            continue
        try:
            features = np.load(source)
            target_len = int(entries[motion_id]["num_frames"])
            converted = retarget_hml263_clip(
                features,
                smpl_rest=smpl_rest,
                device=args.device,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                target_len=target_len,
                floor_align=True,
                refine_iters=args.refine_iters,
                refine_lr=args.refine_lr,
                rotation_init="auto",
            )
            motion135 = np.asarray(converted["motion_135"], dtype=np.float32)
            joints66 = np.asarray(converted["fitted_joints"], dtype=np.float32).reshape(
                target_len, 66
            )
            ms272 = np.asarray(
                motion135_to_motion272(
                    motion135,
                    rotation_space="local",
                    bone_offsets=smpl_offsets,
                ),
                dtype=np.float32,
            )
            if ms272.shape != (target_len, 272):
                raise RuntimeError(f"invalid MS272 output shape {ms272.shape}")
            np.savez_compressed(motion135_path, **converted)
            np.save(joints66_path, joints66)
            np.save(ms272_path, ms272)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"motion_id": motion_id, "error": str(exc)})
            print(f"[failed] id={motion_id} error={exc}", flush=True)
        if (index + 1) % 25 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] seen={index + 1}/{len(motion_ids)} written={written} "
                f"skipped={skipped} failed={failed} rate={(written + skipped) / elapsed:.2f}/s",
                flush=True,
            )

    summary = {
        "representation_source": "HumanML3D-263",
        "representations_written": ["motion135", "joints66", "MotionStreamer-272"],
        "annotation": str(Path(args.annotation).resolve()),
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "refine_iters": args.refine_iters,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "jobs": len(motion_ids),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:50],
        "elapsed_seconds": time.time() - started,
    }
    summary_path = motion135_dir.parent / f"conversion_shard_{args.shard_index:02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
