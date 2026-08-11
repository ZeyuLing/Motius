#!/usr/bin/env python3
"""Generate MaskControl outputs for the Motius temporal-condition protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.pipelines.maskcontrol import MaskControlPipeline


SETTING_TO_MODE = {
    "start_1f": "first_frame",
    "pre20": "prefix",
    "both_1f": "first_last",
    "mid80": "boundary",
    "adaptive_keyframes": "keyframes",
}


def _read_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return [str(item) for item in value]


def _read_map(path: Path) -> dict:
    value = json.loads(path.read_text())
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return {str(key): item for key, item in value.items()}


def _caption(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("selected_caption", "caption", "text"):
            text = value.get(key)
            if isinstance(text, str):
                return text
    raise ValueError(f"invalid caption record: {value!r}")


def _keyframes(value, length: int) -> list[int]:
    if not isinstance(value, dict):
        raise ValueError("adaptive keyframe records must be objects")
    fractions = value.get("fracs")
    if fractions is None:
        source_length = max(1, int(value.get("T", length)) - 1)
        fractions = [float(index) / source_length for index in value["keyframe_indices"]]
    return sorted(
        {
            max(0, min(length - 1, int(round(float(frac) * (length - 1)))))
            for frac in fractions
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="ZeyuLing/motius-maskcontrol-humanml3d")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--gt-hml263-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--setting", required=True, choices=sorted(SETTING_TO_MODE))
    parser.add_argument("--caption-mode", choices=("normal", "blank"), default="normal")
    parser.add_argument("--keyframe-file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimization-profile",
        choices=("paper", "fast"),
        default="paper",
        help="paper=100 iterations per sampling step plus 600 final; fast=0 plus 100.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.setting == "adaptive_keyframes" and not args.keyframe_file:
        parser.error("adaptive_keyframes requires --keyframe-file")
    return args


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_hml263_dir).resolve()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    captions = _read_map(Path(args.captions).resolve())
    keyframe_map = _read_map(Path(args.keyframe_file).resolve()) if args.keyframe_file else {}
    ids = _read_ids(Path(args.ids).resolve())[args.shard_index :: args.num_shards]
    ids = [motion_id for motion_id in ids if (gt_dir / f"{motion_id}.npy").is_file()]
    if args.max_samples:
        ids = ids[: args.max_samples]
    print(
        f"[setup] setting={args.setting} caption={args.caption_mode} "
        f"shard={args.shard_index}/{args.num_shards} cases={len(ids)}",
        flush=True,
    )
    if args.dry_run:
        return

    pipeline = MaskControlPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"device": args.device},
        device=args.device,
    )
    if args.optimization_profile == "paper":
        each_iterations, final_iterations = 100, 600
    else:
        each_iterations, final_iterations = 0, 100

    written = skipped = failed = 0
    started = time.time()
    for offset in range(0, len(ids), max(1, args.batch_size)):
        batch_ids = ids[offset : offset + max(1, args.batch_size)]
        todo = [
            motion_id
            for motion_id in batch_ids
            if not (args.skip_existing and (output / f"{motion_id}.npy").is_file())
        ]
        skipped += len(batch_ids) - len(todo)
        if not todo:
            continue
        motions = [np.load(gt_dir / f"{motion_id}.npy").astype(np.float32) for motion_id in todo]
        lengths = [min(len(value), 196) for value in motions]
        prompts = None
        if args.caption_mode == "normal":
            prompts = [_caption(captions[motion_id]) for motion_id in todo]
        batch_keyframes = None
        if args.setting == "adaptive_keyframes":
            batch_keyframes = [
                _keyframes(keyframe_map[motion_id], length)
                for motion_id, length in zip(todo, lengths)
            ]
        try:
            predictions = pipeline.infer_temporal(
                prompts,
                motions,
                lengths=lengths,
                mode=SETTING_TO_MODE[args.setting],
                keyframe_indices=batch_keyframes,
                prefix_ratio=0.2,
                boundary_ratio=0.1,
                seed=args.seed + args.shard_index * 100_000 + offset,
                each_iterations=each_iterations,
                final_iterations=final_iterations,
            )
            for motion_id, length, prediction in zip(todo, lengths, predictions):
                value = np.asarray(prediction, dtype=np.float32)[:length]
                if value.shape != (length, 263) or not np.isfinite(value).all():
                    raise RuntimeError(f"{motion_id}: invalid output {value.shape}")
                np.save(output / f"{motion_id}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] offset={offset} error={exc}", flush=True)
            traceback.print_exc()
        print(
            f"[progress] seen={min(offset + args.batch_size, len(ids))}/{len(ids)} "
            f"written={written} skipped={skipped} failed={failed}",
            flush=True,
        )

    summary = {
        "method": "MaskControl",
        "setting": args.setting,
        "caption_mode": args.caption_mode,
        "native_condition": "six released HML263 anchor joints at selected frames",
        "control_joint_ids": [0, 10, 11, 15, 20, 21],
        "optimization_profile": args.optimization_profile,
        "each_iterations": each_iterations,
        "final_iterations": final_iterations,
        "jobs": len(ids),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    summary_dir = output / "_generation"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / f"shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
