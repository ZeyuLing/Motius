#!/usr/bin/env python3
"""Generate MotionCLR outputs for the selected-caption HumanML3D protocol."""

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

from motius.pipelines.motionclr import MotionCLRPipeline


def _entries(path: Path) -> list[tuple[str, dict]]:
    raw = json.loads(path.read_text())
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        return [
            (str(value.get("motion_id") or value.get("id") or index), value)
            for index, value in enumerate(data)
        ]
    raise ValueError("annotation must contain a data_list dict or list")


def _caption_from_json(path: Path) -> str | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, dict):
        return None
    for key in ("caption", "text"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for group in ("macro", "meso", "micro"):
        values = raw.get(group, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _caption(entry: dict, annotation_root: Path) -> str | None:
    for key in ("selected_caption", "caption", "text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    caption_path = entry.get("hierarchical_caption_path") or entry.get("caption_path")
    if caption_path:
        path = Path(caption_path)
        if not path.is_absolute():
            path = annotation_root / path
        return _caption_from_json(path)
    return None


def _length(entry: dict, source_fps: float, model_fps: float) -> int | None:
    fps = float(entry.get("fps") or source_fps)
    source_frames = int(
        entry.get("num_frames")
        or round(float(entry.get("duration") or 0.0) * fps)
    )
    if source_frames <= 0:
        return None
    frames = int(round(source_frames * model_fps / fps))
    frames = (frames // 4) * 4
    return max(40, min(196, frames))


def _safe_name(value: str) -> str:
    return value.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="ZeyuLing/motius-motionclr-humanml3d")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    jobs = []
    annotation_root = Path(args.annotation_root).resolve()
    eligible = 0
    for motion_id, entry in _entries(Path(args.annotation).resolve()):
        caption = _caption(entry, annotation_root)
        length = _length(entry, args.source_fps, args.model_fps)
        if caption is None or length is None:
            continue
        if eligible % args.num_shards == args.shard_index:
            jobs.append((eligible, motion_id, caption, length))
            if args.max_samples and len(jobs) >= args.max_samples:
                break
        eligible += 1

    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} jobs={len(jobs)} "
        f"batch={args.batch_size} output={output}",
        flush=True,
    )
    if jobs:
        print(
            f"[first] id={jobs[0][1]} frames={jobs[0][3]} caption={jobs[0][2]}",
            flush=True,
        )
    if args.dry_run:
        return

    pipeline = MotionCLRPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={
            "device": args.device,
            "torch_dtype": "fp16",
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
        },
        device=args.device,
    )
    started = time.time()
    written = skipped = failed = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        todo = []
        for job in chunk:
            path = output / f"{_safe_name(job[1])}.npy"
            if args.skip_existing and path.is_file():
                skipped += 1
            else:
                todo.append(job)
        if not todo:
            continue
        try:
            motions = pipeline.infer_t2m(
                [item[2] for item in todo],
                [item[3] for item in todo],
                seed=args.seed + todo[0][0],
            )
            for (_, motion_id, _caption_text, length), motion in zip(todo, motions):
                value = np.asarray(motion, dtype=np.float32)[:length]
                if value.shape != (length, 263) or not np.isfinite(value).all():
                    raise RuntimeError(f"{motion_id}: invalid MotionCLR output {value.shape}")
                np.save(output / f"{_safe_name(motion_id)}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] batch={start} count={len(todo)} error={exc}", flush=True)
            traceback.print_exc()
        if (start // batch_size + 1) % 5 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] seen={min(start + batch_size, len(jobs))}/{len(jobs)} "
                f"written={written} skipped={skipped} failed={failed} "
                f"rate={(written + skipped) / elapsed:.2f}/s",
                flush=True,
            )

    summary = {
        "artifact": args.artifact,
        "annotation": str(Path(args.annotation).resolve()),
        "caption_protocol": "HumanML3D selected caption",
        "representation": "HumanML3D-263",
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "jobs": len(jobs),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    summary_dir = output / "_generation"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / f"shard_{args.shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
