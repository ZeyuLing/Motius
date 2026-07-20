#!/usr/bin/env python3
"""Generate MaskControl outputs for the selected-caption HumanML3D protocol."""

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
    for key in ("caption", "text", "selected_caption"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for group in ("macro", "meso", "micro"):
        for value in raw.get(group, []):
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _caption(
    motion_id: str,
    entry: dict,
    annotation_root: Path,
    caption_map: dict[str, str],
) -> str | None:
    mapped = caption_map.get(motion_id)
    if mapped:
        return mapped
    for key in ("selected_caption", "caption", "text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = entry.get("hierarchical_caption_path") or entry.get("caption_path")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = annotation_root / path
    return _caption_from_json(path)


def _load_caption_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    records = raw.get("data", raw) if isinstance(raw, dict) else {}
    result = {}
    for motion_id, record in records.items():
        if isinstance(record, str):
            caption = record
        elif isinstance(record, dict):
            selected = record.get("selected", {})
            caption = (
                selected.get("caption")
                if isinstance(selected, dict)
                else None
            ) or record.get("caption")
        else:
            caption = None
        if isinstance(caption, str) and caption.strip():
            result[str(motion_id)] = caption.strip()
    return result


def _length(entry: dict, model_fps: float) -> int | None:
    source_fps = float(entry.get("fps") or 30.0)
    source_frames = int(
        entry.get("num_frames")
        or round(float(entry.get("duration") or 0.0) * source_fps)
    )
    if source_frames <= 0:
        return None
    frames = int(round(source_frames * model_fps / source_fps))
    frames = (frames // 4) * 4
    return max(40, min(196, frames))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="ZeyuLing/motius-maskcontrol-humanml3d")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-root", default=".")
    parser.add_argument(
        "--caption-map",
        help=(
            "Selected-caption map. Defaults to caption_map.json beside the "
            "annotation and avoids thousands of small-file reads on Ceph."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    annotation = Path(args.annotation).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    caption_map_path = (
        Path(args.caption_map).resolve()
        if args.caption_map
        else annotation.with_name("caption_map.json")
    )
    caption_map = _load_caption_map(caption_map_path)
    jobs = []
    eligible = 0
    for motion_id, entry in _entries(annotation):
        caption = _caption(motion_id, entry, annotation_root, caption_map)
        length = _length(entry, args.model_fps)
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
    if args.dry_run:
        return

    pipeline = MaskControlPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"device": args.device},
        device=args.device,
    )
    started = time.time()
    written = skipped = failed = 0
    for start in range(0, len(jobs), max(1, args.batch_size)):
        chunk = jobs[start : start + max(1, args.batch_size)]
        todo = [
            value
            for value in chunk
            if not (
                args.skip_existing
                and (output / f"{value[1].replace('/', '__')}.npy").is_file()
            )
        ]
        skipped += len(chunk) - len(todo)
        if not todo:
            continue
        try:
            motions = pipeline.infer_t2m(
                [value[2] for value in todo],
                [value[3] for value in todo],
                seed=args.seed + todo[0][0],
            )
            for (_, motion_id, _, length), motion in zip(todo, motions):
                motion = np.asarray(motion, dtype=np.float32)[:length]
                if motion.shape != (length, 263) or not np.isfinite(motion).all():
                    raise RuntimeError(f"{motion_id}: invalid output {motion.shape}")
                np.save(output / f"{motion_id.replace('/', '__')}.npy", motion)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] batch={start} error={exc}", flush=True)
            traceback.print_exc()
        if (start // max(1, args.batch_size) + 1) % 5 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] seen={min(start + args.batch_size, len(jobs))}/{len(jobs)} "
                f"written={written} skipped={skipped} failed={failed} "
                f"rate={(written + skipped) / elapsed:.2f}/s",
                flush=True,
            )

    summary = {
        "artifact": args.artifact,
        "annotation": str(annotation),
        "caption_protocol": "HumanML3D selected caption",
        "caption_map": str(caption_map_path) if caption_map else None,
        "representation": "HumanML3D-263",
        "seed": args.seed,
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
