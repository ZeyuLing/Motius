#!/usr/bin/env python3
"""Generate CondMDI predictions for a HumanML3D selected-caption manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path):
    return json.loads(path.read_text())


def _entries(path: Path):
    raw = _load_json(path)
    data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, list):
        return [
            (str(value.get("motion_id") or value.get("id") or index), value)
            for index, value in enumerate(data)
        ]
    raise ValueError("annotation must contain a data_list dict/list")


def _caption_from_json(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = _load_json(path)
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


def _length(entry: dict, args: argparse.Namespace) -> int | None:
    source_fps = float(entry.get("fps") or args.source_fps)
    source_frames = int(
        entry.get("num_frames")
        or round(float(entry.get("duration") or 0.0) * source_fps)
    )
    if source_frames <= 0:
        return None
    length = int(round(source_frames * args.model_fps / source_fps))
    length = (length // 4) * 4
    return max(args.min_length, min(args.max_length, length))


def _jobs(args: argparse.Namespace):
    annotation_root = Path(args.annotation_root).resolve()
    selected = [
        (ordinal, motion_id, entry)
        for ordinal, (motion_id, entry) in enumerate(
            _entries(Path(args.annotation).resolve())
        )
        if ordinal % args.num_shards == args.shard_index
    ]
    if args.max_samples:
        selected = selected[: args.max_samples]

    def build(job):
        ordinal, motion_id, entry = job
        caption = _caption(entry, annotation_root)
        length = _length(entry, args)
        return (ordinal, motion_id, caption, length)

    with ThreadPoolExecutor(max_workers=min(32, max(1, len(selected)))) as pool:
        resolved = list(pool.map(build, selected))
    return [job for job in resolved if job[2] and job[3]]


def _safe_name(value: str) -> str:
    return value.replace("/", "__")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="ZeyuLing/motius-condmdi-humanml3d")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--respacing", default="ddim100")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--model-fps", type=float, default=20.0)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=196)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must satisfy 0 <= index < num_shards")
    return args


def main():
    args = _parse_args()
    jobs = _jobs(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] shard={args.shard_index}/{args.num_shards} jobs={len(jobs)} "
        f"batch={args.batch_size} output={out_dir}",
        flush=True,
    )
    if jobs:
        _, motion_id, caption, length = jobs[0]
        print(f"[first] id={motion_id} frames={length} caption={caption}", flush=True)
    if args.dry_run:
        return

    from motius.pipelines.condmdi import CondMDIPipeline

    pipeline = CondMDIPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"guidance_param": args.guidance, "respacing": args.respacing},
        device=args.device,
    )
    written = skipped = failed = 0
    started = time.time()
    batch_size = max(1, args.batch_size)
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        todo = []
        for job in chunk:
            output_path = out_dir / f"{_safe_name(job[1])}.npy"
            if args.skip_existing and output_path.exists():
                skipped += 1
            else:
                todo.append(job)
        if not todo:
            continue
        try:
            arrays = pipeline.infer_t2m(
                [job[2] for job in todo],
                [job[3] for job in todo],
                seed=args.seed + todo[0][0],
            )
            for (_, motion_id, _caption_text, length), array in zip(todo, arrays):
                value = np.asarray(array, dtype=np.float32)[:length]
                if value.ndim != 2 or value.shape[1] != 263 or not np.isfinite(value).all():
                    raise RuntimeError(f"{motion_id}: invalid HML263 output {value.shape}")
                np.save(out_dir / f"{_safe_name(motion_id)}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] start={start} count={len(todo)} error={exc}", flush=True)
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
        "respacing": args.respacing,
        "guidance": args.guidance,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "jobs": len(jobs),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = out_dir.parent / f"generation_shard_{args.shard_index:02d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {json.dumps(summary)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
