#!/usr/bin/env python3
"""Run OmniControl on dense-root and sparse-waypoint HumanML3D control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _read_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return [str(item) for item in value]


def _read_captions(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if not isinstance(value, dict):
        raise ValueError(f"expected an id-keyed caption map in {path}")
    result = {}
    for motion_id, entry in value.items():
        if isinstance(entry, dict):
            caption = entry.get("caption", entry.get("caption_en", ""))
        else:
            caption = entry
        result[str(motion_id)] = str(caption or "")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--gt-hml263-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--axes", choices=("xz", "xyz"), required=True)
    parser.add_argument(
        "--waypoint-interval",
        type=int,
        default=20,
        help="Sparse waypoint interval at OmniControl's native 20 FPS.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--respacing", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must satisfy 0 <= index < num_shards")
    if args.waypoint_interval < 1:
        parser.error("--waypoint-interval must be positive")
    return args


def main() -> None:
    args = _parse_args()
    gt_dir = Path(args.gt_hml263_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = _read_captions(Path(args.captions).resolve())
    ids = _read_ids(Path(args.ids).resolve())
    ids = ids[args.shard_index :: args.num_shards]
    if args.max_samples:
        ids = ids[: args.max_samples]
    ids = [motion_id for motion_id in ids if (gt_dir / f"{motion_id}.npy").is_file()]
    print(
        f"[setup] mode={args.mode} axes={args.axes} "
        f"shard={args.shard_index}/{args.num_shards} cases={len(ids)}",
        flush=True,
    )
    if args.dry_run:
        return

    from motius.pipelines.omnicontrol import OmniControlPipeline

    pipeline = OmniControlPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"guidance_param": args.guidance, "respacing": args.respacing},
        device=args.device,
    )
    written = skipped = failed = 0
    started = time.time()
    batch_size = max(1, args.batch_size)
    for offset in range(0, len(ids), batch_size):
        batch_ids = ids[offset : offset + batch_size]
        todo = [
            motion_id
            for motion_id in batch_ids
            if not (args.skip_existing and (out_dir / f"{motion_id}.npy").exists())
        ]
        skipped += len(batch_ids) - len(todo)
        if not todo:
            continue
        motions = [np.load(gt_dir / f"{motion_id}.npy").astype(np.float32) for motion_id in todo]
        lengths = [min(len(motion), 196) for motion in motions]
        keyframes = None
        control_mode = "trajectory"
        if args.mode == "sparse":
            control_mode = "keyframes"
            keyframes = []
            for length in lengths:
                frames = list(range(0, length, args.waypoint_interval)) + [length - 1]
                keyframes.append(sorted(set(frames)))
        try:
            arrays = pipeline.infer_control(
                [captions.get(motion_id, "") for motion_id in todo],
                motions,
                lengths=lengths,
                control_mode=control_mode,
                joint_indices=[0],
                axes=args.axes,
                keyframe_indices=keyframes,
                seed=args.seed + args.shard_index * 100_000 + offset,
            )
            for motion_id, length, array in zip(todo, lengths, arrays):
                value = np.asarray(array, dtype=np.float32)[:length]
                if value.ndim != 2 or value.shape[1] != 263 or not np.isfinite(value).all():
                    raise RuntimeError(f"{motion_id}: invalid output {value.shape}")
                np.save(out_dir / f"{motion_id}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] offset={offset} count={len(todo)} error={exc}", flush=True)
        if (offset // batch_size + 1) % 5 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] seen={min(offset + batch_size, len(ids))}/{len(ids)} "
                f"written={written} skipped={skipped} failed={failed} "
                f"rate={(written + skipped) / elapsed:.3f}/s",
                flush=True,
            )

    summary = {
        "method": "OmniControl",
        "mode": args.mode,
        "axes": args.axes.upper(),
        "native_condition": "pelvis position at selected frames and axes",
        "respacing": args.respacing or "official_1000_steps",
        "guidance": args.guidance,
        "jobs": len(ids),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir.parent / f"generation_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {json.dumps(summary)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
