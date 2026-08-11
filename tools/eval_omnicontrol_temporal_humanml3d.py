#!/usr/bin/env python3
"""Run OmniControl on the HumanML3D temporal-condition benchmark.

Temporal frame evidence is represented through OmniControl's native interface:
the XYZ positions of all 22 joints at every observed frame. Predictions are
stored as physical-scale HumanML3D-263 arrays for the shared Motius evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SETTING_TO_MODE = {
    "start_1f": "first_frame",
    "pre20": "prefix",
    "both_1f": "first_last",
    "mid80": "boundary",
    "adaptive_keyframes": "keyframes",
}


def _read_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return [str(item) for item in value]


def _read_map(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "data_list" in value:
        value = value["data_list"]
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return {str(key): item for key, item in value.items()}


def _caption(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("selected_caption", "caption", "caption_en", "text"):
            text = value.get(key)
            if isinstance(text, str):
                return text
    raise ValueError(f"invalid caption record: {value!r}")


def _keyframe_indices(entry: object, length: int) -> list[int]:
    if not isinstance(entry, dict):
        raise ValueError("adaptive keyframe entry must be an object")
    fractions = entry.get("fracs")
    if fractions is None:
        source_length = max(1, int(entry.get("T", length)) - 1)
        fractions = [float(index) / source_length for index in entry["keyframe_indices"]]
    return sorted(
        {max(0, min(length - 1, int(round(float(frac) * (length - 1))))) for frac in fractions}
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--gt-hml263-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--setting", required=True, choices=sorted(SETTING_TO_MODE))
    parser.add_argument("--caption-mode", choices=("normal", "blank"), default="normal")
    parser.add_argument("--keyframe-file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument(
        "--respacing",
        default="",
        help="Empty uses the released 1000-step sampler; e.g. ddim100 is a speed ablation.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must satisfy 0 <= index < num_shards")
    if args.setting == "adaptive_keyframes" and not args.keyframe_file:
        parser.error("adaptive_keyframes requires --keyframe-file")
    return args


def main() -> None:
    args = _parse_args()
    gt_dir = Path(args.gt_hml263_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = _read_map(Path(args.captions).resolve())
    keyframes = _read_map(Path(args.keyframe_file).resolve()) if args.keyframe_file else {}

    ids = _read_ids(Path(args.ids).resolve())
    ids = ids[args.shard_index :: args.num_shards]
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

    from motius.pipelines.omnicontrol import OmniControlPipeline

    pipeline = OmniControlPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"guidance_param": args.guidance, "respacing": args.respacing},
        device=args.device,
    )
    mode = SETTING_TO_MODE[args.setting]
    written = skipped = failed = 0
    started = time.time()
    for offset in range(0, len(ids), max(1, args.batch_size)):
        batch_ids = ids[offset : offset + max(1, args.batch_size)]
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
        prompts = [
            "" if args.caption_mode == "blank" else _caption(captions[motion_id])
            for motion_id in todo
        ]
        batch_keyframes = None
        if mode == "keyframes":
            batch_keyframes = [
                _keyframe_indices(keyframes[motion_id], length)
                for motion_id, length in zip(todo, lengths)
            ]
        try:
            arrays = pipeline.infer_control(
                prompts,
                motions,
                lengths=lengths,
                control_mode=mode,
                joint_indices=range(22),
                axes="xyz",
                keyframe_indices=batch_keyframes,
                prefix_ratio=0.2,
                boundary_ratio=0.1,
                seed=args.seed + args.shard_index * 100_000 + offset,
            )
            for motion_id, length, array in zip(todo, lengths, arrays):
                value = np.asarray(array, dtype=np.float32)[:length]
                if value.ndim != 2 or value.shape[1] != 263 or not np.isfinite(value).all():
                    raise RuntimeError(f"{motion_id}: invalid output shape/content {value.shape}")
                np.save(out_dir / f"{motion_id}.npy", value)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[failed] offset={offset} count={len(todo)} error={exc}", flush=True)
        if (offset // max(1, args.batch_size) + 1) % 5 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] seen={min(offset + args.batch_size, len(ids))}/{len(ids)} "
                f"written={written} skipped={skipped} failed={failed} "
                f"rate={(written + skipped) / elapsed:.3f}/s",
                flush=True,
            )

    summary = {
        "method": "OmniControl",
        "setting": args.setting,
        "caption_mode": args.caption_mode,
        "native_condition": "all-joint XYZ positions at selected frames",
        "respacing": args.respacing or "official_1000_steps",
        "guidance": args.guidance,
        "jobs": len(ids),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = out_dir.parent / f"generation_shard_{args.shard_index:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {json.dumps(summary)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
