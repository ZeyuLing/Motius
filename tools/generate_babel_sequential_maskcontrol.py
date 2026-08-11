#!/usr/bin/env python3
"""Generate the MaskControl-supported subset of the BABEL sequential protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints
from motius.motion.representation.humanml import linear_resample_joints, recover_from_ric
from motius.pipelines.maskcontrol import MaskControlPipeline


SUPPORTED_PROTOCOLS = {
    "babel-official-val-shortmerge30-llm-joints66-v1",
    "babel-official-val-shortmerge30-llm-joints66-multipositive-v2",
    "babel-official-val-shortmerge30-llm-joints66-actiongroups-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact", default="ZeyuLing/motius-maskcontrol-humanml3d")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition-padding", type=int, default=5)
    parser.add_argument("--each-iterations", type=int, default=300)
    parser.add_argument("--final-iterations", type=int, default=300)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def _lengths_at_20fps(case: dict) -> list[int]:
    boundaries_30 = [int(item["start_frame"]) for item in case["segments"]]
    boundaries_30.append(int(case["segments"][-1]["end_frame"]))
    boundaries_20 = [int(round(value * 20.0 / 30.0)) for value in boundaries_30]
    return [
        boundaries_20[index + 1] - boundaries_20[index]
        for index in range(len(boundaries_20) - 1)
    ]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported protocol {manifest.get('protocol')!r}")
    output = Path(args.output_dir).resolve()
    hml_dir = output / "hml263"
    joints_dir = output / "joints66"
    hml_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MaskControlPipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"device": args.device},
        device=args.device,
    )

    selected = [
        (index, case)
        for index, case in enumerate(manifest.get("cases", []))
        if index % args.num_shards == args.shard_index
    ]
    if args.max_samples:
        selected = selected[: args.max_samples]
    generated = skipped = unsupported_length = failed = 0
    for position, (case_index, case) in enumerate(selected, start=1):
        case_id = str(case["case_id"])
        joints_path = joints_dir / f"{case_id}.npy"
        if joints_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        lengths = _lengths_at_20fps(case)
        if min(lengths) < 4 or sum(lengths) > 392:
            unsupported_length += 1
            continue
        try:
            motion = pipeline.infer_sequential(
                [str(item["caption"]) for item in case["segments"]],
                lengths,
                transition_padding=args.transition_padding,
                seed=args.seed + case_index,
                each_iterations=args.each_iterations,
                final_iterations=args.final_iterations,
            )
            np.save(hml_dir / f"{case_id}.npy", motion.astype(np.float32))
            joints = recover_from_ric(torch.from_numpy(motion).unsqueeze(0), 22)[0].numpy()
            expected = int(case["total_frames"])
            joints = linear_resample_joints(joints, 20.0, 30.0)
            if len(joints) != expected:
                source = np.linspace(0.0, 1.0, len(joints), dtype=np.float64)
                target = np.linspace(0.0, 1.0, expected, dtype=np.float64)
                joints = np.stack(
                    [np.interp(target, source, joints[:, joint, axis]) for joint in range(22) for axis in range(3)],
                    axis=-1,
                ).reshape(expected, 22, 3)
            joints = canonicalize_smpl22_joints(joints).reshape(expected, 66)
            if not np.isfinite(joints).all():
                raise RuntimeError("non-finite generated joints")
            np.save(joints_path, joints.astype(np.float32))
            generated += 1
            print(f"[{position}/{len(selected)}] generated {case_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{position}/{len(selected)}] failed {case_id}: {exc}", flush=True)

    summary = {
        "method": "MaskControl",
        "protocol": manifest["protocol"],
        "artifact": args.artifact,
        "source_fps": 30,
        "model_fps": 20,
        "checkpoint_max_frames_20fps": 392,
        "generated": generated,
        "skipped": skipped,
        "unsupported_length": unsupported_length,
        "failed": failed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (output / f"run_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
