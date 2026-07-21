#!/usr/bin/env python3
"""Fit selected Bailando SMPL-24 joint outputs to preview-ready SMPL poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.motion.retarget.hml263_smpl import (  # noqa: E402
    load_smpl_rest,
    retarget_hml263_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="Comma-separated AIST++ sequence ids")
    parser.add_argument(
        "--input-format",
        choices=("prediction_npz", "aistpp_json"),
        default="prediction_npz",
    )
    parser.add_argument("--smpl-model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-source-frames", type=int, default=1_200)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--refine-iters", type=int, default=40)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smpl_rest = load_smpl_rest(args.smpl_model_dir, args.device, gender="neutral")
    records = []
    for sequence_id in (value.strip() for value in args.ids.split(",")):
        if not sequence_id:
            continue
        if args.input_format == "prediction_npz":
            source = args.input_dir / f"{sequence_id}.npz"
            with np.load(source, allow_pickle=False) as payload:
                joints = np.asarray(payload["joints"], dtype=np.float32)
        else:
            source = args.input_dir / f"{sequence_id}.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            joints = np.asarray(payload["dance_array"], dtype=np.float32).reshape(
                -1, 24, 3
            )
        joints = joints[: args.max_source_frames, :22]
        result = retarget_hml263_clip(
            joints,
            smpl_rest=smpl_rest,
            source_fps=60.0,
            target_fps=args.target_fps,
            gender="neutral",
            device=args.device,
            floor_align=True,
            refine_iters=args.refine_iters,
            refine_lr=args.refine_lr,
            rotation_init="position_ik",
        )
        errors = np.asarray(result["fit_mpjpe_mm"], dtype=np.float32)
        destination = args.output_dir / f"{sequence_id}.npz"
        np.savez_compressed(
            destination,
            global_orient=np.asarray(result["global_orient"], dtype=np.float32),
            body_pose=np.asarray(result["body_pose"], dtype=np.float32),
            transl=np.asarray(result["transl"], dtype=np.float32),
            motion_135=np.asarray(result["motion_135"], dtype=np.float32),
            fit_mpjpe_mm=errors,
            source_fps=np.float32(60.0),
            target_fps=np.float32(args.target_fps),
            rotation_init=np.asarray("position_ik"),
        )
        record = {
            "sequence_id": sequence_id,
            "source_frames": len(joints),
            "output_frames": len(result["motion_135"]),
            "fit_mpjpe_mm_mean": float(errors.mean()),
            "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
            "path": str(destination),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / args.manifest_name).write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
