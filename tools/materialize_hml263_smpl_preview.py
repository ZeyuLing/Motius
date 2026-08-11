#!/usr/bin/env python3
"""Fit selected HumanML3D-263 clips to neutral SMPL for release previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion.retarget.hml263_smpl import load_smpl_rest, retarget_hml263_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ids", required=True, help="Comma-separated motion IDs")
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "body_models" / "smpl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--refine-iters", type=int, default=80)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--max-fit-mpjpe-mm", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    smpl_rest = load_smpl_rest(
        args.smpl_model_dir.expanduser(), args.device, gender="neutral"
    )
    records = []
    for motion_id in ids:
        source = input_dir / f"{motion_id}.npy"
        destination = output_dir / f"{motion_id}.npz"
        if destination.is_file() and not args.overwrite:
            records.append({"motion_id": motion_id, "status": "skipped"})
            continue
        motion = np.asarray(np.load(source), dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != 263:
            raise ValueError(f"{source}: expected (T,263), got {motion.shape}")
        target_len = max(
            2,
            int(round(len(motion) * args.target_fps / args.source_fps)),
        )
        converted = retarget_hml263_clip(
            motion,
            smpl_rest=smpl_rest,
            device=args.device,
            gender="neutral",
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            target_len=target_len,
            floor_align=True,
            refine_iters=args.refine_iters,
            refine_lr=args.refine_lr,
            rotation_init="position_ik",
        )
        errors = np.asarray(converted["fit_mpjpe_mm"], dtype=np.float32)
        mean_error = float(errors.mean())
        if not np.isfinite(errors).all() or mean_error > args.max_fit_mpjpe_mm:
            raise RuntimeError(
                f"{motion_id}: SMPL fit MPJPE {mean_error:.2f} mm exceeds "
                f"{args.max_fit_mpjpe_mm:.2f} mm"
            )
        np.savez_compressed(
            destination,
            global_orient=np.asarray(converted["global_orient"], dtype=np.float32),
            body_pose=np.asarray(converted["body_pose"], dtype=np.float32),
            transl=np.asarray(converted["transl"], dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.asarray(args.target_fps, dtype=np.float32),
            fit_mpjpe_mm=errors,
            rotation_init=np.asarray(str(converted["rotation_init"])),
            source_hml263=np.asarray(str(source)),
        )
        records.append(
            {
                "motion_id": motion_id,
                "status": "generated",
                "source_frames": len(motion),
                "target_frames": target_len,
                "fit_mpjpe_mm_mean": mean_error,
                "fit_mpjpe_mm_p95": float(np.percentile(errors, 95)),
                "rotation_init": str(converted["rotation_init"]),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    (output_dir / "materialization.json").write_text(
        json.dumps(
            {
                "source_representation": "HumanML3D-263",
                "target_representation": "neutral SMPL",
                "source_fps": args.source_fps,
                "target_fps": args.target_fps,
                "rotation_init": "position_ik",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
