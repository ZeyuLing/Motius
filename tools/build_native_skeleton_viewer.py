#!/usr/bin/env python3
"""Build a unified Three.js viewer from native joint positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools/templates/native_skeleton_viewer.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--trajectory-key")
    parser.add_argument("--trajectory-joint", type=int, default=0)
    parser.add_argument("--trajectory-person", type=int, default=0)
    parser.add_argument(
        "--trajectory-on-floor",
        action="store_true",
        help="Place a root/path trajectory just above the shared floor.",
    )
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as payload:
        joints = payload["joints"].astype(np.float32, copy=False)
        parents = payload["parents"].astype(np.int32).tolist()
        caption = str(payload["caption"])
        case_id = str(payload["case_id"])
        fps = int(payload["fps"])
        condition_intervals = (
            payload["condition_intervals"].astype(np.int32).tolist()
            if "condition_intervals" in payload
            else []
        )
        segments = []
        if "segment_frames" in payload and "segment_captions" in payload:
            frames = payload["segment_frames"].astype(np.int32, copy=False)
            captions = payload["segment_captions"].astype(str).tolist()
            if len(frames) != len(captions):
                raise ValueError(
                    "segment_frames and segment_captions must have equal length"
                )
            segments = [
                {
                    "caption": text,
                    "start_frame": int(interval[0]),
                    "end_frame": int(interval[1]),
                }
                for interval, text in zip(frames, captions)
            ]
        trajectory = None
        if args.trajectory_key:
            source = payload[args.trajectory_key].astype(
                np.float32,
                copy=False,
            )
            if source.ndim == 4:
                source = source[:, args.trajectory_person]
            if source.ndim == 3:
                source = source[:, args.trajectory_joint]
            if source.ndim != 2 or source.shape[-1] != 3:
                raise ValueError(
                    "trajectory source must resolve to shape (frames, 3)"
                )
            trajectory = source.copy()
    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape (frames, people, joints, 3)")
    joints = joints.copy()
    floor_height = float(joints[..., 1].min())
    joints[..., 1] -= floor_height
    if trajectory is not None:
        if args.trajectory_on_floor:
            trajectory[..., 1] = 0.018
        else:
            trajectory[..., 1] -= floor_height
    minimum = joints.min(axis=(0, 1, 2))
    maximum = joints.max(axis=(0, 1, 2))
    center = (minimum + maximum) / 2
    span = float(max(maximum[0] - minimum[0], maximum[2] - minimum[2]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    joints.tofile(assets / "joints.f32")
    manifest = {
        "schema_version": 1,
        "method": args.label,
        "representation": args.representation,
        "cases": [
            {
                "case_id": case_id,
                "references": [caption],
                "segments": segments,
                "condition_intervals": condition_intervals,
                "frames": int(joints.shape[0]),
                "fps": fps,
                "trajectory": (
                    trajectory.tolist()
                    if trajectory is not None
                    else []
                ),
                "skeleton": {
                    "positions_file": "assets/joints.f32",
                    "people": int(joints.shape[1]),
                    "joints": int(joints.shape[2]),
                    "parents": parents,
                    "center": [float(center[0]), 0.0, float(center[2])],
                    "height": float(maximum[1] - minimum[1]),
                    "horizontal_span": span,
                },
            }
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(TEMPLATE, args.output_dir / "index.html")
    print((args.output_dir / "index.html").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
