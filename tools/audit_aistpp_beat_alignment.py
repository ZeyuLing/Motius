#!/usr/bin/env python3
"""Compare official, 60 fps, and canonical 30 fps AIST++ BeatAlign paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.datasets.aistpp_music_to_dance import AISTPPMusicDanceDataset  # noqa: E402
from motius.evaluation.music_to_dance import (  # noqa: E402
    beat_alignment_score,
    motion_beat_frames,
)
from motius.motion.representation.humanml import linear_resample_joints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--music-feature-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path)
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pred_root is None and not args.ground_truth:
        parser.error("--pred-root is required unless --ground-truth is set")
    return args


def _official_score(beats: np.ndarray, joints: np.ndarray) -> float:
    velocity = np.linalg.norm(joints[1:] - joints[:-1], axis=2).mean(axis=1)
    motion_beats = argrelextrema(gaussian_filter(velocity, 5), np.less)[0]
    music_beats = np.flatnonzero(np.asarray(beats, dtype=bool)[: len(velocity)])
    if not len(music_beats) or not len(motion_beats):
        return 0.0
    distance = np.min(
        (motion_beats[None, :] - music_beats[:, None]) ** 2,
        axis=1,
    )
    return float(np.exp(-distance / 18.0).mean())


def main() -> None:
    args = parse_args()
    dataset = AISTPPMusicDanceDataset(args.data_root, args.music_feature_root)
    rows = []
    for sample in dataset:
        if args.ground_truth:
            joints_60 = np.asarray(sample["gt_joints"], dtype=np.float32)
        else:
            with np.load(args.pred_root / f"{sample['name']}.npz") as payload:
                joints_60 = np.asarray(payload["joints"], dtype=np.float32)
        beats = sample["music_beats"]
        official = _official_score(beats, joints_60)
        motius_60 = beat_alignment_score(
            beats[: len(joints_60) - 1],
            motion_beat_frames(joints_60, motion_fps=60.0),
            music_fps=60.0,
            motion_fps=60.0,
        )
        joints_30 = linear_resample_joints(joints_60, 60.0, 30.0)
        beat_limit = int(np.ceil((len(joints_30) - 1) * 2.0))
        motius_30 = beat_alignment_score(
            beats[:beat_limit],
            motion_beat_frames(joints_30, motion_fps=30.0),
            music_fps=60.0,
            motion_fps=30.0,
        )
        rows.append(
            {
                "name": sample["name"],
                "official_60fps": official,
                "motius_60fps": motius_60,
                "motius_30fps": motius_30,
                "official_delta": abs(official - motius_60),
                "fps_delta": abs(motius_60 - motius_30),
            }
        )
        print(f"[{len(rows)}/{len(dataset)}] {sample['name']}", flush=True)
    report = {
        "schema_version": 1,
        "num_samples": len(rows),
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("official_60fps", "motius_60fps", "motius_30fps")
        },
        "max_official_delta": float(max(row["official_delta"] for row in rows)),
        "mean_fps_delta": float(np.mean([row["fps_delta"] for row in rows])),
        "max_fps_delta": float(max(row["fps_delta"] for row in rows)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "rows": f"{len(rows)} entries"}, indent=2))


if __name__ == "__main__":
    main()
