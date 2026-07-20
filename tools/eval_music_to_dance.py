#!/usr/bin/env python3
"""Evaluate generated AIST++ SMPL-24 joints with the Bailando protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from motius.datasets.aistpp_music_to_dance import AISTPPMusicDanceDataset
from motius.evaluation.music_to_dance import AISTPPMusicDanceEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--music-feature-root", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-frames", type=int, default=1_200)
    parser.add_argument(
        "--reference-features",
        type=Path,
        help="NPZ feature pool built by build_aistpp_reference_features.py.",
    )
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Evaluate GT against itself as a protocol sanity check.",
    )
    parser.add_argument("--no-physical", action="store_true")
    args = parser.parse_args()
    if not args.ground_truth and args.pred_root is None:
        parser.error("--pred-root is required unless --ground-truth is set")
    return args


def main() -> None:
    args = parse_args()
    dataset = AISTPPMusicDanceDataset(
        args.data_root,
        args.music_feature_root,
        max_samples=args.max_samples,
    )
    evaluator = AISTPPMusicDanceEvaluator(
        max_frames=args.max_frames,
        physical=not args.no_physical,
        reference_feature_path=args.reference_features,
    )
    evaluated = []
    for sample in dataset:
        if args.ground_truth:
            pred_joints = sample["gt_joints"]
        else:
            pred_path = args.pred_root / f"{sample['name']}.npz"
            if not pred_path.is_file():
                print(f"skip missing prediction: {pred_path}")
                continue
            with np.load(pred_path, allow_pickle=False) as payload:
                pred_joints = payload["joints"]
        evaluator.process(
            {
                "name": sample["name"],
                "pred_joints": pred_joints,
                "gt_joints": sample["gt_joints"],
                "music_beats": sample["music_beats"],
                "music_fps": sample["music_fps"],
                "motion_fps": sample["motion_fps"],
            }
        )
        evaluated.append(sample["name"])

    metrics = evaluator.compute()
    report = {
        "schema_version": 1,
        "task": "music_to_dance",
        "dataset": "AIST++ official crossmodal test+validation package",
        "protocol": "Bailando CVPR 2022",
        "prediction": "ground_truth" if args.ground_truth else str(args.pred_root),
        "max_frames": args.max_frames,
        "reference_features": (
            str(args.reference_features) if args.reference_features else None
        ),
        "samples": evaluated,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
