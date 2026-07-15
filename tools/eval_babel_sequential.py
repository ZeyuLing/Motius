#!/usr/bin/env python3
"""Evaluate BABEL sequential generations with the Motius joints66 evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.evaluators.tmr import TMRTextMotionEvaluator
from motius.evaluation.sequential import (
    SequentialCase,
    evaluate_sequential_cases,
    load_joints66,
    load_joints66_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-dir")
    parser.add_argument(
        "--evaluator",
        default="ZeyuLing/motius-evaluator-universal-smplh-joints66",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-frames", type=int, default=30)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "val":
        raise ValueError("The FlowMDM-style BABEL sequential protocol requires split='val'.")
    prediction_dir = Path(args.predictions_dir).resolve() if args.predictions_dir else None
    cases = [
        SequentialCase.from_mapping(
            item,
            base_dir=manifest_path.parent,
            prediction_dir=prediction_dir,
        )
        for item in manifest.get("cases", [])
    ]
    def load_pool(key: str, legacy_key: str) -> list:
        pool_value = manifest.get(key)
        if pool_value:
            pool_path = Path(str(pool_value))
            if not pool_path.is_absolute():
                pool_path = manifest_path.parent / pool_path
            return load_joints66_pool(pool_path)
        return [
            load_joints66(
                Path(item["motion_path"])
                if Path(item["motion_path"]).is_absolute()
                else manifest_path.parent / item["motion_path"]
            )
            for item in manifest.get(legacy_key, [])
        ]

    reference_segments = load_pool("reference_segment_pool", "reference_segments")
    reference_transitions = load_pool(
        "reference_transition_pool", "reference_transitions"
    )
    evaluator = TMRTextMotionEvaluator.from_pretrained(
        args.evaluator,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    summary = evaluate_sequential_cases(
        cases,
        evaluator,
        reference_segment_pool=reference_segments or None,
        reference_transition_pool=reference_transitions or None,
        fps=args.fps,
        transition_frames=args.transition_frames,
        chunk_size=args.chunk_size,
        n_repeats=args.n_repeats,
        seed=args.seed,
        protocol=str(manifest["protocol"]),
    )
    summary["method"] = args.method
    summary["split"] = "val"
    summary["manifest"] = str(manifest_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
