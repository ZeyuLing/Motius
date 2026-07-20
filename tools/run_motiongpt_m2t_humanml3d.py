#!/usr/bin/env python3
"""Run MotionGPT motion captioning on the official HumanML3D M2T split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motius.evaluation.m2t import load_humanml3d_m2t_samples, write_prediction_records
from motius.pipelines.motiongpt import MotionGPTPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ZeyuLing/Motius-MotionGPT-HumanML3D",
        help="Hugging Face model id or local MotionGPT artifact directory.",
    )
    parser.add_argument("--data-root", required=True, help="HumanML3D dataset root.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-file", default="test.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--full-motion-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    samples = load_humanml3d_m2t_samples(
        args.data_root,
        args.split_file,
        include_subclips=not args.full_motion_only,
        max_samples=args.max_samples,
    )
    pending = [
        sample
        for sample in samples
        if args.overwrite or not (prediction_dir / f"{sample.sample_id}.json").exists()
    ]
    pipeline = MotionGPTPipeline.from_pretrained(
        args.model,
        bundle_kwargs={
            "device": args.device,
            "local_files_only": args.local_files_only,
        },
    )

    completed = len(samples) - len(pending)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        motions = [sample.load_motion() for sample in batch]
        predictions = pipeline.infer_m2t(
            motions,
            lengths=[sample.length for sample in batch],
        )
        write_prediction_records(output_dir, zip(batch, predictions))
        completed += len(batch)
        print(f"[motiongpt:m2t] {completed}/{len(samples)}", flush=True)

    manifest = {
        "task": "M2T",
        "method": "MotionGPT",
        "model": args.model,
        "dataset": "HumanML3D",
        "split_file": str(args.split_file),
        "include_subclips": not args.full_motion_only,
        "reference_policy": "tm2t_first3_repeat_to3",
        "num_samples": len(samples),
        "predictions_dir": str(prediction_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
