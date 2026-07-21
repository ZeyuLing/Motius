#!/usr/bin/env python3
"""Materialize all AIST++ GT/Bailando SMPL motions across local GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smpl-model-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--refine-iters", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = [str(value) for value in manifest["samples"]]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("materialize_bailando_smpl_preview.py")

    def run_shard(phase: str, gpu: str, shard: int) -> None:
        ids = samples[shard:: len(gpus)]
        output_dir = args.output_root / phase
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(worker),
            "--input-dir",
            str(args.prediction_dir if phase == "bailando" else args.gt_dir),
            "--input-format",
            "prediction_npz" if phase == "bailando" else "aistpp_json",
            "--output-dir",
            str(output_dir),
            "--ids",
            ",".join(ids),
            "--smpl-model-dir",
            str(args.smpl_model_dir),
            "--refine-iters",
            str(args.refine_iters),
            "--manifest-name",
            f"manifest_shard_{shard:02d}.json",
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        log = args.output_root / f"{phase}_shard_{shard:02d}.log"
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for phase in ("bailando", "gt"):
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(run_shard, phase, gpu, shard)
                for shard, gpu in enumerate(gpus)
            ]
            for future in futures:
                future.result()
        print(f"completed {phase}: {len(samples)} clips", flush=True)

    report = {
        "schema_version": 1,
        "samples": samples,
        "methods": {
            "gt": str((args.output_root / "gt").resolve()),
            "bailando": str((args.output_root / "bailando").resolve()),
        },
        "target_fps": 30.0,
        "max_source_frames": 1_200,
        "refine_iters": args.refine_iters,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
