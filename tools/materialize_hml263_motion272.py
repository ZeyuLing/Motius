#!/usr/bin/env python3
"""Materialize MotionStreamer-272 predictions without an SMPL/mesh IK bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import hml263_to_motion272


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ids_file:
        ids = [
            line.strip()
            for line in args.ids_file.resolve().read_text().splitlines()
            if line.strip()
        ]
    else:
        ids = sorted(path.stem for path in input_dir.glob("*.npy"))
    ids = [
        case_id
        for index, case_id in enumerate(ids)
        if index % args.num_shards == args.shard_index
    ]

    generated = skipped = missing = failed = 0
    failures: list[dict[str, str]] = []
    for index, case_id in enumerate(ids):
        source = input_dir / f"{case_id}.npy"
        destination = output_dir / f"{case_id}.npy"
        if not source.is_file():
            missing += 1
            continue
        if args.skip_existing and destination.is_file():
            skipped += 1
            continue
        try:
            converted = hml263_to_motion272(
                np.load(source),
                source_fps=args.source_fps,
                target_fps=args.target_fps,
            )
            np.save(destination, converted)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"case_id": case_id, "error": str(exc)})
        if (index + 1) % 250 == 0:
            print(
                f"[{index + 1}/{len(ids)}] generated={generated} "
                f"skipped={skipped} missing={missing} failed={failed}",
                flush=True,
            )

    result = {
        "source_representation": "HumanML3D-263",
        "target_representation": "MotionStreamer-272",
        "bridge": (
            "native positions plus HumanML incoming-bone rotations mapped to "
            "the canonical MotionStreamer skeleton; no SMPL IK"
        ),
        "requested": len(ids),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generated": generated,
        "skipped": skipped,
        "missing": missing,
        "failed": failed,
        "failures": failures[:100],
    }
    summary = output_dir / f"conversion_shard_{args.shard_index:03d}.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
