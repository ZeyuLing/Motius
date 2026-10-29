#!/usr/bin/env python3
"""Package prepared motion references with the native MotionLib data reader."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motius.datasets.protomotions_motion import MotionLib, MotionLibConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--clip-delta", type=float, default=3.0)
    parser.add_argument("--clip-seed", type=int, default=42)
    args = parser.parse_args()
    config = MotionLibConfig(
        motion_file=args.motion_path,
        max_seconds=args.max_seconds,
        clip_delta=args.clip_delta,
        clip_seed=args.clip_seed,
    )
    MotionLib(config, device=args.device).save_to_file(args.output_file)


if __name__ == "__main__":
    main()
