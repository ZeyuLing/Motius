#!/usr/bin/env python3
"""Convert G1 NPZ trajectories with native FK, resampling and contact data components."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motius.motion.representation.protomotions.qpos_conversion import convert_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--robot-type", default="g1")
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--num-rank", type=int, default=1)
    parser.add_argument("--slurm-rank", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args()
    convert_directory(**vars(args))


if __name__ == "__main__":
    main()
