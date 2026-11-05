"""Prepare native G1 motion references from retargeted CSV trajectories."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from motius.configuration.gentrack import PROTOMOTIONS_ASSETS_ROOT
    from motius.motion.representation.protomotions.csv_conversion import (
        convert_csv_directory,
    )
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input-dir", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=PROTOMOTIONS_ASSETS_ROOT)
    for name, default in [
        ("input-fps", 30),
        ("output-fps", 30),
        ("ignore-first-n-frames", 0),
        ("num-rank", 1),
        ("slurm-rank", 0),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in [
        ("min-height-threshold", -0.05),
        ("max-velocity-threshold", 15.0),
        ("max-dof-vel-threshold", 40.0),
        ("duration-height-filter", 0.1),
        ("duration-height-seconds", 0.6),
    ]:
        parser.add_argument("--" + name, type=float, default=default)
    for name, default in [
        ("force-remake", False),
        ("apply-motion-filter", False),
        ("has-header", True),
        ("has-frame-column", True),
    ]:
        parser.add_argument(
            "--" + name, action=argparse.BooleanOptionalAction, default=default
        )
    for name, default in [
        ("robot-type", "g1"),
        ("pos-units", "cm"),
        ("rot-format", "euler_deg"),
        ("joint-units", "deg"),
        ("euler-order", "xyz"),
    ]:
        parser.add_argument("--" + name, default=default)
    parser.add_argument("--contact-labels-dir", type=Path)
    with torch.no_grad():
        convert_csv_directory(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
