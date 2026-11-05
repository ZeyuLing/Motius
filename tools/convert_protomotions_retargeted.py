"""Prepare native G1 motion references from PyRoki retargeting output."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from motius.configuration.gentrack import PROTOMOTIONS_ASSETS_ROOT
    from motius.motion.representation.protomotions.retargeted_conversion import (
        convert_retargeted_directory,
    )
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retargeted-motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=PROTOMOTIONS_ASSETS_ROOT)
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int)
    parser.add_argument("--ignore-first-n-frames", type=int, default=0)
    parser.add_argument("--force-remake", action="store_true")
    parser.add_argument("--apply-motion-filter", action="store_true")
    for name, default in [
        ("min-height-threshold", -0.05),
        ("max-velocity-threshold", 15.0),
        ("max-dof-vel-threshold", 40.0),
        ("duration-height-filter", 0.1),
        ("duration-height-seconds", 0.6),
    ]:
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--robot-type", default="g1")
    parser.add_argument("--contact-labels-dir", type=Path)
    with torch.no_grad():
        convert_retargeted_directory(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
