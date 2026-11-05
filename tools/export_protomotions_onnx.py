#!/usr/bin/env python3
"""Export a current native tracker through its deployment Pipeline composition."""
from pathlib import Path
import argparse, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motius.pipelines.protomotions.export import export_tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", type=Path)
    args = parser.parse_args()
    print(
        export_tracker(args.checkpoint, args.output, configuration=args.configuration)
    )


if __name__ == "__main__":
    main()
