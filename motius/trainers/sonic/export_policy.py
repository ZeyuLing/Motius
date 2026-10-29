"""Export a native SONIC checkpoint as a Pipeline-loadable artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .train import export_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", type=Path)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(arguments)
    return export_policy(
        args.checkpoint,
        args.output,
        configuration=args.configuration,
    )


if __name__ == "__main__":
    main()
