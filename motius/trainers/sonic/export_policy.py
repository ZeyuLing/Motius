"""Export a native SONIC checkpoint as a Pipeline-loadable artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .train import SonicTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides passed to the native exporter.",
    )
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(arguments)
    return SonicTrainer.export_policy(
        args.checkpoint,
        args.output,
        arguments=args.overrides,
    )


if __name__ == "__main__":
    main()
