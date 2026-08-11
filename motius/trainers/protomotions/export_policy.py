"""Export a ProtoMotions checkpoint to its deployable ONNX package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .train import ProtoMotionsTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(arguments)
    return ProtoMotionsTrainer.export_policy(
        args.checkpoint,
        args.output,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
