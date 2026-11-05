"""Export a ProtoMotions checkpoint to its native current-weight Pipeline artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .train import ProtoMotionsTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> Path:
    args = build_parser().parse_args(arguments)
    return ProtoMotionsTrainer.export_policy(
        args.checkpoint,
        args.output,
        configuration=args.configuration,
        device=args.device,
    )


if __name__ == "__main__":
    main()
