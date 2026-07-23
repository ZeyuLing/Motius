#!/usr/bin/env python3
"""Create or validate the canonical Motius evaluation artifact layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.artifacts import EvaluationArtifactLayout


def _metadata(values: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Metadata must use KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Metadata key must not be empty: {item!r}")
        try:
            metadata[key] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key] = value
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/evaluation")
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    path_parser = subparsers.add_parser("path")
    path_parser.add_argument("--method")
    path_parser.add_argument("--run")

    protocol_parser = subparsers.add_parser("init-protocol")
    protocol_parser.add_argument("--meta", action="append", default=[])

    run_parser = subparsers.add_parser("init-run")
    run_parser.add_argument("--method", required=True)
    run_parser.add_argument("--run", required=True)
    run_parser.add_argument("--meta", action="append", default=[])

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--require-manifest", action="store_true")

    args = parser.parse_args()
    layout = EvaluationArtifactLayout(
        task_id=args.task,
        benchmark_id=args.benchmark,
        protocol_id=args.protocol,
        root=args.root,
    )

    if args.command == "path":
        if bool(args.method) != bool(args.run):
            parser.error("path requires both --method and --run, or neither")
        print(
            layout.run_root(args.method, args.run)
            if args.method
            else layout.protocol_root
        )
    elif args.command == "init-protocol":
        print(layout.init_protocol(_metadata(args.meta)))
    elif args.command == "init-run":
        print(layout.init_run(args.method, args.run, _metadata(args.meta)))
    else:
        errors = layout.validate(require_manifest=args.require_manifest)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            raise SystemExit(1)
        print(f"OK: {layout.protocol_root}")


if __name__ == "__main__":
    main()
