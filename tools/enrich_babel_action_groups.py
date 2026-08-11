#!/usr/bin/env python3
"""Attach official BABEL action-taxonomy positive groups to a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.babel import enrich_manifest_action_groups


PROTOCOL = "babel-official-val-shortmerge30-llm-joints66-actiongroups-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--babel-annotations",
        default="data/babel/babel-teach/val.json",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", default=PROTOCOL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    annotations = json.loads(args.babel_annotations.read_text(encoding="utf-8"))
    output, stats = enrich_manifest_action_groups(
        manifest,
        annotations,
        protocol=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), **stats}, indent=2))


if __name__ == "__main__":
    main()
