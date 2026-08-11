#!/usr/bin/env python3
"""Materialize scattered selected captions into one evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_motionclr_humanml3d import _caption, _entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--annotation-root", default=".")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.annotation).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    materialized = {}
    missing = []
    for motion_id, entry in _entries(source):
        caption = _caption(entry, annotation_root)
        if caption is None:
            missing.append(motion_id)
            continue
        value = dict(entry)
        value["selected_caption"] = caption
        materialized[motion_id] = value
    if missing:
        raise ValueError(f"Missing selected captions for {len(missing)} entries: {missing[:10]}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "caption_protocol": "HumanML3D selected caption",
                "source_annotation": str(source),
                "data_list": materialized,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"entries": len(materialized), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
