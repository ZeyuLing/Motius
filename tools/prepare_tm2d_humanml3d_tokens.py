#!/usr/bin/env python3
"""Precompute deterministic TM2D words for selected HumanML3D captions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.models.tm2d import TM2DTokenizer
from tools.infer_tm2d_humanml3d import selected_caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotation", type=Path)
    source.add_argument("--case-manifest", type=Path)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vocabulary = json.loads(args.vocabulary.read_text())
    tokenizer = TM2DTokenizer(vocabulary)
    if args.case_manifest is not None:
        payload = json.loads(args.case_manifest.read_text())
        rows = [
            (str(row["case_id"]), str(row["references"][0]))
            for row in payload["cases"]
        ]
    else:
        if args.path_root is None:
            parser.error("--annotation requires --path-root")
        protocol = json.loads(args.annotation.read_text())
        rows = []
        for name, record in sorted(protocol["data_list"].items()):
            caption_path = Path(record["hierarchical_caption_path"])
            if not caption_path.is_absolute():
                caption_path = args.path_root / caption_path
            rows.append((name, selected_caption(caption_path)))
    tokenized = tokenizer.tokenize_batch([caption for _, caption in rows])
    result = {name: words for (name, _), words in zip(rows, tokenized)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(result)} tokenized captions to {args.output}")


if __name__ == "__main__":
    main()
