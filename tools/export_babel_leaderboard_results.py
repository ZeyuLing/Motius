#!/usr/bin/env python3
"""Export the BABEL leaderboard's inline rows as machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import json5


ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "docs/leaderboards/hf_space_babel_sequential"
HTML_PATH = SPACE / "index.html"
OUTPUT_PATH = SPACE / "babel_results.json"


def export_payload() -> dict:
    source = HTML_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"\bconst\s+rows\s*=\s*Object\.freeze\((\[.*?\])\);",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not find the BABEL leaderboard rows")
    rows = json5.loads(match.group(1))
    return {
        "schema_version": 1,
        "benchmark": "Sequential Text-to-Motion · BABEL",
        "protocol": "babel-actiongroups-v3",
        "retrieval_batch_size": 32,
        "fid_space": "per-sample L2-normalized uTMR embedding space",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(export_payload(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text() != rendered:
            print(
                f"stale: {OUTPUT_PATH.relative_to(ROOT)}; run "
                "python tools/export_babel_leaderboard_results.py"
            )
            return 1
        print("BABEL Leaderboard JSON matches the published HTML rows")
        return 0
    OUTPUT_PATH.write_text(rendered)
    print(OUTPUT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
