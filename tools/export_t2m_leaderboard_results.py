#!/usr/bin/env python3
"""Export the inline T2M Leaderboard rows as canonical machine-readable JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import json5


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_DIR = ROOT / "docs" / "leaderboards" / "hf_space_t2m_humanml3d"
HTML_PATH = LEADERBOARD_DIR / "index.html"
OUTPUT_PATH = LEADERBOARD_DIR / "t2m_results.json"
ARRAY_NAMES = (
    "semanticRows",
    "physicalRows",
    "paperRows",
    "calibrationRows",
)


def _extract_array(source: str, name: str) -> list[dict]:
    match = re.search(
        rf"\bconst\s+{re.escape(name)}\s*=\s*(\[.*?\]);",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Could not find JavaScript array {name!r}")
    rows = json5.loads(match.group(1))
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise TypeError(f"{name!r} is not an array of objects")
    return rows


def export_payload() -> dict:
    source = HTML_PATH.read_text(encoding="utf-8")
    arrays = {name: _extract_array(source, name) for name in ARRAY_NAMES}
    return {
        "schema_version": 1,
        "benchmark": "Text-to-Motion - SMPL Skeleton (HumanML3D)",
        "source": "index.html",
        "metric_protocol": {
            "caption": "HumanML3D selected-caption test protocol",
            "retrieval_batch_size": 32,
            "motius_fid_field": "utmrFIDNorm",
            "motius_fid_space": "per-sample L2-normalized embedding space",
            "raw_fid_policy": (
                "utmrFID is retained for provenance but must not be displayed "
                "or ranked as Motius FID"
            ),
        },
        "semantic_rows": arrays["semanticRows"],
        "physical_rows": arrays["physicalRows"],
        "paper_rows": arrays["paperRows"],
        "calibration_rows": arrays["calibrationRows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed JSON differs from the HTML rows.",
    )
    args = parser.parse_args()
    rendered = json.dumps(
        export_payload(),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if args.check:
        if not OUTPUT_PATH.is_file():
            print(f"missing: {OUTPUT_PATH.relative_to(ROOT)}")
            return 1
        if OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(
                "stale: "
                f"{OUTPUT_PATH.relative_to(ROOT)}; run "
                "python tools/export_t2m_leaderboard_results.py"
            )
            return 1
        print("T2M Leaderboard JSON matches the published HTML rows")
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(OUTPUT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
