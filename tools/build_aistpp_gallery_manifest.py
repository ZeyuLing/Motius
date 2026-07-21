#!/usr/bin/env python3
"""Build an all-case AIST++ manifest for the shared SMPL gallery packer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GENRES = {
    "BR": "Break",
    "HO": "House",
    "JB": "Jazz Ballet",
    "JS": "Jazz",
    "KR": "Krump",
    "LH": "LA Style Hip-hop",
    "LO": "Lock",
    "MH": "Middle Hip-hop",
    "PO": "Pop",
    "WA": "Waacking",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    cases = []
    for sequence_id in inference["samples"]:
        fields = sequence_id.split("_")
        genre_code = fields[0][1:]
        music_id = next(field for field in fields if field.startswith("m"))
        with np.load(args.motion_dir / f"{sequence_id}.npz", allow_pickle=False) as payload:
            frames = len(payload["motion_135"])
        cases.append(
            {
                "case_id": sequence_id,
                "sample_id": sequence_id,
                "case_key": sequence_id,
                "frames": frames,
                "fps": 30.0,
                "audio": f"audio/{music_id}.mp3",
                "audio_start_seconds": 0.0,
                "audio_end_seconds": frames / 30.0,
                "references": [GENRES.get(genre_code, genre_code), f"Music {music_id}"],
            }
        )
    output = {
        "schema_version": 1,
        "task": "music_to_dance",
        "protocol": "AIST++ public crossmodal 40-case split",
        "reference_label": "Dance style and music ID",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
