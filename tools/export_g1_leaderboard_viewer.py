#!/usr/bin/env python3
"""Export a portable Unitree G1 qualitative-comparison manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLUMNS = (
    ("gt", "GT G1 reference", "gt"),
    ("kimodo", "KIMODO-G1", "kimodo"),
    ("hymotion_g1", "HYMotion-G1", "hymotion_g1"),
)


def export_manifest(
    source: Path,
    output: Path,
    *,
    asset_base_url: str,
    mesh_base_url: str,
) -> dict:
    raw = json.loads(source.read_text())
    cases = []
    for row in raw["rows"]:
        assets = {}
        for source_key, _, public_key in COLUMNS:
            column = row["columns"][source_key]
            if column.get("status") != "ready":
                raise ValueError(
                    f"{row['keyid']} · {source_key}: asset is not ready"
                )
            assets[public_key] = (
                f"frames/{public_key}/{Path(column['path']).name}"
            )
        cases.append(
            {
                "case_id": row["keyid"],
                "prompt": row["prompt"],
                "frames": row["display_frames"],
                "fps": 30,
                "assets": assets,
            }
        )

    manifest = {
        "schema_version": 1,
        "title": "Text-to-Motion · Unitree G1 qualitative comparison",
        "representation": "Unitree G1 robot body transforms",
        "benchmark_population": 1024,
        "population": len(cases),
        "asset_base_url": asset_base_url.rstrip("/") + "/",
        "mesh_base_url": mesh_base_url.rstrip("/") + "/",
        "columns": [
            {"key": public_key, "label": label}
            for _, label, public_key in COLUMNS
        ],
        "cases": cases,
        "provenance": {
            "source_manifest_name": source.name,
            "source_rows": len(raw["rows"]),
            "note": (
                "The public viewer is a fixed qualitative subset. Metrics use "
                "the complete 1,024-case protocol."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asset-base-url",
        default=(
            "https://huggingface.co/datasets/ZeyuLing/"
            "Motius-Leaderboard-Cases/resolve/main/t2m-unitree-g1"
        ),
    )
    parser.add_argument(
        "--mesh-base-url",
        default=(
            "https://huggingface.co/datasets/ZeyuLing/"
            "Motius-Leaderboard-Cases/resolve/main/t2m-unitree-g1/meshes"
        ),
    )
    args = parser.parse_args()
    manifest = export_manifest(
        args.source_manifest,
        args.output,
        asset_base_url=args.asset_base_url,
        mesh_base_url=args.mesh_base_url,
    )
    print(
        json.dumps(
            {"output": str(args.output), "cases": manifest["population"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
