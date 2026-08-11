#!/usr/bin/env python3
"""Split an existing SMPL gallery manifest into a light index and lazy chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smpl_gallery_assets import write_chunked_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    for path in args.manifest:
        manifest_path = path.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("case_descriptor_chunks"):
            print(f"skip already chunked: {manifest_path}")
            continue
        write_chunked_manifest(
            manifest_path.parent,
            manifest,
            chunk_size=args.chunk_size,
        )
        print(f"compacted {manifest_path}: {len(manifest['cases'])} cases")


if __name__ == "__main__":
    main()
