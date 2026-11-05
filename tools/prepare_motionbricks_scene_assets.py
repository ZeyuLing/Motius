#!/usr/bin/env python3
"""Download pinned official scene data with SHA256 verification.

BVH/CSV motions retain the included NVIDIA Sample Data Evaluation License.
These external assets are not part of the native model checkpoint.
"""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path


def prepare(output, workers=4):
    manifest = json.loads(
        files("motius.configuration.motionbricks_assets")
        .joinpath("scene_checksums.json")
        .read_text()
    )
    base = (
        "https://media.githubusercontent.com/media/"
        + manifest["source_repository"]
        + "/"
        + manifest["revision"]
        + "/"
    )

    def fetch(entry):
        destination = output / entry["path"]

        def valid(path):
            return (
                path.is_file()
                and path.stat().st_size == entry["size"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            )

        if valid(destination):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        try:
            # media serves LFS objects; raw serves regular source data.
            for prefix in (
                base,
                base.replace(
                    "media.githubusercontent.com/media", "raw.githubusercontent.com"
                ),
            ):
                try:
                    urllib.request.urlretrieve(prefix + entry["source"], temporary)
                    if valid(temporary):
                        break
                except Exception:
                    temporary.unlink(missing_ok=True)
            if not valid(temporary):
                raise ValueError("Download/hash verification failed: " + entry["path"])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, manifest["files"]))
    print(f"Verified {len(manifest['files'])} pinned assets in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/models/motionbricks_scene_assets"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.output, args.workers)
