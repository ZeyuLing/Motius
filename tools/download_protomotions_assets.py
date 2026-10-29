#!/usr/bin/env python3
"""Materialize only the pinned G1 physics assets, independently of source code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request

MANIFEST = (
    Path(__file__).resolve().parents[1] / "motius/licenses/protomotions/g1-assets.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "checkpoints/robots/protomotions"


def install_assets(output, *, source_dir=None):
    manifest = json.loads(MANIFEST.read_text())
    output = Path(output).expanduser().resolve()
    for item in manifest["files"]:
        original = PurePosixPath(item["file"])
        prefix = PurePosixPath("protomotions/data/assets")
        relative = original.relative_to(prefix)
        if original.is_absolute() or ".." in original.parts:
            raise ValueError("Unsafe asset path: " + str(original))
        destination = output.joinpath(*relative.parts)
        expected = item["sha256"]

        def verified(path):
            return (
                path.is_file()
                and path.stat().st_size == item["size"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected
            )

        if verified(destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            if source_dir is not None:
                shutil.copyfile(
                    Path(source_dir).joinpath(*relative.parts), temporary_path
                )
            else:
                endpoint = (
                    "media.githubusercontent.com/media"
                    if item["lfs_sha256"]
                    else "raw.githubusercontent.com"
                )
                url = f"https://{endpoint}/{manifest['repository']}/{manifest['revision']}/{original}"
                with (
                    urllib.request.urlopen(url, timeout=120) as source,
                    temporary_path.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
            if not verified(temporary_path):
                raise RuntimeError(
                    "Pinned asset size/SHA256 mismatch: " + str(original)
                )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional already materialized asset root; all files are still hash checked",
    )
    args = parser.parse_args()
    print(
        f"Verified {install_assets(args.output, source_dir=args.source_dir)} G1 assets under {args.output}"
    )


if __name__ == "__main__":
    main()
