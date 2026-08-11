#!/usr/bin/env python3
"""Download the source-pinned Unitree G1 assets used by BeyondMimic."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Optional
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "checkpoints/robots"
ASSET_URL = (
    "https://storage.googleapis.com/qiayuanl_robot_descriptions/"
    "unitree_description.tar.gz"
)
ASSET_SHA256 = "b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674"
INCLUDED_PREFIXES = (
    PurePosixPath("unitree_description/urdf/g1"),
    PurePosixPath("unitree_description/meshes/g1"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_selected(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return any(path == prefix or prefix in path.parents for prefix in INCLUDED_PREFIXES)


def extract_assets(archive: Path, output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not _is_selected(member.name):
                continue
            destination = output_root.joinpath(*PurePosixPath(member.name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    if extracted == 0:
        raise RuntimeError("The verified archive contains no Unitree G1 assets")
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    urdf = output / "unitree_description/urdf/g1/main.urdf"
    if urdf.is_file() and not args.force:
        print(f"BeyondMimic assets already exist under {output}")
        return 0

    temporary: Optional[OptionalTemporary] = None
    if args.archive is None:
        temporary = OptionalTemporary()
        archive = temporary.path
        print(f"Downloading {ASSET_URL}")
        with urllib.request.urlopen(ASSET_URL) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    else:
        archive = args.archive.expanduser().resolve()

    try:
        actual = _sha256(archive)
        if actual != ASSET_SHA256:
            raise RuntimeError(
                f"Unitree asset SHA256 mismatch: expected {ASSET_SHA256}, got {actual}"
            )
        count = extract_assets(archive, output)
    finally:
        if temporary is not None:
            temporary.close()

    print(f"Installed {count} verified G1 files under {output}")
    return 0


class OptionalTemporary:
    def __init__(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        self.path = Path(handle.name)
        handle.close()

    def close(self) -> None:
        self.path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
