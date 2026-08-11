#!/usr/bin/env python3
"""Export InterGen or InterMask as inference-only Motius Hub artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from motius.models.intergen import InterGenBundle
from motius.models.intermask import InterMaskBundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(output: Path, source: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (output / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": {"type": "processed official checkpoint"},
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("intergen", "intermask"))
    parser.add_argument("--source", required=True, help="Processed source artifact")
    parser.add_argument("--output", required=True, help="Inference artifact directory")
    parser.add_argument("--dataset", default="interhuman", choices=("interhuman", "interx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method == "intergen":
        bundle = InterGenBundle.from_pretrained(args.source, device=args.device)
    else:
        bundle = InterMaskBundle.from_pretrained(
            args.source,
            dataset_name=args.dataset,
            device=args.device,
        )
    output = Path(args.output)
    bundle.save_pretrained(str(output))
    write_artifact_manifest(output, Path(args.source))
    if args.repo_id:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(output),
            commit_message=f"Publish Motius {args.method} artifact",
        )
    print(output)


if __name__ == "__main__":
    main()
