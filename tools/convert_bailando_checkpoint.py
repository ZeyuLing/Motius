#!/usr/bin/env python3
"""Convert trusted official Bailando checkpoints to a Motius artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.models.bailando.bundle import (
    BAILANDO_SOURCE_REPOSITORY,
    BAILANDO_SOURCE_REVISION,
    BailandoBundle,
)


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqvae", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = {
        "source_repository": BAILANDO_SOURCE_REPOSITORY,
        "source_revision": BAILANDO_SOURCE_REVISION,
        "source_checkpoints": {
            "vqvae": {"filename": args.vqvae.name, "sha256": sha256(args.vqvae)},
            "gpt": {"filename": args.gpt.name, "sha256": sha256(args.gpt)},
        },
    }
    bundle = BailandoBundle(
        vqvae_weights=str(args.vqvae),
        gpt_weights=str(args.gpt),
        provenance=provenance,
        strict=True,
    )
    bundle.save_pretrained(args.output)
    report = {
        "output": str(args.output.resolve()),
        "load_reports": bundle.load_reports,
        "provenance": provenance,
    }
    (args.output / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
