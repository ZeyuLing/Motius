#!/usr/bin/env python3
"""Export exact official GVHMR assets as one Motius HF artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.gvhmr import (
    GVHMR_ARTIFACT_FORMAT,
    GVHMRBundle,
    OFFICIAL_RUNTIME_REVISION,
)


def _asset_path(root: Path, relative: str) -> Path:
    canonical = root / relative
    if canonical.is_file():
        return canonical
    compact = root / Path(relative).relative_to("inputs/checkpoints")
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("--output-dir must live under repository outputs/.") from exc

    asset_root = args.asset_root.expanduser().resolve()
    from motius.pipelines.gvhmr import GVHMRPipeline
    bundle=GVHMRBundle(artifact_root=asset_root)
    GVHMRPipeline(bundle).save_pretrained(output,include_frontends=True)
    print(output)


if __name__ == "__main__":
    main()
