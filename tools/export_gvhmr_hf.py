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
    bootstrap = output.parent / ".gvhmr_export_bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / "gvhmr_config.json").write_text(
        json.dumps(
            {
                "artifact_format": GVHMR_ARTIFACT_FORMAT,
                "source_revision": OFFICIAL_RUNTIME_REVISION,
            }
        )
        + "\n"
    )
    bundle = GVHMRBundle(artifact_root=bootstrap)
    bundle.save_pretrained(
        str(output),
        source_assets={
            relative: _asset_path(asset_root, relative)
            for relative in (
                "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
                "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
                "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
                "inputs/checkpoints/yolo/yolov8x.pt",
            )
        },
    )
    print(output)


if __name__ == "__main__":
    main()
