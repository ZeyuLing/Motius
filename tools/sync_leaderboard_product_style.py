#!/usr/bin/env python3
"""Synchronize the shared product stylesheet into newer leaderboard Spaces."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("leaderboard_product.css")
TARGETS = (
    ROOT / "docs/leaderboards/hf_space_body_part_condition_humanml3d/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_motion_reconstruction/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_monocular_capture/leaderboard.css",
)


def main() -> None:
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, target)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
