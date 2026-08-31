#!/usr/bin/env python3
"""Synchronize the shared product stylesheet into leaderboard Spaces."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("leaderboard_product.css")
TARGETS = (
    ROOT / "docs/leaderboards/hf_space_body_part_condition_humanml3d/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_monocular_capture/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_motion_reconstruction/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_motion_repair/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_motion_tracking_isaaclab/leaderboard.css",
    ROOT / "docs/leaderboards/hf_space_motion_tracking_mujoco/leaderboard.css",
)


def sync_stylesheet(
    source: Path = SOURCE,
    targets: tuple[Path, ...] = TARGETS,
) -> list[Path]:
    """Write the canonical UTF-8 stylesheet with platform-independent LF endings."""
    stylesheet = source.read_text(encoding="utf-8")
    updated = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stylesheet, encoding="utf-8", newline="\n")
        updated.append(target)
    return updated


def main() -> None:
    for target in sync_stylesheet():
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
