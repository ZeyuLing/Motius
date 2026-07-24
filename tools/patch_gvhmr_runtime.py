#!/usr/bin/env python3
"""Apply Motius' deterministic robustness patch to a pinned GVHMR runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


_RELATIVE_SOLVER = Path(
    "hmr4d/utils/preproc/relpose/solver_two_view.py"
)
_ORIGINAL = """\
        # cam2_from_cam1 means T_0_to_1 in our language
        Rt = answer.cam2_from_cam1.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4)
        T[:3] = Rt
        return T
"""
_PATCHED = """\
        # cam2_from_cam1 means T_0_to_1 in our language. Low-texture or
        # near-static pairs can legitimately have no estimated relative pose.
        # Preserve the previous accumulated camera pose for that interval.
        relative_pose = None if answer is None else answer.cam2_from_cam1
        if relative_pose is None:
            return np.eye(4, dtype=np.float32)
        Rt = relative_pose.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3] = Rt
        return T
"""
_DEMO = Path("tools/demo/demo.py")
_DEMO_IMPORT_ORIGINAL = """\
import cv2
import torch
"""
_DEMO_IMPORT_PATCHED = """\
import os

import cv2
import torch
"""
_DEMO_RENDER_ORIGINAL = """\
    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
"""
_DEMO_RENDER_PATCHED = """\
    # ===== Render ===== #
    if os.environ.get("MOTIUS_GVHMR_SKIP_RENDER") == "1":
        Log.info("[Done] Skipping optional demo renders for Motius evaluation")
    else:
        render_incam(cfg)
        render_global(cfg)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
"""


def patch_runtime(runtime_root: str | Path, *, restore: bool = False) -> Path:
    """Patch or restore the pinned runtime, failing on source drift."""

    root = Path(runtime_root).expanduser().resolve()
    solver = root / _RELATIVE_SOLVER
    demo = root / _DEMO
    replacements = (
        (solver, _PATCHED, _ORIGINAL, "two-view solver"),
        (demo, _DEMO_RENDER_PATCHED, _DEMO_RENDER_ORIGINAL, "demo render block"),
        (demo, _DEMO_IMPORT_PATCHED, _DEMO_IMPORT_ORIGINAL, "demo imports"),
    )
    if not restore:
        replacements = tuple(
            (path, destination, source, label)
            for path, source, destination, label in replacements
        )
    for path, source, destination, label in replacements:
        if not path.is_file():
            raise FileNotFoundError(f"GVHMR {label} file not found: {path}")
        text = path.read_text()
        if restore and source in text:
            path.write_text(text.replace(source, destination, 1))
            continue
        if destination in text:
            continue
        if source not in text:
            raise RuntimeError(
                f"Pinned GVHMR {label} changed; refusing a fuzzy runtime patch."
            )
        path.write_text(text.replace(source, destination, 1))
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    print(patch_runtime(args.runtime_root, restore=args.restore))


if __name__ == "__main__":
    main()
