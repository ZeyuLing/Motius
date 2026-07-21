#!/usr/bin/env python3
"""Materialize HumanML3D-263 and MotionStreamer-272 from TM2D joints."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resample_exact(values: np.ndarray, target_frames: int) -> np.ndarray:
    if len(values) == target_frames:
        return values.copy()
    source_times = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_times = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for channel in range(flat.shape[1]):
        result[:, channel] = np.interp(target_times, source_times, flat[:, channel])
    return result.reshape(target_frames, *values.shape[1:]).astype(np.float32)


def _load_joints(path: Path) -> np.ndarray:
    joints = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if joints.ndim == 2 and joints.shape[1] == 66:
        joints = joints.reshape(len(joints), 22, 3)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T,66) or (T,22,3), got {joints.shape}")
    if len(joints) < 3 or not np.isfinite(joints).all():
        raise ValueError("Motion must contain at least three finite frames")
    return joints


def _convert(task: tuple[str, str, str, float, float, float, bool]):
    source_value, hml_value, ms_value, source_fps, hml_fps, ms_fps, overwrite = task
    source = Path(source_value)
    hml_path = Path(hml_value)
    ms_path = Path(ms_value)
    if hml_path.is_file() and ms_path.is_file() and not overwrite:
        return "skipped", None
    try:
        from motius.motion import hml263_to_motion272, joints_to_hml263

        joints = _load_joints(source)

        hml_frames = max(3, int(round(len(joints) * hml_fps / source_fps)))
        hml263 = joints_to_hml263(_resample_exact(joints, hml_frames))

        ms_frames = max(3, int(round(len(joints) * ms_fps / source_fps)))
        ms_source = joints_to_hml263(joints)
        motion272 = hml263_to_motion272(
            ms_source,
            source_fps=source_fps,
            target_fps=ms_fps,
            target_len=ms_frames,
        )
        hml_path.parent.mkdir(parents=True, exist_ok=True)
        ms_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hml_path, np.asarray(hml263, dtype=np.float32))
        np.save(ms_path, np.asarray(motion272, dtype=np.float32))
        return "generated", None
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{source.name}: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints-dir", required=True, type=Path)
    parser.add_argument("--hml263-dir", required=True, type=Path)
    parser.add_argument("--motion272-dir", required=True, type=Path)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--hml-fps", type=float, default=20.0)
    parser.add_argument("--motion272-fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(args.joints_dir.resolve().glob("*.npy"))
    hml_dir = args.hml263_dir.resolve()
    ms_dir = args.motion272_dir.resolve()
    tasks = [
        (
            str(source),
            str(hml_dir / source.name),
            str(ms_dir / source.name),
            args.source_fps,
            args.hml_fps,
            args.motion272_fps,
            args.overwrite,
        )
        for source in sources
    ]
    counts = {"generated": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for status, error in executor.map(_convert, tasks, chunksize=8):
            counts[status] += 1
            if error is not None:
                failures.append(error)

    summary = {
        "source_representation": "canonical SMPL-22 joints66",
        "source_fps": args.source_fps,
        "outputs": {
            "HumanML3D-263": {"fps": args.hml_fps, "directory": str(hml_dir)},
            "MotionStreamer-272": {
                "fps": args.motion272_fps,
                "directory": str(ms_dir),
            },
        },
        "bridge": (
            "position IK on the generated joints; HumanML3D and MotionStreamer "
            "canonicalization are applied independently at their evaluator FPS"
        ),
        "requested": len(tasks),
        **counts,
        "failures": failures[:100],
    }
    hml_dir.mkdir(parents=True, exist_ok=True)
    ms_dir.mkdir(parents=True, exist_ok=True)
    (hml_dir.parent / "representation_conversion.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
