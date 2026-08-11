#!/usr/bin/env python3
"""Convert a directory of Motius motion135 clips to MotionStreamer-272."""

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


def _load_motion135(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if key in payload.files:
                motion = payload[key]
            elif {"global_orient", "body_pose", "transl"}.issubset(payload.files):
                from motius.motion import smpl_to_motion135

                motion = smpl_to_motion135(
                    payload["global_orient"], payload["body_pose"], payload["transl"]
                )
            else:
                raise KeyError(
                    f"{path} contains neither {key!r} nor SMPL parameter arrays"
                )
    else:
        motion = np.load(path, allow_pickle=False)
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 135:
        raise ValueError(f"Expected [T, >=135] motion at {path}, got {motion.shape}")
    if len(motion) < 2 or not np.isfinite(motion).all():
        raise ValueError(f"Motion at {path} must contain at least two finite frames")
    return motion[:, :135]


def _convert(task: tuple[str, str, str, bool]) -> tuple[str, str | None]:
    source_value, output_value, key, overwrite = task
    source = Path(source_value)
    output = Path(output_value)
    if output.is_file() and not overwrite:
        return "skipped", None
    try:
        from motius.motion import motion135_to_motion272

        converted = motion135_to_motion272(_load_motion135(source, key))
        converted = np.asarray(converted, dtype=np.float32)
        if converted.ndim != 2 or converted.shape[1] != 272:
            raise RuntimeError(f"converter returned {converted.shape}")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, converted)
        return "generated", None
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{source.name}: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--npz-key", default="motion_135")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.ids_file is not None:
        stems = [
            line.strip()
            for line in args.ids_file.resolve().read_text().splitlines()
            if line.strip()
        ]
        sources = [
            next(
                (
                    path
                    for path in (input_dir / f"{stem}.npz", input_dir / f"{stem}.npy")
                    if path.is_file()
                ),
                None,
            )
            for stem in stems
        ]
        missing = [stem for stem, source in zip(stems, sources) if source is None]
        sources = [source for source in sources if source is not None]
    else:
        sources = sorted((*input_dir.glob("*.npz"), *input_dir.glob("*.npy")))
        missing = []

    tasks = [
        (str(source), str(output_dir / f"{source.stem}.npy"), args.npz_key, args.overwrite)
        for source in sources
    ]
    counts = {"generated": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for status, error in executor.map(_convert, tasks, chunksize=16):
            counts[status] += 1
            if error is not None:
                failures.append(error)

    summary = {
        "source_representation": "motion135",
        "target_representation": "MotionStreamer-272",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "requested": len(tasks) + len(missing),
        **counts,
        "missing": len(missing),
        "missing_ids": missing[:100],
        "failures": failures[:100],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if missing or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
