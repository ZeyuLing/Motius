#!/usr/bin/env python3
"""Materialize shared Temporal Motion Completion GT as physical HML263 clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CANONICAL_ROOT = Path(
    "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/"
    "outputs/evaluation/temporal/humanml3d_official_test_4012"
)
DEFAULT_BONE_OFFSETS = (
    REPO_ROOT / "motius" / "motion" / "assets" / "bone_offsets_canon272.npy"
)


def _valid_hml263(path: Path) -> bool:
    try:
        value = np.asarray(np.load(path, allow_pickle=False))
    except (OSError, ValueError):
        return False
    return (
        value.ndim == 2
        and value.shape[0] >= 2
        and value.shape[1] == 263
        and bool(np.isfinite(value).all())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--source-setting", default="temporal_start_1f")
    parser.add_argument("--source-method", default="condmdi")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bone-offsets", type=Path, default=DEFAULT_BONE_OFFSETS)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = (
        args.canonical_root.expanduser().resolve()
        / args.source_setting
        / args.source_method
        / "eval_npz"
    )
    files = sorted(source_dir.glob("*.npz"))
    if args.max_samples:
        files = files[: args.max_samples]
    if not files:
        raise FileNotFoundError(f"No canonical evaluation NPZ files in {source_dir}")

    from motius.motion.representation import motion135_to_hml263

    bone_offsets = np.asarray(np.load(args.bone_offsets), dtype=np.float32)
    if bone_offsets.shape != (22, 3):
        raise ValueError(f"Expected 22x3 bone offsets, got {bone_offsets.shape}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    failures: list[dict[str, str]] = []
    started = time.time()
    for index, source_path in enumerate(files):
        destination = output_dir / f"{source_path.stem}.npy"
        if args.skip_existing and _valid_hml263(destination):
            skipped += 1
            continue
        try:
            with np.load(source_path, allow_pickle=False) as payload:
                motion = np.asarray(payload["gt_motion_135"], dtype=np.float32)
            value = np.asarray(
                motion135_to_hml263(
                    motion[:, :135],
                    bone_offsets=bone_offsets,
                    src_fps=args.source_fps,
                    dst_fps=args.target_fps,
                    resample="linear",
                    coordinate_system="humanml",
                ),
                dtype=np.float32,
            )
            if value.ndim != 2 or value.shape[1] != 263 or not np.isfinite(value).all():
                raise RuntimeError(f"invalid HML263 output {value.shape}")
            temporary = destination.with_name(
                f".{destination.stem}.{os.getpid()}-{time.time_ns()}.tmp.npy"
            )
            np.save(temporary, value)
            os.replace(temporary, destination)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"case_id": source_path.stem, "error": str(exc)})
        if (index + 1) % 100 == 0:
            print(
                f"[progress] {index + 1}/{len(files)} written={written} "
                f"skipped={skipped} failed={failed}",
                flush=True,
            )

    summary = {
        "source": str(source_dir),
        "output_dir": str(output_dir),
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "requested": len(files),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
        "failures": failures[:20],
    }
    (output_dir / "materialization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
