#!/usr/bin/env python3
"""Materialize motion272 predictions as canonical SMPL-22 joints66 arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import canonicalize_smpl22_joints, motion272_to_joints


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_motion272(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        motion = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if key not in payload.files:
                raise KeyError(f"{path} does not contain {key!r}; keys={payload.files}")
            motion = payload[key]
    else:
        raise ValueError(f"Unsupported motion272 file: {path}")
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"Expected [T, 272] in {path}, got {motion.shape}")
    return motion


def materialize_case(
    source_path: Path,
    output_path: Path,
    *,
    offsets: np.ndarray,
    expected_frames: int,
    key: str = "motion_272",
) -> None:
    motion = _load_motion272(source_path, key)
    if len(motion) != int(expected_frames):
        raise ValueError(
            f"{source_path} has {len(motion)} frames, expected {expected_frames}"
        )
    joints = motion272_to_joints(motion, bone_offsets=offsets)
    joints = canonicalize_smpl22_joints(joints).reshape(expected_frames, 66)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, joints.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--npz-key", default="motion_272")
    parser.add_argument("--ids", default="", help="Comma-separated case ids.")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    offsets_path = _resolve_manifest_path(str(manifest["smpl22_offsets"]), manifest_path)
    offsets = np.asarray(np.load(offsets_path), dtype=np.float32).copy()
    if offsets.shape != (22, 3):
        raise ValueError(f"Expected SMPL-22 offsets [22, 3], got {offsets.shape}")
    offsets[0] = 0.0

    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    generated = skipped = missing = 0
    cases = manifest.get("cases", [])
    for case in cases:
        case_id = str(case["case_id"])
        if selected and case_id not in selected:
            continue
        output_path = output_dir / f"{case_id}.npy"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        npz_path = source_dir / f"{case_id}.npz"
        npy_path = source_dir / f"{case_id}.npy"
        source_path = npz_path if npz_path.is_file() else npy_path
        if not source_path.is_file():
            if args.skip_missing:
                missing += 1
                continue
            raise FileNotFoundError(f"Missing motion272 prediction for {case_id}")
        materialize_case(
            source_path,
            output_path,
            offsets=offsets,
            expected_frames=int(case["total_frames"]),
            key=args.npz_key,
        )
        generated += 1

    summary = {
        "protocol": manifest.get("protocol"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": skipped,
        "missing": missing,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "materialization.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
