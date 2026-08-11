#!/usr/bin/env python3
"""Recover the official AIST++ SMPL-24 rest offsets from released joints."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.motion.representation.aistpp import (  # noqa: E402
    AISTPP_SMPL24_PARENTS,
    aistpp_smpl24_fk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions-zip", type=Path, required=True)
    parser.add_argument("--joint-json-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-frames-per-sample", type=int, default=1_200)
    return parser.parse_args()


def _load_motion(handle) -> dict:
    try:
        payload = pickle.load(handle, encoding="latin1")
    except TypeError:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("AIST++ motion pickle must contain a dictionary")
    return payload


def _rotation_chain(poses: np.ndarray) -> np.ndarray:
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix()
    local = local.reshape(len(poses), 24, 3, 3)
    global_rotations = np.empty_like(local)
    global_rotations[:, 0] = local[:, 0]
    for joint in range(1, 24):
        parent = int(AISTPP_SMPL24_PARENTS[joint])
        global_rotations[:, joint] = np.einsum(
            "tij,tjk->tik", global_rotations[:, parent], local[:, joint]
        )
    return global_rotations


def main() -> None:
    args = parse_args()
    json_paths = sorted(args.joint_json_root.glob("*.json"))
    if args.max_samples is not None:
        json_paths = json_paths[: args.max_samples]
    if not json_paths:
        raise FileNotFoundError(f"No calibration JSON files under {args.joint_json_root}")

    with zipfile.ZipFile(args.motions_zip) as archive:
        motion_members = {
            Path(item.filename).stem: item.filename
            for item in archive.infolist()
            if item.filename.endswith(".pkl")
        }
        offset_rows: list[list[np.ndarray]] = [[] for _ in range(24)]
        calibration_records = []
        loaded = []
        for json_path in json_paths:
            name = json_path.stem
            member = motion_members.get(name)
            if member is None:
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            joints = np.asarray(payload["dance_array"], dtype=np.float64).reshape(-1, 24, 3)
            with archive.open(member) as handle:
                motion = _load_motion(handle)
            poses = np.asarray(motion["smpl_poses"], dtype=np.float64).reshape(-1, 24, 3)
            frames = min(len(joints), len(poses), args.max_frames_per_sample)
            joints = joints[:frames]
            poses = poses[:frames]
            global_rotations = _rotation_chain(poses)
            for joint in range(1, 24):
                parent = int(AISTPP_SMPL24_PARENTS[joint])
                delta = joints[:, joint] - joints[:, parent]
                local_delta = np.einsum(
                    "tji,tj->ti", global_rotations[:, parent], delta
                )
                offset_rows[joint].append(local_delta)
            loaded.append((name, joints, motion, frames))

        if not loaded:
            raise RuntimeError("No calibration sequence matched the motion archive")
        offsets = np.zeros((24, 3), dtype=np.float64)
        offset_std = np.zeros((24, 3), dtype=np.float64)
        for joint in range(1, 24):
            rows = np.concatenate(offset_rows[joint], axis=0)
            offsets[joint] = np.median(rows, axis=0)
            offset_std[joint] = rows.std(axis=0)

        all_errors = []
        for name, joints, motion, frames in loaded:
            predicted = aistpp_smpl24_fk(
                np.asarray(motion["smpl_poses"])[:frames],
                np.asarray(motion["smpl_trans"])[:frames],
                motion["smpl_scaling"],
                offsets,
            ).astype(np.float64)
            predicted -= predicted[:1, :1]
            target = joints - joints[:1, :1]
            errors = np.linalg.norm(predicted - target, axis=-1)
            all_errors.append(errors.reshape(-1))
            calibration_records.append(
                {
                    "sequence_id": name,
                    "frames": frames,
                    "mpjpe_mm": float(errors.mean() * 1_000.0),
                    "p95_mm": float(np.percentile(errors, 95) * 1_000.0),
                }
            )

    errors = np.concatenate(all_errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rest_offsets=offsets.astype(np.float32),
        offset_std=offset_std.astype(np.float32),
        parents=AISTPP_SMPL24_PARENTS,
        sequence_ids=np.asarray([record["sequence_id"] for record in calibration_records]),
    )
    report = {
        "schema_version": 1,
        "output": str(args.output),
        "motion_archive": str(args.motions_zip),
        "calibration_root": str(args.joint_json_root),
        "num_sequences": len(calibration_records),
        "num_joint_frames": int(len(errors)),
        "mpjpe_mm": float(errors.mean() * 1_000.0),
        "p95_mm": float(np.percentile(errors, 95) * 1_000.0),
        "max_mm": float(errors.max() * 1_000.0),
        "max_offset_std_mm": float(offset_std.max() * 1_000.0),
        "records": calibration_records,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
