#!/usr/bin/env python3
"""Extract the native SMPL-H joint trace from a HYMotion V2M parity archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SMPLH52_PARENTS = np.asarray(
    [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
        16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31,
        32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46,
        47, 21, 49, 50,
    ],
    dtype=np.int32,
)


def _field(archive, manifest: dict, stage_name: str, field_name: str):
    stage = next(
        stage for stage in manifest["stages"]
        if stage["name"] == stage_name
    )
    field = next(
        field for field in stage["fields"]
        if field["field"] == field_name
    )
    return archive[field["storage_key"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"]))
        joints = np.asarray(
            _field(archive, manifest, "10_native_output", "keypoints3d"),
            dtype=np.float32,
        )[0]
        timestamps = np.asarray(
            _field(archive, manifest, "11_public_result", "frame_timestamps"),
            dtype=np.float64,
        )
    fps = (
        round(1.0 / float(np.median(np.diff(timestamps))))
        if len(timestamps) > 1
        else 30
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joints=joints[:, None],
        parents=SMPLH52_PARENTS,
        caption=np.asarray("monocular video"),
        case_id=np.asarray("hymotion_v2m_parity"),
        fps=np.asarray(fps, dtype=np.int32),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
