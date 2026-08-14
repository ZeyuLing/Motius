"""Generate a tiny test-only SMPL-shaped NPZ for Blender retarget smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    offsets = np.asarray(
        [
            [0.0, 0.52, 0.0],
            [0.08, -0.03, 0.0],
            [-0.08, -0.03, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, -0.22, 0.0],
            [0.0, -0.22, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, -0.22, 0.0],
            [0.0, -0.22, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, -0.03, 0.08],
            [0.0, -0.03, 0.08],
            [0.0, 0.10, 0.0],
            [0.08, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.0, 0.10, 0.0],
            [0.12, 0.0, 0.0],
            [-0.12, 0.0, 0.0],
            [0.18, 0.0, 0.0],
            [-0.18, 0.0, 0.0],
            [0.17, 0.0, 0.0],
            [-0.17, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    joints = np.zeros((22, 3), dtype=np.float64)
    for joint, parent in enumerate(PARENTS):
        joints[joint] = (
            offsets[joint] if parent < 0 else joints[parent] + offsets[joint]
        )
    np.savez(
        args.output,
        v_template=joints,
        shapedirs=np.zeros((22, 3, 1), dtype=np.float64),
        J_regressor=np.eye(22, dtype=np.float64),
        kintree_table=np.stack([PARENTS, np.arange(22)]),
        weights=np.eye(22, dtype=np.float64),
        f=np.asarray([[0, 1, 3], [0, 3, 2], [1, 4, 3], [2, 3, 5]], dtype=np.int32),
    )


if __name__ == "__main__":
    main()
