# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Read and write robot motion CSV files."""

import csv
from itertools import chain
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

from motius.motion.runtime.motionbricks.dof_animation_buffer import DOFAnimationBuffer
from motius.motion.runtime.motionbricks.dof_utils import ROOT_COORDINATE_COUNT

CSV_SAMPLE_RATE = 120.0
_CSV_EXPORT_ROOT_TITLE = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]
_CSV_ROOT_COLUMN_COUNT = len(_CSV_EXPORT_ROOT_TITLE)


def load_csv(
    file_path: str | Path, sample_rate: float = CSV_SAMPLE_RATE, *, device=None
) -> DOFAnimationBuffer:
    """Load a robot motion CSV as joint coordinates.

    The file stores root translations in centimeters, root Euler rotations
    in extrinsic XYZ order, and all angles in degrees. The returned buffer
    uses meters, XYZW quaternions, and radians. Joint columns keep their file
    order, which must match the robot's joint_q.

    Args:
        file_path: CSV file to read.
        sample_rate: Sample rate of the motion in Hz. The file does not store it.
        device: Device for new arrays. None uses the current device.
    """
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
        if (
            header is None
            or header[:_CSV_ROOT_COLUMN_COUNT] != _CSV_EXPORT_ROOT_TITLE
            or len(header) <= _CSV_ROOT_COLUMN_COUNT
        ):
            raise ValueError(
                f"{path}: expected the columns Frame and root_translateX to root_rotateZ, then one named column per joint angle."
            )
        names = tuple(header[_CSV_ROOT_COLUMN_COUNT:])
        rows = (line for line in source if line.strip())
        first_row = next(rows, None)
        if first_row is None:
            raise ValueError(f"{path}: CSV contains no motion frames.")
        data = np.loadtxt(chain((first_row,), rows), delimiter=",", ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(f"{path}: row width does not match the CSV header.")
    joint_q = np.empty(
        (len(data), ROOT_COORDINATE_COUNT + len(names)), dtype=np.float32
    )
    joint_q[:, :3] = data[:, 1:4] * 0.01
    joint_q[:, 3:ROOT_COORDINATE_COUNT] = R.from_euler(
        "xyz", data[:, 4:7], degrees=True
    ).as_quat()
    joint_q[:, ROOT_COORDINATE_COUNT:] = np.deg2rad(data[:, _CSV_ROOT_COLUMN_COUNT:])
    if not np.isfinite(joint_q).all():
        raise ValueError(f"{path}: motion coordinates must be finite.")
    return DOFAnimationBuffer(
        names, sample_rate, wp.array(joint_q, dtype=wp.float32, device=device)
    )


def save_csv(file_path: str | Path, buffer: DOFAnimationBuffer) -> None:
    """Save the frames in use of buffer as a robot motion CSV.

    Translations are written in centimeters, root rotations as extrinsic XYZ
    Euler angles, and all angles in degrees. The joint angle columns take the
    buffer's joint names.

    Args:
        file_path: CSV file to write. An existing file is replaced.
        buffer: Joint coordinate animation with at least one frame in use.
    """
    if buffer.num_frames == 0:
        raise ValueError("buffer must contain at least one frame in use.")
    frames = buffer.poses[: buffer.num_frames].numpy()
    with open(file_path, "w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(_CSV_EXPORT_ROOT_TITLE + list(buffer.joint_names))
        for frame_index, pose in enumerate(frames):
            position_cm = wp.vec3(*pose[:3]) * 100.0
            rotation = wp.quat(*pose[3:ROOT_COORDINATE_COUNT])
            euler_degrees = R.from_quat(rotation).as_euler("xyz", degrees=True)
            row = [frame_index, *position_cm, *euler_degrees]
            row.extend(np.rad2deg(pose[ROOT_COORDINATE_COUNT:]))
            writer.writerow(row)
