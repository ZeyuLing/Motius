"""Validation and body mapping for BeyondMimic motion NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


MOTION_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def load_motion_npz(
    motion_file: str | Path,
    *,
    robot_body_indexes: Sequence[int],
    target_body_names: Sequence[str],
    target_joint_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    """Load a full-body or named compact BeyondMimic motion.

    The official converter stores every robot body and omits ``body_names``.
    Motius data tools may store only the configured tracking bodies together
    with explicit names. Both layouts resolve to the same ordered tensors.
    """
    path = Path(motion_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"BeyondMimic motion does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        missing = {"fps", *MOTION_ARRAY_KEYS}.difference(data.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"BeyondMimic motion is missing keys: {names}")

        fps_value = np.asarray(data["fps"])
        if fps_value.size != 1:
            raise ValueError("BeyondMimic motion fps must be a scalar")
        fps = np.asarray(float(fps_value.reshape(-1)[0]), dtype=np.float32)
        if not np.isfinite(fps) or float(fps) <= 0:
            raise ValueError("BeyondMimic motion fps must be positive and finite")

        arrays = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in MOTION_ARRAY_KEYS
        }
        arrays["fps"] = fps

        joint_pos = arrays["joint_pos"]
        joint_vel = arrays["joint_vel"]
        expected_joint_count = len(target_joint_names)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_joint_count:
            raise ValueError(
                "BeyondMimic joint_pos must have shape "
                f"(T, {expected_joint_count}); got {joint_pos.shape}"
            )
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(
                "BeyondMimic joint_vel shape must match joint_pos; got "
                f"{joint_vel.shape} and {joint_pos.shape}"
            )
        if "joint_names" in data.files:
            source_joint_names = tuple(str(name) for name in data["joint_names"])
            if len(set(source_joint_names)) != len(source_joint_names):
                raise ValueError("BeyondMimic joint_names must be unique")
            source_joint_index = {
                name: index for index, name in enumerate(source_joint_names)
            }
            missing_joints = [
                name
                for name in target_joint_names
                if name not in source_joint_index
            ]
            if missing_joints:
                raise ValueError(
                    "BeyondMimic motion is missing robot joints: "
                    + ", ".join(missing_joints)
                )
            joint_indexes = [
                source_joint_index[name] for name in target_joint_names
            ]
            arrays["joint_pos"] = joint_pos[:, joint_indexes]
            arrays["joint_vel"] = joint_vel[:, joint_indexes]

        frame_count = arrays["joint_pos"].shape[0]
        body_specs = {
            "body_pos_w": 3,
            "body_quat_w": 4,
            "body_lin_vel_w": 3,
            "body_ang_vel_w": 3,
        }
        body_count: int | None = None
        for key, width in body_specs.items():
            value = arrays[key]
            if value.ndim != 3 or value.shape[0] != frame_count:
                raise ValueError(
                    f"BeyondMimic {key} must have shape (T, B, {width}); "
                    f"got {value.shape}"
                )
            if value.shape[2] != width:
                raise ValueError(
                    f"BeyondMimic {key} last dimension must be {width}; "
                    f"got {value.shape}"
                )
            if body_count is None:
                body_count = value.shape[1]
            elif value.shape[1] != body_count:
                raise ValueError(
                    "BeyondMimic body arrays must use one shared body count"
                )

        if "body_names" in data.files:
            source_body_names = tuple(str(name) for name in data["body_names"])
            if len(source_body_names) != body_count:
                raise ValueError(
                    "BeyondMimic body_names length does not match body arrays"
                )
            if len(set(source_body_names)) != len(source_body_names):
                raise ValueError("BeyondMimic body_names must be unique")
            source_index = {
                name: index for index, name in enumerate(source_body_names)
            }
            missing_bodies = [
                name for name in target_body_names if name not in source_index
            ]
            if missing_bodies:
                raise ValueError(
                    "BeyondMimic motion is missing tracked bodies: "
                    + ", ".join(missing_bodies)
                )
            motion_body_indexes = tuple(
                source_index[name] for name in target_body_names
            )
        else:
            motion_body_indexes = tuple(int(index) for index in robot_body_indexes)
            if (
                not motion_body_indexes
                or min(motion_body_indexes) < 0
                or max(motion_body_indexes) >= int(body_count or 0)
            ):
                raise ValueError(
                    "BeyondMimic full-body arrays are incompatible with the "
                    "robot body indexes"
                )

        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(
                    f"BeyondMimic motion contains non-finite values in {key}"
                )

    return arrays, motion_body_indexes
