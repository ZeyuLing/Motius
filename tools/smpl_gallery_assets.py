"""Compact motion135 assets shared by the public SMPL comparison galleries."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_motion135(path: Path, *, max_frames: int | None = None) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "motion_135" in loaded.files:
            value = np.asarray(loaded["motion_135"], dtype=np.float32)
        elif {"global_orient", "body_pose"}.issubset(loaded.files) and (
            "transl" in loaded.files or "trans" in loaded.files
        ):
            from motius.motion.representation.rotation import (
                axis_angle_to_matrix,
                matrix_to_rotation_6d,
            )

            global_orient = np.asarray(loaded["global_orient"], dtype=np.float32).reshape(-1, 1, 3)
            body_pose = np.asarray(loaded["body_pose"], dtype=np.float32).reshape(len(global_orient), -1, 3)
            translation_key = "transl" if "transl" in loaded.files else "trans"
            translation = np.asarray(loaded[translation_key], dtype=np.float32).reshape(-1, 3)
            axis_angle = np.concatenate((global_orient, body_pose[:, :21]), axis=1)
            rotations = axis_angle_to_matrix(axis_angle.reshape(-1, 3)).reshape(
                len(axis_angle), 22, 3, 3
            )
            rotation6d = matrix_to_rotation_6d(
                rotations, convention="row"
            ).reshape(len(axis_angle), 132)
            value = np.concatenate((translation[: len(axis_angle)], rotation6d), axis=1)
        else:
            raise KeyError(f"motion_135 or SMPL pose parameters not found in {path}: {loaded.files}")
    else:
        value = np.asarray(loaded, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] < 135:
        raise ValueError(f"Expected (T,>=135) motion135 in {path}, got {value.shape}")
    value = np.ascontiguousarray(value[:max_frames, :135], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite motion135 values in {path}")
    return value


def encode_motion135(motion: np.ndarray, *, stride: int) -> tuple[bytes, dict]:
    display_frames = int(len(motion))
    sampled = np.ascontiguousarray(motion[:: max(1, stride)], dtype=np.float32)
    translation = sampled[:, :3]
    minimum = translation.min(axis=0).astype(np.float32)
    maximum = translation.max(axis=0).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    encoded_translation = np.rint((translation - minimum) / scale).clip(0, 65535).astype("<u2")
    rotations = np.rint(sampled[:, 3:135] * 32767.0).clip(-32767, 32767).astype("<i2")
    translation_bytes = encoded_translation.tobytes()
    rotation_bytes = rotations.tobytes()
    descriptor = {
        "frames": int(len(sampled)),
        "display_frames": display_frames,
        "stride": max(1, int(stride)),
        "translation_count": int(encoded_translation.size),
        "rotation_count": int(rotations.size),
        "translation_minimum": minimum.tolist(),
        "translation_scale": scale.tolist(),
    }
    return translation_bytes + rotation_bytes, descriptor


def motion_path(directory: Path, case_id: str) -> Path:
    for suffix in (".npz", ".npy"):
        path = directory / f"{case_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No motion135 file for {case_id!r} under {directory}")
