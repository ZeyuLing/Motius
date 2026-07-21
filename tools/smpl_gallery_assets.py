"""Compact motion135 assets shared by the public SMPL comparison galleries."""

from __future__ import annotations

import json
import shutil
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


def load_joint_positions(path: Path) -> np.ndarray:
    """Load native joint positions from an inference NPZ or AIST++ JSON."""

    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "dance_array" not in payload:
            raise KeyError(f"dance_array not found in {path}")
        value = np.asarray(payload["dance_array"], dtype=np.float32)
    else:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "joints" not in loaded.files:
                    raise KeyError(f"joints not found in {path}: {loaded.files}")
                value = np.asarray(loaded["joints"], dtype=np.float32)
            finally:
                loaded.close()
        else:
            value = np.asarray(loaded, dtype=np.float32)
    if value.ndim == 2 and value.shape[1] % 3 == 0:
        value = value.reshape(len(value), -1, 3)
    if value.ndim != 3 or value.shape[1:] != (24, 3):
        raise ValueError(f"Expected AIST++ joints shaped (T,24,3) in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite joint positions in {path}")
    return np.ascontiguousarray(value, dtype=np.float32)


def resample_joint_positions(
    joints: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    target_frames: int,
) -> np.ndarray:
    """Linearly sample native joints onto an exact target-frame timeline."""

    value = np.asarray(joints, dtype=np.float32)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    positions = np.arange(target_frames, dtype=np.float64) * source_fps / target_fps
    if positions[-1] > len(value) - 1 + 1e-6:
        raise ValueError(
            f"Joint clip has {len(value)} frames, but {target_frames} frames at "
            f"{target_fps:g} fps require {positions[-1] + 1:.3f} source frames at {source_fps:g} fps"
        )
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(value) - 1)
    alpha = (positions - lower).astype(np.float32)[:, None, None]
    sampled = value[lower] * (1.0 - alpha) + value[upper] * alpha
    return np.ascontiguousarray(sampled, dtype=np.float32)


def encode_joint_positions(joints: np.ndarray, *, stride: int = 1) -> tuple[bytes, dict]:
    """Quantize a joint-position clip for range-addressable web playback."""

    value = np.asarray(joints, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected joints shaped (T,J,3), got {value.shape}")
    display_frames = int(len(value))
    stride = max(1, int(stride))
    sampled = np.ascontiguousarray(value[::stride], dtype=np.float32)
    minimum = sampled.reshape(-1, 3).min(axis=0).astype(np.float32)
    maximum = sampled.reshape(-1, 3).max(axis=0).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    encoded = np.rint((sampled - minimum) / scale).clip(0, 65535).astype("<u2")
    descriptor = {
        "frames": int(len(sampled)),
        "display_frames": display_frames,
        "stride": stride,
        "joint_count": int(sampled.shape[1]),
        "position_count": int(encoded.size),
        "position_minimum": minimum.tolist(),
        "position_scale": scale.tolist(),
    }
    return encoded.tobytes(), descriptor


def motion_path(directory: Path, case_id: str) -> Path:
    for suffix in (".npz", ".npy"):
        path = directory / f"{case_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No motion135 file for {case_id!r} under {directory}")


def write_chunked_manifest(
    output_dir: Path,
    manifest: dict,
    *,
    chunk_size: int,
    descriptor_dir_name: str = "descriptors",
) -> None:
    """Write a lightweight case index plus lazily loaded motion descriptors."""

    output = Path(output_dir)
    descriptor_dir = output / descriptor_dir_name
    if descriptor_dir.exists():
        shutil.rmtree(descriptor_dir)
    descriptor_dir.mkdir(parents=True)

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest cases must be a list")
    size = max(1, int(chunk_size))
    for start in range(0, len(cases), size):
        chunk = cases[start : start + size]
        motions = []
        for item in chunk:
            value = item.pop("motions", None)
            if not isinstance(value, dict):
                raise ValueError(f"case {item.get('case_id')!r} has no motion descriptors")
            motions.append(value)
        payload = {"start": start, "motions": motions}
        (descriptor_dir / f"{start // size:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    manifest["schema_version"] = max(3, int(manifest.get("schema_version", 0)))
    manifest["case_descriptor_chunks"] = {
        "size": size,
        "path": f"{descriptor_dir_name}/{{chunk}}.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
