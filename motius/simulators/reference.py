"""Reference-motion loading and frame-rate normalization for G1 tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from motius.simulators.g1 import G1_JOINT_NAMES


@dataclass(frozen=True)
class TrackingReference:
    """A Unitree G1 reference in MuJoCo generalized coordinates."""

    name: str
    fps: float
    qpos: np.ndarray
    qvel: np.ndarray
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float32)
        qvel = np.asarray(self.qvel, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError(f"G1 qpos must have shape [T, 36], got {qpos.shape}.")
        if qvel.shape != (qpos.shape[0], 35):
            raise ValueError(
                f"G1 qvel must have shape [{qpos.shape[0]}, 35], got {qvel.shape}."
            )
        if qpos.shape[0] < 2:
            raise ValueError("A tracking reference must contain at least two frames.")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Tracking reference contains non-finite values.")
        if float(self.fps) <= 0:
            raise ValueError(f"Reference fps must be positive, got {self.fps}.")
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos))
        object.__setattr__(self, "qvel", np.ascontiguousarray(qvel))

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_seconds(self) -> float:
        return (self.num_frames - 1) / float(self.fps)

    def window(self, start: int, stop: int, *, name: Optional[str] = None) -> "TrackingReference":
        """Return a frame window while preserving the source clock and metadata."""

        start = int(start)
        stop = int(stop)
        if start < 0 or stop > self.num_frames or stop - start < 2:
            raise ValueError(
                f"Invalid tracking window [{start}, {stop}) for {self.num_frames} frames."
            )
        return TrackingReference(
            name=name or f"{self.name}__f{start:06d}_{stop - 1:06d}",
            fps=self.fps,
            qpos=self.qpos[start:stop].copy(),
            qvel=self.qvel[start:stop].copy(),
            source_path=self.source_path,
        )

    def iter_windows(
        self,
        steps: int,
        *,
        minimum_remainder_steps: int = 1,
    ):
        """Yield deterministic, non-overlapping windows measured in control steps."""

        steps = int(steps)
        minimum_remainder_steps = int(minimum_remainder_steps)
        if steps <= 0 or minimum_remainder_steps <= 0:
            raise ValueError("Window and minimum remainder lengths must be positive.")
        total_steps = self.num_frames - 1
        start_step = 0
        while start_step < total_steps:
            window_steps = min(steps, total_steps - start_step)
            if window_steps < minimum_remainder_steps:
                break
            stop_frame = start_step + window_steps + 1
            yield self.window(start_step, stop_frame)
            start_step += window_steps


def _normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Reference contains a zero-length root quaternion.")
    result = quaternions / norms
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0:
            result[index] *= -1
    return result


def _resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if np.isclose(source_fps, target_fps):
        return np.asarray(qpos, dtype=np.float32).copy()
    duration = (len(qpos) - 1) / float(source_fps)
    source_times = np.arange(len(qpos), dtype=np.float64) / float(source_fps)
    # Match OpenTrack's trajectory protocol: scale frame count, then keep both
    # endpoints. This differs by one frame from duration-based rounding on some
    # 40 -> 50 Hz clips.
    target_count = int(round(len(qpos) * float(target_fps) / float(source_fps)))
    target_times = np.linspace(0.0, duration, target_count, dtype=np.float64)

    result = np.empty((target_count, qpos.shape[1]), dtype=np.float64)
    linear_columns = [*range(3), *range(7, qpos.shape[1])]
    interpolation_kind = "cubic" if len(qpos) >= 4 else "linear"
    interpolator = interp1d(
        source_times,
        qpos[:, linear_columns],
        axis=0,
        kind=interpolation_kind,
        bounds_error=True,
    )
    result[:, linear_columns] = interpolator(target_times)

    source_xyzw = qpos[:, [4, 5, 6, 3]]
    rotations = Rotation.from_quat(source_xyzw)
    target_xyzw = Slerp(source_times, rotations)(target_times).as_quat()
    result[:, 3:7] = target_xyzw[:, [3, 0, 1, 2]]
    return result.astype(np.float32)


def _differentiate_qpos(qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((len(qpos), 35), dtype=np.float32)
    qvel[1:, :3] = np.diff(qpos[:, :3], axis=0) * float(fps)

    root_xyzw = qpos[:, [4, 5, 6, 3]]
    rotations = Rotation.from_quat(root_xyzw)
    relative = rotations[:-1].inv() * rotations[1:]
    qvel[1:, 3:6] = relative.as_rotvec().astype(np.float32) * float(fps)
    qvel[1:, 6:] = np.diff(qpos[:, 7:], axis=0) * float(fps)
    qvel[0] = qvel[1]
    return qvel


def _load_csv_qpos(path: Path) -> tuple[np.ndarray, float]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 36:
        raise ValueError(f"LAFAN1-G1 CSV must have 36 columns, got {values.shape}.")
    # The public LAFAN1 retarget stores root quaternion as xyzw. MuJoCo qpos is wxyz.
    qpos = values.copy()
    qpos[:, 3:7] = values[:, [6, 3, 4, 5]]
    return qpos, 30.0


def _load_npz_qpos(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as archive:
        if "qpos" in archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float32)
            if qpos.ndim == 2 and qpos.shape[1] != 36 and "joint_names" in archive:
                joint_names = [str(item) for item in np.asarray(archive["joint_names"])]
                qpos = _extend_named_g1_qpos(qpos, joint_names)
        elif {"dof_positions", "body_positions", "body_rotations"} <= set(
            archive.files
        ):
            root_pos = np.asarray(archive["body_positions"][:, 0], dtype=np.float32)
            root_xyzw = np.asarray(archive["body_rotations"][:, 0], dtype=np.float32)
            root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
            dof = np.asarray(archive["dof_positions"], dtype=np.float32)
            qpos = np.concatenate([root_pos, root_wxyz, dof], axis=-1)
        else:
            raise ValueError(
                f"Unsupported G1 NPZ layout in {path}; expected qpos or the "
                "AMASS-GMR dof/body arrays."
            )
        if "fps" in archive:
            fps_value = archive["fps"]
        elif "frequency" in archive:
            fps_value = archive["frequency"]
        else:
            fps_value = 30.0
        fps = float(np.asarray(fps_value).reshape(-1)[0])
    return qpos, fps


def _extend_named_g1_qpos(qpos: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Extend a named free-root trajectory to the public G1 29-DOF order."""

    if not joint_names or joint_names[0] != "root":
        raise ValueError("Named G1 qpos must start with the free joint named 'root'.")
    expected_width = 7 + len(joint_names) - 1
    if qpos.shape[1] != expected_width:
        raise ValueError(
            f"Named G1 qpos has width {qpos.shape[1]}, but {len(joint_names)} "
            f"joint names imply width {expected_width}."
        )
    unknown = sorted(set(joint_names[1:]) - set(G1_JOINT_NAMES))
    if unknown:
        raise ValueError(f"Named G1 qpos contains unsupported joints: {unknown}.")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Named G1 qpos contains duplicate joint names.")

    dof = np.zeros((len(qpos), len(G1_JOINT_NAMES)), dtype=np.float32)
    target_indices = {name: index for index, name in enumerate(G1_JOINT_NAMES)}
    for source_index, joint_name in enumerate(joint_names[1:]):
        dof[:, target_indices[joint_name]] = qpos[:, 7 + source_index]
    return np.concatenate([qpos[:, :7], dof], axis=-1)


def load_g1_reference(
    path: str | Path,
    *,
    source_fps: Optional[float] = None,
    target_fps: float = 50.0,
    name: Optional[str] = None,
) -> TrackingReference:
    """Load public LAFAN1 CSV, AMASS-GMR NPZ, or qpos NPZ as a G1 reference."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tracking reference does not exist: {source}")
    if source.suffix.lower() == ".csv":
        qpos, detected_fps = _load_csv_qpos(source)
    elif source.suffix.lower() == ".npz":
        qpos, detected_fps = _load_npz_qpos(source)
    else:
        raise ValueError(f"Unsupported tracking reference extension: {source.suffix}")

    fps = float(source_fps if source_fps is not None else detected_fps)
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"G1 reference must decode to [T, 36], got {qpos.shape}.")
    qpos[:, 3:7] = _normalize_quaternions_wxyz(qpos[:, 3:7])
    qpos = _resample_qpos(qpos, fps, float(target_fps))
    qvel = _differentiate_qpos(qpos, float(target_fps))
    return TrackingReference(
        name=name or source.stem,
        fps=float(target_fps),
        qpos=qpos,
        qvel=qvel,
        source_path=str(source),
    )
