"""Joint-level physical plausibility metrics for SMPL-22 motion.

The implementation follows the non-VLM MBench motion-quality protocol used by
the Motius HumanML3D leaderboard. Inputs are world-space SMPL-22 joints in
metres with a Y-up coordinate frame. Finite differences are measured per frame,
so motions must share an FPS before their values are compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np

PhysicalRepresentation = Literal["joints", "joints66", "motion135", "ms272"]

FOOT_INDICES = (10, 11)
HEIGHT_AXIS = 1
HORIZONTAL_AXES = (0, 2)
PHYSICAL_METRIC_KEYS = ("Jitter", "Dynamic", "Penet", "Float", "Slide")


@dataclass(frozen=True)
class PhysicalMetricsConfig:
    """Contact and floor parameters for the joint-level protocol."""

    velocity_threshold: float = 0.01
    contact_height_threshold: float = 0.02
    floor_mode: Literal["min_foot"] = "min_foot"


DEFAULT_PHYSICAL_CONFIG = PhysicalMetricsConfig()


def _pad_velocity(values: np.ndarray) -> np.ndarray:
    velocity = np.diff(values, axis=0)
    return np.concatenate([velocity, velocity[-1:]], axis=0)


def _foot_contact(
    foot_positions: np.ndarray,
    floor: float,
    config: PhysicalMetricsConfig,
) -> np.ndarray:
    velocity = _pad_velocity(foot_positions)
    displacement = np.linalg.norm(velocity, axis=-1)
    height = foot_positions[:, :, HEIGHT_AXIS] - floor
    return (
        (displacement < config.velocity_threshold)
        | (height < config.contact_height_threshold)
    ).astype(np.int32)


def _contiguous_ranges(contact: np.ndarray, state: int) -> list[list[list[int]]]:
    output: list[list[list[int]]] = []
    for foot_index in range(contact.shape[1]):
        ranges: list[list[int]] = []
        start = -1
        end = -1
        for frame_index in range(contact.shape[0]):
            if contact[frame_index, foot_index] != state:
                continue
            if start == -1:
                start = end = frame_index
            elif frame_index - end == 1:
                end += 1
            else:
                ranges.append([start, end])
                start = end = frame_index
        if end != -1:
            ranges.append([start, end])
        output.append(ranges)
    return output


def _angle(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    vector_a = vector_a / (np.linalg.norm(vector_a) + 1e-6)
    vector_b = vector_b / (np.linalg.norm(vector_b) + 1e-6)
    return float(np.arccos(np.clip(np.dot(vector_a, vector_b), -1.0, 1.0)))


def _validate_joints(joints: np.ndarray) -> np.ndarray:
    positions = np.asarray(joints, dtype=np.float32)
    if positions.ndim == 2 and positions.shape[1] == 66:
        positions = positions.reshape(len(positions), 22, 3)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(
            "Expected SMPL-22 joints with shape (T,22,3) or (T,66), "
            f"got {positions.shape}"
        )
    if positions.shape[0] < 4:
        raise ValueError("At least four frames are required for physical metrics")
    if not np.isfinite(positions).all():
        raise ValueError("Physical metric input contains NaN or infinite values")
    return positions


def compute_physical_metrics(
    joints: np.ndarray,
    config: PhysicalMetricsConfig = DEFAULT_PHYSICAL_CONFIG,
) -> dict[str, float]:
    """Compute Slide, Float, Jitter, Dynamic, and Penet for one motion clip.

    ``Dynamic`` is an expressiveness statistic and should be compared with the
    GT reference rather than minimized. ``Penet`` is diagnostic under the
    per-clip minimum-foot floor because that floor makes penetration nearly
    degenerate.
    """

    from scipy.signal import find_peaks

    positions = _validate_joints(joints)
    frames = positions.shape[0]
    foot_positions = positions[:, FOOT_INDICES]
    floor = float(foot_positions[:, :, HEIGHT_AXIS].min())

    def acceleration_mean(values: np.ndarray) -> float:
        acceleration = np.diff(values, n=2, axis=0)
        return (
            float(np.linalg.norm(acceleration, axis=2).mean())
            if acceleration.shape[0]
            else 0.0
        )

    root_relative = positions - positions[:, 0:1]
    jitter = acceleration_mean(positions) + acceleration_mean(root_relative)

    def velocity_mean(values: np.ndarray) -> float:
        velocity = np.diff(values, axis=0)
        return (
            float(np.linalg.norm(velocity, axis=2).mean())
            if velocity.shape[0]
            else 0.0
        )

    dynamic = velocity_mean(positions) + velocity_mean(root_relative)

    foot_height = foot_positions[:, :, HEIGHT_AXIS] - floor
    below_floor = np.abs(foot_height[foot_height < -0.005])
    penetration = float(below_floor.mean()) if below_floor.size else 0.0

    contact = _foot_contact(foot_positions, floor, config)
    foot_velocity = _pad_velocity(foot_positions)
    foot_displacement = np.linalg.norm(
        foot_velocity[:, :, HORIZONTAL_AXES], axis=-1
    )
    left_slide = (
        (foot_displacement[:, 0] * contact[:, 0]).sum()
        / (contact[:, 0].sum() + 1e-6)
    )
    right_slide = (
        (foot_displacement[:, 1] * contact[:, 1]).sum()
        / (contact[:, 1].sum() + 1e-6)
    )
    sliding = float((left_slide + right_slide) / 2)

    displacement_threshold = 0.001
    rate_threshold = 0.6
    high_rate_threshold = 1.75
    root_positions = positions[:, 0]
    root_velocity = _pad_velocity(root_positions)
    relative_feet = foot_positions - root_positions[:, None]
    relative_foot_velocity = _pad_velocity(relative_feet)
    left_rates = np.zeros(frames)
    right_rates = np.zeros(frames)
    valid = np.ones((frames, 2))

    for frame_index in range(frames):
        root_displacement = np.linalg.norm(root_velocity[frame_index])
        left_relative = np.linalg.norm(relative_foot_velocity[frame_index, 0])
        right_relative = np.linalg.norm(relative_foot_velocity[frame_index, 1])
        left_rate = left_relative / (root_displacement + 1e-6)
        right_rate = right_relative / (root_displacement + 1e-6)
        left_rates[frame_index] = left_rate
        right_rates[frame_index] = right_rate
        left_displacement = np.linalg.norm(foot_velocity[frame_index, 0])
        right_displacement = np.linalg.norm(foot_velocity[frame_index, 1])
        if root_displacement < displacement_threshold:
            continue
        left_invalid = (
            (left_rate < rate_threshold and left_displacement > 1.2e-4)
            or (
                left_rate > high_rate_threshold
                and left_displacement > 1.2e-4
                and root_displacement > 1.2e-4
            )
        )
        right_invalid = (
            (right_rate < rate_threshold and right_displacement > 1.2e-4)
            or (
                right_rate > high_rate_threshold
                and right_displacement > 1.2e-4
                and root_displacement > 1.2e-4
            )
        )
        if contact[frame_index].sum() == 2 and left_invalid and right_invalid:
            valid[frame_index] = 0
        elif contact[frame_index, 0] == 1 and contact[frame_index, 1] == 0 and left_invalid:
            valid[frame_index, 0] = 0
        elif contact[frame_index, 1] == 1 and contact[frame_index, 0] == 0 and right_invalid:
            valid[frame_index, 1] = 0

    rates = np.stack([left_rates, right_rates], axis=-1)
    no_contact = _contiguous_ranges(contact, 0)
    floating_lengths = [0]
    for foot_index, ranges in enumerate(no_contact):
        for start, end in ranges:
            segment_rates = rates[start : end + 1, foot_index]
            if len(segment_rates) < 4:
                continue
            skipped = sum(
                np.linalg.norm(root_velocity[index]) < displacement_threshold
                for index in range(start, end + 1)
            )
            if skipped / (end - start + 1) > 0.5:
                continue
            floating = (segment_rates < (rate_threshold - 0.2)).astype(np.float32)
            changes = np.diff(np.concatenate([[0.0], floating, [0.0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts):
                floating_lengths.extend((ends - starts).tolist())

    mass_motion_lengths: list[int] = []
    if no_contact[0] and no_contact[1]:
        for left_start, left_end in no_contact[0]:
            for right_start, right_end in no_contact[1]:
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if end - start + 1 < 4:
                    continue
                baseline = foot_positions[end, 0] - foot_positions[start, 0]
                angles = [
                    np.rad2deg(
                        abs(
                            _angle(
                                foot_positions[index, 0] - foot_positions[start, 0],
                                baseline,
                            )
                        )
                    )
                    for index in range(start + 1, end + 1)
                ]
                peaks, _ = find_peaks(angles)
                if len(peaks) > 2:
                    mass_motion_lengths.append(end - start + 1)

    invalid_frames = (valid[:, 0] + valid[:, 1]) <= 1
    invalid_count = (
        int(invalid_frames.sum())
        + sum(floating_lengths) / 2
        + sum(mass_motion_lengths)
    )

    return {
        "Jitter": float(jitter),
        "Dynamic": float(dynamic),
        "Penet": float(penetration),
        "Float": float(invalid_count / frames),
        "Slide": float(sliding),
    }


def physical_metrics_from_motion(
    motion: np.ndarray,
    representation: PhysicalRepresentation = "joints",
    *,
    bone_offsets: np.ndarray | None = None,
    rotation_space: str = "local",
    config: PhysicalMetricsConfig = DEFAULT_PHYSICAL_CONFIG,
) -> dict[str, float]:
    """Convert a supported representation to joints and compute the metrics."""

    representation = representation.lower().replace("-", "")
    if representation in {"joints", "joints66"}:
        joints = motion
    elif representation == "motion135":
        if bone_offsets is None:
            raise ValueError("bone_offsets are required for motion135 physical metrics")
        from motius.motion.representation.convert import motion135_to_joints

        joints = motion135_to_joints(
            motion,
            bone_offsets=bone_offsets,
            rotation_space=rotation_space,
        )
    elif representation == "ms272":
        from motius.motion.representation.convert import motion272_to_joints

        joints = motion272_to_joints(motion)
    else:
        raise ValueError(
            "representation must be joints, joints66, motion135, or ms272; "
            f"got {representation!r}"
        )
    return compute_physical_metrics(np.asarray(joints), config=config)


def aggregate_physical_metrics(
    rows: Iterable[Mapping[str, float] | None],
) -> dict[str, float | int]:
    """Average per-clip physical metric dictionaries."""

    valid = [row for row in rows if row is not None]
    output: dict[str, float | int] = {"n": len(valid)}
    if not valid:
        output.update({key: 0.0 for key in PHYSICAL_METRIC_KEYS})
        return output
    values = np.asarray(
        [[row[key] for key in PHYSICAL_METRIC_KEYS] for row in valid],
        dtype=np.float64,
    )
    output.update(
        {
            key: float(values[:, index].mean())
            for index, key in enumerate(PHYSICAL_METRIC_KEYS)
        }
    )
    return output


def table_scaled_physical_metrics(
    metrics: Mapping[str, float],
) -> dict[str, float]:
    """Scale raw values to the units displayed by the Motius leaderboard."""

    return {
        "Slide": float(metrics.get("Slide", 0.0)) * 1000.0,
        "Float": float(metrics.get("Float", 0.0)) * 100.0,
        "Jitter": float(metrics.get("Jitter", 0.0)) * 1000.0,
        "Dynamic": float(metrics.get("Dynamic", 0.0)) * 1000.0,
        "Penet": float(metrics.get("Penet", 0.0)) * 1000.0,
    }


# Explicit aliases make the protocol provenance discoverable without forcing
# callers to use benchmark-specific names throughout their code.
MBenchPhysicsConfig = PhysicalMetricsConfig
compute_mbench_physics_from_joints = compute_physical_metrics


__all__ = [
    "DEFAULT_PHYSICAL_CONFIG",
    "FOOT_INDICES",
    "HEIGHT_AXIS",
    "HORIZONTAL_AXES",
    "MBenchPhysicsConfig",
    "PHYSICAL_METRIC_KEYS",
    "PhysicalMetricsConfig",
    "aggregate_physical_metrics",
    "compute_mbench_physics_from_joints",
    "compute_physical_metrics",
    "physical_metrics_from_motion",
    "table_scaled_physical_metrics",
]
