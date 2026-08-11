"""Shared root-aware adversarial scoring for PhysFlow G1 tracking.

The score is used to rank T2M candidates for hard-prompt mining and to select
good tracked motions for the G1 tracker pool. Keeping it in one module prevents
runner/report/best-of tools from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class G1ScoreConfig:
    joint_error_scale: float = 1.0
    root_trajectory_error_weight: float = 1.0
    root_trajectory_error_scale: float = 0.5
    root_displacement_error_weight: float = 0.5
    root_displacement_error_scale: float = 0.5
    score_component_cap: float = 2.0
    fall_penalty: float = 2.0

    def to_dict(self) -> dict[str, float | str]:
        data: dict[str, float | str] = asdict(self)
        data["completion"] = "1 - completion_ratio"
        return data


DEFAULT_G1_SCORE_CONFIG = G1ScoreConfig()
DEFAULT_G1_HARD_PROMPT_MIN_SCORE = 1.0


@dataclass(frozen=True)
class SonicTrackingScoreConfig:
    """Released SONIC/BeyondMimic tracking-reward parameters.

    The five non-position terms match the released SONIC G1 reward config.
    ``global_anchor_pos`` is the public BeyondMimic component from the same
    ProtoMotions implementation and makes the score start-aligned
    trajectory-aware.
    """

    global_anchor_pos_weight: float = 0.5
    global_anchor_pos_sigma: float = 0.3
    global_anchor_ori_weight: float = 0.5
    global_anchor_ori_sigma: float = 0.4
    relative_body_pos_weight: float = 1.0
    relative_body_pos_sigma: float = 0.3
    relative_body_ori_weight: float = 1.0
    relative_body_ori_sigma: float = 0.4
    body_lin_vel_weight: float = 1.0
    body_lin_vel_sigma: float = 1.0
    body_ang_vel_weight: float = 1.0
    body_ang_vel_sigma: float = 3.14
    incomplete_weight: float = 1.0
    fall_penalty: float = 2.0

    def to_dict(self) -> dict[str, float | str]:
        data: dict[str, float | str] = asdict(self)
        data["source"] = (
            "released SONIC G1 + ProtoMotions BeyondMimic reward components"
        )
        return data


DEFAULT_SONIC_TRACKING_SCORE_CONFIG = SonicTrackingScoreConfig()


@dataclass(frozen=True)
class G1TrackerPoolConfig:
    min_completion: float = 0.95
    max_joint_error_rad: float = 0.7
    max_root_trajectory_error_mean_m: float = 0.25
    max_root_displacement_error_m: float = 0.35
    require_root_metrics: bool = True

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


DEFAULT_G1_TRACKER_POOL_CONFIG = G1TrackerPoolConfig()


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lv, lw = left[..., :3], left[..., 3:4]
    rv, rw = right[..., :3], right[..., 3:4]
    vector = lw * rv + rw * lv + np.cross(lv, rv)
    scalar = lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)
    return np.concatenate((vector, scalar), axis=-1)


def _heading_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    x, y, z, w = (quat[..., index] for index in range(4))
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    half = -0.5 * yaw
    zeros = np.zeros_like(half)
    return np.stack((zeros, zeros, np.sin(half), np.cos(half)), axis=-1)


def _quat_rotate_xyzw(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    vectors = np.asarray(vectors, dtype=np.float64)
    qvec = quat[..., :3]
    uv = np.cross(qvec, vectors)
    uuv = np.cross(qvec, uv)
    return vectors + 2.0 * (quat[..., 3:4] * uv + uuv)


def _quat_angle_sq_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = _normalize_quat_xyzw(first)
    second = _normalize_quat_xyzw(second)
    delta = _quat_mul_xyzw(second, _quat_conjugate_xyzw(first))
    vector_norm = np.linalg.norm(delta[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.abs(delta[..., 3]))
    return np.square(angle)


def _quat_angular_velocity_xyzw(quat: np.ndarray, fps: float) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    delta = _quat_mul_xyzw(quat[1:], _quat_conjugate_xyzw(quat[:-1]))
    sign = np.where(delta[..., 3:4] < 0.0, -1.0, 1.0)
    delta *= sign
    vector = delta[..., :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.maximum(delta[..., 3:4], 1e-12))
    axis = vector / np.maximum(vector_norm, 1e-12)
    return axis * angle * float(fps)


def _exp_position_reward(
    current: np.ndarray,
    reference: np.ndarray,
    sigma: float,
) -> np.ndarray:
    squared_distance = np.square(current - reference).sum(axis=-1)
    if squared_distance.ndim == 2:
        squared_distance = squared_distance.mean(axis=-1)
    return np.exp(-squared_distance / float(sigma) ** 2)


def _exp_orientation_reward(
    current: np.ndarray,
    reference: np.ndarray,
    sigma: float,
) -> np.ndarray:
    squared_angle = _quat_angle_sq_xyzw(current, reference)
    if squared_angle.ndim == 2:
        squared_angle = squared_angle.mean(axis=-1)
    return np.exp(-squared_angle / float(sigma) ** 2)


def compute_sonic_tracking_score(
    reference_body_pos: np.ndarray,
    execution_body_pos: np.ndarray,
    reference_body_quat: np.ndarray,
    execution_body_quat: np.ndarray,
    *,
    fps: float,
    completion: float,
    fall_detected: bool,
    anchor_index: int = 0,
    config: SonicTrackingScoreConfig = DEFAULT_SONIC_TRACKING_SCORE_CONFIG,
) -> tuple[float, dict[str, float]]:
    """Return a lower-is-better trajectory score from released reward kernels."""

    ref_pos = np.asarray(reference_body_pos, dtype=np.float64)
    exe_pos = np.asarray(execution_body_pos, dtype=np.float64)
    ref_quat = _normalize_quat_xyzw(reference_body_quat)
    exe_quat = _normalize_quat_xyzw(execution_body_quat)
    frame_count = min(len(ref_pos), len(exe_pos), len(ref_quat), len(exe_quat))
    if frame_count < 2:
        raise ValueError("SONIC tracking score requires at least two aligned frames")
    ref_pos = ref_pos[:frame_count]
    exe_pos = exe_pos[:frame_count]
    ref_quat = ref_quat[:frame_count]
    exe_quat = exe_quat[:frame_count]
    if ref_pos.shape != exe_pos.shape or ref_quat.shape != exe_quat.shape:
        raise ValueError(
            "reference/execution body trajectories must have matching shapes: "
            f"pos={ref_pos.shape}/{exe_pos.shape}, quat={ref_quat.shape}/{exe_quat.shape}"
        )
    if not 0 <= anchor_index < ref_pos.shape[1]:
        raise ValueError(f"anchor_index={anchor_index} outside {ref_pos.shape[1]} bodies")
    if not np.isfinite(ref_pos).all() or not np.isfinite(exe_pos).all():
        raise ValueError("body positions contain non-finite values")

    ref_anchor_pos = ref_pos[:, anchor_index]
    exe_anchor_pos = exe_pos[:, anchor_index]
    # Global XY is start-aligned, matching the paper protocol. Preserve height.
    start_offset = exe_anchor_pos[0] - ref_anchor_pos[0]
    start_offset[2] = 0.0
    exe_anchor_aligned = exe_anchor_pos - start_offset

    ref_anchor_quat = ref_quat[:, anchor_index]
    exe_anchor_quat = exe_quat[:, anchor_index]
    ref_heading_inv = _heading_inverse_xyzw(ref_anchor_quat)
    exe_heading_inv = _heading_inverse_xyzw(exe_anchor_quat)
    ref_relative_pos = _quat_rotate_xyzw(
        np.broadcast_to(ref_heading_inv[:, None], ref_quat.shape),
        ref_pos - ref_anchor_pos[:, None],
    )
    exe_relative_pos = _quat_rotate_xyzw(
        np.broadcast_to(exe_heading_inv[:, None], exe_quat.shape),
        exe_pos - exe_anchor_pos[:, None],
    )
    ref_relative_quat = _quat_mul_xyzw(
        np.broadcast_to(ref_heading_inv[:, None], ref_quat.shape),
        ref_quat,
    )
    exe_relative_quat = _quat_mul_xyzw(
        np.broadcast_to(exe_heading_inv[:, None], exe_quat.shape),
        exe_quat,
    )

    ref_linear_velocity = np.diff(ref_pos, axis=0) * float(fps)
    exe_linear_velocity = np.diff(exe_pos, axis=0) * float(fps)
    ref_angular_velocity = _quat_angular_velocity_xyzw(ref_quat, fps)
    exe_angular_velocity = _quat_angular_velocity_xyzw(exe_quat, fps)

    # Export the paper evaluator's lower-is-better diagnostics alongside the
    # public SONIC reward kernels. E_r removes per-frame pelvis translation;
    # velocity and acceleration use start-aligned global positions (the
    # constant XY offset cancels under finite differences).
    ref_local_pos = ref_pos - ref_anchor_pos[:, None]
    exe_local_pos = exe_pos - exe_anchor_pos[:, None]
    er_mpjpe_mm = float(
        np.linalg.norm(exe_local_pos - ref_local_pos, axis=-1).mean() * 1000.0
    )
    evel_mps = float(
        np.linalg.norm(exe_linear_velocity - ref_linear_velocity, axis=-1).mean()
    )
    if frame_count >= 3:
        ref_acceleration = np.diff(ref_pos, n=2, axis=0) * float(fps) ** 2
        exe_acceleration = np.diff(exe_pos, n=2, axis=0) * float(fps) ** 2
        eacc_mps2 = float(
            np.linalg.norm(
                exe_acceleration - ref_acceleration,
                axis=-1,
            ).mean()
        )
    else:
        eacc_mps2 = 0.0

    components = {
        "global_anchor_pos": float(
            _exp_position_reward(
                exe_anchor_aligned,
                ref_anchor_pos,
                config.global_anchor_pos_sigma,
            ).mean()
        ),
        "global_anchor_ori": float(
            _exp_orientation_reward(
                exe_anchor_quat,
                ref_anchor_quat,
                config.global_anchor_ori_sigma,
            ).mean()
        ),
        "relative_body_pos": float(
            _exp_position_reward(
                exe_relative_pos,
                ref_relative_pos,
                config.relative_body_pos_sigma,
            ).mean()
        ),
        "relative_body_ori": float(
            _exp_orientation_reward(
                exe_relative_quat,
                ref_relative_quat,
                config.relative_body_ori_sigma,
            ).mean()
        ),
        "body_lin_vel": float(
            _exp_position_reward(
                exe_linear_velocity,
                ref_linear_velocity,
                config.body_lin_vel_sigma,
            ).mean()
        ),
        "body_ang_vel": float(
            _exp_position_reward(
                exe_angular_velocity,
                ref_angular_velocity,
                config.body_ang_vel_sigma,
            ).mean()
        ),
        "er_mpjpe_mm": er_mpjpe_mm,
        "evel_mps": evel_mps,
        "eacc_mps2": eacc_mps2,
    }
    reward_component_names = (
        "global_anchor_pos",
        "global_anchor_ori",
        "relative_body_pos",
        "relative_body_ori",
        "body_lin_vel",
        "body_ang_vel",
    )
    weights = {
        "global_anchor_pos": config.global_anchor_pos_weight,
        "global_anchor_ori": config.global_anchor_ori_weight,
        "relative_body_pos": config.relative_body_pos_weight,
        "relative_body_ori": config.relative_body_ori_weight,
        "body_lin_vel": config.body_lin_vel_weight,
        "body_ang_vel": config.body_ang_vel_weight,
    }
    max_reward = float(sum(weights.values()))
    tracking_reward = float(
        sum(weights[name] * components[name] for name in reward_component_names)
    )
    incomplete_cost = config.incomplete_weight * max(0.0, 1.0 - float(completion))
    fall_cost = config.fall_penalty if fall_detected else 0.0
    score = max_reward - tracking_reward + incomplete_cost + fall_cost
    return float(score), {
        **components,
        "tracking_reward": tracking_reward,
        "max_tracking_reward": max_reward,
        "incomplete_cost": float(incomplete_cost),
        "fall_cost": float(fall_cost),
    }


def config_from_args(args: Any) -> G1ScoreConfig:
    return G1ScoreConfig(
        joint_error_scale=float(getattr(args, "joint_error_scale", DEFAULT_G1_SCORE_CONFIG.joint_error_scale)),
        root_trajectory_error_weight=float(
            getattr(args, "root_trajectory_error_weight", DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_weight)
        ),
        root_trajectory_error_scale=float(
            getattr(args, "root_trajectory_error_scale", DEFAULT_G1_SCORE_CONFIG.root_trajectory_error_scale)
        ),
        root_displacement_error_weight=float(
            getattr(args, "root_displacement_error_weight", DEFAULT_G1_SCORE_CONFIG.root_displacement_error_weight)
        ),
        root_displacement_error_scale=float(
            getattr(args, "root_displacement_error_scale", DEFAULT_G1_SCORE_CONFIG.root_displacement_error_scale)
        ),
        score_component_cap=float(getattr(args, "score_component_cap", DEFAULT_G1_SCORE_CONFIG.score_component_cap)),
        fall_penalty=float(getattr(args, "fall_penalty", DEFAULT_G1_SCORE_CONFIG.fall_penalty)),
    )


def tracker_pool_config_from_args(args: Any) -> G1TrackerPoolConfig:
    return G1TrackerPoolConfig(
        min_completion=float(getattr(args, "good_min_completion", DEFAULT_G1_TRACKER_POOL_CONFIG.min_completion)),
        max_joint_error_rad=float(
            getattr(args, "good_max_joint_error", DEFAULT_G1_TRACKER_POOL_CONFIG.max_joint_error_rad)
        ),
        max_root_trajectory_error_mean_m=float(
            getattr(
                args,
                "good_max_root_trajectory_error",
                DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_trajectory_error_mean_m,
            )
        ),
        max_root_displacement_error_m=float(
            getattr(
                args,
                "good_max_root_displacement_error",
                DEFAULT_G1_TRACKER_POOL_CONFIG.max_root_displacement_error_m,
            )
        ),
        require_root_metrics=not bool(getattr(args, "allow_tracker_pool_without_root_metrics", False)),
    )


def compute_g1_adversarial_score(
    completion: float,
    max_joint_error_rad: float,
    root_trajectory_error_mean_m: float,
    root_displacement_error_m: float,
    fall_detected: bool,
    config: G1ScoreConfig = DEFAULT_G1_SCORE_CONFIG,
) -> float:
    score = (
        (1.0 - float(completion))
        + min(float(max_joint_error_rad) / config.joint_error_scale, config.score_component_cap)
        + config.root_trajectory_error_weight
        * min(float(root_trajectory_error_mean_m) / config.root_trajectory_error_scale, config.score_component_cap)
        + config.root_displacement_error_weight
        * min(float(root_displacement_error_m) / config.root_displacement_error_scale, config.score_component_cap)
    )
    if fall_detected:
        score += config.fall_penalty
    return float(score)


def score_record(record: dict[str, Any], config: G1ScoreConfig = DEFAULT_G1_SCORE_CONFIG) -> float:
    return compute_g1_adversarial_score(
        completion=float(record.get("completion_ratio", 0.0)),
        max_joint_error_rad=float(record.get("max_joint_error_rad", 999.0)),
        root_trajectory_error_mean_m=float(record.get("root_trajectory_error_mean_m", 0.0)),
        root_displacement_error_m=float(record.get("root_displacement_error_m", 0.0)),
        fall_detected=bool(record.get("fall_detected", False)),
        config=config,
    )


def has_root_metrics(record: dict[str, Any]) -> bool:
    if "root_metrics_available" in record:
        return bool(record["root_metrics_available"])
    return (
        "root_trajectory_error_mean_m" in record
        or "root_displacement_error_m" in record
        or "root_displacement_track_m" in record
    )


def is_good_tracker_motion(
    record: dict[str, Any],
    config: G1TrackerPoolConfig = DEFAULT_G1_TRACKER_POOL_CONFIG,
) -> bool:
    if record.get("status") != "scored":
        return False
    if bool(record.get("fall_detected", False)):
        return False
    if config.require_root_metrics and not has_root_metrics(record):
        return False
    return (
        float(record.get("completion_ratio", 0.0)) >= config.min_completion
        and float(record.get("max_joint_error_rad", 999.0)) <= config.max_joint_error_rad
        and float(record.get("root_trajectory_error_mean_m", 999.0))
        <= config.max_root_trajectory_error_mean_m
        and float(record.get("root_displacement_error_m", 999.0)) <= config.max_root_displacement_error_m
    )


def is_hard_adversarial_case(record: dict[str, Any], min_score: float = DEFAULT_G1_HARD_PROMPT_MIN_SCORE) -> bool:
    score = record.get("adversarial_score", record.get("root_aware_score"))
    if score is None:
        score = score_record(record)
    return float(score) >= float(min_score)
