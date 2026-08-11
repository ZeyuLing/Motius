#!/usr/bin/env python3
"""Unified GenTrack evaluator contract and lightweight post-processor.

This script owns the paper-facing metric schema for GenTrack AAAI 2027.  It has
two intentionally separate modes:

1. evaluate-dir: compute table metrics from the standard per-case NPZ rollout
   format under outputs/evaluation/gentrack/.
2. evaluate-canonical-dirs: compute the same metrics directly from paired
   canonical reference/execution directories without duplicating large arrays.
3. adapt-legacy-summary: schema dry-run for old baseline summaries.  These
   outputs are not final-table eligible because legacy runs may use old splits,
   old alignment rules, or method-specific success definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "gentrack-unified-eval-v0.21-20260723"
STANDARD_ROOT = "outputs/evaluation/gentrack"

PAPER_SPLITS = ("lafan1_g1", "amass_test_g1", "wild_g1_clean")
PAPER_CONTINUOUS_AGGREGATION = (
    "all_case_all_valid_export_frame_micro_no_split_macro_v0.21"
)
PAPER_MPJPE_PROTOCOL = (
    "sonic_mpjpe_l_pelvis_translation_root_relative_14_body_"
    "all_case_all_valid_export_frame_micro_v0.21"
)
PAPER_EG_PROTOCOL = (
    "sonic_eg_start_xy_only_no_z_alignment_14_body_"
    "all_case_all_valid_export_frame_micro_v0.21"
)
ROW_MPJPE_PROTOCOL = (
    "sonic_mpjpe_l_pelvis_translation_root_relative_14_body_per_frame_v0.21"
)
ROW_EG_PROTOCOL = "sonic_eg_start_xy_only_no_z_alignment_14_body_per_frame_v0.21"
PAPER_PRIMARY_CONTINUOUS_KEYS = (
    "eg_mpjpe_mm",
    "er_mpjpe_mm",
    "evel_mps",
    "eacc_mps2",
    "mpjpe_mm",
    "mpjve_mps",
    "root_vel_err_mps",
    "root_traj_err_m",
)
PAPER_DERIVATIVE_ORDER = {
    "eg_mpjpe_mm": 0,
    "er_mpjpe_mm": 0,
    "evel_mps": 1,
    "eacc_mps2": 2,
    "mpjpe_mm": 0,
    "mpjve_mps": 1,
    "root_vel_err_mps": 1,
    "root_traj_err_m": 0,
}
INPUT_MANIFEST_SCHEMA = "gentrack-paper-input-manifest-v0.21"
CANONICAL_AMASS_SOURCE = "data/amass_gmr_for_g1/g1"
FORBIDDEN_AMASS_SOURCE = "data/amass_retarged_for_g1"
FORBIDDEN_LEGACY_PROTOCOL_FRAGMENTS = (
    "gentrack_aaai2027/table_tracker/unified_protocol_v1",
)

SUCCESS_PROTOCOL_SONIC_PAPER = "sonic_paper"
SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL = "sonic_release_eval"
SUCCESS_PROTOCOL_ANY2TRACK = "any2track"
SUCCESS_PROTOCOL_CHOICES = (
    SUCCESS_PROTOCOL_SONIC_PAPER,
    SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL,
    SUCCESS_PROTOCOL_ANY2TRACK,
)
DEFAULT_SUCCESS_PROTOCOL = SUCCESS_PROTOCOL_SONIC_PAPER

SUMMARY_KEYS = (
    "num_cases",
    "success_rate_unified",
    "completion",
    "unexpected_fall_rate",
    "e_joint_rad",
    "e_key_m",
    "eg_mpjpe_mm",
    "er_mpjpe_mm",
    "evel_mps",
    "eacc_mps2",
    "mpjpe_mm",
    "mpjve_mps",
    "root_vel_err_mps",
    "root_traj_err_m",
    "num_successful_cases",
    "eg_mpjpe_mm_success_only",
    "er_mpjpe_mm_success_only",
    "evel_mps_success_only",
    "eacc_mps2_success_only",
    "mpjpe_mm_success_only",
    "mpjve_mps_success_only",
    "root_vel_err_mps_success_only",
)

GENERATOR_KEYS = (
    "isaaclab_succ",
    "isaaclab_e_joint_rad",
    "isaaclab_e_key_m",
    "mujoco_succ",
    "mujoco_e_joint_rad",
    "mujoco_e_key_m",
    "tmr_g1_r1",
    "tmr_g1_r2",
    "tmr_g1_r3",
    "tmr_g1_mm_dist",
    "fid",
    "diversity",
    "prompt_coverage",
    "action_entropy",
)

ANY2TRACK_SUCCESS_THRESHOLDS = {
    "completion_min": 0.99,
    "mean_link_position_err_m_max": 0.20,
    "mean_root_height_err_m_max": 0.20,
}

SONIC_PAPER_SUCCESS_THRESHOLDS = {
    # SONIC's cross-method MuJoCo comparison uses the relaxed paper protocol:
    # an imitation fails only when root-height deviation exceeds 0.25 m.
    "max_root_height_err_m_max": 0.25,
    "max_missing_export_frames": 1,
}

SONIC_RELEASE_EVAL_SUCCESS_THRESHOLDS = {
    # The released IsaacLab eval config is deliberately stricter than the
    # paper's cross-method comparison and additionally checks end effectors and
    # pelvis orientation. Keep it as a named diagnostic protocol.
    "max_root_height_err_m_max": 0.25,
    "max_end_effector_height_err_m_max": 0.25,
    "max_root_orientation_err_rad_max": 1.0,
    "max_missing_export_frames": 1,
}

SONIC_TERMINATION_END_EFFECTOR_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

CANONICAL_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

SONIC_KEY_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

UNEXPECTED_FALL_THRESHOLDS = {
    "upright_reference_height_m_min": 0.50,
    "upright_reference_up_cos_min": 0.50,
    "root_drop_relative_to_reference_m_min": 0.25,
    "root_orientation_error_rad_min": 1.00,
    "persistence_seconds_min": 0.20,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_provenance_texts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, Path):
        yield str(value)
        return
    if isinstance(value, bytes):
        yield value.decode("utf-8", errors="replace")
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _iter_provenance_texts(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_provenance_texts(item)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in {"O", "S", "U"}:
            return
        if value.shape == ():
            yield from _iter_provenance_texts(value.item())
            return
        for item in value.reshape(-1).tolist():
            yield from _iter_provenance_texts(item)
        return
    if isinstance(value, np.generic):
        yield from _iter_provenance_texts(value.item())


def _normalized_provenance(payload: Any) -> str:
    return "\n".join(_iter_provenance_texts(payload)).replace("\\", "/").lower()


def _assert_allowed_paper_provenance(
    payload: Any,
    *,
    context: str,
    require_canonical_amass: bool = False,
) -> None:
    normalized = _normalized_provenance(payload)
    if FORBIDDEN_AMASS_SOURCE in normalized:
        raise ValueError(
            f"paper provenance gate rejected {context}: forbidden legacy AMASS source "
            f"{FORBIDDEN_AMASS_SOURCE!r}"
        )
    fragment = next(
        (
            candidate
            for candidate in FORBIDDEN_LEGACY_PROTOCOL_FRAGMENTS
            if candidate in normalized
        ),
        None,
    )
    if fragment is not None:
        raise ValueError(
            f"paper provenance gate rejected {context}: legacy GenTrack protocol "
            f"{fragment!r} is not Table 1 v0.21 eligible"
        )
    if require_canonical_amass and CANONICAL_AMASS_SOURCE not in normalized:
        raise ValueError(
            f"paper provenance gate rejected {context}: AMASS case source/metadata "
            f"must reference canonical {CANONICAL_AMASS_SOURCE!r}"
        )


def _update_array_digest(digest: Any, label: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def _paper_input_digest(
    *,
    joints: np.ndarray,
    joint_names: list[str],
    fps: float,
    total_frames: int,
    num_eval_frames: int,
    qpos: Any = None,
    root_quat: Any = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(INPUT_MANIFEST_SCHEMA.encode("ascii"))
    digest.update(
        json.dumps(
            {
                "fps": float(fps),
                "joint_names": joint_names,
                "num_eval_frames": int(num_eval_frames),
                "total_frames": int(total_frames),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _update_array_digest(digest, "joints", joints[:num_eval_frames])
    if qpos is not None:
        _update_array_digest(digest, "qpos", np.asarray(qpos)[:num_eval_frames])
    if root_quat is not None:
        _update_array_digest(digest, "root_quat", np.asarray(root_quat)[:num_eval_frames])
    return digest.hexdigest()


def _write_input_manifests(
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for role in ("reference", "execution"):
        manifest_path = out_dir / f"{role}_input_manifest.json"
        manifest_rows = [
            {
                "case_id": row["case_id"],
                "num_eval_frames": row["num_eval_frames"],
                "num_total_frames": row[f"num_{role}_frames"],
                "sha256": row[f"{role}_input_sha256"],
                "source_path": row.get("source_path"),
            }
            for row in sorted(rows, key=lambda item: str(item["case_id"]))
        ]
        _write_json(
            manifest_path,
            {
                "schema": INPUT_MANIFEST_SCHEMA,
                "role": role,
                "evaluator_version": VERSION,
                "rows": manifest_rows,
            },
        )
        provenance[role] = {
            "manifest": manifest_path.name,
            "manifest_sha256": _sha256_file(manifest_path),
            "num_cases": len(manifest_rows),
        }
    return provenance


def _finite_mean(vals: Iterable[Any]) -> float | None:
    out = []
    for val in vals:
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            out.append(fval)
    return float(np.mean(out)) if out else None


def _as_bool_rate(vals: Iterable[Any]) -> float | None:
    out = []
    for val in vals:
        if val is None:
            continue
        if isinstance(val, str):
            out.append(val.strip().lower() in {"1", "true", "yes", "y"})
        else:
            out.append(bool(val))
    return float(np.mean(out)) if out else None


def _scalar(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        if payload.shape == ():
            return payload.item()
        if payload.size == 1:
            return payload.reshape(-1)[0].item()
    return payload


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _pick_joints(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ref = case.get("reference_joints")
    exe = case.get("execution_joints")
    names = case.get("joint_names")
    if ref is None:
        ref = case.get("reference_body_pos")
    if exe is None:
        exe = case.get("execution_body_pos")
    if ref is None or exe is None:
        raise ValueError("missing reference/execution joint arrays")
    ref = np.asarray(ref, dtype=np.float32)
    exe = np.asarray(exe, dtype=np.float32)
    if ref.ndim != 3 or exe.ndim != 3 or ref.shape[-1] != 3 or exe.shape[-1] != 3:
        raise ValueError(f"joint arrays must be [T,J,3], got {ref.shape} and {exe.shape}")
    if names is None:
        raise ValueError("joint_names/body_names are required for paper-facing evaluation")
    names_in = [str(x) for x in np.asarray(names).tolist()]
    if len(names_in) != ref.shape[1] or len(names_in) != exe.shape[1]:
        raise ValueError(
            f"joint name count {len(names_in)} does not match arrays {ref.shape[1]}/{exe.shape[1]}"
        )
    name_to_index = {name: idx for idx, name in enumerate(names_in)}
    if all(name in name_to_index for name in CANONICAL_G1_BODY_NAMES):
        selected_names = list(CANONICAL_G1_BODY_NAMES)
    elif all(name in name_to_index for name in SONIC_KEY_BODY_NAMES):
        selected_names = list(SONIC_KEY_BODY_NAMES)
    else:
        missing = [name for name in SONIC_KEY_BODY_NAMES if name not in name_to_index]
        raise ValueError(f"missing required SONIC metric bodies: {missing}")
    keep = [name_to_index[name] for name in selected_names]
    return ref[:, keep], exe[:, keep], selected_names


def _safe_text(case: dict[str, Any], key: str) -> str | None:
    if key not in case:
        return None
    value = _scalar(case[key])
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _max_joint_angle_error_rad(case: dict[str, Any], num_frames: int) -> tuple[float | None, str | None]:
    """Return the maximum wrapped G1 joint-angle error from canonical qpos.

    Canonical G1 qpos stores floating-base position/quaternion in the first
    seven entries and the 29 actuated joints afterwards.  Explicit metric
    exports remain authoritative; this value is retained only as a diagnostic
    and does not enter the Any2Track paper success criterion.
    """
    explicit = case.get("max_joint_err_rad")
    if explicit is not None:
        return float(_scalar(explicit)), None

    ref_qpos = case.get("reference_qpos")
    exe_qpos = case.get("execution_qpos")
    if ref_qpos is None or exe_qpos is None:
        return None, "max joint-angle error unavailable: missing canonical qpos"

    ref_qpos = np.asarray(ref_qpos, dtype=np.float32)
    exe_qpos = np.asarray(exe_qpos, dtype=np.float32)
    if ref_qpos.ndim != 2 or exe_qpos.ndim != 2:
        return None, f"max joint-angle error unavailable: invalid qpos shapes {ref_qpos.shape}/{exe_qpos.shape}"
    if ref_qpos.shape[1] != exe_qpos.shape[1] or ref_qpos.shape[1] <= 7:
        return None, f"max joint-angle error unavailable: incompatible qpos shapes {ref_qpos.shape}/{exe_qpos.shape}"

    n = min(num_frames, ref_qpos.shape[0], exe_qpos.shape[0])
    if n == 0:
        return None, "max joint-angle error unavailable: empty canonical qpos"
    delta = exe_qpos[:n, 7:] - ref_qpos[:n, 7:]
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return float(np.max(np.abs(wrapped))), None


def _mean_joint_angle_error_rad(case: dict[str, Any], num_frames: int) -> tuple[float | None, str | None]:
    ref_qpos = case.get("reference_qpos")
    exe_qpos = case.get("execution_qpos")
    if ref_qpos is None or exe_qpos is None:
        return None, "E_joint unavailable: missing canonical qpos"

    ref_qpos = np.asarray(ref_qpos, dtype=np.float32)
    exe_qpos = np.asarray(exe_qpos, dtype=np.float32)
    if ref_qpos.ndim != 2 or exe_qpos.ndim != 2:
        return None, f"E_joint unavailable: invalid qpos shapes {ref_qpos.shape}/{exe_qpos.shape}"
    if ref_qpos.shape[1] != exe_qpos.shape[1] or ref_qpos.shape[1] <= 7:
        return None, f"E_joint unavailable: incompatible qpos shapes {ref_qpos.shape}/{exe_qpos.shape}"

    n = min(num_frames, ref_qpos.shape[0], exe_qpos.shape[0])
    if n == 0:
        return None, "E_joint unavailable: empty canonical qpos"
    delta = exe_qpos[:n, 7:] - ref_qpos[:n, 7:]
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return float(np.mean(np.abs(wrapped))), None


def _root_quaternions(case: dict[str, Any], num_frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    ref_quat = case.get("reference_body_quat")
    exe_quat = case.get("execution_body_quat")
    if ref_quat is not None and exe_quat is not None:
        ref_quat = np.asarray(ref_quat, dtype=np.float64)
        exe_quat = np.asarray(exe_quat, dtype=np.float64)
        names = case.get("body_names", case.get("joint_names"))
        names = [] if names is None else [str(x) for x in np.asarray(names).tolist()]
        if (
            ref_quat.ndim == 3
            and exe_quat.ndim == 3
            and ref_quat.shape[-1] == 4
            and exe_quat.shape[-1] == 4
            and "pelvis" in names
        ):
            root_index = names.index("pelvis")
            n = min(num_frames, len(ref_quat), len(exe_quat))
            return ref_quat[:n, root_index], exe_quat[:n, root_index]

    ref_qpos = case.get("reference_qpos")
    exe_qpos = case.get("execution_qpos")
    if ref_qpos is None or exe_qpos is None:
        return None
    ref_qpos = np.asarray(ref_qpos, dtype=np.float64)
    exe_qpos = np.asarray(exe_qpos, dtype=np.float64)
    if ref_qpos.ndim != 2 or exe_qpos.ndim != 2 or ref_qpos.shape[1] < 7 or exe_qpos.shape[1] < 7:
        return None
    n = min(num_frames, len(ref_qpos), len(exe_qpos))
    return ref_qpos[:n, 3:7], exe_qpos[:n, 3:7]


def _normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def _quaternion_geodesic_error_wxyz(reference: np.ndarray, execution: np.ndarray) -> np.ndarray:
    """Return sign-invariant quaternion geodesic error in radians."""
    reference = _normalize_quaternions(np.asarray(reference, dtype=np.float64))
    execution = _normalize_quaternions(np.asarray(execution, dtype=np.float64))
    dot = np.abs(np.sum(reference * execution, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _quaternion_up_axis_wxyz(quat: np.ndarray) -> np.ndarray:
    """Return each scalar-first quaternion's local +Z axis in world space."""
    quat = _normalize_quaternions(np.asarray(quat, dtype=np.float64))
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    )


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _unexpected_fall(
    case: dict[str, Any],
    ref_root_pos: np.ndarray,
    exe_root_pos: np.ndarray,
    fps: float,
) -> tuple[bool, str, int, int, str | None]:
    explicit = case.get("unexpected_fall")
    if explicit is None:
        explicit = case.get("unexpected_fall_detected")
    if explicit is not None:
        return bool(_scalar(explicit)), "reference_conditioned_simulator_export", 0, 0, None

    n = min(len(ref_root_pos), len(exe_root_pos))
    ref_root_pos = np.asarray(ref_root_pos[:n], dtype=np.float64)
    exe_root_pos = np.asarray(exe_root_pos[:n], dtype=np.float64)
    upright = ref_root_pos[:, 2] >= UNEXPECTED_FALL_THRESHOLDS["upright_reference_height_m_min"]
    root_drop = ref_root_pos[:, 2] - exe_root_pos[:, 2]
    collapse = root_drop > UNEXPECTED_FALL_THRESHOLDS["root_drop_relative_to_reference_m_min"]
    source = "reference_conditioned_root_drop"
    warning = "Unexpected Fall uses root-drop only because root orientations are unavailable"

    root_quat = _root_quaternions(case, n)
    if root_quat is not None:
        ref_quat, exe_quat = root_quat
        quat_frames = min(len(ref_quat), len(exe_quat), len(upright))
        ref_quat = ref_quat[:quat_frames]
        exe_quat = exe_quat[:quat_frames]
        upright = upright[:quat_frames]
        collapse = collapse[:quat_frames]
        ref_up = _quaternion_up_axis_wxyz(ref_quat)
        exe_up = _quaternion_up_axis_wxyz(exe_quat)
        ref_up_cos = ref_up[:, 2]
        upright &= ref_up_cos >= UNEXPECTED_FALL_THRESHOLDS["upright_reference_up_cos_min"]
        # Compare pelvis up axes, not full quaternion geodesic distance.  A
        # yaw difference is a tracking error, but it is not evidence of a fall.
        up_axis_cos = np.clip(np.sum(ref_up * exe_up, axis=-1), -1.0, 1.0)
        tilt_error = np.arccos(up_axis_cos)
        collapse |= tilt_error > UNEXPECTED_FALL_THRESHOLDS["root_orientation_error_rad_min"]
        source = "reference_conditioned_root_drop_or_tilt"
        warning = None

    signal = upright & collapse
    persistence_frames = max(
        1,
        int(math.ceil(UNEXPECTED_FALL_THRESHOLDS["persistence_seconds_min"] * max(fps, 1e-6))),
    )
    longest_run = _longest_true_run(signal)
    return longest_run >= persistence_frames, source, longest_run, persistence_frames, warning


@dataclass
class CaseMetrics:
    row: dict[str, Any]
    warnings: list[str]


def _evaluate_payload(
    case: dict[str, Any],
    path: Path,
    allow_joint_crop: bool = False,
    use_native_rollout_status: bool = False,
    success_protocol: str = DEFAULT_SUCCESS_PROTOCOL,
) -> CaseMetrics:
    if success_protocol not in SUCCESS_PROTOCOL_CHOICES:
        raise ValueError(
            f"unsupported success protocol {success_protocol!r}; "
            f"expected one of {SUCCESS_PROTOCOL_CHOICES}"
        )
    provenance_payload = {
        "input_path": str(path),
        "split": case.get("split"),
        "payload": _provenance_subset(case),
    }
    normalized_provenance = _normalized_provenance(provenance_payload)
    _assert_allowed_paper_provenance(
        provenance_payload,
        context=str(path),
        require_canonical_amass=(
            "amass_test_g1" in normalized_provenance
            or CANONICAL_AMASS_SOURCE in normalized_provenance
            or FORBIDDEN_AMASS_SOURCE in normalized_provenance
        ),
    )
    ref, exe, joint_names = _pick_joints(case)
    warnings: list[str] = []

    if ref.shape[1] != exe.shape[1]:
        if not allow_joint_crop:
            raise ValueError(f"joint count mismatch: reference={ref.shape[1]} execution={exe.shape[1]}")
        keep = min(ref.shape[1], exe.shape[1])
        ref = ref[:, :keep]
        exe = exe[:, :keep]
        joint_names = joint_names[:keep]
        warnings.append(f"cropped joint count to {keep}")

    if ref.shape[0] == 0 or exe.shape[0] == 0:
        raise ValueError("empty motion")
    length_completion = min(float(exe.shape[0]) / max(float(ref.shape[0]), 1.0), 1.0)
    native_completion = case.get("native_completion")
    if native_completion is None:
        native_completion = case.get("completion")
    if native_completion is None:
        native_completion = case.get("execution_progress")
    native_terminated_value = case.get("native_terminated")
    if native_terminated_value is None:
        native_terminated_value = case.get("terminated")
    native_terminated = (
        None if native_terminated_value is None else bool(_scalar(native_terminated_value))
    )

    # Tracker-table success is recomputed from synchronized exported frames for
    # every method.  Method-native callbacks remain diagnostic unless a caller
    # explicitly opts into a native evaluator (e.g. generator executability).
    if use_native_rollout_status and native_completion is not None:
        completion = float(np.clip(float(_scalar(native_completion)), 0.0, 1.0))
        completion_source = "native_simulator_progress"
    else:
        completion = length_completion
        completion_source = "trajectory_length"

    n = min(ref.shape[0], exe.shape[0])
    if completion_source == "native_simulator_progress":
        # Some simulators keep writing fixed-size buffers after an episode has
        # terminated.  Those padded/reset frames are not part of the rollout.
        n = min(n, max(1, int(math.ceil(ref.shape[0] * completion - 1e-6))))
    ref_eval = ref[:n]
    exe_eval = exe[:n]
    fps = float(_scalar(case.get("fps", case.get("frequency", 30.0))))
    dt = 1.0 / max(fps, 1e-6)

    # E_g: start-XY-aligned global MPJPE.  We remove only the initial root XY
    # offset and keep height/trajectory errors visible.
    start_xy_offset = exe_eval[0, 0, :2] - ref_eval[0, 0, :2]
    exe_global = exe_eval.copy()
    exe_global[:, :, :2] -= start_xy_offset[None, None, :]
    global_err = np.linalg.norm(exe_global - ref_eval, axis=-1)

    sonic_indices = [joint_names.index(name) for name in SONIC_KEY_BODY_NAMES]
    sonic_global_err = global_err[:, sonic_indices]

    # E_r remains the canonical-link local diagnostic. MPJPE-L below is the
    # paper metric and always uses the fixed SONIC 14-body set.
    ref_local = ref_eval - ref_eval[:, :1, :]
    exe_local = exe_eval - exe_eval[:, :1, :]
    local_err = np.linalg.norm(exe_local - ref_local, axis=-1)

    # MPJPE-L removes each frame's pelvis translation independently. It does
    # not rotate-align poses and it never expands beyond SONIC's fixed 14
    # command bodies.
    sonic_ref_local = ref_local[:, sonic_indices]
    sonic_exe_local = exe_local[:, sonic_indices]
    sonic_local_err = np.linalg.norm(sonic_exe_local - sonic_ref_local, axis=-1)

    if n >= 2:
        vel_err = np.linalg.norm(np.diff(exe_global, axis=0) - np.diff(ref_eval, axis=0), axis=-1) / dt
        sonic_local_vel_err = (
            np.linalg.norm(
                np.diff(sonic_exe_local, axis=0) - np.diff(sonic_ref_local, axis=0),
                axis=-1,
            )
            / dt
        )
        root_vel_err = (
            np.linalg.norm(
                np.diff(exe_global[:, 0], axis=0) - np.diff(ref_eval[:, 0], axis=0),
                axis=-1,
            )
            / dt
        )
    else:
        vel_err = np.array([], dtype=np.float32)
        sonic_local_vel_err = np.array([], dtype=np.float32)
        root_vel_err = np.array([], dtype=np.float32)
    if n >= 3:
        acc_err = (
            np.linalg.norm(np.diff(exe_global, n=2, axis=0) - np.diff(ref_eval, n=2, axis=0), axis=-1)
            / (dt * dt)
        )
    else:
        acc_err = np.array([], dtype=np.float32)

    root_traj_err = np.linalg.norm(exe_global[:, 0, :2] - ref_eval[:, 0, :2], axis=-1)
    root_height_err = np.abs(exe_global[:, 0, 2] - ref_eval[:, 0, 2])

    unexpected_fall, unexpected_fall_source, fall_run, fall_window, fall_warning = _unexpected_fall(
        case,
        ref_eval[:, 0],
        exe_eval[:, 0],
        fps,
    )
    if fall_warning:
        warnings.append(fall_warning)

    max_joint_err_rad, joint_angle_warning = _max_joint_angle_error_rad(case, n)
    if joint_angle_warning:
        warnings.append(joint_angle_warning)
    e_joint_rad, e_joint_warning = _mean_joint_angle_error_rad(case, n)
    if e_joint_warning:
        warnings.append(e_joint_warning)

    eg = float(np.mean(sonic_global_err) * 1000.0)
    er = float(np.mean(local_err) * 1000.0)
    evel = float(np.mean(vel_err)) if vel_err.size else None
    eacc = float(np.mean(acc_err)) if acc_err.size else None
    mpjpe = float(np.mean(sonic_local_err) * 1000.0)
    mpjve = float(np.mean(sonic_local_vel_err)) if sonic_local_vel_err.size else None
    root_vel = float(np.mean(root_vel_err)) if root_vel_err.size else None

    mean_link_position_err_m = float(np.mean(local_err))
    mean_root_height_err_m = float(np.mean(root_height_err))
    max_root_height_err_m = float(np.max(root_height_err))
    end_effector_indices = [joint_names.index(name) for name in SONIC_TERMINATION_END_EFFECTOR_NAMES]
    end_effector_height_err = np.abs(
        exe_global[:, end_effector_indices, 2] - ref_eval[:, end_effector_indices, 2]
    )
    max_end_effector_height_err_m = float(np.max(end_effector_height_err))

    root_quat = _root_quaternions(case, n)
    max_root_orientation_err_rad = None
    if root_quat is not None:
        ref_root_quat, exe_root_quat = root_quat
        root_orientation_err = _quaternion_geodesic_error_wxyz(ref_root_quat, exe_root_quat)
        max_root_orientation_err_rad = float(np.max(root_orientation_err))
    elif success_protocol == SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL:
        raise ValueError(
            "SONIC release-eval protocol requires reference/execution pelvis quaternions; "
            "refusing a position-only success approximation"
        )
    e_key_m = float(np.mean(sonic_global_err))
    full_export_with_endpoint_tolerance = (
        exe.shape[0]
        >= max(
            1,
            ref.shape[0] - SONIC_PAPER_SUCCESS_THRESHOLDS["max_missing_export_frames"],
        )
    )

    sonic_paper_failure_reasons: list[str] = []
    if not full_export_with_endpoint_tolerance:
        sonic_paper_failure_reasons.append("incomplete_export")
    if max_root_height_err_m > SONIC_PAPER_SUCCESS_THRESHOLDS["max_root_height_err_m_max"]:
        sonic_paper_failure_reasons.append("root_height_termination")

    sonic_release_eval_failure_reasons = list(sonic_paper_failure_reasons)
    if (
        max_end_effector_height_err_m
        > SONIC_RELEASE_EVAL_SUCCESS_THRESHOLDS["max_end_effector_height_err_m_max"]
    ):
        sonic_release_eval_failure_reasons.append("end_effector_height_termination")
    if max_root_orientation_err_rad is not None:
        if (
            max_root_orientation_err_rad
            > SONIC_RELEASE_EVAL_SUCCESS_THRESHOLDS["max_root_orientation_err_rad_max"]
        ):
            sonic_release_eval_failure_reasons.append("root_orientation_termination")

    any2track_failure_reasons: list[str] = []
    if completion < ANY2TRACK_SUCCESS_THRESHOLDS["completion_min"]:
        any2track_failure_reasons.append("incomplete_trajectory")
    if mean_link_position_err_m > ANY2TRACK_SUCCESS_THRESHOLDS["mean_link_position_err_m_max"]:
        any2track_failure_reasons.append("mean_link_position_error")
    if mean_root_height_err_m > ANY2TRACK_SUCCESS_THRESHOLDS["mean_root_height_err_m_max"]:
        any2track_failure_reasons.append("mean_root_height_error")

    protocol_failures = {
        SUCCESS_PROTOCOL_SONIC_PAPER: sonic_paper_failure_reasons,
        SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL: sonic_release_eval_failure_reasons,
        SUCCESS_PROTOCOL_ANY2TRACK: any2track_failure_reasons,
    }
    failure_reasons = list(protocol_failures[success_protocol])
    terminated = native_terminated if use_native_rollout_status else None
    if use_native_rollout_status and (
        completion < ANY2TRACK_SUCCESS_THRESHOLDS["completion_min"] or terminated is True
    ):
        if "native_rollout_failure" not in failure_reasons:
            failure_reasons.append("native_rollout_failure")
    success = not failure_reasons

    if root_quat is None:
        ref_root_quat = None
        exe_root_quat = None
    reference_input_sha256 = _paper_input_digest(
        joints=ref_eval,
        joint_names=joint_names,
        fps=fps,
        total_frames=ref.shape[0],
        num_eval_frames=n,
        qpos=case.get("reference_qpos"),
        root_quat=ref_root_quat,
    )
    execution_input_sha256 = _paper_input_digest(
        joints=exe_eval,
        joint_names=joint_names,
        fps=fps,
        total_frames=exe.shape[0],
        num_eval_frames=n,
        qpos=case.get("execution_qpos"),
        root_quat=exe_root_quat,
    )

    row = {
        "case_id": _safe_text(case, "case_id") or path.stem,
        "caption": _safe_text(case, "caption"),
        "source_path": _safe_text(case, "source_path"),
        "method": _safe_text(case, "method"),
        "fps": fps,
        "num_reference_frames": int(ref.shape[0]),
        "num_execution_frames": int(exe.shape[0]),
        "num_eval_frames": int(n),
        "completion": completion,
        "completion_source": completion_source,
        "terminated": terminated,
        "native_completion": (
            None
            if native_completion is None
            else float(np.clip(float(_scalar(native_completion)), 0.0, 1.0))
        ),
        "native_terminated": native_terminated,
        "native_rollout_status_used": bool(use_native_rollout_status),
        "full_export_with_endpoint_tolerance": bool(full_export_with_endpoint_tolerance),
        "unexpected_fall": unexpected_fall,
        "unexpected_fall_source": unexpected_fall_source,
        "unexpected_fall_longest_run_frames": fall_run,
        "unexpected_fall_persistence_frames": fall_window,
        "success_unified": bool(success),
        "success_protocol": (
            "sonic_paper_mujoco_fall_only_30fps_full_export"
            if success_protocol == SUCCESS_PROTOCOL_SONIC_PAPER
            else (
                "sonic_release_eval_30fps_full_export"
                if success_protocol == SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL
                else "full_export_any2track_mean_link_and_root_height_0p20m"
            )
        ),
        "success_sonic_paper_exported": not sonic_paper_failure_reasons,
        "success_sonic_release_eval_exported": not sonic_release_eval_failure_reasons,
        "success_any2track_exported": not any2track_failure_reasons,
        "failure_reasons_sonic_paper": sonic_paper_failure_reasons,
        "failure_reasons_sonic_release_eval": sonic_release_eval_failure_reasons,
        "failure_reasons_any2track": any2track_failure_reasons,
        "failure_reasons": failure_reasons,
        "eg_mpjpe_mm": eg,
        "er_mpjpe_mm": er,
        "evel_mps": evel,
        "eacc_mps2": eacc,
        "mpjpe_mm": mpjpe,
        "mpjve_mps": mpjve,
        "root_vel_err_mps": root_vel,
        "root_traj_err_m": float(np.mean(root_traj_err)),
        "root_height_err_m": mean_root_height_err_m,
        "max_root_height_err_m": max_root_height_err_m,
        "max_end_effector_height_err_m": max_end_effector_height_err_m,
        "max_root_orientation_err_rad": max_root_orientation_err_rad,
        "mean_link_position_err_m": mean_link_position_err_m,
        "max_joint_err_rad": max_joint_err_rad,
        "e_joint_rad": e_joint_rad,
        "e_key_m": e_key_m,
        "joint_count": int(ref.shape[1]),
        "joint_names": joint_names,
        "reference_input_sha256": reference_input_sha256,
        "execution_input_sha256": execution_input_sha256,
        "eg_protocol": ROW_EG_PROTOCOL,
        "eg_body_count": len(SONIC_KEY_BODY_NAMES),
        "eg_body_names": list(SONIC_KEY_BODY_NAMES),
        "mpjpe_protocol": ROW_MPJPE_PROTOCOL,
        "mpjpe_body_count": len(SONIC_KEY_BODY_NAMES),
        "mpjpe_body_names": list(SONIC_KEY_BODY_NAMES),
        "warnings": warnings,
    }
    return CaseMetrics(row=row, warnings=warnings)


def _evaluate_case(
    path: Path,
    allow_joint_crop: bool = False,
    use_native_rollout_status: bool = False,
    success_protocol: str = DEFAULT_SUCCESS_PROTOCOL,
) -> CaseMetrics:
    return _evaluate_payload(
        _load_npz(path),
        path,
        allow_joint_crop,
        use_native_rollout_status=use_native_rollout_status,
        success_protocol=success_protocol,
    )


def _canonical_body_arrays(data: dict[str, Any], path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names_raw = data.get("body_names", data.get("joint_names"))
    if names_raw is None:
        raise ValueError(f"missing body_names/joint_names: {path}")
    names = [str(x) for x in np.asarray(names_raw).tolist()]
    name_to_index = {name: idx for idx, name in enumerate(names)}
    if all(name in name_to_index for name in CANONICAL_G1_BODY_NAMES):
        selected_names = list(CANONICAL_G1_BODY_NAMES)
    elif all(name in name_to_index for name in SONIC_KEY_BODY_NAMES):
        selected_names = list(SONIC_KEY_BODY_NAMES)
    else:
        missing = [name for name in SONIC_KEY_BODY_NAMES if name not in name_to_index]
        raise ValueError(f"missing required SONIC metric bodies in {path}: {missing}")
    keep = [name_to_index[name] for name in selected_names]
    body_pos = np.asarray(data["body_pos"], dtype=np.float32)[:, keep]
    body_quat = np.asarray(data["body_quat"], dtype=np.float32)[:, keep]
    return body_pos, body_quat, selected_names


def _provenance_subset(data: dict[str, Any]) -> dict[str, Any]:
    tokens = ("source", "metadata", "manifest", "provenance", "protocol", "path")
    return {
        key: value
        for key, value in data.items()
        if any(token in key.lower() for token in tokens)
    }


def _canonical_case_payload(
    ref_path: Path,
    exe_path: Path,
    *,
    method: str,
    split: str,
    fps: float,
    ref_qpos_path: Path | None = None,
    exe_qpos_path: Path | None = None,
) -> dict[str, Any]:
    ref = _load_npz(ref_path)
    exe = _load_npz(exe_path)
    ref_pos, ref_quat, body_names = _canonical_body_arrays(ref, ref_path)
    exe_pos, exe_quat, exe_names = _canonical_body_arrays(exe, exe_path)
    if body_names != exe_names:
        raise ValueError(f"canonical body order mismatch: {ref_path} vs {exe_path}")
    payload = {
        "reference_body_pos": ref_pos,
        "execution_body_pos": exe_pos,
        "reference_body_quat": ref_quat,
        "execution_body_quat": exe_quat,
        "reference_joints": ref_pos,
        "execution_joints": exe_pos,
        "body_names": np.asarray(body_names, dtype=str),
        "joint_names": np.asarray(body_names, dtype=str),
        "fps": np.float32(fps),
        "frequency": np.float32(fps),
        "case_id": np.asarray(exe_path.stem),
        "method": np.asarray(method),
        "split": np.asarray(split),
        "input_paths": {
            "reference": str(ref_path),
            "execution": str(exe_path),
        },
        "metadata": {
            "reference": _provenance_subset(ref),
            "execution": _provenance_subset(exe),
        },
    }
    if (ref_qpos_path is None) != (exe_qpos_path is None):
        raise ValueError("reference and execution qpos paths must be provided together")
    if ref_qpos_path is not None and exe_qpos_path is not None:
        ref_qpos_data = _load_npz(ref_qpos_path)
        exe_qpos_data = _load_npz(exe_qpos_path)
        if "qpos" not in ref_qpos_data or "qpos" not in exe_qpos_data:
            raise ValueError(f"missing qpos array: {ref_qpos_path} or {exe_qpos_path}")
        payload["reference_qpos"] = np.asarray(ref_qpos_data["qpos"], dtype=np.float32)
        payload["execution_qpos"] = np.asarray(exe_qpos_data["qpos"], dtype=np.float32)
        payload["input_paths"]["reference_qpos"] = str(ref_qpos_path)
        payload["input_paths"]["execution_qpos"] = str(exe_qpos_path)
        payload["metadata"]["reference_qpos"] = _provenance_subset(ref_qpos_data)
        payload["metadata"]["execution_qpos"] = _provenance_subset(exe_qpos_data)
    canonical_source = next(
        (
            text
            for text in _iter_provenance_texts(payload["metadata"])
            if CANONICAL_AMASS_SOURCE in text.replace("\\", "/").lower()
        ),
        None,
    )
    if canonical_source is not None:
        payload["source_path"] = np.asarray(
            CANONICAL_AMASS_SOURCE
            if canonical_source.lstrip().startswith(("{", "["))
            else canonical_source
        )
    return payload


def _load_rollout_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load per-case simulator completion metadata from a rollout summary."""
    payload = json.loads(path.read_text())
    _assert_allowed_paper_provenance(
        {"rollout_summary_path": str(path), "payload": payload},
        context=str(path),
    )
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"rollout summary must contain a cases list: {path}")

    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"rollout summary contains a non-object case row: {path}")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"rollout summary case row is missing case_id: {path}")
        if case_id in metadata:
            raise ValueError(f"rollout summary contains duplicate case_id {case_id!r}: {path}")

        completion = row.get("completion")
        if completion is None:
            completion = row.get("execution_progress")
        if completion is None:
            completion = row.get("progress")
        terminated = row.get("terminated")
        if completion is None or terminated is None:
            raise ValueError(
                f"rollout summary case {case_id!r} must contain progress/completion and terminated: {path}"
            )
        metadata[case_id] = {
            "completion": float(np.clip(float(completion), 0.0, 1.0)),
            "terminated": bool(terminated),
        }
    return metadata


def _load_case_ids(path: Path) -> list[str]:
    """Load either a compact ID list or a provenance-rich cases manifest."""
    payload = json.loads(path.read_text())
    _assert_allowed_paper_provenance(
        {"manifest_path": str(path), "payload": payload},
        context=str(path),
    )
    if isinstance(payload, list) and all(isinstance(case_id, str) for case_id in payload):
        case_ids = list(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        case_ids = []
        for index, row in enumerate(payload["rows"]):
            if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
                raise ValueError(f"manifest row {index} is missing a string case_id: {path}")
            case_ids.append(row["case_id"])
    else:
        raise ValueError(f"manifest must be an ID list or an object with rows[].case_id: {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"manifest contains duplicate case IDs: {path}")
    return case_ids


def _summary_from_rows(rows: list[dict[str, Any]], *, table_eligible: bool, warnings: list[str]) -> dict[str, Any]:
    failure_reasons = sorted({reason for row in rows for reason in row.get("failure_reasons", [])})
    successful_rows = [row for row in rows if row.get("success_unified") is True]

    def frame_mean(
        selected_rows: list[dict[str, Any]], key: str, *, derivative_order: int = 0
    ) -> float | None:
        weighted_sum = 0.0
        total_weight = 0
        for row in selected_rows:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                continue
            frames = int(row.get("num_eval_frames", 0)) - derivative_order
            if frames <= 0:
                continue
            weighted_sum += float(value) * frames
            total_weight += frames
        return weighted_sum / total_weight if total_weight else None

    paper_metrics = {
        key: frame_mean(rows, key, derivative_order=PAPER_DERIVATIVE_ORDER[key])
        for key in PAPER_PRIMARY_CONTINUOUS_KEYS
    }
    success_only_metrics = {
        key: frame_mean(
            successful_rows,
            key,
            derivative_order=PAPER_DERIVATIVE_ORDER[key],
        )
        for key in PAPER_PRIMARY_CONTINUOUS_KEYS
    }

    summary = {k: None for k in SUMMARY_KEYS}
    summary.update({k: None for k in GENERATOR_KEYS})
    summary.update(
        {
            "evaluator_version": VERSION,
            "num_cases": len(rows),
            "success_rate_unified": _as_bool_rate(row.get("success_unified") for row in rows),
            "success_rate_sonic_paper": _as_bool_rate(
                row.get("success_sonic_paper_exported") for row in rows
            ),
            "success_rate_sonic_release_eval": _as_bool_rate(
                row.get("success_sonic_release_eval_exported") for row in rows
            ),
            "success_rate_any2track": _as_bool_rate(
                row.get("success_any2track_exported") for row in rows
            ),
            "success_protocol": (
                next(iter({row.get("success_protocol") for row in rows}), None)
                if len({row.get("success_protocol") for row in rows}) <= 1
                else "mixed"
            ),
            "completion": _finite_mean(row.get("completion") for row in rows),
            "unexpected_fall_rate": _as_bool_rate(row.get("unexpected_fall") for row in rows),
            "e_joint_rad": _finite_mean(row.get("e_joint_rad") for row in rows),
            "e_key_m": frame_mean(rows, "e_key_m"),
            # Main-table continuous errors use every exported trajectory and
            # frame. Conditioning on method-specific success sets makes a
            # tracker that attempts harder clips incomparable with one that
            # succeeds only on easy clips.
            **paper_metrics,
            "num_successful_cases": len(successful_rows),
            "eg_mpjpe_mm_success_only": success_only_metrics["eg_mpjpe_mm"],
            "er_mpjpe_mm_success_only": success_only_metrics["er_mpjpe_mm"],
            "evel_mps_success_only": success_only_metrics["evel_mps"],
            "eacc_mps2_success_only": success_only_metrics["eacc_mps2"],
            "mpjpe_mm_success_only": success_only_metrics["mpjpe_mm"],
            "mpjve_mps_success_only": success_only_metrics["mpjve_mps"],
            "root_vel_err_mps_success_only": success_only_metrics[
                "root_vel_err_mps"
            ],
            "success_only_diagnostic": success_only_metrics,
            "success_only_is_diagnostic": True,
            "paper_metric_source": "all_cases",
            "paper_primary_fields": list(PAPER_PRIMARY_CONTINUOUS_KEYS),
            "paper_continuous_aggregation": PAPER_CONTINUOUS_AGGREGATION,
            "paper_splits": list(PAPER_SPLITS),
            "eg_protocol": PAPER_EG_PROTOCOL,
            "eg_body_count": len(SONIC_KEY_BODY_NAMES),
            "eg_body_names": list(SONIC_KEY_BODY_NAMES),
            "mpjpe_protocol": PAPER_MPJPE_PROTOCOL,
            "mpjpe_body_count": len(SONIC_KEY_BODY_NAMES),
            "mpjpe_body_names": list(SONIC_KEY_BODY_NAMES),
            "failure_reasons": {
                reason: sum(1 for row in rows if reason in row.get("failure_reasons", []))
                for reason in failure_reasons
            },
            "final_table_eligible": table_eligible,
            "warnings": sorted(set(w for w in warnings if w)),
        }
    )
    return summary


def _extract_legacy_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    motions = data.get("motions", [])
    if isinstance(motions, dict):
        return [dict(v, motion=k) for k, v in motions.items() if isinstance(v, dict)]
    if isinstance(motions, list):
        return [dict(v) for v in motions if isinstance(v, dict)]
    return []


def _summary_blob(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", data)
    return summary if isinstance(summary, dict) else {}


def _legacy_metric(row: dict[str, Any], summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def _adapt_legacy_summary(input_path: Path, out_dir: Path, method: str, split: str) -> dict[str, Any]:
    data = json.loads(input_path.read_text())
    summary = _summary_blob(data)
    legacy_rows = _extract_legacy_rows(data)
    rows: list[dict[str, Any]] = []
    warnings = [
        "legacy summary adapter dry-run only",
        "standard per-case NPZ rollouts were not used",
        "legacy alignment may differ from start-XY-aligned E_g",
    ]
    if "paper_success_rate" not in summary and not any("paper_success" in r for r in legacy_rows):
        warnings.append("Any2Track-style paper_success missing; success_rate_unified is null")
    if "xy_aligned_mpjpe_mm" in summary or any("xy_aligned_mpjpe_mm" in r for r in legacy_rows):
        warnings.append("legacy E_g is mapped from xy_aligned_mpjpe_mm when no start-aligned metric exists")

    if not legacy_rows:
        legacy_rows = [{} for _ in range(int(float(summary.get("num_motions", 0) or 0)))]

    for i, row in enumerate(legacy_rows):
        case_id = row.get("case_id") or row.get("motion") or row.get("name") or row.get("traj_id") or f"case_{i:06d}"
        adapted = {
            "case_id": str(case_id),
            "method": method,
            "split": split,
            "success_unified": row.get("paper_success") if "paper_success" in row else None,
            "completion": _legacy_metric(row, summary, "completion", "length_ratio"),
            "unexpected_fall": None,
            "legacy_fall": _legacy_metric(row, summary, "fall", "fall_detected"),
            "eg_mpjpe_mm": _legacy_metric(
                row,
                summary,
                "start_aligned_mpjpe_mm",
                "xy_aligned_mpjpe_mm",
                "mpjpe_mm",
            ),
            "er_mpjpe_mm": _legacy_metric(row, summary, "local_mpjpe_mm"),
            "evel_mps": _legacy_metric(row, summary, "mpjve_mps", "body_vel_err_mean"),
            "eacc_mps2": _legacy_metric(row, summary, "mpjae_mps2", "body_acc_err_mean"),
            "legacy_row": True,
            "warnings": warnings,
        }
        rows.append(adapted)

    out_summary = _summary_from_rows(rows, table_eligible=False, warnings=warnings)
    out_summary.update(
        {
            "method": method,
            "split": split,
            "legacy_input": str(input_path),
            "legacy_num_motions": summary.get("num_motions"),
        }
    )
    if out_summary["success_rate_unified"] is None and "paper_success_rate" in summary:
        out_summary["success_rate_unified"] = summary.get("paper_success_rate")
    _write_json(out_dir / "summary.json", out_summary)
    _write_jsonl(out_dir / "case_metrics.jsonl", rows)
    _write_jsonl(out_dir / "failed_cases.jsonl", [])
    _write_json(out_dir / "manifest.json", {"method": method, "split": split, "rows": len(rows), "legacy_input": str(input_path)})
    return out_summary


def _evaluate_dir(
    cases_dir: Path,
    out_dir: Path,
    allow_joint_crop: bool,
    use_native_rollout_status: bool = False,
    success_protocol: str = DEFAULT_SUCCESS_PROTOCOL,
) -> dict[str, Any]:
    rows = []
    failures = []
    warnings = []
    for path in sorted(cases_dir.glob("*.npz")):
        try:
            result = _evaluate_case(
                path,
                allow_joint_crop=allow_joint_crop,
                use_native_rollout_status=use_native_rollout_status,
                success_protocol=success_protocol,
            )
            rows.append(result.row)
            warnings.extend(result.warnings)
        except Exception as exc:  # noqa: BLE001 - failure is recorded per case.
            failures.append({"case_id": path.stem, "path": str(path), "reason": repr(exc)})

    table_eligible = bool(rows) and not failures
    if not rows:
        warnings.append("final-table gate failed: no valid cases were evaluated")
    summary = _summary_from_rows(rows, table_eligible=table_eligible, warnings=warnings)
    input_provenance = _write_input_manifests(out_dir, rows)
    summary.update(
        {
            "cases_dir": str(cases_dir),
            "num_failed_cases": len(failures),
            "input_provenance": input_provenance,
            "reference_manifest": input_provenance["reference"]["manifest"],
            "reference_manifest_sha256": input_provenance["reference"][
                "manifest_sha256"
            ],
            "execution_manifest": input_provenance["execution"]["manifest"],
            "execution_manifest_sha256": input_provenance["execution"][
                "manifest_sha256"
            ],
            "rollout_status_policy": (
                "native_simulator" if use_native_rollout_status else "exported_trajectory"
            ),
        }
    )
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "case_metrics.jsonl", rows)
    _write_jsonl(out_dir / "failed_cases.jsonl", failures)
    _write_json(
        out_dir / "manifest.json",
        {
            "cases_dir": str(cases_dir),
            "rows": len(rows),
            "failed": len(failures),
            "input_provenance": input_provenance,
        },
    )
    return summary


def _evaluate_canonical_dirs(
    reference_dir: Path,
    execution_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    *,
    method: str,
    split: str,
    fps: float,
    workers: int,
    reference_qpos_dir: Path | None = None,
    execution_qpos_dir: Path | None = None,
    rollout_summary_path: Path | None = None,
    use_native_rollout_status: bool = False,
    success_protocol: str = DEFAULT_SUCCESS_PROTOCOL,
) -> dict[str, Any]:
    _assert_allowed_paper_provenance(
        {
            "split": split,
            "reference_dir": str(reference_dir),
            "execution_dir": str(execution_dir),
            "reference_qpos_dir": str(reference_qpos_dir),
            "execution_qpos_dir": str(execution_qpos_dir),
            "manifest_path": str(manifest_path),
            "rollout_summary_path": str(rollout_summary_path),
        },
        context=f"{method}/{split}",
    )
    case_ids = _load_case_ids(manifest_path)

    rollout_metadata = (
        _load_rollout_metadata(rollout_summary_path) if rollout_summary_path is not None else None
    )
    if rollout_metadata is not None:
        missing_metadata = [case_id for case_id in case_ids if case_id not in rollout_metadata]
        if missing_metadata:
            preview = ", ".join(missing_metadata[:5])
            raise ValueError(
                f"rollout summary is missing {len(missing_metadata)} manifest cases "
                f"(first: {preview}): {rollout_summary_path}"
            )

    def evaluate_case_id(case_id: str) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
        ref_path = reference_dir / f"{case_id}.npz"
        exe_path = execution_dir / f"{case_id}.npz"
        ref_qpos_path = (
            reference_qpos_dir / f"{case_id}.npz" if reference_qpos_dir is not None else None
        )
        exe_qpos_path = (
            execution_qpos_dir / f"{case_id}.npz" if execution_qpos_dir is not None else None
        )
        try:
            if not ref_path.is_file():
                raise FileNotFoundError(f"missing reference: {ref_path}")
            if not exe_path.is_file():
                raise FileNotFoundError(f"missing execution: {exe_path}")
            if ref_qpos_path is not None and not ref_qpos_path.is_file():
                raise FileNotFoundError(f"missing reference qpos: {ref_qpos_path}")
            if exe_qpos_path is not None and not exe_qpos_path.is_file():
                raise FileNotFoundError(f"missing execution qpos: {exe_qpos_path}")
            case = _canonical_case_payload(
                ref_path,
                exe_path,
                method=method,
                split=split,
                fps=fps,
                ref_qpos_path=ref_qpos_path,
                exe_qpos_path=exe_qpos_path,
            )
            if rollout_metadata is not None:
                case.update(rollout_metadata[case_id])
            result = _evaluate_payload(
                case,
                exe_path,
                use_native_rollout_status=use_native_rollout_status,
                success_protocol=success_protocol,
            )
            return result.row, result.warnings, None
        except Exception as exc:  # noqa: BLE001 - failure is recorded per case.
            return None, [], {
                "case_id": case_id,
                "reference_path": str(ref_path),
                "execution_path": str(exe_path),
                "reason": repr(exc),
            }

    worker_count = max(1, int(workers))
    if worker_count == 1:
        results = [evaluate_case_id(case_id) for case_id in case_ids]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(evaluate_case_id, case_ids))

    rows = [row for row, _, _ in results if row is not None]
    failures = [failure for _, _, failure in results if failure is not None]
    warnings = [warning for _, case_warnings, _ in results for warning in case_warnings]

    table_eligible = bool(rows) and not failures and len(rows) == len(case_ids)
    if not rows:
        warnings.append("final-table gate failed: no valid cases were evaluated")
    summary = _summary_from_rows(rows, table_eligible=table_eligible, warnings=warnings)
    input_provenance = _write_input_manifests(out_dir, rows)
    summary.update(
        {
            "reference_dir": str(reference_dir),
            "execution_dir": str(execution_dir),
            "reference_qpos_dir": str(reference_qpos_dir) if reference_qpos_dir else None,
            "execution_qpos_dir": str(execution_qpos_dir) if execution_qpos_dir else None,
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _sha256_file(manifest_path),
            "rollout_summary": str(rollout_summary_path) if rollout_summary_path else None,
            "method": method,
            "split": split,
            "num_failed_cases": len(failures),
            "input_provenance": input_provenance,
            "reference_manifest": input_provenance["reference"]["manifest"],
            "reference_manifest_sha256": input_provenance["reference"][
                "manifest_sha256"
            ],
            "execution_manifest": input_provenance["execution"]["manifest"],
            "execution_manifest_sha256": input_provenance["execution"][
                "manifest_sha256"
            ],
            "input_format": (
                "paired_canonical_g1_body30_qpos36"
                if reference_qpos_dir is not None
                else "paired_canonical_g1_body30"
            ),
            "workers": worker_count,
            "rollout_status_policy": (
                "native_simulator" if use_native_rollout_status else "exported_trajectory"
            ),
        }
    )
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "case_metrics.jsonl", rows)
    _write_jsonl(out_dir / "failed_cases.jsonl", failures)
    _write_json(
        out_dir / "manifest.json",
        {
            "reference_dir": str(reference_dir),
            "execution_dir": str(execution_dir),
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": _sha256_file(manifest_path),
            "rollout_summary": str(rollout_summary_path) if rollout_summary_path else None,
            "rows": len(rows),
            "failed": len(failures),
            "input_provenance": input_provenance,
        },
    )
    return summary


def _version_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "standard_output_root": STANDARD_ROOT,
        "summary_required_keys": list(SUMMARY_KEYS),
        "generator_summary_keys": list(GENERATOR_KEYS),
        "paper_protocol": {
            "splits": list(PAPER_SPLITS),
            "continuous_aggregation": PAPER_CONTINUOUS_AGGREGATION,
            "primary_continuous_fields": list(PAPER_PRIMARY_CONTINUOUS_KEYS),
            "eg_protocol": PAPER_EG_PROTOCOL,
            "mpjpe_protocol": PAPER_MPJPE_PROTOCOL,
            "success_only_is_diagnostic": True,
            "input_manifest_schema": INPUT_MANIFEST_SCHEMA,
            "canonical_amass_source": CANONICAL_AMASS_SOURCE,
            "forbidden_amass_source": FORBIDDEN_AMASS_SOURCE,
            "forbidden_legacy_protocol_fragments": list(
                FORBIDDEN_LEGACY_PROTOCOL_FRAGMENTS
            ),
        },
        "standard_case_npz_fields": [
            "reference_qpos",
            "execution_qpos",
            "reference_joints",
            "execution_joints",
            "fps",
            "joint_names",
            "case_id",
            "caption",
            "source_path",
            "method",
            "unexpected_fall",
        ],
        "tracker_metric_definitions": {
            "success_rate_unified": "Default cross-method protocol follows SONIC's paper comparison on synchronized 30-FPS exports: at most one missing endpoint frame, and failure only when the maximum reference-relative pelvis-height deviation exceeds 0.25 m. Method-native termination flags are diagnostic only.",
            "success_rate_sonic_paper": "SONIC paper's relaxed cross-method MuJoCo protocol: full export and maximum reference-relative pelvis-height deviation at most 0.25 m.",
            "success_rate_sonic_release_eval": "Diagnostic released-code evaluation protocol: full export, pelvis-height error at most 0.25 m, ankle/wrist-height error at most 0.25 m, and pelvis-orientation error at most 1.0 rad.",
            "success_rate_any2track": "Diagnostic Any2Track-style success: at least 99% frame coverage, mean root-relative canonical-link position error at most 0.20 m, and mean root-height error at most 0.20 m.",
            "completion": "min(num_execution_frames / num_reference_frames, 1.0).",
            "unexpected_fall_rate": "Persistent root drop or pelvis up-axis tilt relative to an upright reference; yaw error and low-floor target frames are excluded, and absolute execution height is never a failure rule.",
            "e_joint_rad": "Mean wrapped actuated-joint rotation error in radians.",
            "e_key_m": "Mean start-XY-aligned position error over SONIC's fixed official 14 tracked bodies, in meters.",
            "eg_mpjpe_mm": "SONIC fixed-14-body global position error in millimeters after removing only the execution-to-reference pelvis XY offset at frame zero. Z is never pre-aligned, so height error remains visible.",
            "er_mpjpe_mm": "Per-frame root-relative local G1 joint MPJPE in millimeters.",
            "evel_mps": "Mean start-aligned global joint velocity error in meters per second.",
            "eacc_mps2": "Mean start-aligned global joint acceleration error in meters per second squared.",
            "mpjpe_mm": "MPJPE-L over SONIC's fixed 14 bodies after subtracting each frame's pelvis translation independently; no rotational alignment is applied.",
            "mpjve_mps": "Mean root-relative Cartesian G1 joint velocity error in meters per second, computed by finite differences at the exported trajectory rate.",
            "root_vel_err_mps": "Mean Cartesian root linear-velocity error in meters per second, computed by finite differences at the exported trajectory rate.",
            "paper_continuous_aggregation": "Every primary continuous error pools all cases and all valid exported frames. The three splits are concatenated before weighting; equal-weight split macro averages are forbidden.",
            "success_only_metrics": "The *_success_only fields and success_only_diagnostic object are diagnostics only and must never populate a paper table.",
            "input_provenance": "Every paper-facing summary records verified reference_input_manifest.json and execution_input_manifest.json paths plus SHA-256 hashes. Each AMASS row must resolve through source/metadata to data/AMASS_GMR_for_G1/g1; the validated table2_tracker/unified_protocol_v1 quatfix directory is allowed.",
        },
        "generator_metric_definitions": {
            "isaaclab_succ/e_joint_rad/e_key_m": "Native SONIC timeout success plus joint-rotation and fixed-14-body position errors in IsaacLab.",
            "mujoco_succ/e_joint_rad/e_key_m": "The same frozen reference/controller protocol replayed in MuJoCo for cross-simulator validation.",
            "tmr_g1_r1/r2/r3": "Text-motion retrieval precision using repository-native TMR-G1.",
            "tmr_g1_mm_dist": "TMR-G1 multimodal distance with raw latent features.",
            "fid": "FID over unit-normalized TMR-G1 motion embeddings against the frozen held-out GT split.",
            "diversity": "Average pairwise distance in TMR-G1 raw latent motion features.",
            "prompt_coverage": "Fraction of prompt/action buckets represented by valid generations.",
            "action_entropy": "Entropy over action buckets or classifier labels on valid generations.",
        },
        "any2track_success_thresholds": ANY2TRACK_SUCCESS_THRESHOLDS,
        "sonic_paper_success_thresholds": SONIC_PAPER_SUCCESS_THRESHOLDS,
        "sonic_release_eval_success_thresholds": SONIC_RELEASE_EVAL_SUCCESS_THRESHOLDS,
        "unexpected_fall_thresholds": UNEXPECTED_FALL_THRESHOLDS,
        "canonical_g1_body_names": list(CANONICAL_G1_BODY_NAMES),
        "sonic_key_body_names": list(SONIC_KEY_BODY_NAMES),
        "dry_run_policy": {
            "legacy_summary_adapter": "Allowed only to validate schema. Outputs are marked final_table_eligible=false.",
            "final_table_policy": "Final paper values must use v0.21 per-case rows, pool all valid exported frames across all three frozen splits, and pass the legacy-input and manifest-hash provenance gates.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_version = sub.add_parser("write-version")
    p_version.add_argument("--out", type=Path, default=Path(STANDARD_ROOT) / "evaluator_version.json")

    p_eval = sub.add_parser("evaluate-dir")
    p_eval.add_argument("--cases-dir", type=Path, required=True)
    p_eval.add_argument("--out-dir", type=Path, required=True)
    p_eval.add_argument("--allow-joint-crop", action="store_true")
    p_eval.add_argument(
        "--use-native-rollout-status",
        action="store_true",
        help="Use method-native progress/termination instead of exported-frame coverage.",
    )
    p_eval.add_argument(
        "--success-protocol",
        choices=SUCCESS_PROTOCOL_CHOICES,
        default=DEFAULT_SUCCESS_PROTOCOL,
    )

    p_canonical = sub.add_parser("evaluate-canonical-dirs")
    p_canonical.add_argument("--reference-dir", type=Path, required=True)
    p_canonical.add_argument("--execution-dir", type=Path, required=True)
    p_canonical.add_argument("--reference-qpos-dir", type=Path)
    p_canonical.add_argument("--execution-qpos-dir", type=Path)
    p_canonical.add_argument(
        "--rollout-summary",
        type=Path,
        help="Optional per-case simulator progress/termination summary for padded rollout buffers.",
    )
    p_canonical.add_argument(
        "--success-protocol",
        choices=SUCCESS_PROTOCOL_CHOICES,
        default=DEFAULT_SUCCESS_PROTOCOL,
    )
    p_canonical.add_argument("--manifest", type=Path, required=True)
    p_canonical.add_argument("--out-dir", type=Path, required=True)
    p_canonical.add_argument("--method", required=True)
    p_canonical.add_argument("--split", required=True)
    p_canonical.add_argument("--fps", type=float, default=30.0)
    p_canonical.add_argument("--workers", type=int, default=1)
    p_canonical.add_argument(
        "--use-native-rollout-status",
        action="store_true",
        help="Use rollout-summary progress/termination in the success decision.",
    )

    p_legacy = sub.add_parser("adapt-legacy-summary")
    p_legacy.add_argument("--input", type=Path, required=True)
    p_legacy.add_argument("--out-dir", type=Path, required=True)
    p_legacy.add_argument("--method", required=True)
    p_legacy.add_argument("--split", required=True)

    args = parser.parse_args()
    if args.cmd == "write-version":
        payload = _version_payload()
        _write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.cmd == "evaluate-dir":
        summary = _evaluate_dir(
            args.cases_dir,
            args.out_dir,
            args.allow_joint_crop,
            use_native_rollout_status=args.use_native_rollout_status,
            success_protocol=args.success_protocol,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    elif args.cmd == "evaluate-canonical-dirs":
        if (args.reference_qpos_dir is None) != (args.execution_qpos_dir is None):
            parser.error("--reference-qpos-dir and --execution-qpos-dir must be provided together")
        summary = _evaluate_canonical_dirs(
            args.reference_dir,
            args.execution_dir,
            args.manifest,
            args.out_dir,
            method=args.method,
            split=args.split,
            fps=args.fps,
            workers=args.workers,
            reference_qpos_dir=args.reference_qpos_dir,
            execution_qpos_dir=args.execution_qpos_dir,
            rollout_summary_path=args.rollout_summary,
            use_native_rollout_status=args.use_native_rollout_status,
            success_protocol=args.success_protocol,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    elif args.cmd == "adapt-legacy-summary":
        summary = _adapt_legacy_summary(args.input, args.out_dir, args.method, args.split)
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
