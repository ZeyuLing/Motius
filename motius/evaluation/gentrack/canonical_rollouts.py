#!/usr/bin/env python3
"""Canonical PhysFlow Table-2 rollout storage helpers.

The storage contract mirrors the T2M evaluation layout:

    outputs/evaluation/gentrack/table_tracker/<method>/<test_dataset>/<motion_representation>/<case>.npz

where the method is the stable artifact entry point, and the representation
directory names the arrays directly stored in that directory.  For Table-2
tracking we currently materialize:

* g1_qpos30: qpos [T, 36], root xyz + quat WXYZ + 29 G1 DOFs, 30 FPS.
* g1_body30: body_pos/body_quat [T, B, ...], 30 FPS.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SPLIT_ALIASES = {
    "lafan1_fixed600": "lafan1_g1",
    "amass_test_fixed600": "amass_test_g1",
    "wild_clean_fixed600": "wild_g1_clean",
    "wild_g1_clean_1024": "wild_g1_clean",
}


def canonical_split_name(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def _safe_method(method: str) -> str:
    return method.strip().lower().replace("-", "_")


def canonical_task_root(root: Path) -> Path:
    root = Path(root)
    # ``table2_tracker`` is the legacy task name.  New GenTrack AAAI artifacts
    # use ``table_tracker`` to match the paper TODO/result-root contract.
    if root.name in {"table_tracker", "table2_tracker"}:
        return root
    return root / "table_tracker"


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)


def resample_qpos_wxyz(qpos: np.ndarray, source_fps: float, target_fps: float = 30.0) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D, got {qpos.shape}")
    if qpos.shape[0] < 2 or abs(float(source_fps) - float(target_fps)) < 1e-6:
        out = qpos.astype(np.float32, copy=True)
        if out.shape[1] >= 7:
            out[:, 3:7] = _normalize_quat(out[:, 3:7])
        return out

    n = qpos.shape[0]
    duration = (n - 1) / float(source_fps)
    src_t = np.arange(n, dtype=np.float64) / float(source_fps)
    out_n = int(round(duration * float(target_fps))) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / float(target_fps)
    dst_t[-1] = min(dst_t[-1], src_t[-1])

    out = np.empty((out_n, qpos.shape[1]), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    src_xyzw = _normalize_quat(qpos[:, 3:7])[:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out[:, 3:7] = out_xyzw[:, [3, 0, 1, 2]]
    for i in range(7, qpos.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, qpos[:, i])
    return out.astype(np.float32)


def resample_body_frames(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    source_fps: float,
    target_fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    body_pos = np.asarray(body_pos, dtype=np.float32)
    body_quat = np.asarray(body_quat, dtype=np.float32)
    if body_pos.ndim != 3 or body_quat.ndim != 3:
        raise ValueError(f"body arrays must be 3D, got {body_pos.shape}, {body_quat.shape}")
    if body_pos.shape[:2] != body_quat.shape[:2]:
        raise ValueError(f"body_pos/body_quat shape mismatch: {body_pos.shape}, {body_quat.shape}")
    if body_pos.shape[0] < 2 or abs(float(source_fps) - float(target_fps)) < 1e-6:
        return body_pos.astype(np.float32, copy=True), _normalize_quat(body_quat).astype(np.float32)

    n, num_bodies, _ = body_pos.shape
    duration = (n - 1) / float(source_fps)
    src_t = np.arange(n, dtype=np.float64) / float(source_fps)
    out_n = int(round(duration * float(target_fps))) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / float(target_fps)
    dst_t[-1] = min(dst_t[-1], src_t[-1])

    out_pos = np.empty((out_n, num_bodies, 3), dtype=np.float32)
    for b in range(num_bodies):
        for c in range(3):
            out_pos[:, b, c] = np.interp(dst_t, src_t, body_pos[:, b, c])
    out_quat = np.empty((out_n, num_bodies, 4), dtype=np.float32)
    for b in range(num_bodies):
        src_xyzw = _normalize_quat(body_quat[:, b])[:, [1, 2, 3, 0]]
        out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
        out_quat[:, b] = out_xyzw[:, [3, 0, 1, 2]]
    return out_pos, out_quat


def qpos_to_body_arrays(model: Any, qpos: np.ndarray, body_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import mujoco

    qpos = np.asarray(qpos, dtype=np.float32)
    data = mujoco.MjData(model)
    if body_ids is None:
        body_ids = np.array([i for i in range(1, model.nbody)], dtype=np.int32)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) or f"body_{int(i)}"
        for i in body_ids
    ]
    body_pos = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float32)
    body_quat = np.zeros((qpos.shape[0], len(body_ids), 4), dtype=np.float32)
    for i, q in enumerate(qpos):
        data.qpos[: min(model.nq, q.shape[0])] = q[: min(model.nq, q.shape[0])]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[body_ids]
        body_quat[i] = data.xquat[body_ids]
    return body_pos, body_quat, body_names


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def qpos_path(root: Path, split: str, method: str, case_id: str) -> Path:
    return canonical_task_root(root) / _safe_method(method) / canonical_split_name(split) / "g1_qpos30" / f"{case_id}.npz"


def body_path(root: Path, split: str, method: str, case_id: str) -> Path:
    return canonical_task_root(root) / _safe_method(method) / canonical_split_name(split) / "g1_body30" / f"{case_id}.npz"


def metric_body_path(root: Path, split: str, method: str, case_id: str) -> Path:
    return (
        canonical_task_root(root)
        / _safe_method(method)
        / canonical_split_name(split)
        / "g1_metric14_30"
        / f"{case_id}.npz"
    )


def save_qpos(
    root: Path,
    split: str,
    method: str,
    case_id: str,
    qpos: np.ndarray,
    *,
    source_fps: float,
    target_fps: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    qpos30 = resample_qpos_wxyz(qpos, source_fps, target_fps)
    path = qpos_path(root, split, method, case_id)
    _atomic_savez(
        path,
        qpos=qpos30.astype(np.float32),
        frequency=np.float32(target_fps),
        fps=np.float32(target_fps),
        source_fps=np.float32(source_fps),
        case_id=np.array(case_id),
        method=np.array(_safe_method(method)),
        split=np.array(canonical_split_name(split)),
        metadata=np.array(json.dumps(metadata or {}, sort_keys=True)),
    )
    return path


def save_body(
    root: Path,
    split: str,
    method: str,
    case_id: str,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: list[str],
    *,
    source_fps: float,
    target_fps: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    pos30, quat30 = resample_body_frames(body_pos, body_quat, source_fps, target_fps)
    path = body_path(root, split, method, case_id)
    _atomic_savez(
        path,
        body_pos=pos30.astype(np.float32),
        body_quat=quat30.astype(np.float32),
        fps=np.float32(target_fps),
        frequency=np.float32(target_fps),
        source_fps=np.float32(source_fps),
        body_names=np.array(body_names, dtype=str),
        case_id=np.array(case_id),
        method=np.array(_safe_method(method)),
        split=np.array(canonical_split_name(split)),
        metadata=np.array(json.dumps(metadata or {}, sort_keys=True)),
    )
    return path


def save_metric_body(
    root: Path,
    split: str,
    method: str,
    case_id: str,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: list[str],
    *,
    source_fps: float,
    target_fps: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save paper-metric body tensors separately from mesh/FK artifacts."""
    pos30, quat30 = resample_body_frames(body_pos, body_quat, source_fps, target_fps)
    path = metric_body_path(root, split, method, case_id)
    _atomic_savez(
        path,
        body_pos=pos30.astype(np.float32),
        body_quat=quat30.astype(np.float32),
        fps=np.float32(target_fps),
        frequency=np.float32(target_fps),
        source_fps=np.float32(source_fps),
        body_names=np.array(body_names, dtype=str),
        case_id=np.array(case_id),
        method=np.array(_safe_method(method)),
        split=np.array(canonical_split_name(split)),
        metadata=np.array(json.dumps(metadata or {}, sort_keys=True)),
    )
    return path


def write_run_config(root: Path, split: str, representation: str, method: str, payload: dict[str, Any]) -> Path:
    path = canonical_task_root(root) / _safe_method(method) / canonical_split_name(split) / representation / "run_config.json"
    _atomic_json(path, payload)
    return path


def write_reference_from_qpos(
    root: Path,
    split: str,
    case_id: str,
    qpos: np.ndarray,
    *,
    source_fps: float,
    model: Any | None = None,
    target_fps: float = 30.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    frozen_marker = canonical_task_root(root) / "reference" / ".frozen_reference.json"
    existing_qpath = qpos_path(root, split, "reference", case_id)
    existing_bpath = body_path(root, split, "reference", case_id)
    if frozen_marker.is_file():
        required = [existing_qpath]
        if model is not None:
            required.append(existing_bpath)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "canonical reference is frozen but required cases are missing: "
                + ", ".join(missing)
            )
        out = {"qpos": str(existing_qpath)}
        if model is not None:
            out["body"] = str(existing_bpath)
        return out

    qpos30 = resample_qpos_wxyz(qpos, source_fps, target_fps)
    qpath = save_qpos(
        root,
        split,
        "reference",
        case_id,
        qpos30,
        source_fps=target_fps,
        target_fps=target_fps,
        metadata=metadata,
    )
    out = {"qpos": str(qpath)}
    if model is not None:
        body_pos, body_quat, body_names = qpos_to_body_arrays(model, qpos30)
        bpath = save_body(
            root,
            split,
            "reference",
            case_id,
            body_pos,
            body_quat,
            body_names,
            source_fps=target_fps,
            target_fps=target_fps,
            metadata=metadata,
        )
        out["body"] = str(bpath)
    return out
