#!/usr/bin/env python3
"""Materialize standard tracker case NPZ files from canonical rollout folders.

The canonical rollout layout stores reference and method rollouts separately:

  table_tracker/reference/<split>/g1_body30/<case>.npz
  table_tracker/<method>/<split>/g1_body30/<case>.npz

The paper-facing unified evaluator expects one NPZ per case containing both
reference and execution arrays.  This script is the lossless bridge between
those two contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_TO_METADATA = {
    "lafan1_g1": "lafan1_g1_test.jsonl",
    "amass_test_g1": "amass_test_g1.jsonl",
    "wild_g1_clean": "wild_g1_clean.jsonl",
}

CANONICAL_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)


def _safe_stem(name: str) -> str:
    return "__".join(Path(name).with_suffix("").parts).replace(" ", "_")


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
    elif arr.size == 1:
        item = arr.reshape(-1)[0].item()
    else:
        item = arr.tolist()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _metadata_from_array(value: Any) -> dict[str, Any]:
    try:
        return json.loads(_scalar_text(value))
    except Exception:
        return {}


def _select_source_path(
    split_meta: dict[str, Any],
    ref_meta: dict[str, Any],
    exe_meta: dict[str, Any],
) -> str:
    """Prefer immutable dataset provenance over temporary protocol artifacts."""

    return str(
        split_meta.get("source_path")
        or ref_meta.get("canonical_source_path")
        or exe_meta.get("canonical_source_path")
        or ref_meta.get("source")
        or exe_meta.get("source")
        or ""
    )


def _load_metric_rows(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if run_dir is None or not run_dir.is_dir():
        return rows
    for path in sorted(run_dir.glob("eval_shard_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in data.get("motions", []):
            motion = row.get("motion")
            if isinstance(motion, str):
                rows[motion] = row
    return rows


def _load_metric_summary(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    rows: dict[str, dict[str, Any]] = {}
    for row in data.get("cases", []):
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id:
            rows[case_id] = row
    return rows


def _load_protocol_case_ids(protocol_root: Path | None, protocol_split: str) -> list[str] | None:
    if protocol_root is None:
        return None
    path = protocol_root / "inputs" / protocol_split / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing protocol manifest: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"protocol manifest must be a list of case ids: {path}")
    if len(data) != len(set(data)):
        raise ValueError(f"protocol manifest contains duplicate case ids: {path}")
    return data


def _load_manifest_rows(root: Path, split: str) -> dict[str, dict[str, Any]]:
    name = SPLIT_TO_METADATA.get(split)
    if not name:
        return {}
    path = root / "splits" / name
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id") or "")
        if case_id:
            rows[case_id] = row
        source_path = row.get("source_path")
        if isinstance(source_path, str) and source_path:
            rows[_safe_stem(source_path)] = row
    return rows


def _qpos_payload(path: Path | None, key: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    data = _load_npz(path)
    if "qpos" not in data:
        return {}
    return {key: np.asarray(data["qpos"], dtype=np.float32)}


def _body_names(data: dict[str, Any]) -> list[str]:
    names = data.get("body_names")
    if names is None:
        names = data.get("joint_names")
    if names is None:
        raise ValueError("missing body_names/joint_names")
    return [str(x) for x in np.asarray(names).tolist()]


def _align_body_arrays(
    ref: dict[str, Any],
    exe: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    ref_names = _body_names(ref)
    exe_names = _body_names(exe)
    ref_index = {name: idx for idx, name in enumerate(ref_names)}
    exe_index = {name: idx for idx, name in enumerate(exe_names)}
    missing_ref = [name for name in CANONICAL_G1_BODY_NAMES if name not in ref_index]
    missing_exe = [name for name in CANONICAL_G1_BODY_NAMES if name not in exe_index]
    if missing_ref or missing_exe:
        raise ValueError(
            f"canonical G1 body set incomplete: missing reference={missing_ref}, execution={missing_exe}"
        )
    common = list(CANONICAL_G1_BODY_NAMES)
    ref_idx = [ref_index[name] for name in common]
    exe_idx = [exe_index[name] for name in common]
    ref_pos = np.asarray(ref["body_pos"], dtype=np.float32)[:, ref_idx]
    exe_pos = np.asarray(exe["body_pos"], dtype=np.float32)[:, exe_idx]
    ref_quat = np.asarray(ref["body_quat"], dtype=np.float32)[:, ref_idx]
    exe_quat = np.asarray(exe["body_quat"], dtype=np.float32)[:, exe_idx]
    report = {
        "reference_body_count": len(ref_names),
        "execution_body_count": len(exe_names),
        "aligned_body_count": len(common),
        "dropped_reference_bodies": [name for name in ref_names if name not in CANONICAL_G1_BODY_NAMES],
        "dropped_execution_bodies": [name for name in exe_names if name not in CANONICAL_G1_BODY_NAMES],
        "body_name_aligned": ref_names != exe_names,
        "canonical_body_set": True,
    }
    return ref_pos, exe_pos, ref_quat, exe_quat, common, report


def _materialize_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    method = args.method
    task_root = args.root / args.task
    ref_body_dir = task_root / "reference" / split / "g1_body30"
    exe_body_dir = task_root / method / split / "g1_body30"
    ref_qpos_dir = task_root / "reference" / split / "g1_qpos30"
    exe_qpos_dir = task_root / method / split / "g1_qpos30"
    out_dir = task_root / method / split / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_split = args.protocol_split or args.protocol_split_map.get(split, split)
    metric_rows = _load_metric_rows(args.protocol_root / "runs" / method / protocol_split if args.protocol_root else None)
    metric_rows.update(_load_metric_summary(args.metric_summary))
    protocol_case_ids = _load_protocol_case_ids(args.protocol_root, protocol_split)
    manifest_rows = _load_manifest_rows(args.root, split)

    rows = []
    failures = []
    if protocol_case_ids is None:
        execution_paths = sorted(exe_body_dir.glob("*.npz"))
    else:
        execution_paths = [exe_body_dir / f"{case_id}.npz" for case_id in protocol_case_ids]
    for exe_path in execution_paths:
        case_key = exe_path.stem
        if not exe_path.is_file():
            failures.append({"case_id": case_key, "reason": f"missing execution {exe_path}"})
            continue
        ref_path = ref_body_dir / exe_path.name
        if not ref_path.is_file():
            failures.append({"case_id": case_key, "reason": f"missing reference {ref_path}"})
            continue
        try:
            ref = _load_npz(ref_path)
            exe = _load_npz(exe_path)
            ref_pos, exe_pos, ref_quat, exe_quat, body_names, alignment_report = _align_body_arrays(ref, exe)
            ref_meta = _metadata_from_array(ref.get("metadata", np.array("{}")))
            exe_meta = _metadata_from_array(exe.get("metadata", np.array("{}")))
            split_meta = manifest_rows.get(case_key, {})
            metric = metric_rows.get(case_key, {})
            source_path = _select_source_path(split_meta, ref_meta, exe_meta)
            caption = split_meta.get("caption") or ""
            payload: dict[str, Any] = {
                "reference_body_pos": ref_pos,
                "execution_body_pos": exe_pos,
                "reference_body_quat": ref_quat,
                "execution_body_quat": exe_quat,
                "reference_joints": ref_pos,
                "execution_joints": exe_pos,
                "fps": np.float32(args.fps),
                "frequency": np.float32(args.fps),
                "joint_names": np.array(body_names, dtype=str),
                "body_names": np.array(body_names, dtype=str),
                "case_id": np.array(case_key),
                "method": np.array(method),
                "split": np.array(split),
                "caption": np.array(str(caption)),
                "source_path": np.array(str(source_path)),
                "metadata": np.array(json.dumps({
                    "reference_metadata": ref_meta,
                    "execution_metadata": exe_meta,
                    "split_metadata": split_meta,
                    "metric_row": metric,
                    "alignment_report": alignment_report,
                }, sort_keys=True)),
            }
            payload.update(_qpos_payload(ref_qpos_dir / exe_path.name, "reference_qpos"))
            payload.update(_qpos_payload(exe_qpos_dir / exe_path.name, "execution_qpos"))
            if "max_joint_err_max" in metric:
                payload["max_joint_err_rad"] = np.float32(metric["max_joint_err_max"])
            if "strict_success" in metric:
                payload["method_strict_success"] = np.array(bool(metric["strict_success"]))
            if "paper_success" in metric:
                payload["method_paper_success"] = np.array(bool(metric["paper_success"]))
            if "progress" in metric:
                payload["native_completion"] = np.float32(metric["progress"])
                if args.use_native_rollout_status:
                    payload["completion"] = np.float32(metric["progress"])
                    payload["execution_progress"] = np.float32(metric["progress"])
            if "terminated" in metric:
                payload["native_terminated"] = np.array(bool(metric["terminated"]))
                if args.use_native_rollout_status:
                    payload["terminated"] = np.array(bool(metric["terminated"]))
            for fall_key in ("unexpected_fall", "unexpected_fall_detected"):
                if fall_key in metric:
                    payload["unexpected_fall"] = np.array(bool(metric[fall_key]))
                    break
            if "success" in metric:
                payload["method_reported_success"] = np.array(bool(metric["success"]))
            out_path = out_dir / exe_path.name
            tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, **payload)
            tmp.replace(out_path)
            rows.append({"case_id": case_key, "path": str(out_path), "source_path": str(source_path)})
        except Exception as exc:  # noqa: BLE001 - recorded per case for audit.
            failures.append({"case_id": case_key, "path": str(exe_path), "reason": repr(exc)})

    manifest = {
        "method": method,
        "split": split,
        "protocol_split": protocol_split,
        "protocol_case_count": None if protocol_case_ids is None else len(protocol_case_ids),
        "rows": rows,
        "failed": failures,
        "rollout_status_policy": (
            "native_simulator" if args.use_native_rollout_status else "exported_trajectory"
        ),
    }
    if method == "sonic":
        manifest["provenance"] = {
            "official_runner": "user-supplied SONIC evaluator",
            "sonic_evaluator": "official_isaaclab_sonic",
            "source_note": "official_sonic_isaaclab_execution",
            "materialization": "body_names_aligned_canonical_cases",
        }
    (out_dir.parent / "cases_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"split": split, "num_cases": len(rows), "num_failed": len(failures), "cases_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/evaluation/gentrack"),
    )
    ap.add_argument("--task", default="table_tracker")
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--protocol-root", type=Path, default=None)
    ap.add_argument("--protocol-split", default=None)
    ap.add_argument("--metric-summary", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--use-native-rollout-status",
        action="store_true",
        help="Promote native progress/termination into active evaluator fields.",
    )
    args = ap.parse_args()
    args.protocol_split_map = {
        "lafan1_g1": "lafan1_fixed600",
        "amass_test_g1": "amass_test_fixed600",
        "wild_g1_clean": "wild_clean_fixed600",
    }
    results = [_materialize_split(args, split) for split in args.split]
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
