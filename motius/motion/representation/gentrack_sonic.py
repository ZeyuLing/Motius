#!/usr/bin/env python3
"""Convert a unified G1 qpos NPZ into SONIC's official robot-motion pkl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


DOF_AXIS_INDEX = np.array(
    [
        1,
        0,
        2,
        1,
        1,
        0,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        0,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
    ],
    dtype=np.int64,
)

TARGET_G1_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-6 or qpos.shape[0] <= 1:
        return qpos.astype(np.float64, copy=False)

    src_t = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    duration = src_t[-1]
    dst_n = int(round(duration * target_fps)) + 1
    dst_t = np.arange(dst_n, dtype=np.float64) / target_fps
    dst_t[-1] = min(dst_t[-1], duration)

    out = np.empty((dst_n, qpos.shape[1]), dtype=np.float64)
    out[:, :3] = np.stack([np.interp(dst_t, src_t, qpos[:, i]) for i in range(3)], axis=-1)
    src_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    out[:, 3:7] = Slerp(src_t, R.from_quat(src_xyzw))(dst_t).as_quat()[:, [3, 0, 1, 2]]
    out[:, 7:] = np.stack(
        [np.interp(dst_t, src_t, qpos[:, i]) for i in range(7, qpos.shape[1])],
        axis=-1,
    )
    return out


def _load_qpos(npz_path: Path, target_fps: float) -> tuple[np.ndarray, float]:
    pack = np.load(npz_path, allow_pickle=True)
    if "qpos" in pack.files:
        qpos = np.asarray(pack["qpos"], dtype=np.float64)
    else:
        required = {
            "body_positions",
            "body_rotations",
            "body_names",
            "dof_positions",
            "dof_names",
        }
        if not required.issubset(pack.files):
            raise ValueError(f"{npz_path}: unsupported NPZ fields {pack.files}")
        body_names = [str(x) for x in np.asarray(pack["body_names"]).tolist()]
        dof_names = [str(x) for x in np.asarray(pack["dof_names"]).tolist()]
        root_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
        root_pos = np.asarray(pack["body_positions"][:, root_idx], dtype=np.float64)
        # Canonical AMASS-GMR body rotations are stored as xyzw.  Unified G1
        # qpos and MuJoCo free joints use wxyz, so convert explicitly before
        # concatenating the root state.  Copying xyzw into the wxyz slot
        # corrupts every reference orientation consumed by SONIC.
        root_xyzw = np.asarray(pack["body_rotations"][:, root_idx], dtype=np.float64)
        root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
        source_dof = np.asarray(pack["dof_positions"], dtype=np.float64)
        missing = [name for name in TARGET_G1_DOF_NAMES if name not in dof_names]
        if missing:
            raise ValueError(f"{npz_path}: missing G1 DOFs {missing}")
        dof = source_dof[:, [dof_names.index(name) for name in TARGET_G1_DOF_NAMES]]
        qpos = np.concatenate([root_pos, root_wxyz, dof], axis=1)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{npz_path}: expected qpos shape (T, 36), got {qpos.shape}")
    if "frequency" in pack.files:
        source_fps = float(np.asarray(pack["frequency"]).reshape(-1)[0])
    elif "fps" in pack.files:
        source_fps = float(np.asarray(pack["fps"]).reshape(-1)[0])
    else:
        source_fps = 30.0
    return _resample_qpos(qpos, source_fps, target_fps), source_fps


def qpos_to_sonic_motion(qpos: np.ndarray, fps: int) -> dict[str, np.ndarray | int]:
    """Build the motion dict consumed by SONIC's released IsaacLab evaluator.

    Unified G1 qpos uses MuJoCo order: root xyz, root quat wxyz, then 29 dofs.
    SONIC's official robot-motion pkl stores root quat as xyzw plus a 30-body
    axis-angle pose tensor where body 0 is the root and bodies 1: map each
    1-DoF joint onto the axis used by the release sample files.
    """

    root_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    dof = qpos[:, 7:]
    if dof.shape[1] != len(DOF_AXIS_INDEX):
        raise ValueError(f"expected 29 dofs, got {dof.shape[1]}")

    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    pose_aa[:, 0, :] = R.from_quat(root_xyzw).as_rotvec().astype(np.float32)
    for i, axis_idx in enumerate(DOF_AXIS_INDEX):
        pose_aa[:, i + 1, axis_idx] = dof[:, i].astype(np.float32)

    return {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof.astype(np.float32),
        "root_rot": root_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
        "fps": int(fps),
    }


def _convert_one(
    *,
    npz_path: Path,
    out_dir: Path,
    name: str,
    target_fps: float,
    force: bool,
) -> Path:
    out_path = out_dir / f"{name}.pkl"
    if out_path.exists() and not force:
        print(out_path)
        return out_path

    qpos, source_fps = _load_qpos(npz_path, target_fps)
    motion = qpos_to_sonic_motion(qpos, int(round(target_fps)))
    payload = {name: motion}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        "{\n"
        f'  "name": "{name}",\n'
        f'  "source_npz": "{npz_path}",\n'
        f'  "source_fps": {source_fps:.6f},\n'
        f'  "target_fps": {target_fps:.6f},\n'
        f'  "num_frames": {qpos.shape[0]}\n'
        "}\n"
    )
    print(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--npz-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if abs(args.target_fps - round(args.target_fps)) > 1e-6:
        raise ValueError("SONIC official motion pkl expects integer fps")

    if args.npz is not None:
        _convert_one(
            npz_path=args.npz,
            out_dir=args.out_dir,
            name=args.name or args.npz.stem,
            target_fps=args.target_fps,
            force=args.force,
        )
        return

    if args.name is not None:
        raise ValueError("--name is only valid with --npz")
    if args.npz_dir is None:
        raise ValueError("--npz-dir is required with --manifest")
    names = json.loads(args.manifest.read_text())
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{args.manifest}: expected a JSON list of motion names")
    for name in names:
        _convert_one(
            npz_path=args.npz_dir / f"{name}.npz",
            out_dir=args.out_dir,
            name=name,
            target_fps=args.target_fps,
            force=args.force,
        )


if __name__ == "__main__":
    main()
