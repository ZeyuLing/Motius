# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
"""Convert G1 qpos/qvel NPZ files into ProtoMotions .motion files.

This handles LocoMuJoCo/OpenTrack LAFAN1-G1 files, which store a 23-DoF
Unitree-G1 trajectory as qpos/qvel plus joint_names.  ProtoMotions' G1 tracker
uses the 29-DoF G1 layout, so missing waist/wrist joints are filled with the
default zero pose and all joints are reordered by name before FK is recomputed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from contact_detection import compute_contact_labels_from_pos_and_vel

app = typer.Typer()

CANONICAL_G1_DOF_NAMES = [
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


def _scalar(value, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got {np.asarray(value).shape}")
    return float(arr[0])


def _qpos_qvel_slices(joint_names: list[str], jnt_type: np.ndarray) -> tuple[dict, dict]:
    qpos_slices = {}
    qvel_slices = {}
    qpos_i = 0
    qvel_i = 0
    for name, typ in zip(joint_names, jnt_type):
        typ = int(typ)
        if typ == 0:  # mjJNT_FREE
            qpos_slices[name] = slice(qpos_i, qpos_i + 7)
            qvel_slices[name] = slice(qvel_i, qvel_i + 6)
            qpos_i += 7
            qvel_i += 6
        else:
            qpos_slices[name] = slice(qpos_i, qpos_i + 1)
            qvel_slices[name] = slice(qvel_i, qvel_i + 1)
            qpos_i += 1
            qvel_i += 1
    return qpos_slices, qvel_slices


def _resample(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(source_fps - output_fps) < 1e-6:
        return root_pos, root_quat_wxyz, dof_pos

    n = root_pos.shape[0]
    duration = (n - 1) / source_fps
    src_t = np.arange(n, dtype=np.float64) / source_fps
    out_n = int(round(duration * output_fps)) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / output_fps
    dst_t[-1] = min(dst_t[-1], src_t[-1])

    out_root = np.stack(
        [np.interp(dst_t, src_t, root_pos[:, i]) for i in range(3)], axis=-1
    )
    src_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out_quat = out_xyzw[:, [3, 0, 1, 2]]
    out_dof = np.stack(
        [np.interp(dst_t, src_t, dof_pos[:, i]) for i in range(dof_pos.shape[1])],
        axis=-1,
    )
    return (
        out_root.astype(np.float32),
        out_quat.astype(np.float32),
        out_dof.astype(np.float32),
    )


def _load_qpos_npz(
    npz_path: Path,
    output_fps: int,
    expected_dof_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    qpos = data["qpos"].astype(np.float32)
    source_fps = _scalar(data["frequency"] if "frequency" in data.files else None, output_fps)

    if "joint_names" in data.files:
        source_joint_names = [str(x) for x in data["joint_names"].tolist()]
    elif qpos.shape[1] == 7 + len(CANONICAL_G1_DOF_NAMES):
        source_joint_names = ["root"] + CANONICAL_G1_DOF_NAMES
    else:
        raise ValueError(
            f"{npz_path}: missing joint_names for non-canonical qpos shape {qpos.shape}"
        )
    if "jnt_type" in data.files:
        source_jnt_type = np.asarray(data["jnt_type"])
    else:
        source_jnt_type = np.array([0] + [3] * (len(source_joint_names) - 1), dtype=np.int32)
    source_qpos_slices, _ = _qpos_qvel_slices(source_joint_names, source_jnt_type)

    if "root" not in source_qpos_slices:
        raise ValueError(f"{npz_path}: joint_names does not include root")

    root_pos = qpos[:, source_qpos_slices["root"]][:, :3]
    root_rot_wxyz = qpos[:, source_qpos_slices["root"]][:, 3:7]
    dof_pos = np.zeros((qpos.shape[0], len(expected_dof_names)), dtype=np.float32)
    for dst_i, name in enumerate(expected_dof_names):
        if name in source_qpos_slices:
            dof_pos[:, dst_i] = qpos[:, source_qpos_slices[name]].reshape(-1)

    root_pos, root_rot_wxyz, dof_pos = _resample(
        root_pos=root_pos,
        root_quat_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
        source_fps=source_fps,
        output_fps=float(output_fps),
    )
    return (
        torch.from_numpy(root_pos).to(device=device, dtype=dtype),
        torch.from_numpy(root_rot_wxyz).to(device=device, dtype=dtype),
        torch.from_numpy(dof_pos).to(device=device, dtype=dtype),
    )


def _load_manifest_filter(manifest: Optional[Path], input_dir: Path) -> Optional[set[str]]:
    if manifest is None:
        return None
    items = json.loads(manifest.read_text())
    if not isinstance(items, list):
        raise ValueError(f"{manifest}: expected a JSON list")
    wanted: set[str] = set()
    for item in items:
        text = str(item)
        path = Path(text)
        wanted.add(text)
        wanted.add(path.stem)
        wanted.add(str(path.with_suffix("")))
        if not path.is_absolute():
            wanted.add(str((input_dir / path).relative_to(input_dir).with_suffix("")))
    return wanted


def _matches_manifest(npz_path: Path, input_dir: Path, wanted: set[str]) -> bool:
    rel_path = npz_path.relative_to(input_dir)
    return (
        npz_path.stem in wanted
        or str(rel_path) in wanted
        or str(rel_path.with_suffix("")) in wanted
    )


@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Root directory containing recursive G1 qpos NPZ files."),
    output_dir: Path = typer.Option(..., help="Directory to save .motion files."),
    output_fps: int = typer.Option(50, help="Output motion fps."),
    robot_type: str = typer.Option("g1", help="Robot type, default Unitree G1."),
    force_remake: bool = False,
    max_files: Optional[int] = typer.Option(None, help="Optional max files for smoke runs."),
    num_rank: int = typer.Option(1, help="Total deterministic shards."),
    slurm_rank: int = typer.Option(0, help="Shard rank in [0, num_rank)."),
    manifest: Optional[Path] = typer.Option(
        None,
        help="Optional JSON list of motion stems or relative NPZ paths for this shard. When set, rank sharding is ignored.",
    ),
):
    if num_rank <= 0 or not (0 <= slurm_rank < num_rank):
        raise ValueError("--num-rank must be positive and --slurm-rank in range")

    device = torch.device("cpu")
    dtype = torch.float32
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_mjcf_mapping = {"g1": "g1_bm_box_feet.xml", "h1_2": "h1_2.xml"}
    mjcf_filename = robot_mjcf_mapping.get(robot_type, f"{robot_type}.xml")
    mjcf_path = f"protomotions/data/assets/mjcf/{mjcf_filename}"
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")

    kinematic_info = extract_kinematic_info(mjcf_path)
    expected_dof_names = list(kinematic_info.dof_names)
    print(f"Robot type: {robot_type}, expected_dofs={len(expected_dof_names)}, output_fps={output_fps}")

    files = sorted(Path(p) for p in glob.glob(str(input_dir / "**" / "*.npz"), recursive=True))
    manifest_filter = _load_manifest_filter(manifest, input_dir)
    if manifest_filter is not None:
        files = [p for p in files if _matches_manifest(p, input_dir, manifest_filter)]
        missing = sorted(
            name
            for name in manifest_filter
            if not any(_matches_manifest(p, input_dir, {name}) for p in files)
        )
        if missing:
            print(f"WARNING: manifest entries without matching NPZ files: {missing[:10]}")
    if max_files is not None:
        files = files[:max_files]
    print(f"Found {len(files)} NPZ files.")

    converted = 0
    skipped = 0
    for npz_path in tqdm(files, desc="Converting G1 qpos"):
        rel_path = npz_path.relative_to(input_dir)
        if manifest_filter is None:
            file_hash = int(hashlib.sha256(str(rel_path).encode("utf-8")).hexdigest(), 16)
            if file_hash % num_rank != slurm_rank:
                continue
        elif not _matches_manifest(npz_path, input_dir, manifest_filter):
            continue

        outpath = (output_dir / str(rel_path).replace(" ", "_")).with_suffix(".motion")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if outpath.exists() and not force_remake:
            skipped += 1
            continue

        root_pos, root_rot_wxyz, joint_angles = _load_qpos_npz(
            npz_path=npz_path,
            output_fps=output_fps,
            expected_dof_names=expected_dof_names,
            device=device,
            dtype=dtype,
        )

        qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
        root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(
            kinematic_info, qpos
        )
        motion = fk_from_transforms_with_velocities(
            kinematic_info=kinematic_info,
            root_pos=root_pos_from_qpos,
            joint_rot_mats=joint_rot_mats,
            fps=output_fps,
            compute_velocities=True,
            velocity_max_horizon=3,
        )
        qpos_wrapped = extract_qpos_from_transforms(
            kinematic_info, root_pos_from_qpos, joint_rot_mats
        )
        motion.dof_pos = qpos_wrapped[:, 7:]
        motion.dof_vel = compute_cartesian_velocity(
            batched_robot_pos=joint_angles.unsqueeze(1), fps=output_fps
        ).squeeze(1)
        motion.fix_height_per_frame(height_offset=0.02)
        motion.fix_height(height_offset=0.04)
        motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
            positions=motion.rigid_body_pos,
            velocity=motion.rigid_body_vel,
            vel_thres=0.15,
            height_thresh=0.1,
        ).to(torch.bool)
        motion.local_rigid_body_rot = None
        temporary_outpath = outpath.with_name(
            f".{outpath.name}.{os.getpid()}.tmp"
        )
        try:
            torch.save(motion.to_dict(), str(temporary_outpath))
            os.replace(temporary_outpath, outpath)
        finally:
            temporary_outpath.unlink(missing_ok=True)
        converted += 1

    print(f"Converted {converted}, skipped {skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    app()
