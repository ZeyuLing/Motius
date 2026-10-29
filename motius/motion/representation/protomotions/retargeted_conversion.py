# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Native G1 CSV reference preparation, FK, filtering and contact labels."""
import hashlib
import os
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from motius.motion.kinematics.protomotions import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    compute_cartesian_velocity,
    extract_transforms_from_qpos,
    extract_qpos_from_transforms,
)
from motius.configuration.protomotions.g1 import G1RobotConfig
from .contacts import compute_contact_labels_from_pos_and_vel
from .filtering import passes_exclude_motion_filter


import glob
from .fps import downsample_factor, fps_from_mapping, resolve_output_fps


def process_csv_file(csv_path, input_fps, output_fps, device, dtype):
    """
    Process a CSV file and extract motion data.

    CSV format (headerless, comma-separated):
        - Columns 0-2: root position (x, y, z) in meters
        - Columns 3-6: root rotation quaternion (w, x, y, z)
        - Columns 7+: joint angles in radians

    Args:
        csv_path: Path to CSV file
        input_fps: Input motion fps
        output_fps: Output motion fps
        device: torch device
        dtype: torch data type

    Returns:
        tuple: (root_pos, root_rot_wxyz, joint_angles)
    """
    data = np.loadtxt(csv_path, delimiter=",")
    root_pos = data[:, 0:3]
    root_rot_wxyz = data[:, 3:7]
    joint_angles = data[:, 7:]
    actual_input_fps = float(input_fps)
    actual_output_fps = resolve_output_fps(actual_input_fps, output_fps)
    factor = downsample_factor(actual_input_fps, actual_output_fps)
    if factor > 1:
        joint_angles = joint_angles[::factor]
        root_pos = root_pos[::factor]
        root_rot_wxyz = root_rot_wxyz[::factor]
    root_pos = torch.from_numpy(root_pos).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(root_rot_wxyz).to(device, dtype)
    joint_angles = torch.from_numpy(joint_angles).to(device, dtype)
    return (root_pos, root_rot_wxyz, joint_angles, actual_output_fps)


def process_npz_file(npz_path, input_fps, output_fps, device, dtype):
    """
    Process an NPZ file and extract motion data.

    Args:
        npz_path: Path to NPZ file
        device: torch device
        dtype: torch data type

    Returns:
        tuple: (root_pos, root_rot_wxyz, joint_angles)
    """
    data = np.load(npz_path, allow_pickle=True)
    actual_input_fps = fps_from_mapping(data, input_fps)
    actual_output_fps = resolve_output_fps(actual_input_fps, output_fps)
    factor = downsample_factor(actual_input_fps, actual_output_fps)
    base_frame_pos = data["base_frame_pos"][::factor]
    base_frame_wxyz = data["base_frame_wxyz"][::factor]
    joint_angles = data["joint_angles"][::factor]
    root_pos = torch.from_numpy(base_frame_pos).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(base_frame_wxyz).to(device, dtype)
    joint_angles = torch.from_numpy(joint_angles).to(device, dtype)
    return (root_pos, root_rot_wxyz, joint_angles, actual_output_fps)


def apply_contact_labels_to_motion(
    motion,
    contact_labels_dir,
    motion_filename,
    input_fps,
    output_fps,
    left_foot_idx,
    right_foot_idx,
    device,
):
    """
    Load and apply contact labels from source motion to the motion object.

    Args:
        motion: Motion object to update
        contact_labels_dir: Directory containing contact label files
        motion_filename: Name of the motion file (for matching)
        input_fps: Input fps of contact labels
        output_fps: Target output fps
        left_foot_idx: Index of left foot body
        right_foot_idx: Index of right foot body
        device: torch device
    """
    base_filename = os.path.splitext(motion_filename)[0]
    if base_filename.endswith("_retargeted"):
        base_filename = base_filename[: -len("_retargeted")]
    contact_labels_path = contact_labels_dir / f"{base_filename}_contacts.npz"
    if not os.path.exists(contact_labels_path):
        raise FileNotFoundError(
            f"Contact labels file not found: {contact_labels_path}\nPlease ensure contact files are generated for all motions."
        )
    contact_data = np.load(contact_labels_path, allow_pickle=True)
    contact_input_fps = fps_from_mapping(contact_data, input_fps)
    foot_contacts = contact_data["foot_contacts"]
    factor = downsample_factor(contact_input_fps, output_fps)
    if factor > 1:
        foot_contacts = foot_contacts[::factor]
    motion_length = motion.rigid_body_pos.shape[0]
    contact_length = foot_contacts.shape[0]
    assert (
        motion_length == contact_length
    ), f"Motion length ({motion_length}) does not match contact length ({contact_length}) for {motion_filename}. Contact file: {contact_labels_path}"
    num_bodies = motion.rigid_body_pos.shape[1]
    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)
    rigid_body_contacts[:, left_foot_idx] = foot_contacts[:, 0] > 0.5
    rigid_body_contacts[:, right_foot_idx] = foot_contacts[:, 1] > 0.5
    motion.rigid_body_contacts = torch.from_numpy(rigid_body_contacts).to(device)
    print(
        f"Applied contact labels from original motion before re-targeting {contact_labels_path}"
    )


def convert_retargeted_directory(
    retargeted_motion_dir: Path = ...,
    output_dir: Path = ...,
    input_fps: int = 30,
    output_fps: Optional[int] = None,
    force_remake: bool = False,
    ignore_first_n_frames: int = 0,
    apply_motion_filter: bool = False,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.1,
    duration_height_seconds: float = 0.6,
    robot_type: str = "g1",
    contact_labels_dir: Optional[Path] = None,
    asset_root=None,
):
    device = torch.device("cpu")
    dtype = torch.float32
    os.makedirs(output_dir, exist_ok=True)
    robot_mjcf_mapping = {"g1": "g1_bm_box_feet.xml", "h1_2": "h1_2.xml"}
    mjcf_filename = robot_mjcf_mapping.get(robot_type, f"{robot_type}.xml")
    mjcf_path = str(
        Path(asset_root or "checkpoints/robots/protomotions").expanduser().resolve()
        / "mjcf"
        / mjcf_filename
    )
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")
    kinematic_info = extract_kinematic_info(mjcf_path)
    if robot_type != "g1":
        raise ValueError("The released native retargeted-data converter supports G1")
    naming = G1RobotConfig.__dataclass_fields__[
        "common_naming_to_robot_body_names"
    ].default_factory()
    left_foot_name = naming["all_left_foot_bodies"][0]
    right_foot_name = naming["all_right_foot_bodies"][0]
    body_names = kinematic_info.body_names
    left_foot_idx = body_names.index(left_foot_name)
    right_foot_idx = body_names.index(right_foot_name)
    print(f"Robot type: {robot_type}")
    print(f"Left foot: {left_foot_name} (index {left_foot_idx})")
    print(f"Right foot: {right_foot_name} (index {right_foot_idx})")
    npz_files = sorted(list(glob.glob(str(retargeted_motion_dir / "*.npz"))))
    csv_files = sorted(list(glob.glob(str(retargeted_motion_dir / "*.csv"))))
    motion_files = npz_files + csv_files
    print(
        f"Found {len(motion_files)} motion files to process ({len(npz_files)} NPZ, {len(csv_files)} CSV)."
    )
    for motion_file_path in tqdm(motion_files, desc="Processing motions"):
        motion_file = Path(motion_file_path)
        outpath = (output_dir / motion_file.name.replace(" ", "_")).with_suffix(
            ".motion"
        )
        if not force_remake and outpath.exists():
            continue
        print(f"Processing {motion_file.name}")
        try:
            if motion_file.suffix.lower() == ".csv":
                root_pos, root_rot_wxyz, joint_angles, motion_fps = process_csv_file(
                    motion_file_path, input_fps, output_fps, device, dtype
                )
            elif motion_file.suffix.lower() == ".npz":
                root_pos, root_rot_wxyz, joint_angles, motion_fps = process_npz_file(
                    motion_file_path, input_fps, output_fps, device, dtype
                )
            else:
                print(f"Unsupported file format: {motion_file.suffix}")
                continue
            if ignore_first_n_frames > 0:
                root_pos = root_pos[ignore_first_n_frames:]
                root_rot_wxyz = root_rot_wxyz[ignore_first_n_frames:]
                joint_angles = joint_angles[ignore_first_n_frames:]
            qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
            root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(
                kinematic_info, qpos
            )
            motion = fk_from_transforms_with_velocities(
                kinematic_info=kinematic_info,
                root_pos=root_pos_from_qpos,
                joint_rot_mats=joint_rot_mats,
                fps=motion_fps,
                compute_velocities=True,
                velocity_max_horizon=3,
            )
            qpos = extract_qpos_from_transforms(
                kinematic_info, root_pos, joint_rot_mats
            )
            motion.dof_pos = qpos[:, 7:]
            allowed_delta = [0.0, 2 * np.pi, 4 * np.pi]
            delta = (qpos[:, 7:] - joint_angles).abs()
            epsilon = 0.0001
            allowed = torch.zeros_like(delta, dtype=torch.bool)
            for d in allowed_delta:
                allowed |= (delta - d).abs() < epsilon
            assert (
                allowed.all()
            ), "qpos and joint_angles are not allowed (exceeds allowed delta with epsilon tolerance)"
            dof_vel = compute_cartesian_velocity(
                batched_robot_pos=joint_angles.unsqueeze(1), fps=motion_fps
            )
            motion.dof_vel = dof_vel.squeeze(1)
            translation_vecs = motion.fix_height_per_frame(height_offset=0.02)
            if motion.rigid_body_vel is not None and motion.fps is not None:
                vel_delta = torch.zeros(
                    translation_vecs.shape[0],
                    1,
                    3,
                    device=motion.rigid_body_vel.device,
                    dtype=motion.rigid_body_vel.dtype,
                )
                vel_delta[:-1] = (
                    translation_vecs[1:] - translation_vecs[:-1]
                ).unsqueeze(1) / motion.motion_dt
                motion.rigid_body_vel = motion.rigid_body_vel + vel_delta
            motion.fix_height(height_offset=0.04)
            if contact_labels_dir is not None:
                apply_contact_labels_to_motion(
                    motion=motion,
                    contact_labels_dir=contact_labels_dir,
                    motion_filename=motion_file.name,
                    input_fps=input_fps,
                    output_fps=motion_fps,
                    left_foot_idx=left_foot_idx,
                    right_foot_idx=right_foot_idx,
                    device=device,
                )
            else:
                motion.rigid_body_contacts = torch.zeros(
                    motion.rigid_body_pos.shape[0],
                    motion.rigid_body_pos.shape[1],
                    device=device,
                    dtype=torch.bool,
                )
            motion.local_rigid_body_rot = None
            if apply_motion_filter:
                if not passes_exclude_motion_filter(
                    motion,
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                ):
                    print(
                        f"Skipping {motion_file.name} because it does not pass motion filter"
                    )
                    continue
            print(f"motion.dof_pos: {motion.dof_pos.shape}")
            print(f"motion.dof_vel: {motion.dof_vel.shape}")
            print(f"motion.rigid_body_pos: {motion.rigid_body_pos.shape}")
            print(f"motion.rigid_body_rot: {motion.rigid_body_rot.shape}")
            print(f"Saving to {outpath}")
            payload = motion.to_dict()
            payload.pop("state_conversion", None)
            torch.save(payload, str(outpath))
        except Exception as e:
            print(f"Error processing {motion_file.name}: {e}")
            print(f"Skipping {motion_file.name} due to error")
            import traceback

            traceback.print_exc()
            continue
