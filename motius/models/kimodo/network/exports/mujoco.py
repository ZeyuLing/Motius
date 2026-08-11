from __future__ import annotations

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert native KIMODO G1 motions to MuJoCo qpos."""

import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..assets import skeleton_asset_path
from ..geometry import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion
from ..skeleton import G1Skeleton34, SkeletonBase, global_rots_to_local_rots
from ..tools import ensure_batched, to_numpy, to_torch

_converter_cache: dict[tuple[int, str], "MujocoQposConverter"] = {}


def _rotation_from_quat_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    try:
        return Rotation.from_quat(quat_wxyz, scalar_first=True)
    except TypeError:
        return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])


class MujocoQposConverter:
    """Batch converter from KIMODO G1 output dicts to MuJoCo qpos.

    KIMODO uses y-up/z-forward coordinates. The G1 MuJoCo model uses z-up/x-forward
    coordinates and a 36-dim qpos layout: root translation (3), root quaternion
    (4), and 29 hinge DOFs.
    """

    def __new__(
        cls,
        input_skeleton: SkeletonBase,
        xml_path: str = str(skeleton_asset_path("g1skel34", "xml", "g1.xml")),
    ):
        key = (id(input_skeleton), xml_path)
        if key not in _converter_cache:
            inst = object.__new__(cls)
            _converter_cache[key] = inst
        return _converter_cache[key]

    def __init__(
        self,
        input_skeleton: SkeletonBase,
        xml_path: str = str(skeleton_asset_path("g1skel34", "xml", "g1.xml")),
    ):
        if getattr(self, "_initialized", False):
            return
        self.xml_path = xml_path
        self.skeleton = input_skeleton
        self._prepare_transforms()
        self._subtree_joints = {}
        self._initialized = True

    def _prepare_transforms(self):
        self.mujoco_to_kimodo_matrix = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        self.kimodo_to_mujoco_matrix = self.mujoco_to_kimodo_matrix.T

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        xml_classes = [x for x in tree.findall(".//default") if "class" in x.attrib]
        joint_axes = {}
        class_ranges: dict[str, tuple[float, float]] = {}
        for xml_class in xml_classes:
            joints = xml_class.findall("joint")
            if not joints:
                continue
            joint = joints[0]
            joint_axes[xml_class.get("class")] = joint.get("axis")
            range_str = joint.get("range")
            if range_str:
                range_vals = [float(x) for x in range_str.split()]
                if len(range_vals) == 2:
                    class_ranges[xml_class.get("class")] = (range_vals[0], range_vals[1])

        mujoco_hinge_joints = root.find("worldbody").findall(".//joint")
        self._mujoco_joint_axis_values_kimodo_space = torch.zeros(
            (len(mujoco_hinge_joints), 3), dtype=torch.float32
        )
        self._mujoco_joint_axis_values_mujoco_space = torch.zeros(
            (len(mujoco_hinge_joints), 3), dtype=torch.float32
        )
        self._mujoco_indices_to_kimodo_indices = torch.zeros((len(mujoco_hinge_joints),), dtype=torch.int32)
        self._kimodo_indices_to_mujoco_indices = torch.ones((self.skeleton.nbjoints,), dtype=torch.int32) * -1

        self._nb_joints_mujoco = len(mujoco_hinge_joints) + 1
        self._nb_joints_kimodo = self.skeleton.nbjoints
        self._mujoco_joint_including_root_parent_list = torch.full(
            (len(mujoco_hinge_joints) + 1,), -1, dtype=torch.int32
        )
        self._mujoco_joint_including_root_list = ["pelvis_skel"]

        for joint_id_in_csv, joint in enumerate(mujoco_hinge_joints):
            joint_name_in_skeleton = joint.get("name").replace("_joint", "_skel")
            joint_parent_name_in_skeleton = self.skeleton.bone_parents[joint_name_in_skeleton]

            self._mujoco_joint_including_root_list.append(joint_name_in_skeleton)
            self._mujoco_joint_including_root_parent_list[joint_id_in_csv + 1] = (
                self._mujoco_joint_including_root_list.index(joint_parent_name_in_skeleton)
            )

            joint_idx_in_kimodo_skeleton = self.skeleton.bone_order_names.index(joint_name_in_skeleton)
            axis_str = joint.get("axis") or joint_axes[joint.get("class")]
            axis_values = [float(x) for x in axis_str.split(" ")]

            mujoco_joint_axis_mapping_kimodo_space = [
                torch.tensor([0, 0, 1]),
                torch.tensor([1, 0, 0]),
                torch.tensor([0, 1, 0]),
            ][np.argmax(axis_values)]

            self._mujoco_joint_axis_values_kimodo_space[joint_id_in_csv] = (
                mujoco_joint_axis_mapping_kimodo_space
            )
            self._mujoco_joint_axis_values_mujoco_space[joint_id_in_csv] = torch.tensor(axis_values)

            self._mujoco_indices_to_kimodo_indices[joint_id_in_csv] = joint_idx_in_kimodo_skeleton
            self._kimodo_indices_to_mujoco_indices[joint_idx_in_kimodo_skeleton] = joint_id_in_csv + 1
        self._kimodo_indices_to_mujoco_indices[0] = 0

        self._joint_limits_min = torch.full((len(mujoco_hinge_joints),), float("-inf"), dtype=torch.float32)
        self._joint_limits_max = torch.full((len(mujoco_hinge_joints),), float("inf"), dtype=torch.float32)
        for joint_id_in_csv, joint in enumerate(mujoco_hinge_joints):
            range_vals = None
            if joint.get("range"):
                range_vals = [float(x) for x in joint.get("range").split()]
            elif joint.get("class") and joint.get("class") in class_ranges:
                range_vals = list(class_ranges[joint.get("class")])
            if range_vals is not None and len(range_vals) == 2:
                self._joint_limits_min[joint_id_in_csv] = range_vals[0]
                self._joint_limits_max[joint_id_in_csv] = range_vals[1]

        r_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
        mujoco_to_kimodo = r_zup_to_yup * x_forward_to_y_forward

        self._rot_offsets_q2t = torch.zeros(
            len(self._kimodo_indices_to_mujoco_indices), 3, 3, dtype=torch.float32
        )
        self._rot_offsets_q2t[...] = torch.eye(3)[None]

        self._rot_offsets_f2q = torch.zeros(
            len(self._kimodo_indices_to_mujoco_indices), 3, 3, dtype=torch.float32
        )
        self._rot_offsets_f2q[...] = torch.eye(3)[None]
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for i, joint in enumerate(mujoco_hinge_joints):
            body = parent_map[joint]
            if "quat" not in body.attrib:
                continue
            rot = _rotation_from_quat_wxyz([float(x) for x in body.get("quat").strip().split(" ")])
            idx = self._mujoco_indices_to_kimodo_indices[i]
            self._rot_offsets_q2t[idx] = torch.from_numpy(rot.as_matrix())
            rot = mujoco_to_kimodo * rot * mujoco_to_kimodo.inv()
            self._rot_offsets_f2q[idx] = torch.from_numpy(rot.as_matrix().T)

        axis_kimodo = self._mujoco_joint_axis_values_kimodo_space
        self._mujoco_joint_axis_values_f2q_space = torch.zeros_like(axis_kimodo)
        for i in range(len(mujoco_hinge_joints)):
            j = self._mujoco_indices_to_kimodo_indices[i].item()
            axis_f2q = torch.mv(self._rot_offsets_f2q[j], axis_kimodo[i])
            norm = axis_f2q.norm()
            if norm > 1e-8:
                axis_f2q = axis_f2q / norm
            self._mujoco_joint_axis_values_f2q_space[i] = axis_f2q

        rest_rot_f2q = self._rot_offsets_f2q[self._mujoco_indices_to_kimodo_indices]
        rest_rot_f2q = rest_rot_f2q.unsqueeze(0).unsqueeze(0)
        self._rest_dofs = self._local_rots_f2q_to_joint_dofs(rest_rot_f2q).squeeze(0).squeeze(0)

        rest_rot_f2q_flat = self._rot_offsets_f2q[self._mujoco_indices_to_kimodo_indices]
        full_axis_angle = matrix_to_axis_angle(rest_rot_f2q_flat)
        self._rest_dofs_axis_angle = (full_axis_angle * self._mujoco_joint_axis_values_f2q_space).sum(dim=-1)

    def dict_to_qpos(
        self,
        output: dict,
        device: Optional[str] = None,
        root_quat_w_first: bool = True,
        numpy: bool = True,
        mujoco_rest_zero: bool = False,
    ):
        local_rot_mats = to_torch(output["local_rot_mats"], device)
        root_positions = to_torch(output["root_positions"], device)

        qpos = self.to_qpos(
            local_rot_mats,
            root_positions,
            root_quat_w_first=root_quat_w_first,
            mujoco_rest_zero=mujoco_rest_zero,
        )
        if numpy:
            qpos = to_numpy(qpos)
        return qpos

    def save_csv(self, qpos: torch.Tensor | np.ndarray, csv_path):
        qpos = to_numpy(qpos)
        shape = qpos.shape
        if len(shape) == 2:
            np.savetxt(csv_path, qpos, delimiter=",")
        elif len(shape) == 3:
            if shape[0] == 1:
                np.savetxt(csv_path, qpos[0], delimiter=",")
            else:
                csv_path_base, ext = os.path.splitext(csv_path)
                for i in range(shape[0]):
                    self.save_csv(qpos[i], csv_path_base + "_" + str(i).zfill(2) + ext)
        else:
            raise ValueError(f"Expected qpos with 2 or 3 dims, got shape {shape}.")

    def _local_rots_to_joint_dofs(
        self,
        local_rot_mats: torch.Tensor,
        axis_vals: torch.Tensor,
    ) -> torch.Tensor:
        x_joint_dof = torch.atan2(local_rot_mats[..., 2, 1], local_rot_mats[..., 2, 2])
        y_joint_dof = torch.atan2(local_rot_mats[..., 0, 2], local_rot_mats[..., 0, 0])
        z_joint_dof = torch.atan2(local_rot_mats[..., 1, 0], local_rot_mats[..., 1, 1])
        xyz_joint_dofs = torch.stack([x_joint_dof, y_joint_dof, z_joint_dof], dim=-1)
        axis_vals = axis_vals.to(device=local_rot_mats.device, dtype=local_rot_mats.dtype)
        joint_dofs = (xyz_joint_dofs * axis_vals[None, None, :, :]).sum(dim=-1)
        return joint_dofs

    def _local_rots_to_joint_dofs_axis_angle(
        self,
        local_rot_mats: torch.Tensor,
        axis_vals: torch.Tensor,
    ) -> torch.Tensor:
        axis_vals = axis_vals.to(device=local_rot_mats.device, dtype=local_rot_mats.dtype)
        full_axis_angle = matrix_to_axis_angle(local_rot_mats)
        joint_dofs = (full_axis_angle * axis_vals).sum(dim=-1)
        return joint_dofs

    def _local_rots_f2q_to_joint_dofs(self, local_rot_mats_f2q: torch.Tensor) -> torch.Tensor:
        return self._local_rots_to_joint_dofs(
            local_rot_mats_f2q,
            self._mujoco_joint_axis_values_f2q_space,
        )

    def _clamp_to_limits(self, joint_dofs: torch.Tensor) -> torch.Tensor:
        device = joint_dofs.device
        lo = self._joint_limits_min.to(device=device, dtype=joint_dofs.dtype)
        hi = self._joint_limits_max.to(device=device, dtype=joint_dofs.dtype)
        return torch.clamp(joint_dofs, lo[None, None, :], hi[None, None, :])

    def _clamp_joint_dofs(self, joint_dofs: torch.Tensor, rest_dofs: torch.Tensor) -> torch.Tensor:
        device = joint_dofs.device
        rest_dofs = rest_dofs.to(device=device, dtype=joint_dofs.dtype)
        mujoco_dofs = joint_dofs - rest_dofs[None, None, :]
        lo = self._joint_limits_min.to(device=device, dtype=joint_dofs.dtype)
        hi = self._joint_limits_max.to(device=device, dtype=joint_dofs.dtype)
        mujoco_dofs = torch.clamp(mujoco_dofs, lo[None, None, :], hi[None, None, :])
        return mujoco_dofs + rest_dofs[None, None, :]

    def _joint_dofs_to_local_rot_mats(
        self,
        joint_dofs: torch.Tensor,
        original_local_rot_mats: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        use_relative: bool = False,
    ) -> torch.Tensor:
        out = original_local_rot_mats.clone()
        axis_kimodo = self._mujoco_joint_axis_values_kimodo_space.to(device=device, dtype=dtype)
        for i in range(joint_dofs.shape[-1]):
            j = self._mujoco_indices_to_kimodo_indices[i].item()
            angle = joint_dofs[..., i]
            axis = axis_kimodo[i]
            if use_relative:
                axis_angle = angle[..., None] * axis[None, None, :]
                r_local = axis_angle_to_matrix(axis_angle)
            else:
                rot_offsets_f2q = self._rot_offsets_f2q.to(device=device, dtype=dtype)
                axis_in_f2q = torch.mv(rot_offsets_f2q[j], axis)
                axis_angle = angle[..., None] * axis_in_f2q[None, None, :]
                r_f2q = axis_angle_to_matrix(axis_angle)
                r_local = torch.einsum("ij,btjk->btik", rot_offsets_f2q[j].T, r_f2q)
            out[:, :, j, :, :] = r_local
        return out

    @ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
    def project_to_real_robot_rotations(
        self,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
        clamp_to_limits: bool = True,
        mujoco_rest_zero: bool = False,
    ) -> dict:
        device = local_rot_mats.device
        dtype = local_rot_mats.dtype

        local_rot_f2q = torch.matmul(self._rot_offsets_f2q.to(device=device, dtype=dtype), local_rot_mats)
        hinge_rots = local_rot_f2q[:, :, self._mujoco_indices_to_kimodo_indices, :, :]
        axis_f2q = self._mujoco_joint_axis_values_f2q_space.to(device=device, dtype=dtype)
        joint_dofs = self._local_rots_to_joint_dofs_axis_angle(hinge_rots, axis_f2q)

        if mujoco_rest_zero:
            rest_dofs = self._rest_dofs_axis_angle.to(device=device, dtype=dtype)
            angles = joint_dofs - rest_dofs[None, None, :]
            use_relative = True
        else:
            angles = joint_dofs
            use_relative = False

        if clamp_to_limits:
            if mujoco_rest_zero:
                angles = self._clamp_to_limits(angles)
            else:
                rest_dofs_axis_angle = self._rest_dofs_axis_angle.to(device=device, dtype=dtype)
                angles = self._clamp_joint_dofs(angles, rest_dofs_axis_angle)

        local_rot_mats_proj = self._joint_dofs_to_local_rot_mats(
            angles,
            local_rot_mats,
            device,
            dtype,
            use_relative=use_relative,
        )
        global_rot_mats, posed_joints, _ = self.skeleton.fk(local_rot_mats_proj, root_positions)
        return {
            "local_rot_mats": local_rot_mats_proj,
            "global_rot_mats": global_rot_mats,
            "posed_joints": posed_joints,
            "root_positions": root_positions,
        }

    @ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
    def to_qpos(
        self,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
        root_quat_w_first: bool = True,
        mujoco_rest_zero: bool = False,
    ) -> torch.Tensor:
        batch_size, num_frames = root_positions.shape[0], root_positions.shape[1]
        device, dtype = local_rot_mats.device, local_rot_mats.dtype

        local_rot_mats = torch.matmul(self._rot_offsets_f2q.to(device=device, dtype=dtype), local_rot_mats)

        kimodo_to_mujoco_matrix = self.kimodo_to_mujoco_matrix.to(device=device, dtype=dtype)
        qpos = torch.zeros((batch_size, num_frames, 36), dtype=dtype, device=device)

        root_positions_mujoco = torch.matmul(
            kimodo_to_mujoco_matrix[None, None, ...],
            root_positions[..., None],
        )
        qpos[:, :, :3] = root_positions_mujoco.view(batch_size, num_frames, 3)

        root_rot = local_rot_mats[:, :, 0, :]
        mujoco_to_kimodo_matrix = kimodo_to_mujoco_matrix.T
        root_rot_mujoco = torch.matmul(
            torch.matmul(kimodo_to_mujoco_matrix[None, None, ...], root_rot),
            mujoco_to_kimodo_matrix[None, None, ...],
        )
        root_rot_quat = matrix_to_quaternion(root_rot_mujoco)
        if root_quat_w_first:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [0, 1, 2, 3]]
        else:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [1, 2, 3, 0]]

        joint_rot_f2q = local_rot_mats[:, :, self._mujoco_indices_to_kimodo_indices, :, :]
        joint_dofs = self._local_rots_f2q_to_joint_dofs(joint_rot_f2q)
        if mujoco_rest_zero:
            rest_dofs = self._rest_dofs.to(device=device, dtype=dtype)
            qpos[:, :, 7:] = joint_dofs - rest_dofs[None, None, :]
        else:
            qpos[:, :, 7:] = joint_dofs
        return qpos


def apply_g1_real_robot_projection(
    skeleton: G1Skeleton34,
    joints_pos: torch.Tensor,
    joints_rot: torch.Tensor,
    clamp_to_limits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
    root_positions = joints_pos[..., skeleton.root_idx, :]

    single_sequence = local_rot_mats.dim() == 4
    if single_sequence:
        local_rot_mats = local_rot_mats.unsqueeze(0)
        root_positions = root_positions.unsqueeze(0)

    converter = MujocoQposConverter(skeleton)
    projected = converter.project_to_real_robot_rotations(
        local_rot_mats,
        root_positions,
        clamp_to_limits=clamp_to_limits,
    )

    out_pos = projected["posed_joints"]
    out_rot = projected["global_rot_mats"]
    if single_sequence:
        out_pos = out_pos.squeeze(0)
        out_rot = out_rot.squeeze(0)
    return out_pos, out_rot
