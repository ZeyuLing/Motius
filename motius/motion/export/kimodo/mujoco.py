# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert kimodo motion (y-up, z-forward) to MuJoCo qpos (z-up, x-forward) for G1 skeleton."""

import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motius.configuration.kimodo.assets import skeleton_asset_path
from motius.motion.representation.kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
from motius.motion.body_models.kimodo import G1Skeleton34, SkeletonBase, global_rots_to_local_rots
from motius.utils.kimodo import ensure_batched, to_numpy, to_torch

# Cache so that the same (skeleton, xml_path) returns the same converter instance.
_converter_cache: dict[tuple[int, str], "MujocoQposConverter"] = {}


class MujocoQposConverter:
    """Fast batch converter from our dictionary format to mujoco qpos with precomputed transforms.

    In mujoco, the coordination is z up and x forward, right handed.

    Features (30 joints):
    - root (pelvis, 7 = translation + rotation) + 29 dof joints (29)

    In kimodo, the coordinate system is y up and z forward, right handed.
    Features (34 joints):
    - root (pelvis) + (34 - 1) joints; among these joints, 4 are end-effector joints added by kimodo.

    Cached by (input_skeleton id, xml_path); repeated calls with the same args return the same instance.
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
        """Initialize converter with precomputed transforms.

        Args:
            xml_path: Path to the mujoco XML file containing joint definitions
        """
        if getattr(self, "_initialized", False):
            return
        self.xml_path = xml_path
        self.skeleton = input_skeleton
        self._prepare_transforms()
        self._subtree_joints = {}
        self._initialized = True

    def _prepare_transforms(self):
        """Precompute all necessary transforms for efficient batch processing."""
        # Define coordinate transformations between mujoco and kimodo space
        # 1) R_zup_to_yup: rotation around x-axis by -90 degrees
        # 2) x_forward_to_y_forward: rotation around z-axis by -90 degrees
        # Combined transformation matrix: mujoco_to_kimodo = R_zup_to_yup * x_forward_to_y_forward
        self.mujoco_to_kimodo_matrix = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )
        self.kimodo_to_mujoco_matrix = self.mujoco_to_kimodo_matrix.T  # Inverse transformation: kimodo_to_mujoco

        # Parse XML once and extract joint information
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        xml_classes = [x for x in tree.findall(".//default") if "class" in x.attrib]
        joint_axes = dict()
        class_ranges: dict[str, tuple[float, float]] = {}
        for xml_class in xml_classes:
            j = xml_class.findall("joint")
            if j:
                joint_axes[xml_class.get("class")] = j[0].get("axis")
                range_str = j[0].get("range")
                if range_str:
                    range_vals = [float(x) for x in range_str.split()]
                    if len(range_vals) == 2:
                        class_ranges[xml_class.get("class")] = (
                            range_vals[0],
                            range_vals[1],
                        )

        mujoco_hinge_joints = root.find("worldbody").findall(".//joint")  # skip the base joint
        self._mujoco_joint_axis_values_kimodo_space = torch.zeros(
            (len(mujoco_hinge_joints), 3), dtype=torch.float32
        )  # mujoco order but kimodo space
        self._mujoco_joint_axis_values_mujoco_space = torch.zeros(
            (len(mujoco_hinge_joints), 3), dtype=torch.float32
        )  # mujoco order but mujoco space

        # for the below indices, mujoco_indices_to_kimodo_indices does not include mujoco root (30 - 1 = 29 elements),
        # while kimodo_indices_to_mujoco_indices inclues the kimodo root (32 elements).
        self._mujoco_indices_to_kimodo_indices = torch.zeros((len(mujoco_hinge_joints),), dtype=torch.int32)
        self._kimodo_indices_to_mujoco_indices = (
            torch.ones((self.skeleton.nbjoints,), dtype=torch.int32) * -1
        )  # -1 means not in the csv skeleton

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
            axis_values = [float(x) for x in (joint.get("axis") or joint_axes[joint.get("class")]).split(" ")]

            # the mapped axis in kimodo skeleton space is calculated as bones_axis = mujoco_to_kimodo.apply(axis_values)
            # [1, 0, 0] -> [0, 0, 1]; [0, 1, 0] -> [1, 0, 0]; [0, 0, 1] -> [0, 1, 0]
            mujoco_joint_axis_mapping_kimodo_space = [
                torch.tensor([0, 0, 1]),
                torch.tensor([1, 0, 0]),
                torch.tensor([0, 1, 0]),
            ][np.argmax(axis_values)]

            self._mujoco_joint_axis_values_kimodo_space[joint_id_in_csv] = mujoco_joint_axis_mapping_kimodo_space
            self._mujoco_joint_axis_values_mujoco_space[joint_id_in_csv] = torch.tensor(axis_values)

            self._mujoco_indices_to_kimodo_indices[joint_id_in_csv] = joint_idx_in_kimodo_skeleton
            self._kimodo_indices_to_mujoco_indices[joint_idx_in_kimodo_skeleton] = (
                joint_id_in_csv + 1
            )  # +1 for the root
        self._kimodo_indices_to_mujoco_indices[0] = 0  # the root joint mapping

        # Joint limits (min, max) in radians for each mujoco hinge, for clamping
        self._joint_limits_min = torch.full((len(mujoco_hinge_joints),), float("-inf"), dtype=torch.float32)
        self._joint_limits_max = torch.full((len(mujoco_hinge_joints),), float("inf"), dtype=torch.float32)
        for joint_id_in_csv, joint in enumerate(mujoco_hinge_joints):
            range_vals = None
            if joint.get("range"):
                range_vals = [float(x) for x in joint.get("range").split()]
            elif joint.get("class") and joint.get("class") in class_ranges:
                lo, hi = class_ranges[joint.get("class")]
                range_vals = [lo, hi]
            if range_vals is not None and len(range_vals) == 2:
                self._joint_limits_min[joint_id_in_csv] = range_vals[0]
                self._joint_limits_max[joint_id_in_csv] = range_vals[1]

        # load the offset matrices from the xml
        R_zup_to_yup = Rotation.from_euler("x", -90, degrees=True)
        x_forward_to_y_forward = Rotation.from_euler("z", -90, degrees=True)
        mujoco_to_kimodo = R_zup_to_yup * x_forward_to_y_forward

        self._rot_offsets_q2t = torch.zeros(len(self._kimodo_indices_to_mujoco_indices), 3, 3, dtype=torch.float32)
        self._rot_offsets_q2t[...] = torch.eye(3)[None]

        self._rot_offsets_f2q = torch.zeros(len(self._kimodo_indices_to_mujoco_indices), 3, 3, dtype=torch.float32)
        self._rot_offsets_f2q[...] = torch.eye(3)[None]
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for i, joint in enumerate(mujoco_hinge_joints):
            body = parent_map[joint]
            if "quat" in body.attrib:
                rot = Rotation.from_quat(
                    [float(x) for x in body.get("quat").strip().split(" ")],
                    scalar_first=True,
                )
                idx = self._mujoco_indices_to_kimodo_indices[i]
                self._rot_offsets_q2t[idx] = torch.from_numpy(rot.as_matrix())
                rot = mujoco_to_kimodo * rot * mujoco_to_kimodo.inv()
                self._rot_offsets_f2q[idx] = torch.from_numpy(rot.as_matrix().T)

        # Hinge axis in f2q space so extraction uses the same frame as joint_rot_f2q.
        # Then extract(offset) gives the angle s.t. axis_angle(angle * axis_f2q) = offset, and
        # reconstruction R_local = offset.T @ axis_angle(angle * axis_f2q) = I when input is identity.
        axis_kimodo = self._mujoco_joint_axis_values_kimodo_space
        self._mujoco_joint_axis_values_f2q_space = torch.zeros_like(axis_kimodo)
        for i in range(len(mujoco_hinge_joints)):
            j = self._mujoco_indices_to_kimodo_indices[i].item()
            axis_f2q = torch.mv(self._rot_offsets_f2q[j], axis_kimodo[i])
            n = axis_f2q.norm()
            if n > 1e-8:
                axis_f2q = axis_f2q / n
            self._mujoco_joint_axis_values_f2q_space[i] = axis_f2q

        # Rest-pose DOFs: angle we extract when R_local = I (t-pose). MuJoCo limits are
        # relative to joint zero (rest pose), so we must clamp in MuJoCo space: convert
        # joint_dofs to mujoco_angle = joint_dofs - rest_dofs, clamp, then back.
        rest_rot_f2q = self._rot_offsets_f2q[self._mujoco_indices_to_kimodo_indices]
        rest_rot_f2q = rest_rot_f2q.unsqueeze(0).unsqueeze(0)
        self._rest_dofs = self._local_rots_f2q_to_joint_dofs(rest_rot_f2q).squeeze(0).squeeze(0)
        # Axis-angle rest DOFs: angle s.t. axis_angle(angle * axis_f2q) = offset. Used in
        # project_to_real_robot_rotations so extract+reconstruct round-trip and t-pose is preserved.
        rest_rot_f2q_flat = self._rot_offsets_f2q[self._mujoco_indices_to_kimodo_indices]
        full_aa = matrix_to_axis_angle(rest_rot_f2q_flat)
        self._rest_dofs_axis_angle = (full_aa * self._mujoco_joint_axis_values_f2q_space).sum(dim=-1)

    def dict_to_qpos(
        self,
        output: dict,
        device: Optional[str] = None,
        root_quat_w_first: bool = True,
        numpy: bool = True,
        mujoco_rest_zero: bool = False,
    ):
        """Convert kimodo output dict to mujoco qpos format.

        Args:
            output: dict with keys "local_rot_mats" and "root_positions".
            device: device to use for the output.
            root_quat_w_first: If True, quaternion in qpos is (w,x,y,z).
            numpy: If True, convert the output to numpy array.
            mujoco_rest_zero: If True, joint angles are written so that kimodo rest (t-pose)
                maps to q=0 in MuJoCo. If False, write raw joint_dofs.

        Returns:
            qpos: (B, T, 7+J) mujoco qpos format.
        """
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

    def qpos_to_motion_dict(
        self,
        qpos: torch.Tensor | np.ndarray,
        source_fps: float,
        *,
        root_quat_w_first: bool = True,
        mujoco_rest_zero: bool = False,
    ):
        """Inverse of :meth:`to_qpos` / :meth:`dict_to_qpos` for MuJoCo CSV ``(T, 36)`` rows.

        Args:
            qpos: Shape ``(T, 36)`` or ``(1, T, 36)`` (root xyz, root quat wxyz, 29 joint angles).
            source_fps: Source frame rate (Hz) of the qpos data.
            root_quat_w_first: Must match how the CSV was written (default ``True``).
            mujoco_rest_zero: Must match :meth:`dict_to_qpos` / :meth:`to_qpos`.

        Returns:
            Kimodo motion dict (see :func:`kimodo.exports.motion_io.complete_motion_dict`).
        """
        from motius.motion.export.kimodo.motion_io import complete_motion_dict

        qpos = to_torch(qpos, None)
        if qpos.dim() == 2:
            qpos = qpos.unsqueeze(0)
        device = qpos.device
        dtype = qpos.dtype
        batch_size, num_frames, ncols = qpos.shape
        if ncols != 36:
            raise ValueError(f"Expected qpos last dim 36; got {ncols}")

        kimodo_to_mujoco_matrix = self.kimodo_to_mujoco_matrix.to(device=device, dtype=dtype)
        mujoco_to_kimodo_matrix = kimodo_to_mujoco_matrix.T

        root_mujoco = qpos[..., :3]
        root_positions = torch.matmul(mujoco_to_kimodo_matrix[None, None, ...], root_mujoco[..., None]).squeeze(-1)

        quat = qpos[..., 3:7]
        if root_quat_w_first:
            root_rot_mujoco = quaternion_to_matrix(quat)
        else:
            quat_wxyz = quat[..., [3, 0, 1, 2]]
            root_rot_mujoco = quaternion_to_matrix(quat_wxyz)

        O0 = self._rot_offsets_f2q[0].to(device=device, dtype=dtype)
        # root_rot_mujoco is (..., 3, 3) after optional batch unsqueeze (e.g. (1, T, 3, 3)).
        # Use ``...il`` so ``k`` sums with ``kl``; ``...ik`` incorrectly keeps ``k`` in the output.
        R_f2q_root = torch.einsum(
            "ij,...jk,kl->...il",
            mujoco_to_kimodo_matrix,
            root_rot_mujoco,
            kimodo_to_mujoco_matrix,
        )
        R_kimodo_root = torch.einsum("ij,...jk->...ik", O0.T, R_f2q_root)

        joint_dofs = qpos[..., 7:]
        if mujoco_rest_zero:
            rest_dofs = self._rest_dofs.to(device=device, dtype=dtype)
            angles = joint_dofs + rest_dofs[None, None, :]
            use_relative = True
        else:
            angles = joint_dofs
            use_relative = False

        nb_joints = self.skeleton.nbjoints
        template = torch.eye(3, device=device, dtype=dtype).expand(batch_size, num_frames, nb_joints, 3, 3).contiguous()
        template[:, :, 0] = R_kimodo_root

        local_rot_mats = self._joint_dofs_to_local_rot_mats(
            angles,
            template,
            device,
            dtype,
            use_relative=use_relative,
        )

        if batch_size != 1:
            raise ValueError(f"Only a single clip is supported; got batch_size={batch_size}")

        return complete_motion_dict(local_rot_mats[0], root_positions[0], self.skeleton, source_fps)

    def save_csv(self, qpos: torch.Tensor | np.ndarray, csv_path):
        # comment this
        qpos = to_numpy(qpos)
        shape = qpos.shape
        if len(shape) == 2:
            # only one motion: save it
            np.savetxt(csv_path, qpos, delimiter=",")
        if len(shape) == 3:
            # batch of motions
            if shape[0] == 1:
                # if only one motion, just save it
                np.savetxt(csv_path, qpos[0], delimiter=",")
            else:
                csv_path_base, ext = os.path.splitext(csv_path)
                for i in range(shape[0]):
                    self.save_csv(qpos[i], csv_path_base + "_" + str(i).zfill(2) + ext)

    def _local_rots_to_joint_dofs(
        self,
        local_rot_mats: torch.Tensor,
        axis_vals: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-joint single-DoF angles (radians) via Euler projection (for to_qpos/f2q)."""
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
        """Extract per-joint single-DoF angles (radians) via axis-angle; round-trips with
        axis_angle_to_matrix.

        Args:
            local_rot_mats: (..., num_hinges, 3, 3) in same frame as axis_vals.
            axis_vals: (num_hinges, 3) unit axis per hinge.
        Returns:
            joint_dofs: (..., num_hinges) signed angle = dot(axis_angle(R), axis).
        """
        axis_vals = axis_vals.to(device=local_rot_mats.device, dtype=local_rot_mats.dtype)
        full_aa = matrix_to_axis_angle(local_rot_mats)
        joint_dofs = (full_aa * axis_vals).sum(dim=-1)
        return joint_dofs

    def _local_rots_f2q_to_joint_dofs(self, local_rot_mats_f2q: torch.Tensor) -> torch.Tensor:
        """Extract per-joint single-DoF angles from local rotations in f2q space (for to_qpos)."""
        axis_vals = self._mujoco_joint_axis_values_f2q_space
        return self._local_rots_to_joint_dofs(local_rot_mats_f2q, axis_vals)

    def _clamp_to_limits(self, joint_dofs: torch.Tensor) -> torch.Tensor:
        """Clamp joint angles to XML limits (radians).

        Angles are in kimodo convention (0 = rest).
        """
        device = joint_dofs.device
        lo = self._joint_limits_min.to(device=device, dtype=joint_dofs.dtype)
        hi = self._joint_limits_max.to(device=device, dtype=joint_dofs.dtype)
        return torch.clamp(joint_dofs, lo[None, None, :], hi[None, None, :])

    def _clamp_joint_dofs(self, joint_dofs: torch.Tensor, rest_dofs: torch.Tensor) -> torch.Tensor:
        """Clamp joint angles to MuJoCo limits (radians), with rest_dofs conversion."""
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
        """Reconstruct full local rotation matrices from 1-DoF angles."""
        out = original_local_rot_mats.clone()
        axis_kimodo = self._mujoco_joint_axis_values_kimodo_space.to(device=device, dtype=dtype)
        for i in range(joint_dofs.shape[-1]):
            j = self._mujoco_indices_to_kimodo_indices[i].item()
            angle = joint_dofs[..., i]
            axis = axis_kimodo[i]
            if use_relative:
                axis_angle = angle[..., None] * axis[None, None, :]
                R_local = axis_angle_to_matrix(axis_angle)
            else:
                rot_offsets_f2q = self._rot_offsets_f2q.to(device=device, dtype=dtype)
                axis_in_f2q = torch.mv(rot_offsets_f2q[j], axis)
                axis_angle = angle[..., None] * axis_in_f2q[None, None, :]
                R_f2q = axis_angle_to_matrix(axis_angle)
                R_local = torch.einsum("ij,btjk->btik", rot_offsets_f2q[j].T, R_f2q)
            out[:, :, j, :, :] = R_local
        return out

    @ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
    def project_to_real_robot_rotations(
        self,
        local_rot_mats: torch.Tensor,
        root_positions: torch.Tensor,
        clamp_to_limits: bool = True,
        mujoco_rest_zero: bool = False,
    ) -> dict:
        """Project full 3D local rotations to G1 real robot DoF and back to 3D for viz.

        Joint angles are extracted along each hinge axis, optionally clamped to XML limits, then
        reconstructed to 3D rotations. When mujoco_rest_zero=False (default), raw angles are used
        (baked-with-quat). When True, angles are relative to rest (0 = T-pose in MuJoCo).
        """
        device = local_rot_mats.device
        dtype = local_rot_mats.dtype

        # Transform to f2q frame and extract 1-DoF angles (axis-angle projection).
        local_rot_f2q = torch.matmul(self._rot_offsets_f2q.to(device=device, dtype=dtype), local_rot_mats)
        hinge_rots = local_rot_f2q[:, :, self._mujoco_indices_to_kimodo_indices, :, :]
        axis_f2q = self._mujoco_joint_axis_values_f2q_space.to(device=device, dtype=dtype)
        joint_dofs = self._local_rots_to_joint_dofs_axis_angle(hinge_rots, axis_f2q)

        # Optionally express angles relative to rest (MuJoCo q=0 at T-pose).
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
                rest_dofs_aa = self._rest_dofs_axis_angle.to(device=device, dtype=dtype)
                angles = self._clamp_joint_dofs(angles, rest_dofs_aa)

        # Reconstruct 3D local rotations from 1-DoF angles and run FK.
        local_rot_mats_proj = self._joint_dofs_to_local_rot_mats(
            angles, local_rot_mats, device, dtype, use_relative=use_relative
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
        """Fast batch conversion from kimodo features to mujoco qpos format.

        Args:
            local_rot_mats: (B, T, J, 3, 3) local rotation matrices (kimodo convention).
            root_positions: (B, T, 3) root positions.
            root_quat_w_first: If True, quaternion in qpos is (w,x,y,z).
            mujoco_rest_zero: If True, joint angles are written so that kimodo rest (t-pose)
                maps to q=0 in MuJoCo. If False, write raw joint_dofs.

        Returns:
            torch.Tensor of shape [batch, numFrames, 36] containing mujoco qpos data:
            - root_trans (3) + root_quat (4) + joint_dofs (29) = 36 columns
        """

        batch_size, num_frames, nb_joints = local_rot_mats.shape[:3]
        device, dtype = local_rot_mats.device, local_rot_mats.dtype

        local_rot_mats = torch.matmul(self._rot_offsets_f2q.to(device), local_rot_mats)

        batch_size, num_frames = root_positions.shape[0], root_positions.shape[1]

        # Move precomputed matrices to the same device/dtype
        kimodo_to_mujoco_matrix = self.kimodo_to_mujoco_matrix.to(device=device, dtype=dtype)

        # Initialize output tensor: [batch, numFrames, 36]
        qpos = torch.zeros((batch_size, num_frames, 36), dtype=dtype, device=device)

        # Convert root translation: apply coordinate transformation
        root_positions_mujoco = torch.matmul(kimodo_to_mujoco_matrix[None, None, ...], root_positions[..., None])
        qpos[:, :, :3] = root_positions_mujoco.view(batch_size, num_frames, 3)

        # Convert root rotation: apply coordinate transformation to rotation matrix
        root_rot = local_rot_mats[:, :, 0, :]  # [batch, numFrames, 3, 3]

        # Apply coordinate transformation: R_mujoco = kimodo_to_mujoco * R_kimodo * kimodo_to_mujoco^T
        mujoco_to_kimodo_matrix = kimodo_to_mujoco_matrix.T
        root_rot_mujoco = torch.matmul(
            torch.matmul(kimodo_to_mujoco_matrix[None, None, ...], root_rot),
            mujoco_to_kimodo_matrix[None, None, ...],
        )
        root_rot_quat = matrix_to_quaternion(root_rot_mujoco)  # [w, x, y, z]
        if root_quat_w_first:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [0, 1, 2, 3]]  # [w, x, y, z]
        else:
            qpos[:, :, 3:7] = root_rot_quat[:, :, [1, 2, 3, 0]]  # [w, x, y, z] -> [x, y, z, w]

        # Joint DOFs: raw angles or relative to rest (rest = q=0 in MuJoCo).
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
    """Project G1 motion to real robot DoF (1-DoF per joint) with optional axis limits.

    Extracts a single angle per hinge along its axis (1-DoF), optionally clamps to
    joint limits from the MuJoCo XML (when clamp_to_limits=True), then reconstructs
    3D rotations and runs FK. T-pose (identity local rotations) is preserved.

    Args:
        skeleton: G1 skeleton instance.
        joints_pos: (T, J, 3) or (B, T, J, 3) joint positions in global space.
        joints_rot: (T, J, 3, 3) or (B, T, J, 3, 3) global rotation matrices.
        clamp_to_limits: If True, clamp joint angles to XML axis limits (default True).

    Returns:
        (posed_joints, global_rot_mats) as tensors, same shape as inputs (batch preserved).
    """

    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
    root_positions = joints_pos[..., skeleton.root_idx, :]

    # Converter expects batch dim (B, T, ...); add and remove if single sequence.
    single_sequence = local_rot_mats.dim() == 4
    if single_sequence:
        local_rot_mats = local_rot_mats.unsqueeze(0)
        root_positions = root_positions.unsqueeze(0)

    converter = MujocoQposConverter(skeleton)
    projected = converter.project_to_real_robot_rotations(
        local_rot_mats, root_positions, clamp_to_limits=clamp_to_limits
    )

    out_pos = projected["posed_joints"]
    out_rot = projected["global_rot_mats"]
    if single_sequence:
        out_pos = out_pos.squeeze(0)
        out_rot = out_rot.squeeze(0)
    return out_pos, out_rot
