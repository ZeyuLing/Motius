# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 sonic components; see licenses/sonic."""
from collections import OrderedDict
import copy
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ETree
from easydict import EasyDict
from loguru import logger
from lxml.etree import XMLParser
from lxml.etree import parse
import numpy as np
import scipy.ndimage.filters as filters
import torch
from motius.motion.geometry.sonic.rotations import axis_angle_to_quaternion
from motius.motion.geometry.sonic.rotations import matrix_to_quaternion
from motius.motion.geometry.sonic.rotations import quat_angle_axis
from motius.motion.geometry.sonic.rotations import quat_identity_like
from motius.motion.geometry.sonic.rotations import quat_inverse
from motius.motion.geometry.sonic.rotations import quat_mul_norm
from motius.motion.geometry.sonic.rotations import quaternion_to_matrix
from motius.motion.geometry.sonic.rotations import slerp
from motius.motion.geometry.sonic.rotations import wxyz_to_xyzw
from motius.motion.geometry.sonic.transform import quaternion_to_angle_axis


def dof_to_rotation_matrices(
    dof_angles: torch.Tensor, dof_axis: torch.Tensor
) -> torch.Tensor:
    """Convert DOF angles [..., N] to rotation matrices [..., N, 3, 3]."""
    half_angles = dof_angles / 2
    cos_half, sin_half = (torch.cos(half_angles), torch.sin(half_angles))
    axis = dof_axis.to(dof_angles.device)
    for _ in range(dof_angles.dim() - 1):
        axis = axis.unsqueeze(0)
    axis = axis.expand(*dof_angles.shape, 3)
    quaternion = torch.cat(
        [cos_half.unsqueeze(-1), sin_half.unsqueeze(-1) * axis], dim=-1
    )
    return quaternion_to_matrix(quaternion)


def rotation_matrices_to_dof(
    rotation_matrices: torch.Tensor, dof_axis: torch.Tensor
) -> torch.Tensor:
    """Extract DOF angles [..., N] from rotation matrices [..., N, 3, 3]."""
    R = rotation_matrices
    x_angle = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    y_angle = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    z_angle = torch.atan2(R[..., 1, 0], R[..., 1, 1])
    xyz_angles = torch.stack([x_angle, y_angle, z_angle], dim=-1)
    axis = dof_axis.to(rotation_matrices.device)
    for _ in range(xyz_angles.dim() - 2):
        axis = axis.unsqueeze(0)
    axis = axis.expand(*xyz_angles.shape[:-1], 3)
    return (xyz_angles * axis).sum(dim=-1)


def qpos_to_root_and_dof(qpos: torch.Tensor, num_dof: int, root_quat_wxyz: bool = True):
    """Parse qpos into (root_trans, root_quat_wxyz, dof_angles).

    Args:
        qpos: Joint positions tensor [..., 7 + num_dof].
        num_dof: Number of DOF angles.
        root_quat_wxyz: If True, root quaternion is in wxyz order.

    Returns:
        root_trans: Root translation [..., 3].
        root_quat: Root quaternion in wxyz order [..., 4].
        dof_angles: DOF angles [..., num_dof].
    """
    root_trans = qpos[..., :3]
    root_quat = qpos[..., 3:7]
    dof_angles = qpos[..., 7 : 7 + num_dof]
    if not root_quat_wxyz:
        root_quat = root_quat[..., [3, 0, 1, 2]]
    return (root_trans, root_quat, dof_angles)


def root_and_dof_to_qpos(
    root_trans, root_quat, dof_angles, root_quat_wxyz: bool = True
):
    """Assemble qpos from (root_trans, root_quat_wxyz, dof_angles).

    Args:
        root_trans: Root translation [..., 3].
        root_quat: Root quaternion in wxyz order [..., 4].
        dof_angles: DOF angles [..., num_dof].
        root_quat_wxyz: If True, output root quaternion in wxyz order.

    Returns:
        qpos: Joint positions tensor [..., 7 + num_dof].
    """
    if not root_quat_wxyz:
        root_quat = root_quat[..., [1, 2, 3, 0]]
    return torch.cat([root_trans, root_quat, dof_angles], dim=-1)


def _compute_idx_levels(parents):
    """Group joint indices by their depth level in the kinematic tree.

    This enables parallel FK by processing all joints at the same level simultaneously.

    Args:
        parents: Tensor of parent indices for each joint, where -1 indicates root.

    Returns:
        List of tensors, where idx_levels[i] contains indices of all joints at level i.
    """
    idx_levels = []
    level_dict = {}
    for i in range(len(parents)):
        parent_idx = int(parents[i])
        if parent_idx == -1:
            level_dict[i] = -1
        else:
            parent_level = level_dict[parent_idx]
            level = parent_level + 1
            level_dict[i] = level
            while len(idx_levels) <= level:
                idx_levels.append([])
            idx_levels[level].append(i)
    idx_levels = [torch.tensor(indices, dtype=torch.long) for indices in idx_levels]
    return idx_levels


class Humanoid_Batch:

    def __init__(self, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.asset_root = Path(cfg.asset.assetRoot)
        self.asset_file = cfg.asset.assetFileName
        self.mjcf_file = self.asset_root / self.asset_file
        parser = XMLParser(remove_blank_text=True)
        tree = parse(BytesIO(open(self.mjcf_file, "rb").read()), parser=parser)
        self.dof_axis = []
        joints = sorted(
            [
                j.attrib["name"]
                for j in tree.getroot().find("worldbody").findall(".//joint")
            ]
        )
        motors = sorted(
            [m.attrib["name"] for m in tree.getroot().find("actuator").getchildren()]
        )
        assert len(motors) > 0, "No motors found in the mjcf file"
        self.num_dof = len(motors)
        self.num_extend_dof = self.num_dof
        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]
        self.body_names_augment = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)
        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.actuated_joints_idx = np.array(
            [self.body_names.index(k) for k, v in mjcf_data["body_to_joint"].items()]
        )
        for m in motors:
            if m not in joints:
                print(m)
        if (
            "type" in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
            and tree.getroot().find("worldbody").findall(".//joint")[0].attrib["type"]
            == "free"
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint")[1:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        elif (
            "type" not in tree.getroot().find("worldbody").findall(".//joint")[0].attrib
        ):
            for j in tree.getroot().find("worldbody").findall(".//joint"):
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = True
        else:
            for j in tree.getroot().find("worldbody").findall(".//joint")[6:]:
                self.dof_axis.append([int(i) for i in j.attrib["axis"].split(" ")])
            self.has_freejoint = False
        self.dof_axis = torch.tensor(self.dof_axis)
        for extend_config in cfg.extend_config:
            self.body_names_augment += [extend_config.joint_name]
            self._parents = torch.cat(
                [
                    self._parents,
                    torch.tensor([self.body_names.index(extend_config.parent_name)]).to(
                        device
                    ),
                ],
                dim=0,
            )
            self._offsets = torch.cat(
                [self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim=1
            )
            self._local_rotation = torch.cat(
                [self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)],
                dim=1,
            )
            self.num_extend_dof += 1
        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)
        self.joints_range = mjcf_data["joints_range"].to(device)
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float()
        self._idx_levels = _compute_idx_levels(self._parents)
        self._device = device

    @property
    def device(self):
        """Return the device of the humanoid tensors."""
        return self._device

    def to(self, device):
        """Move all tensors to the specified device."""
        self._device = device
        self._offsets = self._offsets.to(device)
        self._local_rotation = self._local_rotation.to(device)
        self._local_rotation_mat = self._local_rotation_mat.to(device)
        self._parents = self._parents.to(device)
        self.dof_axis = self.dof_axis.to(device)
        self.joints_range = self.joints_range.to(device)
        return self

    def from_mjcf(self, path):
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        xml_joint_root = xml_body_root.find("joint")
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            if len(all_joints) == 6:
                all_joints = all_joints[6:]
            for joint in all_joints:
                if joint.attrib.get("range") is not None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                elif not joint.attrib.get("type") == "free":
                    joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint,
        }

    def _lerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(
        self, times: torch.Tensor, duration: torch.Tensor, input_frames: int
    ) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / duration
        index_0 = (phase * (input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1, torch.tensor(input_frames - 1).to(times.device)
        )
        blend = phase * (input_frames - 1) - index_0
        return (index_0, index_1, blend)

    def interploate_pose(self, pose_quat, trans, fps, target_fps):
        in_shape = trans.shape if pose_quat is None else pose_quat.shape
        device = pose_quat.device if pose_quat is not None else trans.device
        assert in_shape[0] == 1, "Only support single sequence for now"
        duration = (in_shape[1] - 1) * 1 / fps
        times = torch.arange(
            0, duration, 1 / target_fps, dtype=torch.float32, device=device
        )
        index_0, index_1, blend = self._compute_frame_blend(
            times=times, duration=duration, input_frames=in_shape[1]
        )
        if pose_quat is not None:
            pose_quat = self._slerp(pose_quat[0, index_0], pose_quat[0, index_1], blend)
            pose_quat = pose_quat.unsqueeze(0)
        if trans is not None:
            trans = self._lerp(trans[0, index_0], trans[0, index_1], blend.unsqueeze(1))
        trans = trans.unsqueeze(0)
        return (pose_quat, trans)

    def fk_batch(
        self,
        pose,
        trans,
        return_full=False,
        fps=30,
        target_fps=50,
        interpolate_data=False,
        use_parallel_fk: bool = False,
    ):
        device, dtype = (pose.device, pose.dtype)
        B, seq_len = pose.shape[:2]
        pose = pose[..., : len(self._parents), :]
        if (
            self.num_bodies_augment > 0
            and pose.shape[2] < self.num_bodies + self.num_bodies_augment
        ):
            pose = torch.cat(
                [
                    pose,
                    torch.zeros(
                        B,
                        seq_len,
                        self.num_bodies_augment - pose.shape[2],
                        pose.shape[3],
                    ).to(device),
                ],
                dim=2,
            )
        pose_quat = axis_angle_to_quaternion(pose.clone())
        if interpolate_data and fps != target_fps:
            pose_quat, trans = self.interploate_pose(pose_quat, trans, fps, target_fps)
            dt = 1 / target_fps
        else:
            dt = 1 / fps
        pose = quaternion_to_angle_axis(pose_quat)
        B, seq_len = pose.shape[:2]
        pose_mat = quaternion_to_matrix(pose_quat)
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat[:, :, 1:],
            pose_mat[:, :, 0:1],
            trans,
            use_parallel_fk=use_parallel_fk,
        )
        return_dict = EasyDict()
        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
        if len(self.cfg.extend_config) > 0:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(
                    wbody_pos, dt
                )
                return_dict.global_angular_velocity_extend = (
                    self._compute_angular_velocity(wbody_rot, dt)
                )
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            wbody_pos = wbody_pos[..., : self.num_bodies, :]
            wbody_mat = wbody_mat[..., : self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., : self.num_bodies, :]
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[
                ..., 0, :
            ]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            if len(self.cfg.extend_config) > 0:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1 : self.num_bodies]
            elif not len(self.actuated_joints_idx) == len(self.body_names):
                return_dict.dof_pos = pose.sum(dim=-1)[..., self.actuated_joints_idx]
            else:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]
            dof_vel = (return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) / dt
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
            return_dict.fps = int(1 / dt)
        return return_dict

    def forward_kinematics_batch(
        self, rotations, root_rotations, root_positions, use_parallel_fk: bool = False
    ):
        """
        Perform forward kinematics using the given trajectory and local rotations.

        Arguments (where B = batch size, T = sequence length, J = number of joints):
            rotations: (B, T, J-1, 3, 3) tensor of rotation matrices for non-root joints.
            root_rotations: (B, T, 1, 3, 3) tensor of root rotation matrix.
            root_positions: (B, T, 3) tensor describing the root joint positions.
            use_parallel_fk: If True, use level-wise parallel FK (faster for large batches).
                             If False, use sequential FK (original implementation).

        Output:
            positions_world: (B, T, J, 3) world positions of all joints
            rotations_world: (B, T, J, 3, 3) world rotation matrices of all joints
        """
        device, dtype = (root_rotations.device, root_rotations.dtype)
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        expanded_offsets = (
            self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype)
        )
        if use_parallel_fk:
            eye = (
                torch.eye(3, device=device, dtype=dtype)
                .view(1, 1, 1, 3, 3)
                .expand(B, seq_len, 1, 3, 3)
            )
            local_rot_mat = self._local_rotation_mat.to(device, dtype).unsqueeze(1)
            local_transforms = torch.matmul(
                local_rot_mat, torch.cat([eye, rotations], dim=2)
            )
            positions_world = torch.zeros(B, seq_len, J, 3, device=device, dtype=dtype)
            rotations_world = torch.zeros(
                B, seq_len, J, 3, 3, device=device, dtype=dtype
            )
            root_idx = (self._parents == -1).nonzero(as_tuple=True)[0].item()
            positions_world[:, :, root_idx] = root_positions
            rotations_world[:, :, root_idx] = root_rotations[:, :, 0]
            for level_indices in self._idx_levels:
                if len(level_indices) == 0:
                    continue
                level_indices = level_indices.to(device)
                parent_indices = self._parents[level_indices].long().to(device)
                parent_pos = positions_world[:, :, parent_indices]
                parent_rot = rotations_world[:, :, parent_indices]
                local_rot = local_transforms[:, :, level_indices]
                offsets = expanded_offsets[:, :, level_indices]
                world_pos = parent_pos + torch.matmul(
                    parent_rot, offsets.unsqueeze(-1)
                ).squeeze(-1)
                world_rot = torch.matmul(parent_rot, local_rot)
                positions_world[:, :, level_indices] = world_pos
                rotations_world[:, :, level_indices] = world_rot
        else:
            positions_world = []
            rotations_world = []
            for i in range(J):
                if self._parents[i] == -1:
                    positions_world.append(root_positions)
                    rotations_world.append(root_rotations)
                else:
                    try:
                        jpos = (
                            torch.matmul(
                                rotations_world[self._parents[i]][:, :, 0],
                                expanded_offsets[:, :, i, :, None],
                            ).squeeze(-1)
                            + positions_world[self._parents[i]]
                        )
                        rot_mat = torch.matmul(
                            rotations_world[self._parents[i]],
                            torch.matmul(
                                self._local_rotation_mat[:, i : i + 1],
                                rotations[:, :, i - 1 : i, :],
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Error at joint index {i}")
                        logger.error(f"Parent index: {self._parents[i]}")
                        logger.error(f"Error details: {str(e)}")
                    positions_world.append(jpos)
                    rotations_world.append(rot_mat)
            positions_world = torch.stack(positions_world, dim=2)
            rotations_world = torch.cat(rotations_world, dim=2)
        return (positions_world, rotations_world)

    def global_to_local_rotations(self, global_rotations: torch.Tensor) -> torch.Tensor:
        """Convert global rotations to local rotations.

        Args:
            global_rotations: Global rotation matrices [..., J, 3, 3].

        Returns:
            local_rotations: Local rotation matrices [..., J, 3, 3].
        """
        parents = self._parents[: global_rotations.shape[-3]]
        root_mask = parents == -1
        parent_indices = parents.clone()
        parent_indices[root_mask] = 0
        parent_rot = global_rotations[..., parent_indices.long(), :, :]
        local_rotations = torch.matmul(parent_rot.transpose(-1, -2), global_rotations)
        if root_mask.any():
            local_rotations[..., root_mask, :, :] = global_rotations[
                ..., root_mask, :, :
            ]
        return local_rotations

    def qpos_to_global_transforms(
        self,
        qpos: torch.Tensor,
        root_quat_wxyz: bool = True,
        include_extended: bool = False,
        use_parallel_fk: bool = True,
    ):
        """Convert qpos to global positions and rotations.

        Args:
            qpos: Joint positions tensor [..., D].
                  Format: [root_trans(3), root_quat(4), dof_angles(N)].
                  Supports arbitrary leading batch dimensions.
            root_quat_wxyz: If True, root quaternion is in wxyz order.
            include_extended: If True, include extended bodies in output.
            use_parallel_fk: If True, use level-wise parallel FK.

        Returns:
            global_pos: Global positions [..., J, 3].
            global_rot: Global rotation matrices [..., J, 3, 3].
        """
        orig_shape = qpos.shape[:-1]
        D = qpos.shape[-1]
        qpos = qpos.reshape(-1, 1, D)
        flat_B = qpos.shape[0]
        root_trans, root_quat, dof_angles = qpos_to_root_and_dof(
            qpos, self.num_dof, root_quat_wxyz
        )
        root_rot_mat = quaternion_to_matrix(root_quat).unsqueeze(2)
        joint_rot_mat = dof_to_rotation_matrices(dof_angles, self.dof_axis)
        num_extended = self.num_bodies_augment - self.num_bodies
        if num_extended > 0:
            eye = torch.eye(3, device=joint_rot_mat.device, dtype=joint_rot_mat.dtype)
            eye = eye.view(1, 1, 1, 3, 3).expand(flat_B, 1, num_extended, 3, 3)
            joint_rot_mat = torch.cat([joint_rot_mat, eye], dim=2)
        global_pos, global_rot = self.forward_kinematics_batch(
            joint_rot_mat, root_rot_mat, root_trans, use_parallel_fk=use_parallel_fk
        )
        if not include_extended:
            global_pos = global_pos[..., : self.num_bodies, :]
            global_rot = global_rot[..., : self.num_bodies, :, :]
        J_pos = global_pos.shape[-2]
        global_pos = global_pos.squeeze(1).reshape(*orig_shape, J_pos, 3)
        global_rot = global_rot.squeeze(1).reshape(*orig_shape, J_pos, 3, 3)
        return (global_pos, global_rot)

    def global_transforms_to_qpos(
        self,
        global_rotations: torch.Tensor,
        global_positions: torch.Tensor,
        root_quat_wxyz: bool = True,
    ) -> torch.Tensor:
        """Convert global transforms back to qpos.

        Args:
            global_rotations: Global rotation matrices [B, T, J, 3, 3] or [B, J, 3, 3].
            global_positions: Global positions [B, T, J, 3] or [B, J, 3].
            root_quat_wxyz: If True, output root quaternion in wxyz order.

        Returns:
            qpos: Joint positions tensor [B, T, D] or [B, D].
        """
        squeeze_time = global_rotations.dim() == 4
        if squeeze_time:
            global_rotations = global_rotations.unsqueeze(1)
            global_positions = global_positions.unsqueeze(1)
        root_trans = global_positions[..., 0, :]
        local_rotations = self.global_to_local_rotations(global_rotations)
        root_quat = matrix_to_quaternion(local_rotations[..., 0, :, :])
        local_rot_mat = self._local_rotation_mat.to(local_rotations.device)
        joint_rot_mat = torch.matmul(
            local_rot_mat[:, 1 : self.num_bodies].transpose(-1, -2),
            local_rotations[..., 1 : self.num_bodies, :, :],
        )
        dof_angles = rotation_matrices_to_dof(joint_rot_mat, self.dof_axis)
        qpos = root_and_dof_to_qpos(root_trans, root_quat, dof_angles, root_quat_wxyz)
        return qpos.squeeze(1) if squeeze_time else qpos

    def append_extended_transforms(
        self, base_positions: torch.Tensor, base_rotations: torch.Tensor
    ):
        """Append extended body transforms to base body transforms (global frame).

        Computes world-frame transforms for extended bodies from their parent's
        global transforms and concatenates them with base transforms.
        All extended joints have base-body parents, so processed in one batched pass.

        Args:
            base_positions: Base body positions in global frame [..., num_bodies, 3].
            base_rotations: Base body rotation matrices in global frame [..., num_bodies, 3, 3].

        Returns:
            full_positions: All body positions [..., num_bodies_augment, 3].
            full_rotations: All body rotation matrices [..., num_bodies_augment, 3, 3].
        """
        if (
            self.num_bodies_augment - self.num_bodies == 0
            or base_positions.shape[-2] == self.num_bodies_augment
        ):
            return (base_positions, base_rotations)
        assert (
            base_positions.shape[-2] == self.num_bodies
        ), "Must provide the non-extended base body positions"
        device = base_positions.device
        dtype = base_positions.dtype
        offsets = self._offsets[0].to(device, dtype)
        local_rot = self._local_rotation_mat[0].to(device, dtype)
        ext_indices = torch.arange(
            self.num_bodies, self.num_bodies_augment, device=device
        )
        parent_indices = self._parents[ext_indices].long().to(device)
        parent_pos = base_positions[..., parent_indices, :]
        parent_rot = base_rotations[..., parent_indices, :, :]
        ext_offsets = offsets[ext_indices]
        ext_local_rot = local_rot[ext_indices]
        ext_positions = parent_pos + torch.matmul(
            parent_rot, ext_offsets.unsqueeze(-1)
        ).squeeze(-1)
        ext_rotations = torch.matmul(parent_rot, ext_local_rot)
        full_positions = torch.cat([base_positions, ext_positions], dim=-2)
        full_rotations = torch.cat([base_rotations, ext_rotations], dim=-3)
        return (full_positions, full_rotations)

    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(
                filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")
            ).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        return velocity

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        diff_quat_data = quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(
            r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :], w_last=True), w_last=True
        )
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(
                filters.gaussian_filter1d(
                    angular_velocity.numpy(), 2, axis=-3, mode="nearest"
                )
            )
        return angular_velocity
