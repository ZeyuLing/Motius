import torch
import numpy as np
from motius.motion.representation.mogendit.animo.utils.rotation_conversions import *
from motius.motion.representation.mogendit.animo.utils.angular import *


class AnimoSkeleton:
    def __init__(self):
        self.kinematic_tree = None
        self.joint_offset = None  # 每个节点到其父节点的距离
        self.joint_name_dict = {}

    def get_joint_num(self):
        """
        根据joint_idx_dict计算joint数量
        Returns:

        """
        joint_idx = []
        for k, layer in self.kinematic_tree.items():
            for edge in layer:
                for node in edge:
                    if node not in joint_idx:
                        joint_idx.append(node)
        # print(joint_idx)
        return len(joint_idx)

    def kinematic_params_init(self):
        self.n_joint = self.get_joint_num()
        self.q_dof = 3 + self.n_joint * 3
        # 父节点-子节点映射, 用于IK
        self.pc_mapping = []
        # 分层次的父节点-子节点映射, 用于FK
        self.layered_pc_mapping = {}
        #
        self.propagation_matrix = torch.zeros(self.n_joint, self.n_joint)
        for k, v in self.kinematic_tree.items():
            self.pc_mapping += v
            v = torch.LongTensor(v)
            p_id = np.array(v[:, 0]).tolist()
            c_id = np.array(v[:, 1]).tolist()
            self.layered_pc_mapping.update({k: [p_id, c_id]})
            self.propagation_matrix[c_id, p_id] += 1
            self.propagation_matrix[c_id] += self.propagation_matrix[p_id]
        self.pc_mapping = torch.LongTensor(self.pc_mapping)
        self.pc_mapping = [
            np.array(self.pc_mapping[:, 0]).tolist(),
            np.array(self.pc_mapping[:, 1]).tolist(),
        ]
        self._symmetrize_joint_offsets()

        # Jacobian相关
        self.propagation_matrix = self.propagation_matrix == 1
        self.propagation_link = [[]]
        for i in range(1, self.n_joint):
            self.propagation_link.append(
                np.array(
                    torch.nonzero(self.propagation_matrix[i], as_tuple=True)[0]
                ).tolist()
            )
        self.joint_end_idx, self.joint_through_idx = [], []
        for i in range(1, self.n_joint):
            self.joint_end_idx += [i for _ in range(len(self.propagation_link[i]))]
            # 速度传递途径的关节点
            self.joint_through_idx += self.propagation_link[i]

        # 质心计算相关
        self.parent_child_pairs = []
        for layer in self.kinematic_tree.values():
            self.parent_child_pairs.extend(layer)
        self.parent_child_pairs = torch.tensor(
            self.parent_child_pairs, dtype=torch.long
        )  # (N_bones, 2)

        # 2. 骨骼密度配置（不变）
        self.bone_density = None
        self.default_density = 1.0
        self.calc_bone_mass()

        # 计算ik_ambiguity_joints和ik_ambiguity_joints_children
        # 1. 首先统计每个关节的直接子节点数量
        self.ik_ambiguity_joints = []
        self.ik_ambiguity_joints_children = []
        child_count = {}
        for layer_edges in self.kinematic_tree.values():
            for p_idx, c_idx in layer_edges:
                if p_idx not in child_count:
                    child_count[p_idx] = []
                child_count[p_idx].append(c_idx)

        # 2. 找出只有一个直接子节点的关节
        for joint_idx, children in child_count.items():
            if len(children) == 1:
                self.ik_ambiguity_joints.append(joint_idx)
                self.ik_ambiguity_joints_children.append(children[0])

    def set_joint_offset(self, pose, joint):
        assert pose.shape[-1] == 3 and pose.shape[-2] == 3
        global_pose = self.forward_kinematics(R=pose).reshape(self.n_joint, 3, 3)
        joint = joint.reshape(self.n_joint, 3)
        for _, edges in self.kinematic_tree.items():
            for edge in edges:
                p_idx, c_idx = edge[0], edge[1]
                self.joint_offset[c_idx] = (
                    global_pose[p_idx]
                    .t()
                    .matmul((joint[c_idx] - joint[p_idx]).unsqueeze(-1))
                    .squeeze(-1)
                )
        self.calc_bone_mass()

    def calc_bone_mass(self):
        bone_mass = []
        for p, c in self.parent_child_pairs:
            p, c = p.item(), c.item()
            offset = self.joint_offset[c]
            bone_length = torch.norm(offset).item()
            if self.bone_density is not None:
                density = self.bone_density[(p, c)]
            else:
                density = self.default_density
            bone_mass.append(bone_length * density)
        self.bone_mass = torch.tensor(bone_mass).view(-1, 1)

    def get_skeleton_offsets(self, pose, joint, global_pose=False, require_grad=False):
        pose = pose.clone()
        n_joint = pose.shape[-3]
        assert pose.shape[-1] == 3 and pose.shape[-2] == 3
        if not global_pose:
            global_pose = self.forward_kinematics(
                R=pose, require_grad=require_grad
            ).reshape(-1, n_joint, 3, 3)
        else:
            global_pose = pose.reshape(-1, n_joint, 3, 3)
        joint = joint.reshape(-1, n_joint, 3)
        joint_offsets = torch.zeros_like(joint)
        for _, pc_map in self.layered_pc_mapping.items():
            p_idx, c_idx = pc_map[0], pc_map[1]
            joint_offsets[:, c_idx] = (
                global_pose[:, p_idx]
                .transpose(-1, -2)
                .matmul((joint[:, c_idx] - joint[:, p_idx]).unsqueeze(-1))
                .squeeze(-1)
            )
        return joint_offsets

    def forward_kinematics(self, R, trans=None, calc_joint=False, require_grad=False):
        R = R.clone()
        if not require_grad:
            R = R.detach()
        if calc_joint:
            return self._forward_kinematics_with_joint(
                R, trans, require_grad=require_grad
            )

        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            R[..., c_idx, :, :] = R[..., p_idx, :, :].matmul(R[..., c_idx, :, :])
        return R

    def _forward_kinematics_with_joint(self, R, trans, require_grad=False):
        R = R.clone()
        if not require_grad:
            R = R.detach()
        # positions n x self.n_joint x 3
        positions = torch.zeros_like(R[..., -1]) + self.joint_offset.to(R.device)

        if trans is not None:
            positions[..., 0, :] += trans

        # n x self.n_joint x 3 x 4
        Rk = torch.cat([R, positions.unsqueeze(-1)], dim=-1)
        padding = torch.zeros_like(Rk[..., [-1], :])
        padding[..., -1] += 1

        # 构建传递矩阵: [[R, pos],
        #              [0,  1]]
        # n x self.n_joint x 4 x 4
        Rk = torch.cat([Rk, padding], dim=-2)

        # 前向运动学
        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            Rk[..., c_idx, :, :] = Rk[..., p_idx, :, :].matmul(Rk[..., c_idx, :, :])

        # 获取global的R与pos
        # n x self.n_joint x 3 x 4
        Rk = Rk[..., :-1, :]
        # n x self.n_joint x 3 x 3
        R = Rk[..., :, :-1]
        # n x self.n_joint x 3
        joint = Rk[..., :, -1]

        return R, joint

    def inverse_kinematics(self, R, require_grad=False):
        R = R.clone()
        if not require_grad:
            R = R.detach()
        p_idx, c_idx = self.pc_mapping[0], self.pc_mapping[1]
        R[..., c_idx, :, :] = (
            R[..., p_idx, :, :].transpose(-2, -1).matmul(R[..., c_idx, :, :])
        )
        return R

    def joint_fk_local(self, R, skeleton_offsets, trans=None, require_grad=False):
        positions = torch.zeros_like(R[..., -1]) + skeleton_offsets.to(R.device)

        if trans is not None:
            positions[..., 0, :] += trans

        # n x 24 x 3 x 4
        Rk = torch.cat([R, positions.unsqueeze(-1)], dim=-1)
        padding = torch.zeros_like(Rk[..., [-1], :])
        padding[..., -1] += 1

        # 构建传递矩阵: [[R, pos],
        #              [0,  1]]
        # n x 24 x 4 x 4
        Rk = torch.cat([Rk, padding], dim=-2)
        if not require_grad:
            Rk = Rk.detach()

        # 前向运动学
        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            Rk[..., c_idx, :, :] = Rk[..., p_idx, :, :].matmul(Rk[..., c_idx, :, :])

        # 获取global的R与pos
        # n x 24 x 3 x 4
        # Rk = Rk[..., :-1, :]
        # n x 24 x 3 x 3
        # R = Rk[..., :, :-1]
        # n x 24 x 3
        joint = Rk[..., :, -1]

        return joint

    def joint_fk_global(self, R, skeleton_offsets, trans=None, require_grad=False):
        positions = torch.zeros_like(R[..., -1]) + skeleton_offsets.to(R.device)
        if trans is not None:
            positions[..., 0, :] += trans
        positions = positions.unsqueeze(-1)  # 转换成列向量
        if not require_grad:
            positions = positions.detach()
            R = R.detach()

        # 前向运动学
        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            positions[..., c_idx, :, :] = positions[..., p_idx, :, :] + R[
                ..., p_idx, :, :
            ].matmul(positions[..., c_idx, :, :])

        return positions.squeeze(-1)

    @torch.no_grad()
    def get_Jacobian(self, R, q_idx=None, p_idx=None):
        J = torch.zeros(self.n_joint, self.n_joint, 3, 3).to(R.device)
        J_root_pos = torch.eye(3).unsqueeze(0).repeat(self.n_joint, 1, 1).to(R.device)

        global_R, global_Joint = self._forward_kinematics_with_joint(R, trans=None)

        r = global_Joint[self.joint_end_idx] - global_Joint[self.joint_through_idx]
        # print(global_R[joint_end_idx, 0].shape, r.shape)
        ds_theta_1 = torch.cross(global_R[self.joint_end_idx, 0], r).unsqueeze(-1)
        ds_theta_2 = torch.cross(global_R[self.joint_end_idx, 1], r).unsqueeze(-1)
        ds_theta_3 = torch.cross(global_R[self.joint_end_idx, 2], r).unsqueeze(-1)
        J_through = torch.cat([ds_theta_1, ds_theta_2, ds_theta_3], dim=-1)
        # print(ds_theta_3.shape)
        J[self.joint_end_idx, self.joint_through_idx] += J_through

        if q_idx is not None:
            n_joint = len(q_idx)
            J_root_pos = J_root_pos[p_idx]
            if p_idx is not None:
                J = J[p_idx][:, q_idx]
                J = J.transpose(1, 2).reshape(len(p_idx) * 3, n_joint * 3)
            else:
                J = J[p_idx][:, q_idx]
                J = J.transpose(1, 2).reshape(n_joint * 3, n_joint * 3)
        else:
            J = J.transpose(1, 2).reshape(self.n_joint * 3, self.n_joint * 3)

        J = torch.cat([J_root_pos.reshape(-1, 3), J], dim=-1)
        return J

    @torch.no_grad()
    def q_encode(self, pose, trans):
        r"""
        Convert smpl poses and translations to robot configuration q. (numpy, batch)

        :param poses: Array that can reshape to [n, 24, 3, 3].
        :param trans: Array that can reshape to [n, 3].
        :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
        """
        poses = pose.reshape(-1, self.n_joint, 3, 3)
        trans = trans.reshape(-1, 3)
        euler_angle = matrix_to_euler_angles(poses.reshape(-1, 3, 3), "XYZ").reshape(
            -1, self.n_joint * 3
        )
        qs = torch.cat([trans, euler_angle], dim=1)
        qs[:, 3:] = normalize_angle(qs[:, 3:])
        return qs

    @torch.no_grad()
    def q_decode(self, q):
        r"""
        Convert robot configuration q to smpl poses and translations. (numpy, batch)

        :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
        :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
        """
        q = q.reshape(-1, self.n_joint * 3 + 3)
        trans, euler_poses = q[:, :3], q[:, 3:]
        poses = euler_angles_to_matrix(euler_poses.reshape(-1, 3).cpu(), "XYZ").reshape(
            -1, self.n_joint, 3, 3
        )
        return poses.to(q.device), trans

    def calc_com_position(self, joint_position):
        """
        并行化（向量化）计算骨骼段质心，兼容原输入接口
        参数:
            joint_position: torch.Tensor，形状(24, 3)，24个关节的世界坐标
        返回:
            com: torch.Tensor，形状(3,)，身体质心坐标
        """
        assert len(joint_position.shape) == 2 and joint_position.shape[1] == 3
        device = joint_position.device

        # ---------------------- 1. 预处理批量数据（并行化核心） ----------------------
        # 将父子对转为Tensor，方便批量索引 (N_bones, 2)，N_bones是骨骼段数量
        parent_child = self.parent_child_pairs.to(
            device
        )  # 提前在__init__中转为Tensor更高效
        parents = parent_child[:, 0]  # (N_bones,) 所有父关节索引
        children = parent_child[:, 1]  # (N_bones,) 所有子关节索引

        # 批量获取所有骨骼段的父/子关节位置 (N_bones, 3)
        parent_pos = joint_position[parents]
        child_pos = joint_position[children]

        # ---------------------- 2. 批量计算骨骼质心 ----------------------
        # 批量计算骨骼段质心（父子关节中点） (N_bones, 3)
        bone_coms = (parent_pos + child_pos) / 2.0

        # ---------------------- 5. 批量累加计算总质心 ----------------------
        # 总质量 (标量)
        total_mass = self.bone_mass.sum().to(device)

        # 总质心 = (Σ 质量×质心) / 总质量 (3,)
        com_sum = (self.bone_mass.to(device) * bone_coms).sum(dim=0)
        com = com_sum / total_mass

        return com
