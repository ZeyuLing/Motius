import torch

import numpy as np

from motius.motion.representation.mogendit.articulate.math.angular import r6d_norm

from motius.motion.representation.mogendit.articulate.math.angular import r6d_to_rotation_matrix

from motius.motion.representation.mogendit.articulate.math.angular import rotation_matrix_to_r6d

from motius.motion.representation.mogendit.rotation_conversions import euler_angles_to_matrix

from motius.motion.representation.mogendit.animo.skeleton.smpl_body import AnimoSMPLBody

from motius.motion.representation.mogendit.utils import get_ego_gv

import random

import time

class OccamMotionRep:
    def __init__(self, keep_hand=False, global_pose=True, fps=30):
        self.n_joint = 22
        assert keep_hand == False, "OccamMotionRep 手部关节支持还在开发中"
        self.global_pose = global_pose

        d_pose = self.n_joint * 6
        d_joint = self.n_joint * 3
        d_trans = 3

        self.fps = fps

        self.data_dim = d_pose + d_joint + d_trans
        self.pose_mask, self.joint_mask, self.trans_mask = (
            torch.zeros(self.data_dim, dtype=torch.bool) for _ in range(3)
        )
        self.pose_mask[:d_pose] = True
        self.pose_wo_hand_mask = self.pose_mask.clone()
        self.pose_wo_hand_mask[20*6:] = False
        self.joint_mask[d_pose : d_pose + d_joint] = True
        self.trans_mask[d_pose + d_joint : d_pose + d_joint + d_trans] = True

        self.mask_dict = {
            "pose": self.pose_mask,
            "joint": self.joint_mask,
            "trans": self.trans_mask,
        }
        self.body_model = AnimoSMPLBody()

    def encode(self, pose: torch.Tensor, joint: torch.Tensor, trans: torch.Tensor):
        """
        pose: (T, self.n_joint, 3, 3) rotation matrix representation
        joint: (T, self.n_joint/22, 3) 3d global joint positions
        trans: (T, 3) 3d root joint translations
        """
        assert (
            pose.ndim == 4
            and pose.shape[1] == self.n_joint
            and pose.shape[2] == 3
            and pose.shape[3] == 3
        ), f"pose shape {pose.shape} is not correct"
        assert (
            joint.ndim == 3 and joint.shape[1] == self.n_joint and joint.shape[2] == 3
        ), f"joint shape {joint.shape} is not correct"
        assert (
            trans.ndim == 2 and trans.shape[1] == 3
        ), f"trans shape {trans.shape} is not correct"

        if self.global_pose:
            pose = self.body_model.forward_kinematics(pose)
        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint, 6)

        pose = pose.flatten(1)
        joint = joint.flatten(1)
        trans = trans.flatten(1)

        motion = torch.cat([pose, joint, trans], dim=-1)

        return motion

    def encode_batch(
        self, pose: torch.Tensor, joint: torch.Tensor, trans: torch.Tensor
    ):
        """
        处理 batch 数据的编码函数，输入增加 batch 维度
        pose: (B, T, self.n_joint, 3, 3) 旋转矩阵表示，B 为 batch size，T 为时间步
        joint: (B, T, n_ric, 3) 3D 全局关节位置（n_ric 为 self.n_joint 或 22）
        """
        # 维度和形状检查（增加对 batch 维度的支持）
        assert (
            pose.ndim == 5
            and pose.shape[2] == self.n_joint
            and pose.shape[3] == 3
            and pose.shape[4] == 3
        ), f"pose 形状 {pose.shape} 不正确，应为 (B, T, {self.n_joint}, 3, 3)"
        assert (
            joint.ndim == 4 and joint.shape[2] == self.n_joint and joint.shape[3] == 3
        ), f"joint 形状 {joint.shape} 不正确，应为 (B, T, {self.n_joint}, 3)"
        assert (
            trans.ndim == 3 and trans.shape[2] == 3
        ), f"trans 形状 {trans.shape} 不正确，应为 (B, T, 1, 3)"

        # 如果需要全局姿态，对每个 batch 中的每个时间步计算前向运动学
        if self.global_pose:
            # 注意：需确保 body_model.forward_kinematics 支持 batch 输入
            # 若原函数不支持，可能需要遍历 batch 或调整实现
            pose = self.body_model.forward_kinematics(pose)

        # 将旋转矩阵转换为 6D 表示（保持 batch 和时间维度）
        # rotation_matrix_to_r6d 需支持 (B, T, self.n_joint, 3, 3) 输入，输出 (B, T, self.n_joint, 6)
        pose = rotation_matrix_to_r6d(pose).reshape(
            pose.shape[0], pose.shape[1], self.n_joint, 6
        )

        # 展平特征维度（保留 batch 和时间维度）
        pose = pose.flatten(2)  # (B, T, self.n_joint*6) = (B, T, 144)
        joint = joint.flatten(2)  # (B, T, n_ric*3)
        trans = trans.flatten(2)  # (B, T, 1*3) = (B, T, 3)

        # 在特征维度拼接所有信息
        motion = torch.cat([pose, joint, trans], dim=-1)  # 形状: (B, T, 总特征数)

        return motion

    def decode(self, motion: torch.Tensor):
        """
        返回:
          pose: (T, self.n_joint, 6) 6d 表示
          joint: (T, self.n_joint/22, 3) 3d joint positions
          vel: (T, self.n_joint/22, 3) 3d joint velocities
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        T = motion.shape[0]
        # 直接分割出非 padding 部分
        x = motion[:, : self.data_dim]  # (T, data_dim)

        # 还原顺序：pose(self.n_joint*6) -> joint(n_ric*3) -> vel(n_ric*3)
        pose_flat = x[:, self.pose_mask.to(x.device)]
        joint_flat = x[:, self.joint_mask.to(x.device)]
        trans_flat = x[:, self.trans_mask.to(x.device)]

        # 重构形状
        pose = pose_flat.view(T, self.n_joint, 6)
        pose = r6d_to_rotation_matrix(pose).reshape(T, self.n_joint, 3, 3)
        if self.global_pose:
            pose = self.body_model.inverse_kinematics(pose)
        joint = joint_flat.view(T, self.n_joint, 3)
        trans = trans_flat.view(T, 3)

        return pose, joint, trans

    def normalization(self, motion: torch.Tensor, ref_idx=0, height_reset=False):
        motion = motion.clone()
        """
        对数据进行标准化, 保障ref_idx的motion数据帧面朝z轴, 且位于x-z平面原点
        ref_idx默认为0 (起始帧)
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        if ref_idx < 0:
            ref_idx = len(motion) + ref_idx

        T = motion.shape[0]

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[:, self.pose_mask.to(motion.device)]
        ).reshape(T, self.n_joint, 3, 3)
        R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
        if not self.global_pose:
            pose[:, 0] = R_ego_gv_inv.matmul(pose[:, 0])
        else:
            pose = R_ego_gv_inv.matmul(pose)

        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

        # 提提取并处理joint vel trans
        joint = motion[:, self.joint_mask.to(motion.device)].reshape(
            -1, self.n_joint, 3
        )
        trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3)

        global_joint = joint + trans
        global_joint[:, :, [0, 2]] -= global_joint[ref_idx : ref_idx + 1, :1, [0, 2]]
        global_joint = R_ego_gv_inv.matmul(global_joint.unsqueeze(-1)).view_as(
            global_joint
        )
        if height_reset:
            # 重置高度 让最低点为0
            init_h = global_joint[:60, :, 1].min()
            global_joint[:, :, 1] -= init_h

        trans = global_joint[:, 0]
        joint = global_joint - global_joint[:, [0]]

        # 覆写
        motion[:, self.pose_mask] = pose.flatten(1)
        motion[:, self.joint_mask] = joint.flatten(1)
        motion[:, self.trans_mask] = trans.flatten(1)

        return motion

    def pre_stitch(
        self,
        motion: torch.Tensor,
        ref_motion: torch.Tensor,
        ref_idx=0,
        stitch_joint_idx=None,
        reset_height=False,
        sync_skeleton=False,
    ):
        motion = motion.clone()
        """
        将数据与参考帧进行预缝合
        """
        if motion.dim() != 2:
            raise ValueError(f"hm263x_motion must be 2D batch, got {motion.shape}")

        if ref_idx < 0:
            ref_idx = len(motion) + ref_idx

        if reset_height:
            trans_reset_axis = [0, 2]
        else:
            trans_reset_axis = [0, 1, 2]

        T = motion.shape[0]

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[:, self.pose_mask.to(motion.device)]
        ).reshape(T, self.n_joint, 3, 3)

        ref_motion = ref_motion.to(motion.device)
        # decode出来的pose固定是local pose
        ref_pose, ref_joint, _ = self.decode(ref_motion)

        # 把motion的骨架与ref的骨架进行同步
        if sync_skeleton:
            # 这里用的ref_pose是decode的pose 默认为local 所以global_pose=False
            ref_skel_offsets = self.body_model.get_skeleton_offsets(
                pose=ref_pose, joint=ref_joint, global_pose=False
            )
            if self.global_pose:
                fk = self.body_model.joint_fk_global
            else:
                fk = self.body_model.joint_fk_local
            joint_sync = fk(R=pose.clone(), skeleton_offsets=ref_skel_offsets)
            motion[:, self.joint_mask.to(motion.device)] = joint_sync.reshape(
                -1, self.n_joint * 3
            )
        # import pdb

        # pdb.set_trace()

        # global joint x-z初始值清零 朝向对齐
        R_ego_gv_inv = get_ego_gv(pose[ref_idx, 0]).transpose(-2, -1)
        R_ego_gv_ref = get_ego_gv(ref_pose[0, 0])
        ori_transform = R_ego_gv_ref.matmul(R_ego_gv_inv)

        if not self.global_pose:
            pose[:, 0] = ori_transform.matmul(pose[:, 0])
        else:
            pose = ori_transform.matmul(pose)

        pose = rotation_matrix_to_r6d(pose).reshape(-1, self.n_joint * 6)

        joint = (
            motion[:, self.joint_mask.to(motion.device)]
            .reshape(-1, self.n_joint, 3)
            .clone()
        )
        trans = motion[:, self.trans_mask.to(motion.device)].reshape(-1, 1, 3).clone()
        trans[:, :, [0, 2]] -= trans[:1, :, [0, 2]].clone()

        global_joint = joint + trans
        global_joint = ori_transform.matmul(global_joint.unsqueeze(-1)).squeeze(-1)

        # 方向已对齐 初始xz位移置0

        global_joint_ref = (
            self.get_component(ref_motion, "joint").reshape(-1, self.n_joint, 3).clone()
            + self.get_component(ref_motion, "trans").reshape(-1, 1, 3).clone()
        )

        # import pdb

        # pdb.set_trace()

        if stitch_joint_idx is None:
            stitch_joint_idx = torch.argmin(
                global_joint[0, :, 1]
            )  # 取ref的最低点作为对齐关节

        global_joint += (
            global_joint_ref[:1, [stitch_joint_idx]]
            - global_joint[:1, [stitch_joint_idx]]
        ).clone()

        if reset_height:
            # pass
            init_height = global_joint[0, 0, 1] - global_joint[0, :, 1].min()
            ground_ref = ref_joint[
                :60, :, 1
            ].min()  # 假设ref的最低点触地 其值视为地面高度
            global_joint[:, :, 1] -= global_joint[
                :1, [0], 1
            ].clone()  # motion初始高度置0
            global_joint[:, :, 1] += init_height + ground_ref  # 地面为ref时的高度

        trans = global_joint[:, 0].clone()
        joint = global_joint - global_joint[:, [0]]

        # 覆写
        motion[:, self.pose_mask] = pose.flatten(1)
        motion[:, self.joint_mask] = joint.flatten(1)
        motion[:, self.trans_mask] = trans.flatten(1)

        return motion

    def get_component(self, motion, component_name: str):
        assert component_name in ["pose", "joint", "trans"]
        return motion[..., self.mask_dict[component_name].to(motion.device)]
