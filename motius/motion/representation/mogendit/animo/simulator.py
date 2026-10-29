import pdb

import torch
import numpy as np
from motius.motion.representation.mogendit.animo.utils.angular import *
from motius.motion.representation.mogendit.animo.utils.general import *

from qpsolvers import solve_qp
from motius.motion.representation.mogendit.animo.tools import *
from motius.motion.representation.mogendit.animo.skeleton import AnimoSkeleton
from tqdm import tqdm

# pip install qpsolvers
# pip install qpsolvers[quadprog]


class FlatGroundSimulator:
    # Human Body
    # contact_joint_set = torch.LongTensor([0, 10, 11, 22, 23])
    # adj_contact_joint_set = torch.LongTensor([10, 11, 20, 21])
    # adj_chain = {0: [], 7: [4, 1], 8: [5, 2], 10: [7, 4, 1], 11: [8, 5, 2],
    #              22: [20, 18, 16, 13], 23: [21, 19, 17, 14]}
    adj_chain = {0: [], 7: [4, 1, 0], 8: [5, 2, 0], 10: [7, 4, 1, 0], 11: [8, 5, 2, 0]}
    # force_effect_set = torch.LongTensor([0, 1, 2, 1, 2, 16, 17])
    float_prob = 1

    # Physical World
    G = 9.8
    mu = 0.8

    def __init__(self, skeleton: AnimoSkeleton, fps=30, eps=1e-2):
        # configuration维度 3 position + 72 euler angle
        self.init = False
        self.q_dof = skeleton.q_dof

        # 根据fps进行自适应调整 线速度p的增益需要考虑fps
        self.kp_q, self.kd_q = 1 * fps**2, fps
        self.kp_p, self.kd_p = 1 * fps**2, fps

        # 同理 优化器输出权重也需要调整
        # self.w_qp = 0.5 * min(1.0, fps / 60)
        self.w_qp = 0.5

        # y轴参考速度的权重 设置为0时将完全通过物理规则推导
        self.w_vel_ref = 0.2

        self.dt = 1 / fps
        self.eps = eps

        self.body_model = skeleton
        self.reset_states()

    def set_joint_offsets(self, pose, joint):
        assert pose.shape[-1] == 3 and pose.shape[-2] == 3
        self.body_model.set_joint_offsets(pose, joint)

    def reset_states(self):
        # for angular control
        self.pose = None

        self.q = torch.zeros(self.q_dof)
        self.qdot = torch.zeros(self.q_dof)
        self.qddot = torch.zeros(self.q_dof)
        self.q_bias = torch.zeros(self.q_dof)

        # for local positional control
        self.p = None
        self.pdot = torch.zeros(self.q_dof - 3)
        # self.pddot = np.zeros(self.q_dof - 3)
        self.p_bias = torch.zeros(self.q_dof - 3)

        # for momentum calculation
        self.p_com = torch.zeros(3)
        self.p_com_dot = torch.zeros(3)
        self.p_com_ddot = torch.zeros(3)

        # for global translation simulation
        self.t = torch.zeros(3)
        self.tdot = torch.zeros(3)
        self.tddot = torch.zeros(3)

        self.tdot_ref = torch.zeros(3)

        self.last_x = None
        self.last_t_ref = None
        self.init = False
        self.float_flag = False
        self.contact_joint = []

    def init_state(self, pose, vel=None, trans=None):
        self.pose = pose
        if vel is not None:
            self.tdot = vel[0]
        _, joint = self.body_model.forward_kinematics(pose, calc_joint=True)
        # vel = torch.FloatTensor(self.pdot).view_as(vel) + vel
        # joint += vel
        height = self.get_minimal_height(joint)
        self.contact_joint = [int(torch.argmin(joint[:, 1]))]

        q_ref = self.body_model.q_encode(pose, torch.zeros(3).to(pose.device)).flatten()
        p_ref = joint.flatten(0)
        self.q = q_ref
        self.p = p_ref
        self.p_com = self.body_model.calc_com_position(
            p_ref.reshape(self.body_model.n_joint, 3)
        )

        self.t = torch.FloatTensor([0, height, 0]).to(pose.device)
        if trans is not None:
            self.trans_offset = trans - self.t
        self.last_t_ref = self.t.clone()

        self.init = True

    def update_state(self, pose, vel=None, trans=None):
        # 1. ======先优化angular motion=======
        # pdb.set_trace()

        if not self.init:
            self.init_state(pose, trans)
            return

        q_ref = self.body_model.q_encode(pose, torch.zeros(3).to(pose.device)).flatten()
        if vel is not None:
            assert vel.shape[0] == 24 and vel.shape[1] == 3
            vel_root = vel[0]
            vel = vel - vel_root.unsqueeze(0)
            p_ref = self.p + vel.flatten(0) * self.dt
        else:
            _, p_ref = self.body_model.forward_kinematics(
                pose, torch.zeros(3).to(pose.device), calc_joint=True
            )
            p_ref = p_ref.flatten()

        # Dual-level pd control
        q_delta = q_ref - self.q
        # pdb.set_trace()
        q_delta[3:] = normalize_angle(q_delta[3:])
        # pdb.set_trace()
        des_qddot = self.kp_q * (q_delta) - self.kd_q * self.qdot
        # pdb.set_trace()
        des_pddot = self.kp_p * (p_ref - self.p) - self.kd_p * self.pdot

        des_qdot = self.qdot + des_qddot * self.dt
        # des_qdot = q_delta / self.dt
        des_pdot = self.pdot + des_pddot * self.dt

        # 计算当前姿态（还未更新）的雅各比矩阵
        Js = np.array(self.body_model.get_Jacobian(R=self.pose.squeeze(0).cpu()))

        Js[:, :3] *= 0
        # minimize   ||A1 * q_dot - b1||^2     for A1, b1 in zip(As1, bs1)
        As1, bs1 = [np.zeros((0, self.q_dof))], [np.empty(0)]

        A_, b_ = None, None

        # joint position controller (using joint velocity to determine target joint position)
        if True:
            A = Js
            b = np.array(des_pdot.cpu())
            As1.append(A)
            bs1.append(b)

        # # joint rotation controller (using joint velocity to determine target joint position)
        # if True:
        #     A2 = np.hstack((np.zeros((self.q_dof - 3, 3)), np.eye((self.q_dof - 3))))
        #     b2 = des_qdot[3:]
        #     As1.append(A2 * 1)  # 72 * 75
        #     bs1.append(b2 * 1)  # 72

        # pdb.set_trace()

        As1, bs1 = np.vstack(As1), np.concatenate(bs1)

        # 注意 这里不用手动乘1/2 算法包内部会自动乘
        P_ = block_diagonal_matrix_np([np.dot(As1.T, As1)])
        q_ = -np.dot(As1.T, bs1)
        # 正则项 防止非正定
        P_ += np.eye(P_.shape[0]) * self.eps

        # fast solvers are less accurate/robust, and may fail

        init = des_qdot
        x = solve_qp(P_, q_, solver="quadprog", initvals=None)

        if x is None or np.linalg.norm(x) > 100:
            x = init
        qdot_qp = torch.FloatTensor(x[: self.q_dof]).to(self.q.device)

        qdot_fusion = qdot_qp * self.w_qp + des_qdot * (1 - self.w_qp)
        q_fusion = self.q + qdot_fusion * self.dt
        q_fusion[3:] = normalize_angle(q_fusion[3:])
        q_fusion[:3] *= 0
        pose_optim, _ = self.body_model.q_decode(q_fusion)

        _, p_optim = self.body_model.forward_kinematics(pose_optim, calc_joint=True)

        # 计算优化后姿态的重心
        p_com = self.body_model.calc_com_position(
            p_optim.reshape(self.body_model.n_joint, 3)
        )
        # pdb.set_trace()
        p_com_dot = (p_com - self.p_com) / self.dt
        p_com_ddot = (p_com_dot - self.p_com_dot) / self.dt

        # 2. ======优化完动作后, 再处理全局位移=======

        # 我们假设物理环境是纯平面
        tdot_fusion = self.tdot.clone()
        # 上一时刻触地
        if self.t[1] <= self.get_minimal_height(
            self.p.reshape(self.body_model.n_joint, 3)
        ):
            # 支持关节判断
            self.float_flag = False
            contact_joint_idx = self.contact_judge(p_com)
            # 垂直动量计算 这里混入参考tdot来计算 缓解难以跳起的情况
            # total_momentum = ((0.5 * (self.tdot[1] + tdot_ref[1]) + self.p_com_dot[:, 1])).sum()
            total_y_momentum = self.tdot[1] + self.p_com_dot[1]
            gravity_impulse = self.G * self.dt
            # 垂直动量不足以起跳
            if total_y_momentum - gravity_impulse <= 0:
                com_tdot_fusions = []
                for i in range(len(contact_joint_idx)):
                    tdot = (
                        self.p.reshape(self.body_model.n_joint, 3)[contact_joint_idx[i]]
                        - p_optim.reshape(self.body_model.n_joint, 3)[
                            contact_joint_idx[i]
                        ]
                    ) / self.dt
                    tddot = (tdot - self.tdot) / self.dt  # 静摩擦的加速度

                    max_tddot_xz_value = (
                        tddot[1] + p_com_ddot[1] + self.G
                    ).abs() * self.mu  # 物理环境可提供的极限水平加速度
                    com_tddot_xz_value = torch.norm(
                        tddot[[0, 2]] + p_com_ddot[[0, 2]], dim=-1, p=2
                    )

                    if com_tddot_xz_value > max_tddot_xz_value:
                        tddot[[0, 2]] = (tddot[[0, 2]] + p_com_ddot[[0, 2]]) * (
                            max_tddot_xz_value / (com_tddot_xz_value + 1e-5)
                        ) * 0.8 - p_com_ddot[[0, 2]]

                    com_tdot_fusions.append(tdot_fusion + tddot * self.dt)
                com_tdot_fusions = torch.stack(com_tdot_fusions)
                tdot_fusion = com_tdot_fusions.mean(dim=0)
            else:
                # print('jump')
                fk_delta_h = self.get_minimal_height(
                    p_optim.reshape(self.body_model.n_joint, 3)
                ) - self.get_minimal_height(self.p.reshape(self.body_model.n_joint, 3))
                freefall_delta_h = float((self.tdot[1] - self.G * self.dt) * self.dt)
                if abs(freefall_delta_h) > abs(fk_delta_h):
                    tdot_fusion[1] -= self.G * self.dt
                    tdot_fusion -= p_com_ddot * self.dt
                    self.is_float = True
                else:
                    tdot_fusion[1] = fk_delta_h / self.dt
            if not self.float_flag:
                self.contact_joint = contact_joint_idx
            else:
                self.contact_joint = []
        else:
            self.float_flag = True
            self.contact_joint = []
            # 自由落体模拟
            tdot_fusion[1] -= self.G * self.dt
            tdot_fusion -= p_com_dot
            min_height = self.get_minimal_height(
                p_optim.reshape(self.body_model.n_joint, 3)
            )
            if self.t[1] + tdot_fusion[1] * self.dt < self.get_minimal_height(
                p_optim.reshape(self.body_model.n_joint, 3)
            ):
                tdot_fusion[1] = (min_height - self.t[1]) / self.dt

        # updates
        self.pose = pose_optim

        self.qddot = (qdot_fusion - self.qdot) / self.dt
        self.qdot = qdot_fusion
        self.q += qdot_fusion * self.dt
        self.q[3:] = normalize_angle(self.q[3:])

        pdot_fusion = (p_optim.flatten() - self.p) / self.dt
        self.pddot = (pdot_fusion - self.pdot) / self.dt
        self.pdot = pdot_fusion
        self.p += pdot_fusion * self.dt

        self.p_com_ddot = p_com_ddot
        self.p_com_dot = p_com_dot
        self.p_com += p_com_dot * self.dt

        self.tddot = (tdot_fusion - self.tdot) / self.dt
        self.tdot = tdot_fusion
        self.t += tdot_fusion * self.dt

        self.t[1] = max(
            self.get_minimal_height(self.p.reshape(self.body_model.n_joint, 3)),
            self.t[1],
        )

    def update_state_with_vel(self, pose, vel, state_optim=True):
        """
        基于关节速度估计值更新状态的简化版本
        融合运动学计算与关节速度估计值的全局位移更新策略

        Args:
            pose: 当前姿态 [n_joint, 3, 3] 或 [1, n_joint, 3, 3]
            vel: 关节速度估计值 [n_joint, 3]
        """
        if not self.init:
            self.init_state(pose, vel)
            return

        # 确保速度张量格式正确
        if vel is not None:
            assert vel.shape[0] == self.body_model.n_joint and vel.shape[1] == 3, \
                f"速度维度应为({self.body_model.n_joint}, 3)，但得到{vel.shape}"

        # 1. ======姿态优化（简化版本）======
        # 使用与update_state类似的姿态优化逻辑，但更简化
        q_ref = self.body_model.q_encode(pose, torch.zeros(3).to(pose.device)).flatten()

        if vel is not None:
            # 提取根关节速度
            vel_root = vel[0]
            # 计算相对速度（相对于根关节）
            vel_relative = vel - vel_root.unsqueeze(0)
            # 基于速度预测关节位置
            p_ref = self.p + vel_relative.flatten(0) * self.dt
        else:
            # 无速度输入时，使用运动学计算
            _, p_ref = self.body_model.forward_kinematics(
                pose, torch.zeros(3).to(pose.device), calc_joint=True
            )
            p_ref = p_ref.flatten()

        # 简化的PD控制
        q_delta = q_ref - self.q
        q_delta[3:] = normalize_angle(q_delta[3:])
        des_qddot = self.kp_q * q_delta - self.kd_q * self.qdot
        des_pddot = self.kp_p * (p_ref - self.p) - self.kd_p * self.pdot

        des_qdot = self.qdot + des_qddot * self.dt
        des_pdot = self.pdot + des_pddot * self.dt

        # ============
        if state_optim:
            # 计算当前姿态（还未更新）的雅各比矩阵
            Js = np.array(self.body_model.get_Jacobian(R=self.pose.squeeze(0).cpu()))

            Js[:, :3] *= 0
            # minimize   ||A1 * q_dot - b1||^2     for A1, b1 in zip(As1, bs1)
            As1, bs1 = [np.zeros((0, self.q_dof))], [np.empty(0)]

            A_, b_ = None, None

            # joint position controller (using joint velocity to determine target joint position)
            if True:
                A = Js
                b = np.array(des_pdot.cpu())
                As1.append(A)
                bs1.append(b)

            As1, bs1 = np.vstack(As1), np.concatenate(bs1)

            # 注意 这里不用手动乘1/2 算法包内部会自动乘
            P_ = block_diagonal_matrix_np([np.dot(As1.T, As1)])
            q_ = -np.dot(As1.T, bs1)
            # 正则项 防止非正定
            P_ += np.eye(P_.shape[0]) * self.eps


            init = des_qdot
            x = solve_qp(P_, q_, solver="quadprog", initvals=init)

            if x is None or np.linalg.norm(x) > 100:
                x = init
            qdot_qp = torch.FloatTensor(x[: self.q_dof]).to(self.q.device)

            qdot_fusion = qdot_qp * self.w_qp + des_qdot * (1 - self.w_qp)
        else:
            qdot_fusion = des_qdot
        q_fusion = self.q + qdot_fusion * self.dt
        q_fusion[3:] = normalize_angle(q_fusion[3:])
        q_fusion[:3] *= 0

        # ============

        pose_optim, _ = self.body_model.q_decode(q_fusion)
        pose_optim = pose_optim.reshape(self.body_model.n_joint, 3, 3)
        _, p_optim = self.body_model.forward_kinematics(pose_optim, calc_joint=True)

        # # 基于predict_result_optim函数中的策略
        # # 提取左右脚速度（关节10和11）
        # vel_left = (vel[7] + vel[10]) / 2  # 左脚
        # vel_right = (vel[8] + vel[11]) / 2  # 右脚
        # vel_left_hand = vel[20]  # 左手
        # vel_right_hand = vel[21]  # 右手
        # # vel_left = vel[10] # 左脚
        # # vel_right = vel[11] # 右脚

        # # 计算接触概率（基于速度幅值）
        # contact_left = 1 - (torch.norm(vel_left, 2) - 0.02).clamp(min=0, max=0.15) / 0.15
        # contact_right = 1 - (torch.norm(vel_right, 2) - 0.02).clamp(min=0, max=0.15) / 0.15

        # # 计算手部接触概率，加入高度范围约束 [-2.5cm, 2.5cm]
        # hand_left_height = p_optim[20, 1]
        # hand_right_height = p_optim[21, 1]
        # height_range = 0.025  # 2.5cm
        # hand_left_in_range = (hand_left_height >= -height_range) & (hand_left_height <= height_range)
        # hand_right_in_range = (hand_right_height >= -height_range) & (hand_right_height <= height_range)

        # contact_left_hand = (1 - (torch.norm(vel_left_hand, 2) - 0.02).clamp(min=0, max=0.15) / 0.15) * hand_left_in_range.float()
        # contact_right_hand = (1 - (torch.norm(vel_right_hand, 2) - 0.02).clamp(min=0, max=0.15) / 0.15) * hand_right_in_range.float()

        # # print(contact_left, contact_right)

        # vel_left_norm = torch.norm(vel_left, 2)
        # vel_right_norm = torch.norm(vel_right, 2)

        # # 计算接触强度（选择最大值，包括手部）
        # contact_strength = max(contact_left, contact_right, contact_left_hand, contact_right_hand)

        # # 计算FK位移（基于关节位置变化）
        # # 保存上一次关节位置用于FK位移计算
        # if hasattr(self, 'last_joint_pos') and self.last_joint_pos is not None:
        #     # 选择速度较小的脚作为参考
        #     if vel_left_norm < vel_right_norm:
        #         d_trans_fk = (self.last_joint_pos[[7, 10]] - p_optim[[7, 10]]).mean(dim=0)
        #     else:
        #         d_trans_fk = (self.last_joint_pos[[8, 11]] - p_optim[[8, 11]]).mean(dim=0)
        # else:
        #     d_trans_fk = torch.zeros(3).to(pose.device)

        # 获取关节半径（如果存在）
        if hasattr(self.body_model, 'joint_radii'):
            joint_radii = self.body_model.joint_radii.to(p_optim.device)
        else:
            joint_radii = torch.zeros(self.body_model.n_joint, device=p_optim.device)
        # 确保半径张量与关节数匹配
        if len(joint_radii) < self.body_model.n_joint:
            # 填充缺失的半径为零
            padding = torch.zeros(self.body_model.n_joint - len(joint_radii), device=p_optim.device)
            joint_radii = torch.cat([joint_radii, padding])
        elif len(joint_radii) > self.body_model.n_joint:
            # 截断多余的半径
            joint_radii = joint_radii[:self.body_model.n_joint]
        vel_norm = torch.norm(vel, 2, dim=-1)
        stationary = 1 - (vel_norm - 0.02).clamp(min=0, max=0.15) / 0.15
        joint_bottom_heights = p_optim[:, 1] - joint_radii + self.t[1]
        in_contact_range = (joint_bottom_heights >= -0.1) & (joint_bottom_heights <= 0.1)
        contact = stationary * in_contact_range
        # import pdb; pdb.set_trace()
        if in_contact_range.any():
            vel_norm += ~in_contact_range * 100

        # 计算接触强度（选择最大值，包括手部）
        contact_strength = max(contact)

        # 计算FK位移（基于关节位置变化）
        # 保存上一次关节位置用于FK位移计算
        if hasattr(self, 'last_joint_pos') and self.last_joint_pos is not None:
            # 选择速度较小的脚作为参考
            contact_joint_idx = torch.argmin(vel_norm)
            if contact_joint_idx in [7, 10]:
                d_trans_fk = (self.last_joint_pos[[7, 10]] - p_optim[[7, 10]]).mean(dim=0)

            elif contact_joint_idx in [8, 11]:
                d_trans_fk = (self.last_joint_pos[[8, 11]] - p_optim[[8, 11]]).mean(dim=0)
            # 使用接触最强的关节索引
            else:
                d_trans_fk = (self.last_joint_pos[contact_joint_idx] - p_optim[contact_joint_idx])
        else:
            d_trans_fk = torch.zeros(3).to(pose.device)

        # 保存当前关节位置用于下一帧
        self.last_joint_pos = p_optim.clone()

        # 计算NN位移（基于根关节速度）
        root_vel = vel[0]  # 根关节速度
        d_trans_nn = root_vel * self.dt  # 根关节速度乘以时间步长
        d_root_height_nn = d_trans_nn[1]  # NN高度变化

        # 计算FK高度（相对于最低关节的高度，考虑关节半径）

        # 计算每个关节的底部高度（y坐标减去半径）
        joint_bottom_heights = p_optim[:, 1] - joint_radii
        lowest_bottom_height, _ = torch.min(joint_bottom_heights, dim=-1)
        root_height_fk = p_optim[0][1] - lowest_bottom_height

        # 初始化浮空概率（如果需要）
        if not hasattr(self, 'floating_prob'):
            self.floating_prob = 0.0

        # 根据接触状态更新浮空概率
        if contact_strength < 0.1:  # 接触较弱，增加浮空概率
            self.floating_prob = min(self.floating_prob + 0.33, 1.0)
            d_trans = d_trans_nn  # 使用NN位移
        else:  # 接触较强，减少浮空概率
            self.floating_prob = max(self.floating_prob - 0.33, 0.0)
            # 混合位移：根据浮空概率在FK和NN之间混合
            d_trans = self.floating_prob * d_trans_nn + (1 - self.floating_prob) * d_trans_fk
            # d_trans = d_trans_nn

        # 更新根关节高度（混合FK高度和NN高度）
        if not hasattr(self, 'root_height'):
            self.root_height = root_height_fk

        # import pdb; pdb.set_trace()

        self.root_height = (1 - self.floating_prob) * root_height_fk + \
                            self.floating_prob * (self.root_height + d_root_height_nn)
        self.root_height = max(root_height_fk, self.root_height)

        # 更新全局位移
        tdot_fusion = torch.zeros_like(self.tdot)
        tdot_fusion[[0, 2]] = d_trans[[0, 2]] / self.dt  # XZ平面位移
        tdot_fusion[1] = (self.root_height - self.t[1]) / self.dt if self.t[1] != 0 else 0

        # 3. ======状态更新======
        self.pose = pose_optim

        # 角度状态更新
        self.qddot = (des_qdot - self.qdot) / self.dt
        self.qdot = des_qdot
        self.q = q_fusion

        # 位置状态更新
        pdot_fusion = (p_optim.flatten() - self.p) / self.dt
        self.pddot = (pdot_fusion - self.pdot) / self.dt
        self.pdot = pdot_fusion
        self.p = p_optim.flatten()

        # 全局位移更新
        self.tddot = (tdot_fusion - self.tdot) / self.dt
        self.tdot = tdot_fusion
        self.t += tdot_fusion * self.dt

        # 确保不穿地
        self.t[1] = self.root_height

        # 初始化last_joint_pos（如果需要）
        if not hasattr(self, 'last_joint_pos'):
            self.last_joint_pos = p_optim.clone()

    def get_state(self):
        return self.pose.clone(), self.t.clone()

    def get_minimal_height(self, p):
        lowest_position = p[:, 1].min()
        return float(p[0][1] - lowest_position)

    def contact_judge(self, p_com, vel=None):
        """
        判断质心投影是否被支撑点的支撑形包围，筛选有效支撑点集合
        新增逻辑：当传入vel时，将候选支撑点中速度数值最低的关节视为唯一接触点

        Args:
            p_com: 质心坐标 [3,]（x, y, z 全球坐标系）
            vel: 可选，关节速度张量 [num_joints, 3]（全球坐标系下每个关节的速度），
                传入时会优先选择候选支撑点中速度模长最小的关节作为唯一接触点

        Returns:
            contact_idx: 有效支撑点索引列表（满足支撑条件）
        """
        # 1. 提取关节全球坐标系下的 y 高度（离地高度）和 x/z 坐标
        global_joint_pos = self.p.reshape(-1, 3) + self.t.unsqueeze(
            0
        )  # shape: [num_joints, 3]（x,y,z）
        global_joint_height = global_joint_pos[:, 1]  # 关节y坐标（离地高度）
        global_joint_xz = global_joint_pos[:, [0, 2]]  # 关节x-z平面坐标（支撑判断用）

        # 规则一：筛选离地低于5cm的候选支撑点
        candidate_mask = global_joint_height < 0.05  # 布尔掩码
        candidate_idx = torch.where(candidate_mask)[0].tolist()  # 候选支撑点索引

        # ========= 新增逻辑：vel输入时选择速度最低的候选关节作为唯一接触点 ==========
        if vel is not None and len(candidate_idx) > 0:
            # 校验速度张量维度与关节位置匹配
            assert (
                vel.shape == global_joint_pos.shape
            ), f"vel维度{vel.shape}必须与关节位置维度{global_joint_pos.shape}一致"
            # 确保vel与关节位置在同一设备（CPU/GPU）
            vel = vel.to(global_joint_pos.device)

            # 提取候选关节的速度，并计算每个关节的速度模长（数值大小）
            candidate_vels = vel[candidate_idx]  # [num_candidate, 3]
            vel_norms = torch.norm(
                candidate_vels, dim=1
            )  # [num_candidate,] 每个候选关节的速度幅值

            # 找到速度幅值最小的候选关节（唯一接触点）
            min_vel_idx_in_candidate = torch.argmin(vel_norms).item()
            min_vel_joint_idx = candidate_idx[min_vel_idx_in_candidate]

            # 返回唯一的接触点索引
            return [min_vel_joint_idx]
        # ============================================================================

        # 无vel输入时，执行原支撑形包围判断逻辑
        if len(candidate_idx) <= 1:
            return candidate_idx  # 不足2个候选点，直接返回（无法形成支撑形）

        # 规则二：按离地高度升序排序（越低的支撑点优先级越高）
        candidate_sorted_idx = sorted(
            candidate_idx, key=lambda idx: global_joint_height[idx]
        )

        # 支撑参数：每个支撑点的支撑半径（6cm）
        support_radius = 0.06

        # 3. 逐次加入支撑点，判断质心投影是否被包围
        contact_idx = []
        p_com_xz = p_com[[0, 2]]  # 质心x-z平面投影（判断目标）
        num_candidates = len(candidate_sorted_idx)

        for i in range(num_candidates):
            # 加入当前优先级最高的支撑点
            current_idx = candidate_sorted_idx[i]
            contact_idx.append(current_idx)

            # 若仅1个支撑点：判断质心投影是否在该点的支撑圆内
            if len(contact_idx) == 1:
                contact_xz = global_joint_xz[contact_idx[0]]  # 单个支撑点x-z坐标
                distance = torch.norm(p_com_xz - contact_xz)  # 质心到支撑点的欧氏距离
                if distance <= support_radius:
                    return contact_idx  # 被单个支撑圆包围，返回
                continue

            # 若多个支撑点：简化为「膨胀最小包围矩形」判断（高效且满足需求）
            # 3.1 计算当前支撑点集合的x-z最小包围矩形（MBR）
            contact_xz = global_joint_xz[contact_idx]  # [num_contact, 2]
            mbr_x_min = contact_xz[:, 0].min() - support_radius  # 向左膨胀
            mbr_x_max = contact_xz[:, 0].max() + support_radius  # 向右膨胀
            mbr_z_min = contact_xz[:, 1].min() - support_radius  # 向前膨胀
            mbr_z_max = contact_xz[:, 1].max() + support_radius  # 向后膨胀

            # 3.2 判断质心投影是否在膨胀后的MBR内
            in_mbr = (
                (p_com_xz[0] >= mbr_x_min)
                & (p_com_xz[0] <= mbr_x_max)
                & (p_com_xz[1] >= mbr_z_min)
                & (p_com_xz[1] <= mbr_z_max)
            )

            # 若包围则返回当前支撑点集合，否则继续加入下一个候选点
            if in_mbr:
                return contact_idx

            # 4. 遍历完所有候选点仍未包围，返回全部候选点（兜底）
            return candidate_sorted_idx

    def simulate(self, pose, vel=None):
        self.to(pose.device)
        optim_pose = []
        optim_trans = []
        contact_flag = []
        for i in tqdm(range(len(pose))):
            if vel is not None:
                self.update_state(pose=pose[i], vel=vel[i])
            else:
                self.update_state(pose=pose[i])
            _optim_pose, _trans = self.get_state()
            optim_pose.append(_optim_pose.reshape(1, self.body_model.n_joint, 3, 3))
            optim_trans.append(_trans.reshape(1, 3))
            _contact = torch.zeros(self.body_model.n_joint)
            if len(self.contact_joint) > 0:
                _contact[self.contact_joint] = 1
            contact_flag.append(_contact)
        pose = torch.cat(optim_pose, dim=0)
        trans = torch.cat(optim_trans, dim=0)
        contact_flag = torch.stack(contact_flag, dim=0)
        return pose, trans, contact_flag

    def to(self, device: torch.device):
        """
        将类中所有Tensor移动到指定设备（GPU/CPU）
        """
        # 调用通用递归函数，处理自身所有属性
        move_tensor_to_device(self, device)


def move_tensor_to_device(obj, device: torch.device):
    """
    递归遍历对象，将所有torch.Tensor移动到指定设备（GPU/CPU）
    支持的对象类型：Tensor、列表、元组、字典、集合、自定义类实例等
    """
    # 1. 处理Tensor：直接移动
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)  # non_blocking加速GPU拷贝

    # 2. 处理自定义类实例：遍历__dict__中的属性
    elif hasattr(obj, "__dict__"):
        for attr_name, attr_value in obj.__dict__.items():
            # 跳过内置属性（避免修改__class__/__dict__等）
            if attr_name.startswith("__"):
                continue
            # 递归处理属性值，并重新赋值
            obj.__dict__[attr_name] = move_tensor_to_device(attr_value, device)
        return obj

    # 3. 处理列表（可变）
    elif isinstance(obj, list):
        return [move_tensor_to_device(item, device) for item in obj]

    # 4. 处理元组（不可变，转列表处理后转回元组）
    elif isinstance(obj, tuple):
        return tuple(move_tensor_to_device(item, device) for item in obj)

    # 5. 处理字典（遍历key-value）
    elif isinstance(obj, dict):
        return {k: move_tensor_to_device(v, device) for k, v in obj.items()}

    # 6. 处理集合（可变）
    elif isinstance(obj, set):
        return {move_tensor_to_device(item, device) for item in obj}

    # 7. 其他类型（int/str/float等）：直接返回
    else:
        return obj
