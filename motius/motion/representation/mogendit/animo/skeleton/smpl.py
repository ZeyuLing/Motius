# SMPL骨架
from motius.motion.representation.mogendit.animo.skeleton.base_skeleton import *
import torch

RIGHT_CHAIN = [2, 5, 8, 11, 14, 17, 19, 21, 23]
LEFT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20, 22]

SYMMETRIC_JOINT_PAIRS = [
    (1, 2),  # 左髋 ↔ 右髋
    (4, 5),  # 左大腿 ↔ 右大腿
    (7, 8),  # 左小腿 ↔ 右小腿
    (10, 11),  # 左足 ↔ 右足
    (13, 14),  # 左肩 ↔ 右肩
    (16, 17),  # 左大臂 ↔ 右大臂
    (18, 19),  # 左小臂 ↔ 右小臂
    (20, 21),  # 左手 ↔ 右手
    (22, 23),  # 左手指 ↔ 右手指
]

KINE_TREE = {
    1: [[0, 1], [0, 2], [0, 3]],
    2: [[1, 4], [2, 5], [3, 6]],
    3: [[4, 7], [5, 8], [6, 9]],
    4: [[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
    5: [[12, 15], [13, 16], [14, 17]],
    6: [[16, 18], [17, 19]],
    7: [[18, 20], [19, 21]],
    8: [[20, 22], [21, 23]],
}

BONE_DENSITY = {
    (0, 3): 1.0,
    (3, 6): 0.8,
    (6, 9): 0.7,
    (9, 12): 0.4,
    (12, 15): 0.2,
    (0, 1): 0.3,
    (1, 4): 0.4,
    (4, 7): 0.2,
    (7, 10): 0.0,
    (0, 2): 0.3,
    (2, 5): 0.4,
    (5, 8): 0.2,
    (8, 11): 0.0,
    (9, 13): 0.2,
    (13, 16): 0.2,
    (16, 18): 0.15,
    (18, 20): 0.1,
    (20, 22): 0.0,
    (9, 14): 0.2,
    (14, 17): 0.2,
    (17, 19): 0.15,
    (19, 21): 0.0,
    (21, 23): 0.0,
}
# 以上数值按目测半径计算 换算为密度时需取平方
for k in list(BONE_DENSITY.keys()):
    BONE_DENSITY[k] = BONE_DENSITY[k] ** 2

# local关节位置
JOINT_OFFSETS = 0.001 * torch.FloatTensor(
    [
        [0.0000, 0.0000, 0.0000],
        [58.5813, -82.2800, -17.6641],
        [-60.3097, -90.5133, -13.5425],
        [4.4394, 124.4036, -38.3852],
        [43.4514, -386.4695, 8.0370],
        [-43.2566, -383.6879, -4.8430],
        [4.4884, 137.9564, 26.8203],
        [-14.7903, -426.8745, -37.4280],  # 7
        [19.0555, -420.0456, -34.5617],  # 8
        [-2.2646, 56.0324, 2.8550],
        [41.0544, -60.2859, 122.0424],  # 10
        [-34.8399, -62.1055, 130.3233],  # 11
        [-13.3902, 211.6355, -33.4676],
        [71.7025, 113.9997, -18.8982],
        [-82.9537, 112.4724, -23.7074],
        [10.1132, 88.9373, 50.4099],
        [122.9214, 45.2051, -19.0460],
        [-113.2283, 46.8532, -8.4721],
        [255.3319, -15.6490, -22.9465],
        [-260.1275, -14.3692, -31.2687],
        [265.7092, 12.6981, -7.3747],
        [-269.1084, 6.7937, -6.0268],
        [86.6905, -10.6360, -15.5943],
        [-88.7537, -8.6516, -10.1071],
    ]
)

joint_mass_weights = torch.tensor(
    [
        0.15,  # pelvis (骨盆)
        0.05,
        0.05,  # left_hip, right_hip (左髋, 右髋)
        0.10,  # spine1 (脊柱1)
        0.08,
        0.08,  # left_knee, right_knee (左膝, 右膝)
        0.10,  # spine2 (脊柱2)
        0.05,
        0.05,  # left_ankle, right_ankle (左脚踝, 右脚踝)
        0.10,  # spine3 (脊柱3)
        0.02,
        0.02,  # left_foot, right_foot (左脚, 右脚) - 高权重，因为通常是支持点
        0.05,  # neck (颈部)
        0.03,
        0.03,  # (左锁骨, 右锁骨)
        0.05,  # head (头部)
        0.04,
        0.04,  # left_shoulder, right_shoulder (左肩, 右肩)
        0.03,
        0.03,  # left_elbow, right_elbow (左肘, 右肘)
        0.02,
        0.02,  # left_wrist, right_wrist (左手腕, 右手腕)
        0.01,
        0.01,  # left_hand, right_hand (左手, 右手) - 可能作为支持点
    ]
)
joint_mass_weights = joint_mass_weights / joint_mass_weights.sum()


class AnimoSMPL(AnimoSkeleton):
    def __init__(self):
        # super().__init__()
        self.joint_offset = JOINT_OFFSETS
        self.kinematic_tree = KINE_TREE
        # 关节重量权重
        self.joint_mass_weights = joint_mass_weights
        self.kinematic_params_init()
        # 2. 骨骼密度配置（不变）
        self.bone_density = BONE_DENSITY

    @torch.no_grad()
    def mirror_pose(self, R):
        "对pose进行镜像处理"
        pose_glob = self.forward_kinematics(R)
        pose_glob[..., :, 0, :] *= -1  # 三轴x坐标镜像
        pose_glob[..., :, :, 0] *= -1  # x轴镜像 转回右手系
        # 结构镜像
        pose_glob[..., LEFT_CHAIN, :, :], pose_glob[..., RIGHT_CHAIN, :, :] = (
            pose_glob[..., RIGHT_CHAIN, :, :],
            pose_glob[..., LEFT_CHAIN, :, :],
        )
        pose = self.inverse_kinematics(pose_glob)

        return pose

    def _symmetrize_joint_offsets(self):
        """
        关节偏移量对称化后处理（核心新增函数）：
        1. 对每对左右关节，计算偏移量的平均长度
        2. 保持左关节方向不变，将长度调整为平均长度
        3. 右关节方向取左关节的X轴镜像（符合人体左右对称），长度同样调整为平均长度
        """
        # return
        # 克隆一份偏移量避免原地修改冲突
        sym_offsets = self.joint_offset.clone()

        for left_idx, right_idx in SYMMETRIC_JOINT_PAIRS:
            # 1. 计算左右关节当前偏移量的长度（L2范数）
            left_len = torch.norm(sym_offsets[left_idx])
            right_len = torch.norm(sym_offsets[right_idx])

            # 2. 计算平均长度（保证左右长度一致）
            avg_len = (left_len + right_len) / 2.0

            # 3. 处理左关节：保持方向，调整长度为平均长度
            if left_len > 1e-6:  # 避免除以0（偏移量为0时无需调整）
                sym_offsets[left_idx] = sym_offsets[left_idx] / left_len * avg_len

            # 4. 处理右关节：X轴镜像（人体左右对称特征）+ 调整长度为平均长度
            # 原理：右关节相对于父节点的位置，是左关节的X轴反向（假设X轴为左右方向）
            left_dir = sym_offsets[left_idx] / torch.norm(sym_offsets[left_idx] + 1e-6)
            right_dir = left_dir * torch.tensor(
                [-1, 1, 1], device=left_dir.device
            )  # X轴镜像
            sym_offsets[right_idx] = right_dir * avg_len

        # 更新为对称后的偏移量
        self.joint_offset = sym_offsets
