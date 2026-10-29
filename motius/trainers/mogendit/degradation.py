import torch
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F
from motius.motion.representation.mogendit.articulate.math.angular import r6d_norm
from motius.motion.representation.mogendit.articulate.math.angular import r6d_to_rotation_matrix
from motius.motion.representation.mogendit.articulate.math.angular import rotation_matrix_to_r6d
from motius.motion.representation.mogendit.articulate.math.angular import axis_angle_to_rotation_matrix
from motius.motion.representation.mogendit.rotation_conversions import euler_angles_to_matrix
from motius.motion.representation.mogendit.animo.skeleton.smpl_body import AnimoSMPLBody


class GlobalMotionDegradation:
    """
    动作捕捉数据降质处理工具类

    该类实现了光学动作捕捉（optical mocap）和视觉动作捕捉（video mocap）
    系统中常见的6种核心降质模式。每种模式都以独立的成员函数实现，
    便于单独调用或组合使用。

    这些降质模式源于两种主要动作捕捉系统的固有误差：
    1. 光学动作捕捉系统：标记点混淆、遮挡、三角测量噪声等
    2. 视觉动作捕捉系统：2D检测噪声、深度模糊、视角依赖等

    使用方法：
        degradation = MotionDegradation()

        # 单独应用某种降质
        degraded_pose, degraded_trans = degradation.apply_joint_orientation_pops(pose, trans)

        # 组合多种降质
        result_pose = pose.clone()
        result_trans = trans.clone()
        result_pose, result_trans = degradation.apply_pose_twist(result_pose, result_trans)
        result_pose, result_trans = degradation.apply_translation_drift(result_pose, result_trans)
        result_pose, result_trans = degradation.apply_translation_ratio_distortion(result_pose, result_trans)
    """

    def __init__(self, device: str = None):
        """
        初始化降质处理器

        参数:
            device: PyTorch设备 ('cpu', 'cuda', 'cuda:0'等)
                  如果为None，则自动检测可用设备
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 降质参数配置：密度范围、尺度范围、选择概率
        self.degradation_configs = {
            "joint_orientation_pops": {
                "density_range": [0.05, 0.5],  # 密度参数随机范围
                "scale_range": [0.1, 1.0],  # 尺度参数随机范围
                "probability": 1/7,  # 被随机选择的概率
            },
            "joint_rotation_pops": {
                "density_range": [0.05, 0.2],  # 密度参数随机范围
                "scale_range": [0.1, 1.0],  # 尺度参数随机范围
                "probability": 1/7,  # 被随机选择的概率
            },
            "pose_twist": {
                "density_range": [0.25, 0.5],
                "scale_range": [0.5, 1.0],
                "probability": 1/7,
            },
            "candy_wrapper_twist": {
                "density_range": [0.2, 0.5],
                "scale_range": [0.5, 1.0],
                "probability": 1/7,
            },
            "frozen_frame": {
                "density_range": [0.1, 0.5],
                "scale_range": [0.0, 0.0],
                "probability": 1/7,
            },
            "d_translation_drift": {
                "density_range": [0.1, 1.0],
                "scale_range": [0.12, 0.32],
                "probability": 1/7,
            },
            "d_translation_distortion": {
                "density_range": [0.0, 0.0],
                "scale_range": [0.1, 1.0],
                "probability": 1/7,
            },
            "identity": {
                "density_range": [0.0, 0.0],  # identity不应用密度参数
                "scale_range": [0.0, 0.0],  # identity不应用尺度参数
                "probability": 0.5,  # 保持原始数据的概率
            },
        }

        self.body_model = AnimoSMPLBody()

        self.candy_joint = self.body_model.ik_ambiguity_joints
        self.candy_joint_children = self.body_model.ik_ambiguity_joints_children
        self.n_candy_joint = len(self.candy_joint)

        self.degradation_functions = [
            "identity",
            "joint_orientation_pops",
            "joint_rotation_pops",
            "pose_twist",
            "candy_wrapper_twist",
            "frozen_frame",
            "d_translation_drift",
            "d_translation_distortion",  # 修正名称，与degradation_configs中的键名保持一致
        ]

        # 预计算基于概率的随机选择所需的概率张量
        self._init_degradation_probabilities()

        # self.degradation_functions = [
        #     # "joint_orientation_pops",
        #     # "pose_twist",
        #     "candy_wrapper_twist",
        #     # "frozen_frame",
        #     # "d_translation_drift",
        #     # "d_translation_ratio_distortion",
        # ]

    def _init_degradation_probabilities(self):
        """
        预计算基于概率的随机选择所需的概率张量

        该方法在初始化时调用，用于从degradation_configs中提取每个降质函数的概率，
        构建归一化的概率张量，以便在apply_random_degradations中高效使用。
        """
        # 从degradation_configs中提取概率，构建概率向量
        probs = []
        for func_name in self.degradation_functions:
            if func_name in self.degradation_configs:
                # 确保所有降质函数在配置中都有对应的概率
                probs.append(self.degradation_configs[func_name]["probability"])
            else:
                # 如果配置中缺少某个函数，使用默认概率1/len(functions)
                probs.append(1.0 / len(self.degradation_functions))

        # 将概率列表转换为PyTorch张量
        prob_tensor = torch.tensor(probs, dtype=torch.float32, device=self.device)

        # 归一化概率，确保总和为1
        prob_tensor = prob_tensor / prob_tensor.sum()

        # 存储预计算的概率张量
        self.degradation_probabilities = prob_tensor

        # 验证概率配置
        print(f"[INFO] 降质函数概率配置：")
        for i, func_name in enumerate(self.degradation_functions):
            prob = self.degradation_probabilities[i].item()
            print(f"  {func_name}: {prob:.3f} ({prob*100:.1f}%)")

    def apply_joint_pops(
        self,
        pose: torch.Tensor,
        pop_type: 'ori'
    ) -> torch.Tensor:
        """
        应用关节旋转突变降质（joint orientation pops）

        模拟光学/视觉动捕中关节方向突然变化的情况：
        1. [光学动捕]：多个光学标记点过于靠近导致识别混淆
        2. [视觉动捕]：2D关键点检测噪声导致3D方向估计跳变

        这种降质表现为关节旋转在相邻帧间的非连续性变化，
        通常在标记点密度高的区域（如肩部、髋部）或快速运动时更常见。

        实现原理：
        1. 随机选择部分关节和时间点
        2. 对选中的关节施加随机的欧拉角扰动
        3. 将扰动限制在合理范围内，避免不自然的旋转

        参数:
            pose: 输入姿态数据，形状为 [batch_size, seq_len, n_joints, 3, 3]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            torch.Tensor: 降质处理后的姿态数据
        """
        b, seq_len, n_joints = pose.shape[:3]
        device = pose.device
        assert pop_type in ['ori', 'rot']

        density_range = self.degradation_configs["joint_orientation_pops"][
            "density_range"
        ]
        scale_range = self.degradation_configs["joint_orientation_pops"]["scale_range"]

        density_mask = torch.bernoulli(
            torch.ones(b, seq_len, n_joints, 1, device=device)
            * torch.empty(b, 1, 1, 1, device=device).uniform_(*density_range)
        )

        degradation_scale = torch.ones(
            b, seq_len, n_joints, 1, device=device
        ) * torch.empty(b, 1, 1, 1, device=device).uniform_(*scale_range)
        degraded_pose = pose.clone()

        # 生成随机旋转扰动（欧拉角），基于geometric_degradation_batch中的逻辑
        rotation_degradation = (
            torch.empty(b, seq_len, n_joints, 3, device=device).uniform_(-1, 1)
            * (np.pi / 180.0)  # 转换为弧度
            * 90.0  # 最大角度90度
        )

        rotation_degradation *= degradation_scale * density_mask

        rotation_degradation_matrix = euler_angles_to_matrix(
            rotation_degradation.reshape(-1, 3), convention="XYZ"
        ).reshape(b, seq_len, n_joints, 3, 3)

        # 通过前向运动学进行旋转累积
        if pop_type == 'rot':
            rotation_degradation_matrix = pose = self.body_model.forward_kinematics(rotation_degradation_matrix).view_as(rotation_degradation_matrix)

        # 模拟平滑后处理
        rotation_degradation_6d = rotation_matrix_to_r6d(rotation_degradation_matrix).reshape(b, seq_len, n_joints, 6)
        # 应用随机的smooth factor对rotation_degradation_6d进行平滑处理(待实现)

        # 应用随机的smooth factor对rotation_degradation_6d进行平滑处理
        # 使用 torch.randint 决定是否应用平滑（50%概率）
        if torch.randint(0, 2, (1,)).item() > 0:
            # 生成随机的平滑因子（0.01-1.0），0.01表示几乎不进行平滑，1.0表示最大程度的平滑
            smooth_factor = torch.empty(b, 1, 1, 1, device=device).uniform_(0.01, 1.0)

            # 随机选择内核大小（奇数：3, 5, 7帧），增加平滑效果的多样性
            kernel_sizes = [3, 5, 7]
            kernel_size = kernel_sizes[torch.randint(0, len(kernel_sizes), (1,)).item()]
            padding = kernel_size // 2

            # 重新排列维度以便进行一维卷积：[b, seq_len, n_joints, 6] -> [b, 6 * n_joints, seq_len]
            # 这样我们可以一次性对所有关节和维度进行时间平滑
            original_shape = rotation_degradation_6d.shape
            reshaped_data = rotation_degradation_6d.reshape(b, seq_len, -1)  # [b, seq_len, 6 * n_joints]
            reshaped_data = reshaped_data.permute(0, 2, 1)  # [b, 6 * n_joints, seq_len]

            n_channels = reshaped_data.shape[1]
            weight = torch.ones(n_channels, 1, kernel_size, device=device) / kernel_size

            # 对每个通道进行独立卷积（深度可分离卷积）
            smoothed_data = F.conv1d(
                reshaped_data,
                weight,
                padding=padding,
                groups=n_channels  # 深度可分离卷积，每个通道独立处理
            )

            smoothed_data = smoothed_data.permute(0, 2, 1)  # [b, seq_len, 6 * n_joints]
            smoothed_6d = smoothed_data.reshape(original_shape)  # [b, seq_len, n_joints, 6]
            smooth_factor_expanded = smooth_factor.expand_as(rotation_degradation_6d)
            rotation_degradation_6d = rotation_degradation_6d * (1 - smooth_factor_expanded) + \
                                        smoothed_6d * smooth_factor_expanded

            # 还原为旋转矩阵
            rotation_degradation_matrix = r6d_to_rotation_matrix(rotation_degradation_6d).reshape(b, seq_len, n_joints, 3, 3)

        degraded_pose = degraded_pose @ rotation_degradation_matrix


        return degraded_pose


    def apply_pose_twist(
        self,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用姿态异常降质（pose twist）

        模拟光学/视觉动捕中因遮挡或部分视野丢失导致的姿态异常：
        1. [光学/视觉动捕]：身体自遮挡（如交叉手臂、转身等）
        2. [视觉动捕]：关键点被其他物体或人物遮挡
        3. [视觉动捕]：光照变化或运动模糊影响检测质量

        这种降质通常表现为局部的、时间上连续的姿态扭曲，
        与关节朝向突变的突然变化不同，它更强调持续的畸变。

        实现原理：
        1. 在时间上生成连续平滑的旋转扰动
        2. 对受影响的关节应用逐渐变化的旋转
        3. 保持相邻帧间的运动连续性，避免不自然的跳变

        参数:
            pose: 输入姿态数据，形状为 [batch_size, seq_len, n_joints, rotation_dim]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            torch.Tensor: 降质处理后的姿态数据
        """
        # 待实现
        # 获取输入数据的形状
        b, seq_len, n_joints = pose.shape[:3]
        device = pose.device

        # 克隆输入数据以避免原地修改
        degraded_pose = pose.clone()

        density_range = self.degradation_configs["pose_twist"]["density_range"]
        scale_range = self.degradation_configs["pose_twist"]["scale_range"]
        # 生成密度掩码，控制哪些位置应用扰动（0-50%的概率出现关节跳变）
        density_mask = torch.bernoulli(
            torch.ones(b, 1, n_joints, 1, device=device)
            * torch.empty(b, 1, 1, 1, device=device).uniform_(*density_range)
        )

        # 生成扰动尺度（基于geometric_degradation_batch的逻辑）
        degradation_scale = torch.empty(b, 1, n_joints, 1, device=device).uniform_(
            *scale_range
        )

        # 生成随机旋转扰动（欧拉角），基于geometric_degradation_batch中的逻辑
        rotation_degradation = (
            torch.empty(b, 1, n_joints, 3, device=device).uniform_(-1, 1)
            * (np.pi / 180.0)  # 转换为弧度
            * 60.0  # 最大角度60度
        )
        # root expection
        rotation_degradation[:, :, 0] *= 0

        # 生成随机掩码，控制哪些位置应用扰动（0-50%的概率出现关节跳变）
        rotation_degradation_mask = torch.bernoulli(
            torch.ones(b, 1, 1, 1, device=device)
            * 0.5
            * torch.rand(b, 1, 1, 1, device=device)
        ).repeat(1, 1, n_joints, 1)

        rotation_degradation *= rotation_degradation_mask

        # 将欧拉角转换为旋转矩阵
        rotation_degradation_matrix = (
            euler_angles_to_matrix(
                rotation_degradation.reshape(-1, 3), convention="XYZ"
            )
            .reshape(b, 1, n_joints, 3, 3)
            .repeat(1, seq_len, 1, 1, 1)
        )

        # 将旋转矩阵应用于姿态数据
        # 假设pose的格式是[batch, seq, n_joints, 3, 3]的旋转矩阵格式
        # 如果pose是其他格式，需要先转换为旋转矩阵

        degraded_pose = degraded_pose @ rotation_degradation_matrix

        return degraded_pose

    def apply_candy_wrapper_twist(
        self,
        pose: torch.Tensor,
        joint: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用糖果包装纸扭转降质（Candy Wrapper Twist）

        模拟逆运动学（IK）多解问题导致的关节旋转方向错误。
        在光学/视觉动捕中，当多个不同的关节旋转组合能产生
        相同的末端位置时，算法可能选择错误的解。

        这种降质的特点是：关节位置正确，但关节旋转方向错误。
        在球形关节（如肩部、髋部）上尤其常见，因为这些关节
        的旋转有更大的自由度。

        实现原理：
        1. 为受影响关节生成看似合理的错误旋转
        2. 保持关节位置不变，只改变关节旋转
        3. 确保新旋转在数学上是有效的（保持正交性等）

        参数:
            pose: 输入姿态数据，形状为 [batch_size, seq_len, n_joints, rotation_dim]
            joint: 输入关节位置数据，形状为 [batch_size, seq_len, n_joints, 3]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            torch.Tensor: 降质处理后的姿态数据
        """
        # 待实现
        b, seq_len, n_joints = pose.shape[:3]
        device = pose.device

        density_range = self.degradation_configs["candy_wrapper_twist"]["density_range"]
        scale_range = self.degradation_configs["candy_wrapper_twist"]["scale_range"]
        # 生成密度掩码，控制哪些位置应用扰动（0-50%的概率出现关节跳变）
        density_mask = torch.bernoulli(
            torch.ones(b, 1, self.n_candy_joint, 1, device=device)
            * torch.empty(b, 1, 1, 1, device=device).uniform_(*density_range)
        )

        # 生成扰动尺度（基于geometric_degradation_batch的逻辑）
        degradation_scale = torch.empty(
            b, 1, self.n_candy_joint, 1, device=device
        ).uniform_(*scale_range)

        candy_axis = (
            joint[:, :, self.candy_joint_children] - joint[:, :, self.candy_joint]
        )

        candy_axis = candy_axis / torch.norm(candy_axis, dim=-1, keepdim=True).clip(
            min=1e-3
        )
        degradation_angle = (
            torch.empty(b, 1, self.n_candy_joint, 1, device=device).uniform_(-1, 1)
        ) * np.pi

        candy_twist_axis_angle = (
            candy_axis * degradation_angle * degradation_scale * density_mask
        )
        candy_twist_matrix = axis_angle_to_rotation_matrix(
            candy_twist_axis_angle
        ).reshape(b, seq_len, self.n_candy_joint, 3, 3)

        pose[:, :, self.candy_joint] = candy_twist_matrix.matmul(
            pose[:, :, self.candy_joint]
        )

        return pose

    def apply_frozen_frame(
        self,
        pose: torch.Tensor,
        trans: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用帧静止降质（Frozen Frame）

        模拟光学/视觉动捕中因人物完全超出视野而无法识别的情况。
        在真实系统中，这会导致系统重复使用最后检测到的有效姿态。

        这种降质表现为：
        1. 单帧或多帧完全静止
        2. 相邻帧间没有变化
        3. 可能影响序列的连续部分

        实现原理：
        1. 随机选择帧作为"冻结"帧
        2. 将冻结帧的内容替换为前一个有效帧
        3. 可模拟单点冻结或连续区域冻结

        参数:
            pose: 输入姿态数据，形状为 [batch_size, seq_len, n_joints, rotation_dim]
            trans: 输入平移数据，形状为 [batch_size, seq_len, 3]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 降质处理后的姿态和平移数据
        """
        # 待实现
        b, seq_len, n_joints = pose.shape[:3]
        device = pose.device
        density_range = self.degradation_configs["frozen_frame"]["density_range"]
        n_frozen_frame = int(
            torch.empty(1, device=device).uniform_(*density_range) * (seq_len - 2)
        )

        pose[:, 1 : 1 + n_frozen_frame] = (
            pose[:, [0]].clone().repeat(1, n_frozen_frame, 1, 1, 1)
        )
        trans[:, 1 : 1 + n_frozen_frame] = (
            trans[:, [0]].clone().repeat(1, n_frozen_frame, 1)
        )

        return pose, trans

    def apply_d_translation_drift(
        self,
        d_trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用位移漂移降质（Translation Drift）

        模拟视觉动捕中相机位置估计误差的累积效应。
        在单目视觉动捕中，缺乏全局参考会导致根节点位置
        随时间逐渐漂移，形成系统性偏差。

        这种降质表现为：
        1. 整体运动在X、Y、Z方向的缓慢偏移
        2. 漂移量随时间线性或累积增加
        3. 不影响关节相对位置，只影响全局平移

        实现原理：
        1. 生成随时间线性增加的漂移向量
        2. 将漂移应用于根节点平移
        3. 保持漂移的平滑性和连续性

        参数:
            d_trans: 输入位移增量数据，形状为 [batch_size, seq_len, 3]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            torch.Tensor: 降质处理后的位移增量数据
        """

        b, seq_len, dim = d_trans.shape[:3]
        device = d_trans.device

        # 确保d_trans是3D数据
        if dim != 3:
            # 如果输入不是3D，调整到3D
            d_trans = d_trans.reshape(b, seq_len, 3)
            dim = 3

        density_range = self.degradation_configs["d_translation_drift"]["density_range"]
        scale_range = self.degradation_configs["d_translation_drift"]["scale_range"]
        # 生成密度掩码，控制哪些位置应用扰动（0-50%的概率出现关节跳变）
        density_mask = torch.bernoulli(
            torch.empty(b, 1, 1, device=device).uniform_(*density_range)
            * torch.ones(b, seq_len, 1, device=device)
        )
        # 生成扰动尺度（基于geometric_degradation_batch的逻辑）
        degradation_scale = torch.empty(b, 1, 1, device=device).uniform_(*scale_range)

        # 生成漂移 线性分量+高斯分量（3D数据）
        gaussian_delta_drift = (
            (
                torch.empty(b, 1, 3, device=device).uniform_(-1, 1) * 0.05
                + torch.randn(b, seq_len, 3, device=device) * 0.02
            )
            * degradation_scale
            * density_mask
        )
        # 参考 apply_joint_orientation_pops 中的平滑处理，为 gaussian_delta_drift 添加随机平滑效果
        # 使用 torch.randint 决定是否应用平滑（50%概率）
        if torch.randint(0, 2, (1,)).item() > 0:
            # 生成随机的平滑因子（0.01-1.0），0.01表示几乎不进行平滑，1.0表示最大程度的平滑
            smooth_factor = torch.empty(b, 1, 1, device=device).uniform_(0.01, 1.0)

            # 随机选择内核大小（奇数：3, 5, 7帧），增加平滑效果的多样性
            kernel_sizes = [3, 5, 7]
            kernel_size = kernel_sizes[torch.randint(0, len(kernel_sizes), (1,)).item()]
            padding = kernel_size // 2

            # 调整维度以便进行一维卷积：[b, seq_len, 3] -> [b, 3, seq_len]
            # gaussian_delta_drift 是3D位移数据，在时间维度上进行平滑
            original_shape = gaussian_delta_drift.shape
            reshaped_data = gaussian_delta_drift.permute(0, 2, 1)  # [b, 3, seq_len]

            # 使用F.conv1d进行滑动平均
            n_channels = reshaped_data.shape[1]  # 3个通道（X,Y,Z）
            weight = torch.ones(n_channels, 1, kernel_size, device=device) / kernel_size

            # 对每个通道进行独立卷积（深度可分离卷积）
            smoothed_data = F.conv1d(
                reshaped_data,
                weight,
                padding=padding,
                groups=n_channels  # 深度可分离卷积，每个通道独立处理
            )

            # 恢复原始维度顺序
            smoothed_data = smoothed_data.permute(0, 2, 1)  # [b, seq_len, 3]
            smoothed_drift = smoothed_data.reshape(original_shape)

            # 混合原始数据和平滑数据，根据smooth_factor调整混合比例
            # smooth_factor形状为[b, 1, 1]，需要扩展以匹配gaussian_delta_drift的维度
            smooth_factor_expanded = smooth_factor.expand_as(gaussian_delta_drift)
            gaussian_delta_drift = gaussian_delta_drift * (1 - smooth_factor_expanded) + \
                                  smoothed_drift * smooth_factor_expanded

        return d_trans + gaussian_delta_drift

    def apply_d_translation_distortion(
        self,
        d_trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用位移比例失真降质（Translation Distortion）

        模拟视觉动捕中深度估计误差和相机姿态估计误差的影响。
        由于2D到3D映射需要估计未知的绝对尺度，可能导致：
        1. 整个运动的全局尺度错误
        2. 不同方向的缩放比例不一致
        3. 深度方向的失真比水平方向更严重

        这种降质表现为：
        1. 运动幅度在深度方向上失真
        2. 不同轴向上的缩放比例不同
        3. 全局运动的比例不准确

        实现原理：
        1. 为X、Y、Z轴生成独立的缩放因子
        2. 深度方向（通常Z轴）设置更大的失真
        3. 保持关节相对比例，只改变绝对尺度

        参数:
            d_trans: 输入位移增量数据，形状为 [batch_size, seq_len, 3]
            density: 降质密度系数（0.0-1.0），如果为None则使用默认值
            scale: 降质尺度系数（0.0-1.0），如果为None则使用默认值

        返回:
            torch.Tensor: 降质处理后的位移增量数据
        """
        # 写到这里 需要对delta trans进行[-30， 30]度随机yaw旋转与xyz三轴独立的[0.75-1.25倍缩放]
        b, seq_len, dim = d_trans.shape[:3]
        device = d_trans.device

        scale_range = self.degradation_configs["d_translation_drift"]["scale_range"]

        # 生成密度掩码，控制哪些位置应用扰动（0-50%的概率出现关节跳变）
        degradation_scale_affine = torch.empty(b, 1, 1, device=device).uniform_(
            *scale_range
        )
        degradation_scale_yaw_drift = torch.empty(b, 1, 1, device=device).uniform_(
            *scale_range
        )

        # 确保d_trans是3D数据
        if dim != 3:
            # 如果输入不是3D，调整到3D
            d_trans = d_trans.reshape(b, seq_len, 3)
            dim = 3

        # 应用缩放因子到d_trans
        # 使用3维的scale_factor以避免不必要的维度扩展
        affine_factor = 1 + torch.empty(b, 1, 3, device=device).uniform_(-0.25, 0.25)
        d_trans *= affine_factor * degradation_scale_affine

        # 添加随机yaw旋转
        rotation_degradation = (
            torch.empty(b, 1, 3, device=device).uniform_(-1, 1)
            * (np.pi / 180.0)  # 转换为弧度
            * 30.0  # 最大角度30度
        ) * degradation_scale_yaw_drift

        rotation_degradation[..., [0, 2]] *= 0  # 仅绕Y轴旋转
        yaw_rotation_matrix = (
            euler_angles_to_matrix(
                rotation_degradation.reshape(-1, 3), convention="XYZ"
            )
            .reshape(b, 1, 3, 3)
            .repeat(1, seq_len, 1, 1)
        )

        # 确保d_trans是3D向量，进行矩阵乘法
        # 先将d_trans转换为合适的形状进行矩阵乘法
        d_trans_reshaped = d_trans.unsqueeze(-1)  # [b, seq_len, 3, 1]
        d_trans_rotated = torch.matmul(
            yaw_rotation_matrix, d_trans_reshaped
        )  # [b, seq_len, 3, 1]
        d_trans = d_trans_rotated.squeeze(-1)  # [b, seq_len, 3]

        return d_trans

    def apply_random_degradations(
        self,
        pose: torch.Tensor,
        joint: torch.Tensor,
        trans: torch.Tensor,
        global_pose: str = True,
        min_segment_length: int = 10,
        max_segment_length: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用随机降质到动作序列中

        该函数采用分段降质策略：从序列开始处遍历，随机选择降质模式
        和段落长度，依次应用到序列的连续段落中，直到覆盖整个序列。
        这种实现方式避免了段落重叠问题，并确保序列被完全覆盖。

        算法流程：
        1. 初始化当前帧位置为0
        2. 当当前帧位置 < 序列总长度时：
           a. 基于预计算的概率张量随机选择一个降质模式（包括identity）
           b. 随机确定段落长度（在[min_segment_length, max_segment_length]范围内）
           c. 确保段落长度不超过剩余序列长度
           d. 在[current_position, current_position+segment_length]范围内应用降质
           e. 更新当前帧位置
        3. 确保最后一个段落能覆盖到序列末尾

        参数:
            pose: 输入姿态数据，形状为 [batch_size, seq_len, n_joints, rotation_dim]
            joint: 输入关节位置数据，形状为 [batch_size, seq_len, n_joints, 3]
            trans: 输入平移数据，形状为 [batch_size, seq_len, 3]
            min_segment_length: 段落最小长度（帧数），默认为10
            max_segment_length: 段落最大长度（帧数），默认为30

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 降质处理后的姿态和平移数据
        """
        # 获取输入数据的基本信息
        assert pose.ndim == 5  # 确保pose是旋转矩阵格式
        assert joint.ndim == 4  # 确保joint是3D向量格式
        assert trans.ndim == 3  # 确保trans是3D向量格式
        b, seq_len, n_joints = pose.shape[:3]
        device = pose.device

        if not global_pose:
            pose = self.body_model.forward_kinematics(pose).view_as(pose)

        # 克隆输入数据以避免原地修改
        degraded_pose = pose.clone()
        degraded_joint = joint.clone()
        degraded_trans = trans.clone()
        degraded_d_trans = torch.zeros_like(trans, device=device)
        degraded_d_trans[:, 1:] = (trans[:, 1:] - trans[:, :-1]).clone()

        joint_offset = self.body_model.get_skeleton_offsets(
            pose=pose[:, [0]], joint=joint[:, [0]], global_pose=True
        ).reshape(b, 1, -1, 3)

        # 1. 初始化当前帧位置
        current_position = 0

        # 2. 循环处理序列，直到覆盖整个序列长度
        while current_position < seq_len:
            # a. 基于预计算的概率张量随机选择一个降质模式（包括identity）
            # 使用multinomial进行加权随机选择
            degradation_idx = torch.multinomial(
                self.degradation_probabilities, num_samples=1
            ).item()
            degradation_type = self.degradation_functions[degradation_idx]

            # b. 随机确定段落长度（在[min_segment_length, max_segment_length]范围内）
            seg_length = torch.randint(
                min_segment_length, max_segment_length + 1, (1,)
            ).item()

            # c. 确保段落长度不超过剩余序列长度
            remaining_length = seq_len - current_position
            if seg_length > remaining_length:
                seg_length = remaining_length

            # d. 计算段落结束位置
            seg_end = current_position + seg_length

            # 提取当前段落的数据
            segment_pose = degraded_pose[:, current_position:seg_end]
            segment_joint = degraded_joint[:, current_position:seg_end]
            segment_trans = degraded_trans[:, current_position:seg_end]
            segment_d_trans = (
                degraded_d_trans[:, current_position + 1 : seg_end]
                if seg_end > current_position + 1
                else None
            )

            b_seg, seg_len_actual, n_joints = segment_pose.shape[:3]

            # 根据降质类型调用相应的函数
            if degradation_type == "identity":
                # 恒等映射：不应用任何降质
                pass
            elif degradation_type == "joint_orientation_pops":
                # 应用关节朝向突变降质
                segment_pose = self.apply_joint_pops(segment_pose, pop_type='ori')
            elif degradation_type == "joint_rotation_pops":
                # 应用关节朝向突变降质
                segment_pose = self.apply_joint_pops(segment_pose, pop_type='rot')
            elif degradation_type == "pose_twist":
                # 应用姿态异常降质
                segment_pose = self.apply_pose_twist(segment_pose)
            elif degradation_type == "candy_wrapper_twist":
                # 应用糖果包装纸扭转降质
                segment_pose = self.apply_candy_wrapper_twist(
                    segment_pose,
                    segment_joint,
                )
            elif degradation_type == "frozen_frame":
                # 应用帧静止降质
                segment_pose, segment_trans = self.apply_frozen_frame(
                    segment_pose,
                    segment_trans,
                )
            elif degradation_type == "d_translation_drift":
                if seg_len_actual > 1:
                    if current_position > 0:
                        prev_trans = degraded_trans[
                            :, current_position - 1 : current_position
                        ]
                    else:
                        prev_trans = degraded_trans[:, [0]]

                    if segment_d_trans is not None and segment_d_trans.shape[1] > 0:
                        # 应用降质到增量
                        segment_d_trans = self.apply_d_translation_drift(
                            segment_d_trans,
                        )

                        segment_d_trans = torch.cat(
                            [degraded_d_trans[:, [current_position]], segment_d_trans],
                            dim=1,
                        )

                        # 从增量重新计算相对平移
                        cumsum_d_trans = torch.cumsum(segment_d_trans, dim=1)

                        # 维度检查与修复
                        if cumsum_d_trans.dim() == 4:
                            cumsum_d_trans = cumsum_d_trans.squeeze(2)

                        # 转换回绝对平移
                        segment_trans = prev_trans + cumsum_d_trans

                        # 累积位移误差
                        degraded_trans[:, seg_end:] += (
                            segment_trans[:, [-1]]
                            - degraded_trans[:, seg_end - 1 : seg_end]
                        )

            elif degradation_type == "d_translation_distortion":
                if seg_len_actual > 1:
                    if current_position > 0:
                        prev_trans = degraded_trans[
                            :, current_position - 1 : current_position
                        ]
                    else:
                        prev_trans = degraded_trans[:, [0]]

                    if segment_d_trans is not None and segment_d_trans.shape[1] > 0:
                        # 应用降质到增量
                        segment_d_trans = self.apply_d_translation_distortion(
                            segment_d_trans,
                        )

                        segment_d_trans = torch.cat(
                            [degraded_d_trans[:, [current_position]], segment_d_trans],
                            dim=1,
                        )

                        # 从增量重新计算相对平移
                        cumsum_d_trans = torch.cumsum(segment_d_trans, dim=1)

                        # 维度检查与修复
                        if cumsum_d_trans.dim() == 4:
                            cumsum_d_trans = cumsum_d_trans.squeeze(2)

                        # 转换回绝对平移
                        segment_trans = prev_trans + cumsum_d_trans

                        # 累积位移误差
                        degraded_trans[:, seg_end:] += (
                            segment_trans[:, [-1]]
                            - degraded_trans[:, seg_end - 1 : seg_end]
                        )
            else:
                raise ValueError(f"未知的降质类型: {degradation_type}")

            # 将处理后的段落数据放回原位置
            degraded_pose[:, current_position:seg_end] = segment_pose
            degraded_trans[:, current_position:seg_end] = segment_trans

            # e. 更新当前帧位置
            current_position = seg_end

        degraded_joint = self.body_model.joint_fk_global(degraded_pose, joint_offset)
        if not global_pose:
            degraded_pose = self.body_model.inverse_kinematics(degraded_pose).view_as(
                degraded_pose
            )
        return degraded_pose, degraded_joint, degraded_trans

    def get_degradation_info(self) -> dict:
        """
        获取当前降质配置信息

        返回:
            dict: 包含所有降质配置的字典
        """
        return {
            "device": str(self.device),
            "scales": self.degradation_scales.copy(),
            "default_params": self.default_params.copy(),
        }
