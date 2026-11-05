import torch


def get_ego_gv(root_ori):
    """
    用root orientation计算ego gv坐标系的x,y,z基向量. ego gv定义为y轴沿垂直方向, z轴沿人体朝向的坐标系
    :param root_ori: 根节点在SMPL坐标系(Y axis up)的朝向, 旋转矩阵格式
    :return: ego-gv坐标系的x,y,z基向量, 旋转矩阵格式
    """
    assert len(root_ori.shape) == 2
    # 1. y轴固定为垂直方向 [0,1,0]（保持与输入张量相同的设备和数据类型）
    y_axis = torch.tensor(
        [0.0, 1.0, 0.0], device=root_ori.device, dtype=root_ori.dtype
    ).view(3, 1)

    # 2. 从root_ori中提取x轴和z轴向量 (旋转矩阵的列分别对应x、y、z轴)
    x_vec = root_ori[:, 0]  # x轴向量 (3,)
    z_vec = root_ori[:, 2]  # z轴向量 (3,)

    # 3. 将x和z轴向量投影到标准世界坐标系的x-z平面（y=0平面）
    x_proj = torch.stack(
        [
            x_vec[0],
            torch.tensor(0.0, device=root_ori.device, dtype=root_ori.dtype),
            x_vec[2],
        ]
    )
    z_proj = torch.stack(
        [
            z_vec[0],
            torch.tensor(0.0, device=root_ori.device, dtype=root_ori.dtype),
            z_vec[2],
        ]
    )

    # 4. 计算投影后的模长
    x_norm = torch.norm(x_proj)
    z_norm = torch.norm(z_proj)

    # 5. 确定模长更大的轴作为主要轴
    if x_norm < 0.1:
        # z轴模长更大,直接归一化作为z轴基向量
        z_axis = z_proj / z_norm
        # x轴通过y轴与z轴叉乘得到（保证正交）
        x_axis = torch.cross(y_axis.squeeze(), z_axis, dim=0)
    else:
        # x轴模长更大,直接归一化作为x轴基向量
        x_axis = x_proj / x_norm
        # z轴通过x轴与y轴叉乘得到（保证正交）
        z_axis = torch.cross(x_axis, y_axis.squeeze(), dim=0)

    # 确保基向量都是单位向量
    x_axis = x_axis / torch.norm(x_axis)
    z_axis = z_axis / torch.norm(z_axis)

    # 6. 拼接三个基向量形成旋转矩阵（每列对应一个轴）
    ego_gv = torch.column_stack([x_axis, y_axis.squeeze(), z_axis])

    return ego_gv


def vec_seq_resample(vec_seq, original_fps, target_fps):
    """
    对任意维度的向量序列进行重采样（优化版）
    - 当原始fps是目标fps的整数倍时,使用降采样（抽取关键帧）
    - 其他情况使用线性插值

    Args:
        vec_seq: 输入向量序列,形状为 (T, ..., D)
            T: 原始时间帧数量,...: 可选中间维度,D: 向量维度
        original_fps: 原始序列的帧率（如60fps）
        target_fps: 目标序列的帧率（如30fps）

    Returns:
        resampled_vec: 重采样后的向量序列,形状为 (T_new, ..., D)
        sample_points: 采样点在原始序列中的索引（torch.Tensor）
    """
    # 输入验证
    assert isinstance(vec_seq, torch.Tensor), "输入必须是torch.Tensor"
    assert vec_seq.dim() >= 2, "输入至少需要2个维度 (时间帧, 向量维度)"
    T_original = vec_seq.shape[0]  # 原始时间帧数量

    # 处理单帧特殊情况
    if T_original < 2:
        T_new = max(1, int((T_original - 1) * target_fps / original_fps) + 1)
        return vec_seq.repeat(T_new, *[1] * (vec_seq.dim() - 1)), torch.zeros(
            T_new, device=vec_seq.device
        )

    # 1. 计算总时长和目标帧数量
    total_time = (T_original - 1) / original_fps  # 总时长（秒）
    T_new = int(total_time * target_fps) + 1  # 目标帧数量
    T_new = max(1, T_new)  # 确保至少1帧

    # 2. 检查是否可通过整数倍降采样实现
    # 计算原始fps与目标fps的比值（ratio = 原始/fps目标）
    ratio = original_fps / target_fps
    is_integer_multiple = ratio.is_integer() and ratio >= 1.0

    # 3. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
    if is_integer_multiple:
        downsample_step = int(ratio)
        return vec_seq[::downsample_step]

    # 4. 非整数倍情况：使用线性插值
    # 生成目标时间轴（秒）
    target_times = torch.linspace(0.0, total_time, T_new, device=vec_seq.device)
    sample_points = target_times * original_fps  # 转换为原始序列索引（非整数）

    # 确定插值所需的前后帧索引
    idx0 = torch.floor(sample_points).long()  # 前一帧索引
    idx1 = idx0 + 1  # 后一帧索引
    idx1 = torch.clamp(idx1, max=T_original - 1)  # 防止越界

    # 提取前后帧向量并计算插值权重
    vec0 = vec_seq[idx0]  # 前一帧向量 (T_new, ..., D)
    vec1 = vec_seq[idx1]  # 后一帧向量 (T_new, ..., D)
    t = (sample_points - idx0.float()).unsqueeze(1)  # 插值比例（0~1）,扩展维度适配广播

    # 线性插值计算
    resampled_vec = (1 - t) * vec0 + t * vec1

    return resampled_vec


def quat_seq_resample(quats_seq, original_fps, target_fps, method="slerp"):
    """
    对四元数序列进行重采样（输入为torch.Tensor）

    Args:
        quats_seq: 输入四元数序列,形状为 (T, J, 4)
            T: 原始时间帧数量,J: 关节数量,4: 四元数(x,y,z,w)
        original_fps: 原始序列的帧率（如30fps）
        target_fps: 目标序列的帧率（如20fps）
        method: 插值方法,"nlerp"（归一化线性插值）或 "slerp"（球面线性插值）

    Returns:
        q_new: 重采样后的四元数序列,形状为 (T_new, J, 4),T_new为目标帧数量
        sample_points: 采样点在原始序列中的索引（torch.Tensor）
    """
    assert method in ["nlerp", "slerp"], "方法必须是'nlerp'或'slerp'"
    assert isinstance(quats_seq, torch.Tensor), "输入必须是torch.Tensor"
    assert quats_seq.shape[-1] == 4, "输入形状必须为(T, None, 4)"

    ratio = original_fps / target_fps
    is_integer_multiple = ratio.is_integer() and ratio >= 1.0
    # 0. 整数倍降采样逻辑（当原始fps是目标fps的整数倍时）
    if is_integer_multiple:
        downsample_step = int(ratio)
        return quats_seq[::downsample_step]

    J = quats_seq.shape[1] if quats_seq.dim() > 2 else 1
    quats_seq = quats_seq.reshape(-1, J, 4)  # 确保形状为 (T, J, 4)

    # --------------------------
    # 1. 计算目标序列参数
    # --------------------------
    T_original = quats_seq.shape[0]  # 原始帧数量
    total_time = (T_original - 1) / original_fps  # 序列总时长（秒）
    T_target = int(total_time * target_fps) + 1  # 目标帧数量（保证总时长一致）

    # 生成目标时间轴（单位：秒）,均匀采样
    target_times = torch.linspace(0.0, total_time, T_target, device=quats_seq.device)

    # 将目标时间转换为原始序列的索引（采样点）：t时刻对应原始序列的索引 = t * original_fps
    sample_points = target_times * original_fps  # 形状为 (T_target,)

    # --------------------------
    # 2. 获取插值所需的原始帧（q0为前一帧,q1为后一帧）
    # --------------------------
    # 前一帧索引（向下取整）,后一帧索引（前一帧+1）
    idx0 = torch.floor(sample_points).long()
    idx1 = idx0 + 1

    # 处理边界：避免后一帧超出原始序列长度
    idx1 = torch.clamp(idx1, max=T_original - 1)

    # 提取q0和q1（原始序列中前后帧的四元数）
    q0 = quats_seq[idx0]  # 形状 (T_target, J, 4)
    q1 = quats_seq[idx1]  # 形状 (T_target, J, 4)

    # 计算插值比例：采样点在前、后帧之间的比例（0~1）
    sample_percent = sample_points - idx0.float()  # 形状 (T_target,)
    sample_percent = sample_percent.unsqueeze(1).unsqueeze(
        2
    )  # 扩展为 (T_target, 1, 1),方便广播

    # --------------------------
    # 3. 对每个关节进行插值
    # --------------------------
    q_new = torch.zeros_like(q0)  # 初始化输出序列

    for j in range(J):
        # 提取单个关节的前后帧四元数
        q0_j = q0[:, j, :]  # (T_target, 4)
        q1_j = q1[:, j, :]  # (T_target, 4)

        if method == "nlerp":
            # 归一化线性插值（Nlerp）
            q_new_j = _quat_nlerp(q0_j, q1_j, sample_percent.squeeze(1))  # 挤压维度适配
        elif method == "slerp":
            # 球面线性插值（Slerp）
            q_new_j = _quat_slerp(q0_j, q1_j, sample_percent.squeeze(1))
        q_new[:, j, :] = q_new_j
    if J == 1:
        q_new = q_new.reshape(-1, 4)
    return q_new


def _quat_nlerp(q0, q1, t, ensure_shortest=True):
    """
    归一化线性插值（适配torch.Tensor）

    Args:
        q0: 起始四元数,形状 (T, 4)
        q1: 结束四元数,形状 (T, 4)
        t: 插值比例,形状 (T,)
        ensure_shortest: 是否选择最短路径（反转四元数）
    """
    # 确保四元数为单位长度（容错处理）
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    # 选择最短路径：点积为负时反转q1
    if ensure_shortest:
        dot = torch.sum(q0 * q1, dim=1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)  # 点积为负则取反

    # 线性插值 + 归一化
    t = t.unsqueeze(1)  # 扩展为 (T, 1) 适配广播
    q_interp = (1 - t) * q0 + t * q1
    q_interp = q_interp / torch.norm(q_interp, dim=-1, keepdim=True)  # 归一化

    return q_interp


def _quat_slerp(q0, q1, t, ensure_shortest=True):
    """
    球面线性插值（适配torch.Tensor）

    Args:
        q0: 起始四元数,形状 (T, 4)
        q1: 结束四元数,形状 (T, 4)
        t: 插值比例,形状 (T,)
        ensure_shortest: 是否选择最短路径（反转四元数）
    """
    # 确保四元数为单位长度
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    # 计算点积（夹角余弦）
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # 选择最短路径：点积为负时反转q1
    if ensure_shortest:
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.where(dot < 0, -dot, dot)  # 修正点积符号c

    # 处理接近平行的情况（用Nlerp替代,避免除零）
    theta = torch.acos(torch.clamp(dot, -1.0 + 1e-6, 1.0 - 1e-6))  # 夹角
    sin_theta = torch.sin(theta)
    near_zero = sin_theta < 1e-6

    # 球面插值公式
    # t = t.unsqueeze(1)
    sin_t_theta = torch.sin(t * theta)
    sin_1mt_theta = torch.sin((1 - t) * theta)

    q_interp = (sin_1mt_theta / sin_theta) * q0 + (sin_t_theta / sin_theta) * q1

    # 接近平行时用Nlerp
    q_nlerp = (1 - t) * q0 + t * q1
    q_nlerp = q_nlerp / torch.norm(q_nlerp, dim=-1, keepdim=True)
    q_interp = torch.where(near_zero, q_nlerp, q_interp)

    return q_interp
