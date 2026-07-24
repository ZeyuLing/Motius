from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..bodymodels.smpl_skeleton import SMPLMesh, SMPLSkeleton
from ..datasets.geometry import rot6d_to_rotation_matrix
from ..datasets.v2m_generation.geometry import process_r_t
from ..utils import matrix
from ..utils.motion_process import get_rifke


def calculate_jerk(pose_seqs: Tensor, lengths: Tensor) -> float:
    """
    计算动作序列的急动度(Jerk)指标，衡量动作的平滑性。
    与原版numpy实现的区别：直接返回对batch和时间维度取均值的float而非计算与原值的区别

    Args:
        pose_seqs: [B, T, J, 3] 姿态序列，B=batch_size, T=time, J=joints, 3=xyz
        lengths: [B] 每个序列的有效长度

    Returns:
        float: 全局急动度指标，值越小表示动作越平滑
    """
    assert pose_seqs.ndim == 4, f"pose_seqs [B,T,J,3], got {pose_seqs.shape}"
    jerk = torch.diff(pose_seqs, n=3, dim=1)  # [B,T-3,J,3] 三阶差分
    jerk_l1 = jerk.abs().sum(dim=-1)  # [B,T-3,J] L1范数
    bsz, Tm3, _ = jerk_l1.shape
    valid_T = torch.arange(Tm3, device=lengths.device).unsqueeze(0)  # [1,T-3]
    valid = valid_T < (lengths.clamp_min(3) - 3).unsqueeze(1)  # [bsz,T-3] 有效时间mask
    jerk_l1 = jerk_l1 * valid.to(jerk_l1.device).unsqueeze(-1)  # 屏蔽无效帧
    return (jerk_l1.max(dim=-1).values).mean().item()  # 取关节最大值再全局均值


def calculate_motion_similarity(
    pose_seqs_1: Tensor,
    pose_seqs_2: Tensor,
    use_rifke: bool = True,
    root_rotations_mat_1: Optional[Tensor] = None,
    root_rotations_mat_2: Optional[Tensor] = None,
) -> tuple[float, float]:
    """
    计算两个动作序列之间的相似性，基于MPJPE(Mean Per Joint Position Error)。
    与原版的区别：直接在姿态空间计算MPJPE，而非特征空间的欧氏距离，这里是为了用RIFKE变换消除根节点平移和旋转的影响，专注于姿态相似性

    Args:
        pose_seqs_1: [B1, T, J, 3] 第一组姿态序列
        pose_seqs_2: [B2, T, J, 3] 第二组姿态序列，B2可以为1
        use_rifke: 是否使用RIFKE变换进行根节点不变处理
        root_rotations_mat_1: [B1, T, 3, 3] 第一组的根节点旋转矩阵
        root_rotations_mat_2: [B2, T, 3, 3] 第二组的根节点旋转矩阵

    Returns:
        tuple[float, float]: (平均MPJPE, 最小MPJPE)
    """
    x, y = pose_seqs_1, pose_seqs_2
    if use_rifke:
        # NOTE: 注意这里我们用了yaw模式，即【同时】消除根节点平移和旋转的影响
        assert root_rotations_mat_1 is not None and root_rotations_mat_2 is not None
        x = get_rifke(x, root_rotations_mat_1, mode="yaw")
        y = get_rifke(y, root_rotations_mat_2, mode="yaw")
    if y.size(0) == 1:
        y = y.expand(x.size(0), -1, -1, -1)  # 广播到相同batch size
    mpjpe_per = (x - y).norm(dim=-1).mean(dim=(1, 2))  # [B] 每样本的MPJPE
    return mpjpe_per.mean().item(), mpjpe_per.min().item()


def calculate_motion_diversity(
    pose_seqs: Tensor,
    use_rifke: bool = True,
    root_rotations_mat: Optional[Tensor] = None,
) -> float:
    """
    计算一批动作序列的多样性，基于两两距离的平均值。
    与原版的区别： 计算全量两两距离而非随机采样，并且提供RIFKE选项确保根节点不变性

    Args:
        pose_seqs: [B, T, J, 3] 姿态序列批次
        use_rifke: 是否使用RIFKE变换
        root_rotations_mat: [B, T, 3, 3] 根节点旋转矩阵

    Returns:
        float: 多样性指标，值越大表示动作越多样化
    """
    # NOTE: 注意这里我们用了yaw模式，即【同时】消除根节点平移和旋转的影响
    x = (
        get_rifke(pose_seqs, root_rotations_mat, mode="yaw")
        if use_rifke and root_rotations_mat is not None
        else pose_seqs
    )
    B = x.size(0)
    if B <= 1:
        return 0.0  # 单个样本无法计算多样性
    flat = x.reshape(B, -1)  # 展平为向量
    flat_fp32 = flat.to(dtype=torch.float32)
    D = torch.cdist(flat_fp32, flat_fp32, p=2)  # 计算两两欧氏距离矩阵
    iu = torch.triu_indices(B, B, 1)  # 上三角索引，避免重复计算
    return D[iu[0], iu[1]].mean().item()  # 返回平均距离


def calculate_R_precision(embedding1: Tensor, embedding2: Tensor, top_k: int, sum_all: bool = False):
    """
    计算检索精度(R-precision)，用于文本-动作检索任务评估。

    Args:
        embedding1: [N1, D] 查询嵌入向量
        embedding2: [N2, D] 数据库嵌入向量
        top_k: int Top-K精度计算的K值
        sum_all: bool 是否对所有样本求和

    Returns:
        Tensor: [N1, top_k]或[top_k,] 精度结果矩阵
    """
    dist_mat = _euclidean_distance_matrix(embedding1, embedding2)
    argmax = torch.argsort(dist_mat, dim=1)  # 按距离排序
    top_k_mat = _calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(dim=0)
    else:
        return top_k_mat


def _euclidean_distance_matrix(matrix1: Tensor, matrix2: Tensor) -> Tensor:
    """
    ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩

    Args:
        matrix1: [N1, D] 第一组向量
        matrix2: [N2, D] 第二组向量

    Returns:
        Tensor: [N1, N2] 距离矩阵，dist[i,j] = ||matrix1[i] - matrix2[j]||₂
    """
    assert matrix1.shape[1] == matrix2.shape[1]

    # (X - Y)^2 = X^2 - 2XY^T + Y^2
    d1 = -2 * torch.mm(matrix1, matrix2.t())  # [N1, N2]
    d2 = torch.sum(matrix1.square(), dim=1, keepdim=True)  # [N1, 1]
    d3 = torch.sum(matrix2.square(), dim=1)  # [N2]
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting: [N1, N2]
    return dists


def _calculate_top_k(mat: Tensor, top_k: int) -> Tensor:
    """
    计算Top-K精度矩阵，用于检索任务评估。

    Args:
        mat: [N, N] 排序后的索引矩阵 (argsort的输出)
        top_k: int 计算前k个结果的精度

    Returns:
        Tensor: [N, top_k] 累积精度矩阵，值为0/1的浮点数
    """
    size = mat.shape[0]
    device = mat.device

    # Create ground truth matrix: gt_mat[i, j] = i (diagonal pattern)
    gt_mat = torch.arange(size, device=device).unsqueeze(1).repeat(1, size)  # [N, N]

    # Check if retrieved indices match ground truth
    bool_mat = mat == gt_mat  # [N, N]

    correct_vec = torch.zeros(size, dtype=torch.bool, device=device)  # [N]
    top_k_list = []

    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]  # 累积逻辑或操作
        top_k_list.append(correct_vec.unsqueeze(1))  # [N, 1]

    top_k_mat = torch.cat(top_k_list, dim=1)  # [N, top_k]
    return top_k_mat.float()  # 转换为浮点数便于精度计算


def calculate_frechet_distance(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> float:
    """
    计算Fréchet距离(FID)，衡量两个多元高斯分布之间的距离。
    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

    Args:
        mu1: [D] 生成样本的均值向量
        sigma1: [D, D] 生成样本的协方差矩阵
        mu2: [D] 真实样本的均值向量
        sigma2: [D, D] 真实样本的协方差矩阵
        eps: float 数值稳定性参数

    Returns:
        float: Fréchet距离，值越小表示两分布越相似
    """
    mu1 = mu1.flatten() if mu1.dim() > 1 else mu1
    mu2 = mu2.flatten() if mu2.dim() > 1 else mu2
    if sigma1.dim() == 1:
        sigma1 = sigma1.unsqueeze(0)
    if sigma2.dim() == 1:
        sigma2 = sigma2.unsqueeze(0)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    try:
        # 使用特征值分解计算矩阵平方根，避免复数运算
        e1, U1 = torch.linalg.eigh(sigma1)
        e1 = e1.clamp(min=eps)  # 确保正定性
        sqrt_sigma1 = U1 @ torch.diag(torch.sqrt(e1)) @ U1.T

        # 夹心矩阵方法：M = Σ₁^{1/2} Σ₂ Σ₁^{1/2}
        M = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
        eM, _ = torch.linalg.eigh(M)
        eM = eM.clamp(min=0)
        tr_covmean = torch.sqrt(eM).sum()  # trace(√M) = trace(√(Σ₁Σ₂))
    except Exception:
        # 数值不稳定时的正则化处理
        I = torch.eye(sigma1.shape[0], device=sigma1.device, dtype=sigma1.dtype)
        sigma1_reg = sigma1 + eps * I
        sigma2_reg = sigma2 + eps * I
        e1, U1 = torch.linalg.eigh(sigma1_reg)
        e1 = e1.clamp(min=eps)
        sqrt_sigma1 = U1 @ torch.diag(torch.sqrt(e1)) @ U1.T
        M = sqrt_sigma1 @ sigma2_reg @ sqrt_sigma1
        eM, _ = torch.linalg.eigh(M)
        eM = eM.clamp(min=0)
        tr_covmean = torch.sqrt(eM).sum()

    return (diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean).item()


def _rotation_to_angle(rotations: Tensor) -> Tensor:
    """将旋转矩阵转换为角度"""
    traces = rotations[..., 0, 0] + rotations[..., 1, 1] + rotations[..., 2, 2]
    return torch.acos(torch.clamp((traces - 1) / 2, -1.0, 1.0)) * 180 / torch.pi


def calculate_velocity_o6d(rot6d: Tensor, transl: Tensor, fps: int) -> dict:
    """
    rot6d: [B, T, J, 6]
    transl: [B, T, 3]
    fps: 采样频率(帧率)，用于将“每帧角/距变化”换算到“每秒”
    返回度/秒与米/秒（若坐标单位为米）
    """
    assert rot6d.ndim == 4 and rot6d.shape[-1] == 6, f"rot6d shape should be [B,T,J,6], got {rot6d.shape}"
    assert transl.ndim == 3 and transl.shape[-1] == 3, f"transl shape should be [B,T,3], got {transl.shape}"
    # [B,T,J,6] -> [B,T,J,3,3]
    R = rot6d_to_rotation_matrix(rot6d.reshape(-1, 6)).reshape(*rot6d.shape[:-1], 3, 3)
    # 相对旋转：R_t * R_{t-1}^T -> [B, T-1, J, 3, 3]
    rel = R[:, 1:] @ R[:, :-1].transpose(-1, -2)
    # 角速度（度/秒）
    ang = fps * _rotation_to_angle(rel)  # [B, T-1, J]
    non_root = ang[..., 1:]  # 除根以外关节
    root = ang[..., 0]  # 根关节
    # 线速度（单位/秒）
    trans_v = (transl[:, 1:] - transl[:, :-1]).norm(dim=-1) * fps  # [B, T-1]
    return {
        "avg_ang_vel": non_root.mean().item(),
        "max_ang_vel": non_root.max().item(),
        "avg_root_ang_vel": root.mean().item(),
        "max_root_ang_vel": root.max().item(),
        "avg_speed": trans_v.mean().item(),
        "max_speed": trans_v.max().item(),
    }


def calculate_translation_error(
    transl_pred: Tensor,
    transl_gt: Tensor,
    lengths: Tensor,
    align_origin: bool = False,
    xz_only: bool = False,
    return_per_sample: bool = False,
) -> Tensor | float:
    """
    平移绝对误差（ATE）
    Args:
        transl_pred: [1, T, 3]
        transl_gt:   [B, T, 3]
        lengths:     [B]
        align_origin: 是否各自减去第1帧以消除全局偏移（对比轨迹形状）
                      True 时：pred 与 gt 各自对齐到自身起点
                      False 时：直接对齐到数据坐标系
        xz_only:     仅评估 XZ 平面
        return_per_sample: 返回每个样本的平均误差 [B]；否则返回全局均值 float
    """
    assert transl_pred.shape[1:] == transl_gt.shape[1:] and transl_pred.ndim == 3 and transl_pred.shape[-1] == 3
    B, T, _ = transl_pred.shape
    device = transl_pred.device

    if align_origin:
        transl_pred = transl_pred - transl_pred[:, :1, :]
        transl_gt = transl_gt - transl_gt[:, :1, :]

    if xz_only:
        transl_pred = transl_pred[..., [0, 2]]
        transl_gt = transl_gt[..., [0, 2]]

    diff = transl_pred - transl_gt  # [B,T,2/3]
    err_t = diff.norm(dim=-1)  # [B,T]

    valid = torch.arange(T, device=device).unsqueeze(0) < lengths.clamp_min(1).unsqueeze(1)  # [B,T]
    err_t = err_t * valid.float()

    per_sample = err_t.sum(dim=1) / valid.float().sum(dim=1).clamp_min(1.0)  # [B]
    return per_sample if return_per_sample else per_sample.mean().item()


# evaluation utilities for v2m model
def get_joints_from_smpl_params(smpl_skeleton, batch, joint_num=24):
    rot6d = batch["rot6d"]
    transl = batch["trans"]
    shapes = batch["shapes"]
    rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
    transl_flat = transl.reshape(rot6d.shape[0] * rot6d.shape[1], 3)
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, rot6d.shape[1], 1)
    shapes_flat = shapes.reshape(rot6d.shape[0] * rot6d.shape[1], 16)

    local_params = {
        "rot6d": rot6d_flat,
        "trans": torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype),
        "shapes": shapes_flat,
    }
    global_params = {
        "rot6d": rot6d_flat,
        "trans": transl_flat,
        "shapes": shapes_flat,
    }

    output_joints = {
        "local_joints": smpl_skeleton(local_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
        "global_joints": smpl_skeleton(global_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
    }
    for key in output_joints:
        output_joints[key] = output_joints[key][:, :, :joint_num, :]
    return output_joints


def get_vertices_from_smpl_params(smpl_mesh, batch):
    rot6d = batch["rot6d"]
    transl = batch["trans"]
    shapes = batch["shapes"]
    rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
    transl_flat = transl.reshape(rot6d.shape[0] * rot6d.shape[1], 3)
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, rot6d.shape[1], 1)
    shapes_flat = shapes.reshape(rot6d.shape[0] * rot6d.shape[1], -1)

    params = {
        "rot6d": rot6d_flat,
        "trans": transl_flat,
        "shapes": shapes_flat,
    }

    out_vertices = smpl_mesh(params)
    out_vertices["local_vertices"] = out_vertices.pop("vertices_wotrans")
    out_vertices["global_vertices"] = out_vertices.pop("vertices")
    return out_vertices


def get_fkmat_from_smpl_params(smpl_skeleton, batch):
    B, L = batch["rot6d"].shape[:2]
    rotmat = rot6d_to_rotation_matrix(batch["rot6d"])[..., :22, :, :]
    parents = smpl_skeleton.parents[:22]

    shapes = batch["shapes"]
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, L, 1)
    shapes = shapes.reshape(B * L, 16)
    skeleton = smpl_skeleton.compute_j_shaped(shapes)[..., :22, :].reshape(B, L, 22, 3)  # (B, L, 22, 3)
    local_skeleton = skeleton - skeleton[:, :, parents]
    local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
    local_skeleton[..., 0, :] += batch["trans"]

    mat = matrix.get_TRS(rotmat, local_skeleton)
    fk_mat = matrix.forward_kinematics(mat, parents)
    joints = matrix.get_position(fk_mat)

    return joints, mat, fk_mat


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1, device=Y.device, dtype=Y.dtype) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3, device=Y.device, dtype=Y.dtype).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=Y.dtype)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).cpu().numpy()


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr_sta=1e-2):
    """compute foot sliding error
    The foot ground contact label is computed by the threshold of 1 cm/frame
    Args:
        target_verts (N, 6890, 3).
        pred_verts (N, 6890, 3).
    Returns:
        error (N frames in contact).
    """
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr_sta
    dynamic_005 = foot_disp[:] > 0.05
    dynamic_01 = foot_disp[:] > 0.1

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)
    pred_contact = pred_disp[:] < thr_sta
    pred_dynamic_005 = pred_disp[:] > 0.05
    pred_dynamic_01 = pred_disp[:] > 0.1

    error = pred_disp[contact]
    correct_contact = ((pred_contact == contact).sum() / contact.numel()).reshape(1, 1)
    correct_dynamic_01 = ((pred_dynamic_01 == dynamic_01).sum() / dynamic_01.numel()).reshape(1, 1)
    correct_dynamic_005 = ((pred_dynamic_005 == dynamic_005).sum() / dynamic_005.numel()).reshape(1, 1)

    return (
        error.cpu().numpy(),
        correct_contact.cpu().numpy(),
        correct_dynamic_01.cpu().numpy(),
        correct_dynamic_005.cpu().numpy(),
    )


def compute_foot_sliding_by_pred_vel(target_j3d, pred_vel, fps=30, thr_sta=1e-2):
    assert target_j3d.shape[0] == pred_vel.shape[0]
    assert target_j3d.shape[-2] == 24

    # Foot vertices idxs
    # foot_idxs = [3216, 3387, 6617, 6787]  # for mesh vertices
    foot_idxs = [7, 10, 8, 11]  # for joints

    # Compute contact label
    foot_loc = target_j3d[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr_sta
    dynamic_005 = foot_disp[:] > 0.05
    dynamic_01 = foot_disp[:] > 0.1

    pred_disp = (pred_vel[:, : len(foot_idxs)] / fps)[:-1].norm(2, dim=-1)
    pred_contact = pred_disp[:] < thr_sta
    pred_dynamic_005 = pred_disp[:] > 0.05
    pred_dynamic_01 = pred_disp[:] > 0.1

    error = pred_disp[contact]
    correct_contact = ((pred_contact == contact).sum() / contact.numel()).reshape(1, 1)
    correct_dynamic_01 = ((pred_dynamic_01 == dynamic_01).sum() / dynamic_01.numel()).reshape(1, 1)
    correct_dynamic_005 = ((pred_dynamic_005 == dynamic_005).sum() / dynamic_005.numel()).reshape(1, 1)

    return (
        error.cpu().numpy(),
        correct_contact.cpu().numpy(),
        correct_dynamic_01.cpu().numpy(),
        correct_dynamic_005.cpu().numpy(),
    )


def batch_align_by_pelvis(data_list, pelvis_idxs=[1, 2]):
    """
    Assumes data is given as [pred_j3d, target_j3d, pred_verts, target_verts].
    Each data is in shape of (frames, num_points, 3)
    Pelvis is notated as one / two joints indices.
    Align all data to the corresponding pelvis location.
    """

    pred_j3d, target_j3d, pred_verts, target_verts = data_list

    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()

    # Align to the pelvis
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis

    return (pred_j3d, target_j3d, pred_verts, target_verts)


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
    """
    Use [i-1, i, i+1] to compute acc at frame_i. The acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries(-1, 0, +1) in the
    acceleration error will be zero'd out.
    Args:
        joints_gt : (F, J, 3)
        joints_pred : (F, J, 3)
        valid_mask : (F)
    Returns:
        error_accel (F-2) when valid_mask is None, else (F'), F' <= F-2
    """
    # (F, J, 3) -> (F-2) per-joint
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = torch.norm(accel_pred - accel_gt, dim=-1).mean(dim=-1)
    if fps is not None:
        normed = normed * fps**2

    if valid_mask is None:
        new_vis = torch.ones(len(normed)).to(dtype=torch.bool, device=joints_gt.device)
    else:
        invis = torch.logical_not(valid_mask)
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]
        new_vis = torch.logical_not(new_invis)
        if new_vis.sum() == 0:
            print("Warning!!! no valid acceleration error to compute.")

    return normed[new_vis]


def compute_camcoord_metrics(batch, pelvis_idxs=[1, 2], fps=30, mask=None):
    """
    Args:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
            "pred_verts":
            "target_verts":
        }
    Returns:
        cam_coord_metrics (dict): {
            "pa_mpjpe": (..., ) numpy array
            "mpjpe":
            "pve":
            "accel":
        }
    """
    # All data is in camera coordinates
    pred_j3d = batch["pred_j3d"].cpu()  # (..., J, 3)
    target_j3d = batch["target_j3d"].cpu()
    pred_verts = batch["pred_verts"].cpu()
    target_verts = batch["target_verts"].cpu()

    if mask is not None:
        mask = mask.cpu()
        pred_j3d = pred_j3d[mask].clone()
        target_j3d = target_j3d[mask].clone()
        pred_verts = pred_verts[mask].clone()
        target_verts = target_verts[mask].clone()
    assert "mask" not in batch

    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )

    # Metrics
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    pve = compute_jpe(pred_verts, target_verts) * m2mm
    accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d, fps=fps)

    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
        "pve": pve,
        "accel": accel,
    }
    return camcoord_metrics


def compute_local_metrics(pred_j3d, target_j3d, pred_verts, target_verts, pelvis_idxs=[1, 2], fps=30):
    """
    Args:
        pred_j3d_local (F, J, 3)
        target_j3d_local (F, J, 3)
        pred_verts_local (F, 6890, 3)
        target_verts_local (F, 6890, 3)
    """
    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    # TODO: align by first frame
    # 我们输出的不是相机系的
    mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    pve = compute_jpe(pred_verts, target_verts) * m2mm
    accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d, fps=fps)

    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
        "pve": pve,
        "accel": accel,
    }
    return camcoord_metrics


def compute_global_metrics(smpl_skeleton, smpl_mesh, J_regressor, batch, mask=None, enable_timer=False):
    """
    Args:
        batch (dict): {
            "pred": {
                "rot6d": (bs, F, J, 6) tensor
                "shapes": (bs, F, 16) tensor
                "trans": (bs, F, 3) tensor
            }
            "gt": {
                "rot6d": (bs, F, J, 6) tensor
                "shapes": (bs, F, 16) tensor
                "trans": (bs, F, 3) tensor
                "joints": (bs, F, 52, 3) tensor
                "vertices": (bs, F, 6890, 3) tensor
            }
        }
        enable_timer: bool, 是否启用计时统计
    Returns:
        global_metrics (dict): {
            "wa2_mpjpe": (F, ) numpy array
            "waa_mpjpe":
            "rte":
            "jitter":
            "fs":
        }
    """
    import time

    timings = {}

    def timer_start():
        if enable_timer:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return time.perf_counter()
        return None

    def timer_end(name, start_time):
        if enable_timer and start_time is not None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time
            timings[name] = elapsed

    # Step 1: Get pred joints from SMPL params
    # Step 2: Get pred vertices from SMPL params
    t0 = timer_start()
    pred_verts = get_vertices_from_smpl_params(smpl_mesh, batch["pred"])
    pred_verts_local = pred_verts["local_vertices"].squeeze()
    pred_verts_glob = pred_verts["global_vertices"].squeeze()
    pred_j3d_local = J_regressor[None] @ pred_verts_local
    pred_j3d_glob = J_regressor[None] @ pred_verts_glob
    timer_end("2_get_pred_vertices", t0)

    # Step 3: Get target joints/vertices
    t0 = timer_start()
    if False:
        target_joints = get_joints_from_smpl_params(smpl_skeleton, batch["gt"])
        target_j3d_local = target_joints["local_joints"].squeeze()
        target_j3d_glob = target_joints["global_joints"].squeeze()
        target_verts = get_vertices_from_smpl_params(smpl_mesh, batch["gt"])
        target_verts_local = target_verts["local_vertices"].squeeze()
        target_verts_glob = target_verts["global_vertices"].squeeze()
    else:
        target_verts_glob = batch["gt"]["vertices"].squeeze()
        target_transl = batch["gt"]["trans"].squeeze()

        target_verts_local = target_verts_glob - target_transl[:, None]

        target_j3d_glob = J_regressor[None] @ target_verts_glob
        target_j3d_local = J_regressor[None] @ target_verts_local
        target_j3d_glob = target_j3d_glob[:, : pred_j3d_glob.shape[-2], :]
        target_j3d_local = target_j3d_local[:, : pred_j3d_local.shape[-2], :]

    timer_end("3_get_target_joints_verts", t0)

    # print(f'target_j3d_glob.shape: {target_j3d_glob.shape}, pred_j3d_glob.shape: {pred_j3d_glob.shape}')
    # print(f'target_j3d_local.shape: {target_j3d_local.shape}, pred_j3d_local.shape: {pred_j3d_local.shape}')
    # print(f'target_verts_glob.shape: {target_verts_glob.shape}, pred_verts_glob.shape: {pred_verts_glob.shape}')
    # print(f'target_verts_local.shape: {target_verts_local.shape}, pred_verts_local.shape: {pred_verts_local.shape}')

    # Step 4: Apply mask
    t0 = timer_start()
    # All data is in global coordinates
    if mask is not None:
        mask = mask
        pred_j3d_glob = pred_j3d_glob[mask].clone()
        target_j3d_glob = target_j3d_glob[mask].clone()
        pred_verts_glob = pred_verts_glob[mask].clone()
        target_verts_glob = target_verts_glob[mask].clone()
    assert "mask" not in batch
    timer_end("4_apply_mask", t0)

    seq_length = pred_j3d_glob.shape[0]

    # Step 5: Compute local metrics
    t0 = timer_start()
    local_metrics = compute_local_metrics(pred_j3d_local, target_j3d_local, pred_verts_local, target_verts_local)
    timer_end("5_compute_local_metrics", t0)

    # Step 6: Use chunk to compare (alignment metrics)
    t0 = timer_start()
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone()
        pred_j3d = pred_j3d_glob[start:end].clone()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        # visualization
        if False:
            from hmr4d.utils.wis3d_utils import add_motion_as_lines, make_wis3d

            wis3d = make_wis3d(name="debug-metric_utils")
            add_motion_as_lines(target_j3d, wis3d, name="target_j3d")
            add_motion_as_lines(pred_j3d, wis3d, name="pred_j3d")
            add_motion_as_lines(w_j3d, wis3d, name="pred_w2_j3d")
            add_motion_as_lines(wa_j3d, wis3d, name="pred_wa_j3d")

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))
    timer_end("6_chunk_alignment_metrics", t0)

    # Metrics
    t0 = timer_start()
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm
    timer_end("7_concat_mpjpe", t0)

    # Step 8: Compute RTE
    t0 = timer_start()
    rte = compute_rte(target_j3d_glob[:, 0], pred_j3d_glob[:, 0]) * 1e2
    timer_end("8_compute_rte", t0)

    # Step 9: Compute jitter
    t0 = timer_start()
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    timer_end("9_compute_jitter", t0)

    # Step 10: Compute foot sliding
    t0 = timer_start()
    if True:
        # if '64_outdoor_skateboard_s001200_e001500' in batch["meta"]["sequence_name"][0]:
        #     breakpoint()
        foot_sliding, cc_ratio, cd_ratio_01, cd_ratio_005 = compute_foot_sliding(target_verts_glob, pred_verts_glob)
    else:
        # NOTE: to check fs metrics derived from predicted end effector velocity, use this branch
        foot_sliding, cc_ratio, cd_ratio_01, cd_ratio_005 = compute_foot_sliding_by_pred_vel(
            target_j3d_glob, batch["pred"]["end_effector_vel"].squeeze()
        )
    foot_sliding = foot_sliding * m2mm
    cc_ratio = cc_ratio * 100
    cd_ratio_01 = cd_ratio_01 * 100
    cd_ratio_005 = cd_ratio_005 * 100
    timer_end("10_compute_foot_sliding", t0)

    # Print timing summary if enabled
    if enable_timer:
        print("\n" + "=" * 60)
        print("compute_global_metrics 计时统计:")
        print("=" * 60)
        total_time = sum(timings.values())
        for name, elapsed in timings.items():
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {name:35s}: {elapsed*1000:8.2f} ms ({pct:5.1f}%)")
        print("-" * 60)
        print(f"  {'Total':35s}: {total_time*1000:8.2f} ms")
        print("=" * 60 + "\n")

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": foot_sliding,
        "correct_contact": cc_ratio,
        "correct_dynamic_0.1": cd_ratio_01,
        "correct_dynamic_0.05": cd_ratio_005,
        "pred_j3d_glob": pred_j3d_glob,
    }
    global_metrics.update(local_metrics)
    return global_metrics


def compute_2dkp_metrics(smpl_skeleton, smpl_mesh, J_regressor, J_regressor_25, batch, enable_timer=False):
    """
    Args:
        smpl_skeleton: SMPL skeleton model
        smpl_mesh: SMPL mesh model
        J_regressor_25: Joint regressor matrix (25, 6890)
        batch (dict): {
            "gt": {
                "keypoints3d": (bs, F, 133, 3) tensor, COCO133 format, last dimension is (x, y, confidence)
                "K": (bs, 3, 3) tensor, camera intrinsics matrix
            }
            "pred": {
                "rot6d": (bs, F, J, 6) tensor
                "shapes": (bs, F, 16) tensor
                "trans": (bs, F, 3) tensor
            }
        }
        enable_timer: bool, whether to enable timing statistics
    Returns:
        metrics (dict): {
            "reproj_error_px": Mean Per Joint Reprojection Error (pixels)
            "pck@0.05 ": Percentage of Correct Keypoints @ threshold 0.05
            "pck@0.1 ": Percentage of Correct Keypoints @ threshold 0.1
            "pck@0.2 ": Percentage of Correct Keypoints @ threshold 0.2
            "pnp_success_rate (%)": Percentage of successful PnP
        }
    """
    import time

    import cv2

    from .vertex_ids import coco133tobody25, smpl_to_openpose

    timings = {}

    def timer_start():
        if enable_timer:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return time.perf_counter()
        return None

    def timer_end(name, start_time):
        if enable_timer and start_time is not None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time
            timings[name] = elapsed

    # Step 1: Get 3D joints from predicted SMPL params via J_regressor @ vertices
    t0 = timer_start()
    pred_verts = get_vertices_from_smpl_params(smpl_mesh, batch["pred"])
    pred_verts_glob = pred_verts["global_vertices"]  # (bs*F, 6890, 3)

    pred_j3d = torch.einsum("jv,bvc->bjc", J_regressor_25, pred_verts_glob)  # (bs*F, num_joints, 3) 25,3

    # pred_j3d = torch.cat([pred_j3d_all, pred_j3d_extra], dim=1)  # (bs*F, num_joints, 3) 35, 3

    # Reshape to (bs, F, num_joints, 3)
    rot6d = batch["pred"]["rot6d"]
    bs, num_frames = rot6d.shape[:2]
    num_regressor_joints = pred_j3d.shape[1]
    pred_j3d = pred_j3d.reshape(bs, num_frames, num_regressor_joints, 3)

    timer_end("1_get_pred_joints", t0)

    # Step 2: Extract GT 2D keypoints and convert to Body25 format
    gt_kp2d_coco133 = batch["gt"]["keypoints3d"]  # (bs, F, 133, 3)
    K_mat = batch["gt"]["K"]  # (bs, 3, 3)

    bs, num_frames, num_kps, _ = gt_kp2d_coco133.shape

    # Convert COCO133 to Body25 for each batch and frame
    t0 = timer_start()
    gt_kp2d_body25_list = []
    for b in range(bs):
        # coco133tobody25 expects (F, 133, 3)
        kp_np = gt_kp2d_coco133[b].cpu().numpy()  # (F, 133, 3)
        kp_body25 = coco133tobody25(kp_np)  # (F, 25, 3)
        gt_kp2d_body25_list.append(kp_body25)
    gt_kp2d_body25 = np.stack(gt_kp2d_body25_list, axis=0)  # (bs, F, 25, 3)
    timer_end("2_convert_to_body25", t0)

    gt_xy = gt_kp2d_body25[..., :2]  # (bs, F, 25, 2)
    gt_conf = gt_kp2d_body25[..., 2]  # (bs, F, 25)

    # Step 3: PnP alignment and projection for each sample and frame
    t0 = timer_start()

    all_proj_kp2d = []
    all_pnp_success = []

    K_mat_np = K_mat.cpu().numpy().astype(np.float64)
    K_per_frame = K_mat_np.ndim == 4  # (bs, F, 3, 3)

    for b in range(bs):
        sample_proj = []
        sample_success = []

        for f in range(num_frames):
            # 获取当前帧的相机内参
            if K_per_frame:
                K = np.ascontiguousarray(K_mat_np[b, f])  # (3, 3)
            else:
                K = np.ascontiguousarray(K_mat_np[b])  # (3, 3)

            # Get corresponding 2D (Body25) and 3D (SMPL regressor joints) points
            # pts_2d: 全部25个Body25关键点
            # pts_3d: 通过smpl_to_openpose映射到的SMPL关节
            pts_2d = gt_xy[b, f].astype(np.float64)  # (25, 2)
            pts_3d = pred_j3d[b, f].cpu().numpy().astype(np.float64)  # (25, 3)
            conf = gt_conf[b, f]  # (25,)

            # Filter by confidence
            valid_mask = conf > 0.3
            if valid_mask.sum() < 4:
                # Not enough points for PnP, use all points with conf > 0
                valid_mask = conf > 0

            if valid_mask.sum() < 4:
                # Still not enough, skip this frame
                sample_proj.append(np.zeros((25, 2)))
                sample_success.append(False)
                continue

            pts_2d_valid = np.ascontiguousarray(pts_2d[valid_mask])
            pts_3d_valid = np.ascontiguousarray(pts_3d[valid_mask])

            # Solve PnP with RANSAC
            dist_coeffs = np.zeros(4, dtype=np.float64)
            # success, rvec, tvec, inliers = cv2.solvePnPRansac(
            #     pts_3d_valid,
            #     pts_2d_valid,
            #     K,
            #     dist_coeffs,
            #     flags=cv2.SOLVEPNP_EPNP,
            #     reprojectionError=8.0,
            #     confidence=0.99,
            #     iterationsCount=100
            # )
            success, rvec, tvec = cv2.solvePnP(pts_3d_valid, pts_2d_valid, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

            if success:
                # Project all 25 Body25 joints (mapped from SMPL regressor joints) to 2D
                proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
                proj_pts_body25 = proj_pts.reshape(-1, 2)  # (25, 2)

                sample_proj.append(proj_pts_body25)
                sample_success.append(True)
            else:
                sample_proj.append(np.zeros((25, 2)))
                sample_success.append(False)

        all_proj_kp2d.append(np.stack(sample_proj, axis=0))  # (F, 25, 2)
        all_pnp_success.append(sample_success)

    proj_kp2d = np.stack(all_proj_kp2d, axis=0)  # (bs, F, 25, 2)
    pnp_success = np.array(all_pnp_success)  # (bs, F)

    timer_end("3_pnp_alignment", t0)

    # Step 4: Compute metrics for all 25 Body25 joints
    t0 = timer_start()

    errors = []
    pck_05_list = []
    pck_10_list = []
    pck_20_list = []

    # Pre-compute bbox sizes for all frames
    bbox_sizes = np.zeros((bs, num_frames))
    for b in range(bs):
        for f in range(num_frames):
            visible_kps = gt_conf[b, f] > 0.3
            if visible_kps.sum() > 1:
                visible_xy = gt_xy[b, f, visible_kps]
                bbox_sizes[b, f] = max(
                    visible_xy[:, 0].max() - visible_xy[:, 0].min(), visible_xy[:, 1].max() - visible_xy[:, 1].min()
                )

    for body25_idx in list(range(25)):
        pred_2d = proj_kp2d[:, :, body25_idx, :]  # (bs, F, 2)
        gt_2d = gt_xy[:, :, body25_idx, :]  # (bs, F, 2)
        conf = gt_conf[:, :, body25_idx]  # (bs, F)

        # Compute per-joint error
        err = np.sqrt(((pred_2d - gt_2d) ** 2).sum(axis=-1))  # (bs, F)

        # Mask by confidence and PnP success
        valid = (conf > 0.3) & pnp_success

        if valid.sum() > 0:
            errors.append(err[valid])

            # Compute PCK with pre-computed bbox sizes
            for b in range(bs):
                for f in range(num_frames):
                    if valid[b, f] and bbox_sizes[b, f] > 0:
                        normalized_err = err[b, f] / bbox_sizes[b, f]
                        pck_05_list.append(normalized_err < 0.05)
                        pck_10_list.append(normalized_err < 0.1)
                        pck_20_list.append(normalized_err < 0.2)

    timer_end("4_compute_metrics", t0)

    # Aggregate metrics
    if len(errors) > 0:
        all_errors = np.concatenate(errors)
        reproj_error = float(all_errors.mean())  # 单位: 像素 (pixels)
    else:
        reproj_error = float("nan")

    # PCK 用百分比显示 (0-100%)
    pck_05 = float(np.mean(pck_05_list)) * 100 if len(pck_05_list) > 0 else float("nan")
    pck_10 = float(np.mean(pck_10_list)) * 100 if len(pck_10_list) > 0 else float("nan")
    pck_20 = float(np.mean(pck_20_list)) * 100 if len(pck_20_list) > 0 else float("nan")

    kp2d_metrics = {
        "reproj_error_px": reproj_error,  # Mean Per Joint Reprojection Error (pixels)
        "pck@0.05 ": pck_05,  # Percentage of Correct Keypoints @ threshold 0.05
        "pck@0.1 ": pck_10,
        "pck@0.2 ": pck_20,
        "pnp_success_rate (%)": float(pnp_success.mean()) * 100,
    }

    if enable_timer:
        print("Timings:", timings)

    return kp2d_metrics, proj_kp2d, gt_xy, gt_conf


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)
