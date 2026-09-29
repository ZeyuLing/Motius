from __future__ import annotations
import torch
from torch.cuda.amp import autocast
import numpy as np
import os
from datetime import datetime

from ..datasets.geometry import rotation_matrix_to_rot6d, rot6d_to_rotation_matrix

from . import matrix
from .ccd_ik import CCD_IK
from .ccd_ik_full import CCDIKFull, Activation
from ..utils.net_utils import gaussian_smooth
from ..evaluation.metrics import get_joints_from_smpl_params, get_fkmat_from_smpl_params


def _save_ik_debug_comparison(fk_j3d_original, post_target_j3d, fk_j3d_after_ik, joint_ids, joint_names, debug_dir, timestamp):
    """Save a combined comparison plot overlaying original FK, IK target, and post-IK FK trajectories.

    Args:
        fk_j3d_original: (B, L, J, 3) original FK joint positions before any processing
        post_target_j3d: (B, L, J, 3) smoothed IK target positions
        fk_j3d_after_ik: (B, L, J, 3) FK joint positions after IK solve
        joint_ids: list of joint indices to visualize
        joint_names: list of joint names corresponding to joint_ids
        debug_dir: directory to save the debug plots
        timestamp: timestamp string for unique filenames
    """
    import matplotlib.pyplot as plt

    os.makedirs(debug_dir, exist_ok=True)
    axis_names = ['X', 'Y', 'Z']

    data_orig = fk_j3d_original[0].cpu().numpy()   # (L, J, 3)
    data_target = post_target_j3d[0].cpu().numpy()  # (L, J, 3)
    data_after = fk_j3d_after_ik[0].cpu().numpy()   # (L, J, 3)

    num_joints = len(joint_ids)
    fig, axes = plt.subplots(3, num_joints, figsize=(num_joints * 3, 9), squeeze=False)
    fig.suptitle('End-Effector Trajectories: Original FK vs IK Target vs Post-IK FK', fontsize=13)

    for col_idx, (jid, jname) in enumerate(zip(joint_ids, joint_names)):
        for axis_idx in range(3):
            ax = axes[axis_idx, col_idx]
            ax.plot(data_orig[:, jid, axis_idx], linewidth=0.8, label='Original FK', alpha=0.7)
            ax.plot(data_target[:, jid, axis_idx], linewidth=0.8, label='IK Target', linestyle='--')
            ax.plot(data_after[:, jid, axis_idx], linewidth=0.8, label='Post-IK FK', alpha=0.7)
            if axis_idx == 0:
                ax.set_title(f'{jname} (j{jid})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{axis_names[axis_idx]} axis', fontsize=10)
            if axis_idx == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.grid(True, alpha=0.3)
            if axis_idx == 0 and col_idx == num_joints - 1:
                ax.legend(fontsize=7)

    # unify ylim across all joints for each axis (each row shares the same y range)
    for axis_idx in range(3):
        y_min = min(axes[axis_idx, col].get_ylim()[0] for col in range(num_joints))
        y_max = max(axes[axis_idx, col].get_ylim()[1] for col in range(num_joints))
        for col in range(num_joints):
            axes[axis_idx, col].set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = os.path.join(debug_dir, f'{timestamp}_ik_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[process_ik debug] Saved IK comparison to {save_path}")


def save_fk_end_effector_debug(body_model, output, name="fk_end_effector", debug_dir="debug/process_ik",
                               joint_ids=None, joint_names=None):
    """Do FK and save end-effector joint trajectories visualization.

    Args:
        body_model: SMPL body model
        output: dict with 'rot6d', 'trans', 'shapes'
        name: prefix for the saved file
        debug_dir: directory to save debug plots
        joint_ids: list of joint indices to visualize (default: foot + wrist)
        joint_names: list of joint names (default: matches joint_ids)
    """
    import matplotlib.pyplot as plt

    if joint_ids is None:
        joint_ids = [7, 10, 8, 11, 20, 21]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot", "L_Wrist", "R_Wrist"]
    if joint_names is None:
        joint_names = [f"Joint_{jid}" for jid in joint_ids]

    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    data = fk_j3d[0].detach().cpu().numpy()  # (L, J, 3)
    axis_names = ['X', 'Y', 'Z']

    num_joints = len(joint_ids)
    fig, axes = plt.subplots(3, num_joints, figsize=(num_joints * 3, 9), squeeze=False)
    fig.suptitle(f'{name}: End-Effector Positions over Time', fontsize=13)

    for col_idx, (jid, jname) in enumerate(zip(joint_ids, joint_names)):
        for axis_idx in range(3):
            ax = axes[axis_idx, col_idx]
            ax.plot(data[:, jid, axis_idx], linewidth=0.8)
            if axis_idx == 0:
                ax.set_title(f'{jname} (j{jid})', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{axis_names[axis_idx]} axis', fontsize=10)
            if axis_idx == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.grid(True, alpha=0.3)

    # unify ylim across all joints for each axis
    for axis_idx in range(3):
        y_min = min(axes[axis_idx, col].get_ylim()[0] for col in range(num_joints))
        y_max = max(axes[axis_idx, col].get_ylim()[1] for col in range(num_joints))
        for col in range(num_joints):
            axes[axis_idx, col].set_ylim(y_min, y_max)

    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(debug_dir, f'{timestamp}_{name}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[debug] Saved FK end-effector visualization to {save_path}")


@autocast(enabled=False)
def pp_static_joint(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    # fowrward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    L = fk_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]  # (B, L, 6, 3)

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]  # (B, L-1, 6, 3)
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1] / fps  # (B, L-1, 6, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]  # (B, L-1, 3)

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    if False:
        # NOTE: 元旦前的版本，逻辑正确性上有点问题，fk_pred_diff 置零的部分也参与了mean计算
        # 但是在 wa，wa2，rte等关键指标上表现会好一点
        fk_pred_diff = fk_pred_diff.mean(dim=-2)
    else:
        # NOTE: 元旦后更新的版本，逻辑正确，fk_pred_diff 置零的部分不参与trans更新
        # 但是在 wa，wa2，rte等关键指标上表现会差一点，pampjpe、正确静止和正确运动的指标好一点
        non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)  # (B, L-1, 1)
        # fk_pred_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(non_dynamic_cnt, min=1.0)
        # 增加一个与 non_dynamic_cnt 相关的衰减系数
        valid_avg_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(
            non_dynamic_cnt, min=1.0
        )  # if all joints are dynamic in current frame，clamp denominator to 1
        total_joints = fk_pred_diff.shape[-2]
        confidence = non_dynamic_cnt / total_joints
        fk_pred_diff = valid_avg_diff * confidence

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def pp_static_joint_footonly(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    # fowrward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    L = fk_j3d.shape[1]
    joint_ids = [7, 10, 8, 11]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]  # (B, L, 4, 3)

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]  # (B, L-1, 4, 3)
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :4] / fps  # (B, L-1, 4, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]  # (B, L-1, 3)

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    if False:
        # NOTE: 元旦前的版本，逻辑正确性上有点问题，fk_pred_diff 置零的部分也参与了mean计算
        # 但是在 wa，wa2，rte等关键指标上表现会好一点
        fk_pred_diff = fk_pred_diff.mean(dim=-2)
    else:
        # NOTE: 元旦后更新的版本，逻辑正确，fk_pred_diff 置零的部分不参与trans更新
        # 但是在 wa，wa2，rte等关键指标上表现会差一点，pampjpe、正确静止和正确运动的指标好一点
        non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)  # (B, L-1, 1)
        # fk_pred_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(non_dynamic_cnt, min=1.0)
        # 增加一个与 non_dynamic_cnt 相关的衰减系数
        valid_avg_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(
            non_dynamic_cnt, min=1.0
        )  # if all joints are dynamic in current frame，clamp denominator to 1
        total_joints = fk_pred_diff.shape[-2]
        confidence = non_dynamic_cnt / total_joints
        fk_pred_diff = valid_avg_diff * confidence

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def pp_static_joint_footonly_v2(body_model, output, fps=30, l_thres=1e-2, u_thres=5e-2, replace_ground_y=True):
    """Fixed version of pp_static_joint_footonly.
    Fix: the old confidence scaling (sum/cnt * cnt/N) was mathematically equivalent to mean (sum/N),
    so dynamic joints with zeroed fk_pred_diff still diluted the average. Now we use sum/cnt directly
    so that only non-dynamic joints contribute to the average correction.
    """
    # forward kinematics to get global end effector joints
    fk_j3d = get_joints_from_smpl_params(body_model, output, joint_num=52)["global_joints"]
    joint_ids = [7, 10, 8, 11]  # [L_Ankle, L_foot, R_Ankle, R_foot]
    fk_end_j3d = fk_j3d.clone()[:, :, joint_ids]  # (B, L, 4, 3)

    # calculate end effector velocity of fk and prediction, also the root translation velocity
    fk_end_vel = fk_end_j3d[:, 1:] - fk_end_j3d[:, :-1]  # (B, L-1, 4, 3)
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :4] / fps  # (B, L-1, 4, 3)
    trans = output["trans"].clone()  # (B, L, 3)
    root_vel = trans[:, 1:] - trans[:, :-1]  # (B, L-1, 3)

    # determine static and dynamic frames by thresholding predicted end effector velocity
    static_label_ = pred_end_vel.norm(2, dim=-1) < l_thres
    dynamic_label_ = pred_end_vel.norm(2, dim=-1) > u_thres

    # for static frames (< l_thres), zero-out predicted end effector velocity
    # for dynamic frames (> u_thres), ignore the fk_pred_diff
    pred_end_vel = pred_end_vel - (static_label_[..., None] * pred_end_vel)
    fk_pred_diff = pred_end_vel - fk_end_vel
    fk_pred_diff = fk_pred_diff - (dynamic_label_[..., None] * fk_pred_diff)

    # average fk_pred_diff over non-dynamic joints only (dynamic joints are zeroed and excluded)
    non_dynamic_cnt = (~dynamic_label_).float().sum(dim=-1, keepdim=True)  # (B, L-1, 1)
    fk_pred_diff = fk_pred_diff.sum(dim=-2) / torch.clamp(non_dynamic_cnt, min=1.0)  # (B, L-1, 3)

    # update root translation to make fk close to predicted end effector velocity
    root_vel_new = root_vel + fk_pred_diff
    updated_trans = torch.cumsum(torch.cat([trans[:, :1], root_vel_new], dim=1), dim=1)

    # gaussian smooth x and z of updated root translation
    updated_trans[..., 0] = gaussian_smooth(updated_trans[..., 0], dim=-1)
    updated_trans[..., 2] = gaussian_smooth(updated_trans[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    updated_fk_j3d = fk_j3d - trans.unsqueeze(-2) + updated_trans.unsqueeze(-2)
    if replace_ground_y:
        ground_y = updated_fk_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
        updated_trans[..., 1] -= ground_y[:, None]

    return updated_trans


@autocast(enabled=False)
def process_ik(body_model, output, fps=30, use_fk_vel=False,
               debug=False, force_align_to_first_frame=True, use_ccd_ik_full=True, optimize_wrist=False):
    # forward kinematics to get global joint positions and joint rotmat
    fk_j3d, local_rotmat, fk_rotmat = get_fkmat_from_smpl_params(body_model, output)

    if optimize_wrist:
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot", "L_Wrist", "R_Wrist"]
    else:
        joint_ids = [7, 10, 8, 11]  # [L_Ankle, L_Foot, R_Ankle, R_Foot]
        joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot"]

    num_ee = len(joint_ids)

    # determine magnitude of predicted end effector velocity
    if use_fk_vel:
        fk_end_vel = fk_j3d[:, 1:, joint_ids] - fk_j3d[:, :-1, joint_ids]
        end_vel_mag = fk_end_vel.norm(2, dim=-1)
        pred_end_vel = fk_end_vel  # for static_conf computation
    else:
        pred_end_vel = output["end_effector_vel"].clone()[:, :-1, :num_ee] / fps  # (B, L-1, num_ee, 3)
        end_vel_mag = pred_end_vel.norm(2, dim=-1)

    # non-linear mapping from end vel to static confidence by transformed sigmoid
    static_conf = end_vel_to_static_conf(pred_end_vel)

    # save original FK for debug comparison (before any target smoothing)
    if debug:
        fk_j3d_original = fk_j3d.clone()

    post_target_j3d = fk_j3d.clone()
    for i in range(1, fk_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = fk_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None].float()
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1.0 - c_prev)

    # force_align_to_first_frame: 假设第一帧静止，当脚尖或脚跟在第一帧下方时
    # 计算需要往上挪多少才能让脚尖和脚跟都挪上去，然后脚尖脚跟都移动这个数值
    if force_align_to_first_frame:
        # ============ 左脚处理 ============
        first_frame_l_ankle_y = fk_j3d[:, 0, 7, 1]   # (B,)
        first_frame_l_foot_y = fk_j3d[:, 0, 10, 1]    # (B,)
        curr_l_ankle_y = post_target_j3d[:, :, 7, 1]   # (B, L)
        curr_l_foot_y = post_target_j3d[:, :, 10, 1]   # (B, L)
        l_ankle_diff = torch.clamp(first_frame_l_ankle_y[:, None] - curr_l_ankle_y, min=0)
        l_foot_diff = torch.clamp(first_frame_l_foot_y[:, None] - curr_l_foot_y, min=0)
        l_shift_y = torch.max(l_ankle_diff, l_foot_diff)  # (B, L)
        post_target_j3d[:, :, 7, 1] += l_shift_y   # L_Ankle
        post_target_j3d[:, :, 10, 1] += l_shift_y  # L_Foot

        # ============ 右脚处理 ============
        first_frame_r_ankle_y = fk_j3d[:, 0, 8, 1]    # (B,)
        first_frame_r_foot_y = fk_j3d[:, 0, 11, 1]    # (B,)
        curr_r_ankle_y = post_target_j3d[:, :, 8, 1]   # (B, L)
        curr_r_foot_y = post_target_j3d[:, :, 11, 1]   # (B, L)
        r_ankle_diff = torch.clamp(first_frame_r_ankle_y[:, None] - curr_r_ankle_y, min=0)
        r_foot_diff = torch.clamp(first_frame_r_foot_y[:, None] - curr_r_foot_y, min=0)
        r_shift_y = torch.max(r_ankle_diff, r_foot_diff)  # (B, L)
        post_target_j3d[:, :, 8, 1] += r_shift_y   # R_Ankle
        post_target_j3d[:, :, 11, 1] += r_shift_y  # R_Foot

    # ik
    global_rot = matrix.get_rotation(fk_rotmat)
    parents = body_model.parents[:22]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        if use_ccd_ik_full:
            IK_solver = CCDIKFull(
                local_mat,
                parents.tolist(),
                target_ind,
                target_pos=target_pos,
                target_rot=target_rot,
                kinematic_chain=chain,
                iterations=25,
                threshold=0.001,
                activation=Activation.LINEAR,
                pos_weight=1.0,
                rot_weight=0.0,
                debug=True,
            )

            # --- DEBUG: before IK solve ---
            pre_ik_positions = []
            for ti_idx, ti in enumerate(target_ind):
                ee_pos = IK_solver.get_global_position(ti)  # remapped index
                tgt_pos = target_pos[..., ti_idx, :]
                dist = (ee_pos - tgt_pos).norm(dim=-1).mean().item()
                pre_ik_positions.append(dist)
            print(f"[DEBUG IK] chain={chain}, target_ind={target_ind}")
            print(f"[DEBUG IK]   BEFORE solve: ee-target distances = {['%.6f' % d for d in pre_ik_positions]}")

            chain_local_mat = IK_solver.solve()

            # --- DEBUG: after IK solve ---
            post_ik_positions = []
            for ti_idx, ti in enumerate(target_ind):
                ee_pos = IK_solver.get_global_position(ti)  # remapped index
                tgt_pos = target_pos[..., ti_idx, :]
                dist = (ee_pos - tgt_pos).norm(dim=-1).mean().item()
                post_ik_positions.append(dist)
                print(f"[DEBUG IK]   AFTER  solve: target[{ti_idx}] (chain joint {ti}): "
                      f"ee_pos_mean={ee_pos.mean(dim=tuple(range(ee_pos.dim()-1))).tolist()}, "
                      f"tgt_pos_mean={tgt_pos.mean(dim=tuple(range(tgt_pos.dim()-1))).tolist()}, "
                      f"dist={dist:.6f}")
            print(f"[DEBUG IK]   AFTER  solve: ee-target distances = {['%.6f' % d for d in post_ik_positions]}")
            print(f"[DEBUG IK]   converged={IK_solver.is_converged()}, threshold={IK_solver.threshold}")
            print()

            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        else:
            IK_solver = CCD_IK(
                local_mat,
                parents,
                target_ind,
                target_pos,
                target_rot,
                kinematic_chain=chain,
                max_iter=2,
                reg_weight=0.,
            )
            chain_local_mat = IK_solver.solve()
            chain_rotmat = matrix.get_rotation(chain_local_mat)
            local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    # foot IK (always applied)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)

    # wrist IK (only when optimize_wrist is enabled)
    if optimize_wrist:
        local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
        local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = rotation_matrix_to_rot6d(matrix.get_rotation(local_rotmat[:, :, 1:]))  # (B, L, J-1, 6)
    new_pose = torch.cat([output['rot6d'][..., :1, :], body_pose], dim=-2)
    fk_j3d_after_ik, _, _ = get_fkmat_from_smpl_params(body_model, {
        "rot6d": new_pose,
        "trans": output["trans"],
        "shapes": output["shapes"],
    })

    # debug: save comparison of pre-IK and post-IK end-effector trajectories
    if debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join("debug", "process_ik")
        _save_ik_debug_comparison(
            fk_j3d_original, post_target_j3d, fk_j3d_after_ik,
            joint_ids, joint_names, debug_dir, timestamp,
        )

    return body_pose


@autocast(enabled=False)
def process_ik_mc(body_model, output, fps=30):
    # fowrward kinematics to get global joint positions and joint rotmat
    fk_j3d, local_rotmat, fk_rotmat = get_fkmat_from_smpl_params(body_model, output)
    joint_ids = [7, 10, 8, 11]  # [L_Ankle, L_foot, R_Ankle, R_foot]

    # determine magnitude of predicted end effector velocity
    pred_end_vel = output["end_effector_vel"].clone()[:, :-1] / fps  # (B, L-1, 6, 3)
    end_vel_mag = pred_end_vel[:, :, : len(joint_ids)].norm(2, dim=-1)  # (B, L-1, 4)

    post_target_j3d = fk_j3d.clone()
    for i in range(1, fk_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = fk_j3d[:, i, joint_ids]
        # print("prev: ", prev.shape)
        # print("end_vel_mag[:, i, None]: ", end_vel_mag[:, i][:, :, None].shape)
        # print("this: ", this.shape)
        # exit()
        post_target_j3d[:, i, joint_ids] = prev + end_vel_mag[:, i][:, :, None] * (this - prev)

    # ik
    global_rot = matrix.get_rotation(fk_rotmat)
    parents = body_model.parents[:22]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_rotmat = ik(local_rotmat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = rotation_matrix_to_rot6d(matrix.get_rotation(local_rotmat[:, :, 1:]))  # (B, L, J-1, 6)

    return body_pose


def end_vel_to_static_conf(end_vel, l_thres=1e-2, conf_percentile=80):
    end_vel_mag = end_vel.norm(2, dim=-1)

    static_conf = torch.sigmoid(end_vel_mag / l_thres)
    static_conf = 1.0 - static_conf
    static_conf = static_conf * 2
    nonzero_conf_mask = static_conf != 0
    if nonzero_conf_mask.any():
        nonzero_conf = static_conf[nonzero_conf_mask].flatten()
        perc_values = np.percentile(nonzero_conf.cpu().numpy(), conf_percentile)
        static_conf *= 1 / perc_values
        static_conf = torch.clamp(static_conf, max=1.0)

    return static_conf
