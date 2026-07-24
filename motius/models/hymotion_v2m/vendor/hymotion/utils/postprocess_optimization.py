from __future__ import annotations
from typing import Any


import os
import torch
import numpy as np
import torch.nn.functional as F

from .postprocess import end_vel_to_static_conf
from ..datasets.geometry import matrix_to_rotation_6d, rotation_6d_to_matrix
from ..evaluation.metrics import get_joints_from_smpl_params, get_vertices_from_smpl_params


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def post_optimization(
    body_model,
    sparse_body_model,
    output,
    batch,
    fps=30,
    run_post_opt_cam=True,
    postopt_lr=1e-2,
    opt_contact=True,
    loss_kp_w=0.5
):
    # optimize global translation, camera height, and camera orientation using reprojection error
    # print(batch["meta"].keys())
    # # ['start_frame', 'sequence_name', 'feature_name', 'motion_name', 'camera_name', 'video_name', 'camera_origin_K', 'camera_origin_RT', 'camera_wv_RT']

    # pred_cam = batch["meta"]['camera_wv_RT']   # (B, L, 4, 4)
    # Rwc = torch.tensor(pred_cam['Rwc']).float().cuda()
    # Twc = torch.tensor(pred_cam['Twc']).float().cuda()

    # img_focal = pred_cam['img_focal']
    # img_center = pred_cam['img_center']
    # K = np.eye(3)
    # K[0, 0] = img_focal
    # K[1, 1] = img_focal
    # K[0, 2] = img_center[0]
    # K[1, 2] = img_center[1]
    # K = torch.tensor(K).float().cuda()
    K = batch["meta"]["camera_origin_K"]    # (B, L, 4, 4)
    device = K.device

    # convert from dict to mperson
    bs, seq_len, _, _ = K.shape

    bboxes = batch["inputs"]["feature"]["bbox_value"]   # (B, L, 4)
    pred_kp_2d = batch["inputs"]["feature"]["keypoints2d"]  # (B, L, 25, 3)
    transl_init = output["trans"]   # (B, L, 3)
    mask = torch.ones(bs, seq_len).to(device)  # (B, L)
    smplh_pose = output["rot6d"]    # (B, L, 52, 6)
    smplh_betas = output["shapes"]  # (B, L, 16)
    rcw = batch["meta"]['camera_wv_RT'][:, :, :3, :3]   # (B, L, 3, 3)
    tcw = batch["meta"]['camera_wv_RT'][:, :, :3, -1]   # (B, L, 3)

    bbox_height = bboxes[:, :, 3] - bboxes[:, :, 1]
    bbox_height = bbox_height.clamp(min=1e-6)

    transl_init.requires_grad = False
    transl = torch.zeros_like(transl_init)
    transl.requires_grad = True

    cam_height_offset = torch.zeros(1).to(device)
    cam_init_r6d = matrix_to_rotation_6d(torch.eye(3)).to(device)

    if run_post_opt_cam:
        cam_height_offset.requires_grad = True
        cam_init_r6d.requires_grad = True
        optim = torch.optim.Adam([transl, cam_height_offset, cam_init_r6d], lr=postopt_lr)
    else:
        cam_height_offset.requires_grad = False
        cam_init_r6d.requires_grad = False
        optim = torch.optim.Adam([transl], lr=postopt_lr)

    contact_joint_ids = [7, 10, 8, 11] #, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    joint_mapping = np.array([
        55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
        8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
        63, 64, 65,
    ], dtype=np.int32)

    ignore_joints = [9, 8, 12]
    joint_idxs = [i for i in range(25) if i not in ignore_joints]

    with torch.no_grad():
        B = bs * seq_len
        # smplx_out = smplx(
        #     body_pose=smplx_pose[:, :, 3:66].reshape(-1, 21*3),
        #     global_orient=smplx_pose[:, :, :3].reshape(-1, 3),
        #     betas=smplx_betas.reshape(-1, 10),
        #     left_hand_pose=smplx_pose[:, :, 75:120].reshape(-1, 15*3),
        #     right_hand_pose=smplx_pose[:, :, 120:165].reshape(-1, 15*3),
        #     transl=transl_init.reshape(-1, 3),
        #     jaw_pose=torch.zeros(B, 3).to(smplx_pose),
        #     leye_pose=torch.zeros(B, 3).to(smplx_pose),
        #     reye_pose=torch.zeros(B, 3).to(smplx_pose),
        #     expression=torch.zeros(B, 10).to(smplx_pose),
        #     pose2rot=True
        # )

        # joints = smplx_out.joints[:, joint_mapping]
        # contact_joints = smplx_out.joints[:, contact_joint_ids]
        # joints = torch.cat([joints, contact_joints], dim=1)
        # joints = joints.reshape(bs, seq_len, -1, 3)

        joints = get_vertices_from_smpl_params(sparse_body_model, output)["global_vertices"].reshape(bs, seq_len, 25, 3)    # (B, L, 25, 3)
        contact_joints = get_joints_from_smpl_params(body_model, output)["global_joints"][:, :, contact_joint_ids]  # (B, L, 4, 3)
        joints = torch.cat([joints, contact_joints], dim=2)

    loss_dict = {}
    for i in range(1000):
        optim.zero_grad()

        j_world = joints + transl[:, :, None, :]
        cam_init_rotmat = rotation_6d_to_matrix(cam_init_r6d)
        rcw_mod = cam_init_rotmat[None, None, :, :] @ rcw
        tcw_mod = (cam_init_rotmat[None, None, :, :] @ tcw[:, :, :, None])[..., 0]
        tcw_mod[:, :, 1] += cam_height_offset

        j_cam = (rcw_mod[:, :, None] @ j_world[..., None])[..., 0] + tcw_mod[:, :, None]  # camera space
        # print(K[None, None, None, ...].shape)
        # print(j_cam[..., None].shape)
        # exit()

        # Correct perspective projection
        # pj = (K[None, None, None, ...] @ j_cam[..., None])[..., 0]  # apply intrinsics first
        # 1, 300, 3, 3  @ 1, 300, 29, 3
        pj = torch.einsum("blij,blnj->blni", K, j_cam)  # apply intrinsics first
        pj = pj / (pj[..., 2:3] + 1e-6)  # then divide by z
        pj = pj[..., :2]  # get x,y coordinates

        # loss_kp_w = 0.5
        loss_smooth_w = 0.25
        loss_cont_vel_w = 1e3
        loss_cont_height_w = 10.0
        loss_below_floor_w = 1e2

        gt = pred_kp_2d[..., :2]
        loss_kp = gmof(pj[:, :, joint_idxs] - gt[:, :, joint_idxs]) / bbox_height[:, :, None, None]
        loss_kp = loss_kp * mask[:, :, None, None]
        loss_kp = ((pred_kp_2d[:, :, joint_idxs, 2] > 0.9) * loss_kp.mean(-1)).mean(-1).sum() / mask.sum()
        loss_kp = loss_kp_w * loss_kp

        f_transl = transl_init + transl
        loss_smooth = ((f_transl[:, 1:] - f_transl[:, :-1]) * fps).pow(2).mean(-1)
        loss_smooth = (loss_smooth * mask[:, 1:]).sum() / mask[:, 1:].sum()
        loss_smooth_vel = loss_smooth_w * loss_smooth

        loss_smooth = torch.linalg.norm((f_transl[:, 2:] + f_transl[:, :-2] - 2 * f_transl[:, 1:-1]) * fps, dim=-1)
        loss_smooth = (loss_smooth * mask[:, 1:-1]).sum() / mask[:, 1:-1].sum()
        loss_smooth_acc = loss_smooth_w * loss_smooth
        loss_smooth = loss_smooth_vel + loss_smooth_acc

        if opt_contact:
            # contact vel loss
            end_vel = output["end_effector_vel"].clone()[:, :-1] / fps  # (B, L-1, 6, 3)
            contacts_conf = end_vel_to_static_conf(end_vel)[..., :len(contact_joint_ids)] # (B, L-1, 4)
            # contacts_conf = torch.sigmoid(pred_contact)[..., :len(contact_joint_ids)]

            delta_pos = (j_world[:, 1:, 25:] - j_world[:, :-1, 25:])**2 # (B, L-1, 4, 3)
            loss_contact_vel = ((delta_pos.sum(dim=-1) * contacts_conf) * mask[:, 1:, None]).mean(-1).sum() / mask[:, 1:].sum()
            loss_contact_vel = loss_cont_vel_w * loss_contact_vel

            # contact height loss
            floor_diff = torch.abs(j_world[:, :-1, 25:, 1] - 0.08)
            loss_contact_height = ((floor_diff * contacts_conf) * mask[:, 1:, None]).mean(-1).sum() / mask[:, 1:].sum()
            loss_contact_height = loss_cont_height_w * loss_contact_height

        # no joints below the floor
        floor_diff = F.relu(-j_world[:, :, :, 1]) * mask[:, :, None]
        floor_diff = floor_diff.mean(-1).sum() / mask.sum()
        loss_below_floor = loss_below_floor_w * floor_diff

        loss = loss_kp + loss_smooth + loss_below_floor

        if opt_contact:
            loss += loss_contact_vel + loss_contact_height

        if 'loss' not in loss_dict.keys():
            loss_dict['loss'] = []
            loss_dict['loss_kp'] = []
            loss_dict['loss_smooth'] = []
            loss_dict['loss_below_floor'] = []
            if opt_contact:
                loss_dict['loss_contact_vel'] = []
                loss_dict['loss_contact_height'] = []

        loss_dict['loss'].append(loss.item())
        loss_dict['loss_kp'].append(loss_kp.item())
        loss_dict['loss_smooth'].append(loss_smooth.item())
        loss_dict['loss_below_floor'].append(loss_below_floor.item())

        if opt_contact:
            loss_dict['loss_contact_vel'].append(loss_contact_vel.item())
            loss_dict['loss_contact_height'].append(loss_contact_height.item())

        loss.backward()
        optim.step()

        # print("=" * 60)
        # print(loss_dict)
        # print("=" * 60)
        # breakpoint()

    final_transl = transl_init + transl # (B, L, 3)
    output["trans"] = final_transl

    # for pidx, person_id in enumerate(people_ids):
    #     transl_p = final_transl[pidx][mask[pidx].bool()].detach().cpu().numpy()
    #     output['people'][person_id][f'smplx_world']['trans'] = transl_p

    # cam_init_rotmat = cam_init_rotmat.detach().cpu().numpy()
    # Tcw = output['camera_world']['Tcw']
    # Tcw = (cam_init_rotmat[None, :, :] @ Tcw[..., None])[..., 0]
    # Tcw[:, 1] += cam_height_offset.clone().detach().cpu().numpy()
    # Rcw = output['camera_world']['Rcw']
    # Rcw = cam_init_rotmat[None, :, :] @ Rcw

    # Rwc = Rcw.transpose(0, 2, 1)
    # Twc = -(Rcw @ Tcw[..., None])[..., 0]  # Twc = -R_cw * t_cw

    # output['camera_world']['Rcw'] = Rcw
    # output['camera_world']['Tcw'] = Tcw
    # output['camera_world']['Rwc'] = Rwc
    # output['camera_world']['Twc'] = Twc

    return output