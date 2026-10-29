"""Published temporal condition dropout and camera/world supervision objectives."""
import torch
from torch.nn import functional as F
from motius.motion.representation.gvhmr.camera import get_a_pred_cam,project_to_bi01
from motius.motion.representation.prompthmr.camera import inv_compute_transl_full_cam_prompthmr,inv_compute_transl_full_cam_metric3d
from motius.motion.representation.prompthmr.video_motion import rollout_local_transl_vel,get_static_joint_mask

def randomly_set_null_condition(f_condition, uncond_prob=0.1):
    """Conditions are in shape (B, L, *)"""
    keys = list(f_condition.keys())
    for k in keys:
        if f_condition[k] is None:
            continue
        f_condition[k] = f_condition[k].clone()
        mask = torch.rand(f_condition[k].shape[:2]) < uncond_prob
        f_condition[k][mask] = 0.0
    return f_condition


def compute_extra_incam_loss(inputs, outputs, ppl):
    model_output = outputs["model_output"]
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    camt_type = ppl.args.get("camt_type", 'cliff')

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"]  # effective length mask
    mask_reproj = ~inputs["mask"]["spv_incam_only"]  # do not supervise reproj for 3DPW

    # Incam FK
    # prediction
    output_incam = outputs["pred_smpl_params_incam"]
    if camt_type == 'prompthmr':
        output_incam['transl'] = torch.zeros_like(output_incam['transl'])

    pred_c_j3d = endecoder.fk_v2(**output_incam)
    pred_cr_j3d = pred_c_j3d - pred_c_j3d[:, :, :1]  # (B, L, J, 3)

    # gt
    gt_c_j3d = endecoder.fk_v2(**inputs["smpl_params_c"])  # (B, L, J, 3)
    gt_cr_j3d = gt_c_j3d - gt_c_j3d[:, :, :1]  # (B, L, J, 3)

    # Root aligned C-MPJPE Loss
    if weights.cr_j3d > 0.0:
        cr_j3d_loss = F.mse_loss(pred_cr_j3d, gt_cr_j3d, reduction="none")
        cr_j3d_loss = (cr_j3d_loss * mask[..., None, None]).mean()
        extra_loss += cr_j3d_loss * weights.cr_j3d
        extra_loss_dict["cr_j3d_loss"] = cr_j3d_loss

    # Reprojection (to align with image)
    if weights.transl_c > 0.0:
        # pred_transl = outputs["pred_smpl_params_incam"]["transl"]  # (B, L, 3)
        # gt_transl = inputs["smpl_params_c"]["transl"]
        # transl_c_loss = F.mse_loss(pred_transl, gt_transl, reduction="none")
        # gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
        # valid_mask = (
        #     (gt_j3d_z_min > 0.3)
        # )[..., None]

        # transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()

        if camt_type == 'prompthmr':
            pred_cam = model_output["pred_cam"]  # (B, L, 3)
            gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
            gt_pred_cam = inv_compute_transl_full_cam_prompthmr(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)

            gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
            valid_mask = (
                (gt_j3d_z_min > 0.3)
                * (inputs["bbx_xys"][..., 2] > 0)
            )[..., None]
            transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
            transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()
        elif camt_type == 'metric3d':
            pred_cam = model_output["pred_cam"]  # (B, L, 3)
            gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
            gt_pred_cam = inv_compute_transl_full_cam_prompthmr(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
            gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask

            gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
            valid_mask = (
                (gt_j3d_z_min > 0.3)
                * (gt_pred_cam[..., 0] > 0.0)
                * (gt_pred_cam[..., 0] < 1.0)
                * (gt_pred_cam[..., 1] > 0.0)
                * (gt_pred_cam[..., 1] < 1.0)
                # * (gt_pred_cam[..., 2] > -3.0)
                # * (gt_pred_cam[..., 2] < 3.0)
                * (inputs["bbx_xys"][..., 2] > 0)
            )[..., None]
            transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
            transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()
        elif camt_type == 'cliff':
            # Instead of supervising transl, we convert gt to pred_cam (prevent divide 0)
            pred_cam = model_output["pred_cam"]  # (B, L, 3)
            gt_transl = inputs["smpl_params_c"]["transl"]  # (B, L, 3)
            gt_pred_cam = get_a_pred_cam(gt_transl, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 3)
            gt_pred_cam[gt_pred_cam.isinf()] = -1  # this will be handled by valid_mask
            # (compute_transl_full_cam(gt_pred_cam, inputs["bbx_xys"], inputs["K_fullimg"]) - gt_transl).abs().max()

            # Skip gts that are not good during random construction
            gt_j3d_z_min = inputs["gt_j3d"][..., 2].min(dim=-1)[0]
            valid_mask = (
                (gt_j3d_z_min > 0.3)
                * (gt_pred_cam[..., 0] > 0.3)
                * (gt_pred_cam[..., 0] < 5.0)
                * (gt_pred_cam[..., 1] > -3.0)
                * (gt_pred_cam[..., 1] < 3.0)
                * (gt_pred_cam[..., 2] > -3.0)
                * (gt_pred_cam[..., 2] < 3.0)
                * (inputs["bbx_xys"][..., 2] > 0)
            )[..., None]
            transl_c_loss = F.mse_loss(pred_cam, gt_pred_cam, reduction="none")
            transl_c_loss = (transl_c_loss * mask[..., None] * valid_mask).mean()

        extra_loss_dict["transl_c_loss"] = transl_c_loss
        extra_loss += transl_c_loss * weights.transl_c

    if weights.j2d > 0.0:
        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_j3d_z0_mask = pred_c_j3d[..., 2].abs() <= reproj_z_thr
        pred_c_j3d[pred_c_j3d_z0_mask] = reproj_z_thr
        gt_c_j3d_z0_mask = gt_c_j3d[..., 2].abs() <= reproj_z_thr
        gt_c_j3d[gt_c_j3d_z0_mask] = reproj_z_thr

        pred_j2d_01 = project_to_bi01(pred_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_j2d_01 = project_to_bi01(gt_c_j3d, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, J, 2)

        valid_mask = (
            (gt_c_j3d[..., 2] > reproj_z_thr)
            * (pred_c_j3d[..., 2] > reproj_z_thr)  # Be safe
            * (gt_j2d_01[..., 0] > 0.0)
            * (gt_j2d_01[..., 0] < 1.0)
            * (gt_j2d_01[..., 1] > 0.0)
            * (gt_j2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        j2d_loss = F.mse_loss(pred_j2d_01, gt_j2d_01, reduction="none")
        j2d_loss = (j2d_loss * mask[..., None, None] * valid_mask).mean()
        if camt_type == 'prompthmr':
            if inputs['nb_epoch'] > 5:
                extra_loss += j2d_loss * weights.j2d
                extra_loss_dict["j2d_loss"] = j2d_loss
        else:
            extra_loss += j2d_loss * weights.j2d
            extra_loss_dict["j2d_loss"] = j2d_loss

    if weights.cr_verts > 0:
        # SMPL forward
        pred_c_verts437, pred_c_j17 = endecoder.smplx_model(**output_incam)
        root_ = pred_c_j17[:, :, [11, 12], :].mean(-2, keepdim=True)
        pred_cr_verts437 = pred_c_verts437 - root_

        gt_cr_verts437 = inputs["gt_cr_verts437"]  # (B, L, 437, 3)
        cr_vert_loss = F.mse_loss(pred_cr_verts437, gt_cr_verts437, reduction="none")
        cr_vert_loss = (cr_vert_loss * mask[:, :, None, None]).mean()
        extra_loss += cr_vert_loss * weights.cr_verts
        extra_loss_dict["cr_vert_loss"] = cr_vert_loss

    if weights.verts2d > 0:
        gt_c_verts437 = inputs["gt_c_verts437"]  # (B, L, 437, 3)

        # prevent divide 0 or small value to overflow(fp16)
        reproj_z_thr = 0.3
        pred_c_verts437_z0_mask = pred_c_verts437[..., 2].abs() <= reproj_z_thr
        pred_c_verts437[pred_c_verts437_z0_mask] = reproj_z_thr
        gt_c_verts437_z0_mask = gt_c_verts437[..., 2].abs() <= reproj_z_thr
        gt_c_verts437[gt_c_verts437_z0_mask] = reproj_z_thr

        pred_verts2d_01 = project_to_bi01(pred_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])
        gt_verts2d_01 = project_to_bi01(gt_c_verts437, inputs["bbx_xys"], inputs["K_fullimg"])  # (B, L, 437, 2)

        valid_mask = (
            (gt_c_verts437[..., 2] > reproj_z_thr)
            * (pred_c_verts437[..., 2] > reproj_z_thr)  # Be safe
            * (gt_verts2d_01[..., 0] > 0.0)
            * (gt_verts2d_01[..., 0] < 1.0)
            * (gt_verts2d_01[..., 1] > 0.0)
            * (gt_verts2d_01[..., 1] < 1.0)
        )[..., None]
        valid_mask[~mask_reproj] = False  # Do not supervise on 3dpw
        verts2d_loss = F.mse_loss(pred_verts2d_01, gt_verts2d_01, reduction="none")
        verts2d_loss = (verts2d_loss * mask[..., None, None] * valid_mask).mean()
        if camt_type == 'prompthmr':
            if inputs['nb_epoch'] > 5:
                extra_loss += verts2d_loss * weights.verts2d
                extra_loss_dict["verts2d_loss"] = verts2d_loss
        else:
            extra_loss += verts2d_loss * weights.verts2d
            extra_loss_dict["verts2d_loss"] = verts2d_loss

    return extra_loss, extra_loss_dict


def compute_extra_global_loss(inputs, outputs, ppl):
    decode_dict = outputs["decode_dict"]
    endecoder = ppl.endecoder
    weights = ppl.weights
    args = ppl.args

    extra_loss_dict = {}
    extra_loss = 0
    mask = inputs["mask"]["valid"].clone()  # (B, L)
    mask[inputs["mask"]["spv_incam_only"]] = False

    if weights.transl_w > 0:
        # compute pred_transl_w by rollout
        gt_transl_w = inputs["smpl_params_w"]["transl"]
        gt_global_orient_w = inputs["smpl_params_w"]["global_orient"]
        local_transl_vel = decode_dict["local_transl_vel"]
        pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, gt_transl_w[:, [0]])

        trans_w_loss = F.l1_loss(pred_transl_w, gt_transl_w, reduction="none")
        trans_w_loss = (trans_w_loss * mask[..., None]).mean()
        extra_loss += trans_w_loss * weights.transl_w
        extra_loss_dict["transl_w_loss"] = trans_w_loss

    # Static-Conf loss
    if weights.static_conf_bce > 0:
        # Compute gt by thresholding velocity
        vel_thr = args.static_conf.vel_thr
        assert vel_thr > 0
        joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
        gt_w_j3d = endecoder.fk_v2(**inputs["smpl_params_w"])  # (B, L, J=22, 3)
        static_gt = get_static_joint_mask(gt_w_j3d, vel_thr=vel_thr, repeat_last=True)  # (B, L, J)
        static_gt = static_gt[:, :, joint_ids].float()  # (B, L, J')
        pred_static_conf_logits = outputs["model_output"]["static_conf_logits"]

        static_conf_loss = F.binary_cross_entropy_with_logits(pred_static_conf_logits, static_gt, reduction="none")
        static_conf_loss = (static_conf_loss * mask[..., None]).mean()
        extra_loss += static_conf_loss * weights.static_conf_bce
        extra_loss_dict["static_conf_loss"] = static_conf_loss

    return extra_loss, extra_loss_dict
